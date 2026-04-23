"""
Two-Stage Training for MIND: Point-wise Pre-training -> Ranking Fine-tuning

This script implements a two-stage training approach:
1. Stage 1: Point-wise SFT - Train model on Yes/No classification (learns relevance)
2. Stage 2: Ranking SFT - Fine-tune on ranking task (learns ordering)

The hypothesis is that point-wise pre-training provides better initialization
for the ranking task by first learning absolute relevance signals.

Usage:
    torchrun --nproc_per_node 4 src/sft_mind_twostage.py \
        --base_model Qwen/Qwen3-1.7B \
        --train_behaviors_path ../data/MIND/train/behaviors.tsv \
        --train_news_path ../data/MIND/train/news.tsv \
        --eval_behaviors_path ../data/MIND/dev/behaviors.tsv \
        --eval_news_path ../data/MIND/dev/news.tsv \
        --output_dir output_dir/sft_mind_twostage
"""

import os
import sys
import random
import numpy as np
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============== Stage 1: Point-wise Dataset ==============

class MINDPointwiseDataset:
    """Point-wise dataset for Stage 1 training."""

    def __init__(
        self,
        behaviors_path: str,
        news_path: str,
        tokenizer,
        max_len: int = 2048,
        sample: int = -1,
        seed: int = 42,
        max_history: int = 0,
        neg_ratio: float = 1.0,
        use_abstract: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_history = max_history if max_history > 0 else None
        self.neg_ratio = neg_ratio
        self.use_abstract = use_abstract
        self.seed = seed

        self.news = self._load_news(news_path)
        self.samples = self._load_and_expand_behaviors(behaviors_path, sample)

        print(f"[Stage1-Pointwise] Loaded {len(self.samples)} samples")

    def _load_news(self, news_path: str):
        news = {}
        with open(news_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                news_id = parts[0]
                category = parts[1] if len(parts) > 1 else ""
                title = parts[3]
                abstract = parts[4] if len(parts) > 4 else ""

                if self.use_abstract and abstract:
                    text = f"{title} {abstract}"
                else:
                    text = title

                news[news_id] = {'text': text, 'category': category}
        return news

    def _load_and_expand_behaviors(self, behaviors_path: str, sample_limit: int):
        all_samples = []
        rng = random.Random(self.seed)

        with open(behaviors_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue

                history_ids = parts[3].split()
                impressions = parts[4].split()

                if self.max_history is not None:
                    history_ids = history_ids[-self.max_history:]

                history = [self.news[nid] for nid in history_ids if nid in self.news]

                positives, negatives = [], []
                for imp in impressions:
                    if '-' not in imp:
                        continue
                    news_id, label = imp.rsplit('-', 1)
                    if news_id not in self.news:
                        continue
                    if int(label) == 1:
                        positives.append(news_id)
                    else:
                        negatives.append(news_id)

                if not positives:
                    continue

                for pos_id in positives:
                    all_samples.append({
                        'history': history,
                        'candidate': self.news[pos_id],
                        'label': 1
                    })

                num_neg = int(len(positives) * self.neg_ratio)
                if num_neg > 0 and negatives:
                    sampled = rng.sample(negatives, min(num_neg, len(negatives)))
                    for neg_id in sampled:
                        all_samples.append({
                            'history': history,
                            'candidate': self.news[neg_id],
                            'label': 0
                        })

        rng.shuffle(all_samples)
        if sample_limit > 0 and sample_limit < len(all_samples):
            all_samples = all_samples[:sample_limit]

        return all_samples

    def _build_prompt(self, history, candidate, label):
        prompt = "Role: You are a news recommendation assistant.\n"
        prompt += "Task: Determine if the candidate article matches the user's interests.\n\n"

        prompt += "User History:\n"
        if history:
            for i, h in enumerate(history, 1):
                cat = f" ({h.get('category', '')})" if h.get('category') else ""
                prompt += f"{i}. [Title] {h['text']}{cat}\n"
        else:
            prompt += "(No reading history)\n"

        prompt += "\nCandidate Article:\n"
        cat = f" ({candidate.get('category', '')})" if candidate.get('category') else ""
        prompt += f"[Title] {candidate['text']}{cat}\n"

        prompt += "\nBased on the user's reading history, is this article relevant to them?\n"
        prompt += "Answer with Yes or No.\n\nAnswer:"

        target = " Yes" if label == 1 else " No"
        return prompt, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt, target = self._build_prompt(sample['history'], sample['candidate'], sample['label'])

        full_text = prompt + target
        input_ids = self.tokenizer.encode(full_text, max_length=self.max_len, truncation=True, add_special_tokens=True)
        prompt_ids = self.tokenizer.encode(prompt, max_length=self.max_len, truncation=True, add_special_tokens=True)

        train_labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            train_labels = train_labels + [-100] * pad_len

        return {
            'input_ids': torch.tensor(input_ids[:self.max_len], dtype=torch.long),
            'labels': torch.tensor(train_labels[:self.max_len], dtype=torch.long),
            'attention_mask': torch.tensor(
                [1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids[:self.max_len]],
                dtype=torch.long
            )
        }


# ============== Stage 2: Ranking Dataset ==============

class MINDRankingDataset:
    """Ranking dataset for Stage 2 training."""

    def __init__(
        self,
        behaviors_path: str,
        news_path: str,
        tokenizer,
        max_len: int = 4096,
        sample: int = -1,
        seed: int = 42,
        max_history: int = 0,
        neg_ratio: float = 4.0,
        use_abstract: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_history = max_history if max_history > 0 else None
        self.neg_ratio = neg_ratio
        self.use_abstract = use_abstract
        self.seed = seed

        self.news = self._load_news(news_path)
        self.samples = self._load_behaviors(behaviors_path, sample)

        print(f"[Stage2-Ranking] Loaded {len(self.samples)} samples")

    def _load_news(self, news_path: str):
        news = {}
        with open(news_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                news_id = parts[0]
                category = parts[1] if len(parts) > 1 else ""
                title = parts[3]
                abstract = parts[4] if len(parts) > 4 else ""

                if self.use_abstract and abstract:
                    text = f"{title} {abstract}"
                else:
                    text = title

                news[news_id] = {'text': text, 'category': category}
        return news

    def _load_behaviors(self, behaviors_path: str, sample_limit: int):
        all_samples = []
        rng = random.Random(self.seed)

        with open(behaviors_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue

                history_ids = parts[3].split()
                impressions = parts[4].split()

                if self.max_history is not None:
                    history_ids = history_ids[-self.max_history:]

                history = [self.news[nid] for nid in history_ids if nid in self.news]

                positives, negatives = [], []
                for imp in impressions:
                    if '-' not in imp:
                        continue
                    news_id, label = imp.rsplit('-', 1)
                    if news_id not in self.news:
                        continue
                    if int(label) == 1:
                        positives.append(news_id)
                    else:
                        negatives.append(news_id)

                if not positives:
                    continue

                # Create one sample per positive
                for pos_id in positives:
                    num_neg = int(self.neg_ratio)
                    if negatives:
                        sampled_neg = rng.sample(negatives, min(num_neg, len(negatives)))
                        candidates = [pos_id] + sampled_neg
                        rng.shuffle(candidates)
                        correct_idx = candidates.index(pos_id)

                        all_samples.append({
                            'history': history,
                            'candidates': [self.news[cid] for cid in candidates],
                            'correct_idx': correct_idx
                        })

        rng.shuffle(all_samples)
        if sample_limit > 0 and sample_limit < len(all_samples):
            all_samples = all_samples[:sample_limit]

        return all_samples

    def _build_prompt(self, history, candidates, correct_idx):
        prompt = "Role: You are a news recommendation assistant.\n"
        prompt += "Task: Select the article that best matches the user's reading interests.\n\n"

        prompt += "User History:\n"
        if history:
            for i, h in enumerate(history, 1):
                cat = f" ({h.get('category', '')})" if h.get('category') else ""
                prompt += f"{i}. [Title] {h['text']}{cat}\n"
        else:
            prompt += "(No reading history)\n"

        prompt += "\nCandidate Articles:\n"
        for i, cand in enumerate(candidates, 1):
            cat = f" ({cand.get('category', '')})" if cand.get('category') else ""
            prompt += f"{i}. [Title] {cand['text']}{cat}\n"

        prompt += "\nWhich article number would this user most likely click on?\n"
        prompt += "Answer with the article number only.\n\nAnswer:"

        target = f" {correct_idx + 1}"
        return prompt, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt, target = self._build_prompt(sample['history'], sample['candidates'], sample['correct_idx'])

        full_text = prompt + target
        input_ids = self.tokenizer.encode(full_text, max_length=self.max_len, truncation=True, add_special_tokens=True)
        prompt_ids = self.tokenizer.encode(prompt, max_length=self.max_len, truncation=True, add_special_tokens=True)

        train_labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            train_labels = train_labels + [-100] * pad_len

        return {
            'input_ids': torch.tensor(input_ids[:self.max_len], dtype=torch.long),
            'labels': torch.tensor(train_labels[:self.max_len], dtype=torch.long),
            'attention_mask': torch.tensor(
                [1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids[:self.max_len]],
                dtype=torch.long
            )
        }


def train(
    base_model: str = "",
    train_behaviors_path: str = "",
    train_news_path: str = "",
    eval_behaviors_path: str = "",
    eval_news_path: str = "",
    output_dir: str = "",
    use_abstract: bool = False,
    max_history: int = 0,
    # Stage 1 settings
    stage1_epochs: int = 1,
    stage1_neg_ratio: float = 1.0,
    stage1_cutoff_len: int = 2048,
    # Stage 2 settings
    stage2_epochs: int = 2,
    stage2_neg_ratio: float = 4.0,
    stage2_cutoff_len: int = 4096,
    # Common settings
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    learning_rate: float = 3e-4,
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    """Two-stage training: Point-wise -> Ranking"""

    set_seed(seed)

    if not base_model:
        raise ValueError("Please specify --base_model")

    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    print("=" * 60)
    print("Two-Stage Training: Point-wise -> Ranking")
    print("=" * 60)
    print(f"Base model: {base_model}")
    print(f"Stage 1: {stage1_epochs} epochs, neg_ratio={stage1_neg_ratio}")
    print(f"Stage 2: {stage2_epochs} epochs, neg_ratio={stage2_neg_ratio}")
    print("=" * 60)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ==================== STAGE 1: Point-wise ====================
    print("\n" + "=" * 60)
    print("STAGE 1: Point-wise SFT (Learning Relevance)")
    print("=" * 60)

    stage1_dir = os.path.join(output_dir, "stage1_pointwise")

    train_data_s1 = MINDPointwiseDataset(
        behaviors_path=train_behaviors_path,
        news_path=train_news_path,
        tokenizer=tokenizer,
        max_len=stage1_cutoff_len,
        sample=sample,
        seed=seed,
        max_history=max_history,
        neg_ratio=stage1_neg_ratio,
        use_abstract=use_abstract,
    )

    val_data_s1 = MINDPointwiseDataset(
        behaviors_path=eval_behaviors_path,
        news_path=eval_news_path,
        tokenizer=tokenizer,
        max_len=stage1_cutoff_len,
        sample=min(2000, len(train_data_s1) // 10) if sample <= 0 else min(500, sample // 10),
        seed=seed,
        max_history=max_history,
        neg_ratio=stage1_neg_ratio,
        use_abstract=use_abstract,
    )

    training_args_s1 = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=50,
        num_train_epochs=stage1_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir=stage1_dir,
        save_total_limit=2,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if wandb_project else "none",
        run_name=f"{wandb_run_name}_stage1" if wandb_run_name else None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer_s1 = transformers.Trainer(
        model=model,
        train_dataset=train_data_s1,
        eval_dataset=val_data_s1,
        args=training_args_s1,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False
    trainer_s1.train()

    # Save Stage 1 checkpoint
    model.save_pretrained(os.path.join(stage1_dir, "checkpoint"))
    tokenizer.save_pretrained(os.path.join(stage1_dir, "checkpoint"))
    print(f"\nStage 1 completed. Checkpoint: {stage1_dir}/checkpoint")

    # ==================== STAGE 2: Ranking ====================
    print("\n" + "=" * 60)
    print("STAGE 2: Ranking SFT (Learning Ordering)")
    print("=" * 60)

    stage2_dir = os.path.join(output_dir, "stage2_ranking")

    train_data_s2 = MINDRankingDataset(
        behaviors_path=train_behaviors_path,
        news_path=train_news_path,
        tokenizer=tokenizer,
        max_len=stage2_cutoff_len,
        sample=sample,
        seed=seed,
        max_history=max_history,
        neg_ratio=stage2_neg_ratio,
        use_abstract=use_abstract,
    )

    val_data_s2 = MINDRankingDataset(
        behaviors_path=eval_behaviors_path,
        news_path=eval_news_path,
        tokenizer=tokenizer,
        max_len=stage2_cutoff_len,
        sample=min(2000, len(train_data_s2) // 10) if sample <= 0 else min(500, sample // 10),
        seed=seed,
        max_history=max_history,
        neg_ratio=stage2_neg_ratio,
        use_abstract=use_abstract,
    )

    training_args_s2 = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=50,
        num_train_epochs=stage2_epochs,
        learning_rate=learning_rate * 0.5,  # Lower LR for fine-tuning
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir=stage2_dir,
        save_total_limit=2,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if wandb_project else "none",
        run_name=f"{wandb_run_name}_stage2" if wandb_run_name else None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer_s2 = transformers.Trainer(
        model=model,
        train_dataset=train_data_s2,
        eval_dataset=val_data_s2,
        args=training_args_s2,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer_s2.train()

    # Save final model
    final_dir = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("\n" + "=" * 60)
    print("Two-Stage Training Completed!")
    print("=" * 60)
    print(f"Stage 1 checkpoint: {stage1_dir}/checkpoint")
    print(f"Final model: {final_dir}")
    print(f"\nTo evaluate:")
    print(f"  bash scripts/eval_mind_ranking.sh {final_dir} dev")


if __name__ == "__main__":
    fire.Fire(train)
