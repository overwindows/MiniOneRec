"""
Multi-Task Learning for MIND: Joint Point-wise + Ranking Training

This script trains a single model on both tasks simultaneously:
- Task 1: Point-wise classification (Yes/No) - learns absolute relevance
- Task 2: Ranking (select best candidate) - learns relative ordering

The training interleaves samples from both tasks, allowing the model
to learn complementary signals.

Usage:
    torchrun --nproc_per_node 4 src/sft_mind_multitask.py \
        --base_model Qwen/Qwen3-1.7B \
        --train_behaviors_path ../data/MIND/train/behaviors.tsv \
        --train_news_path ../data/MIND/train/news.tsv \
        --eval_behaviors_path ../data/MIND/dev/behaviors.tsv \
        --eval_news_path ../data/MIND/dev/news.tsv \
        --output_dir output_dir/sft_mind_multitask \
        --pointwise_ratio 0.5
"""

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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


def load_news(news_path: str, use_abstract: bool):
    """Load news articles from news.tsv"""
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

            if use_abstract and abstract:
                text = f"{title} {abstract}"
            else:
                text = title

            news[news_id] = {'text': text, 'category': category}
    return news


class MINDMultiTaskDataset(Dataset):
    """
    Multi-task dataset that interleaves point-wise and ranking samples.

    Each sample is either:
    - Point-wise: (history, candidate, Yes/No)
    - Ranking: (history, [candidates], correct_number)
    """

    def __init__(
        self,
        behaviors_path: str,
        news_path: str,
        tokenizer,
        max_len: int = 4096,
        sample: int = -1,
        seed: int = 42,
        max_history: int = 0,
        pointwise_neg_ratio: float = 1.0,
        ranking_neg_ratio: float = 4.0,
        pointwise_ratio: float = 0.5,  # Ratio of point-wise vs ranking samples
        use_abstract: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_history = max_history if max_history > 0 else None
        self.pointwise_neg_ratio = pointwise_neg_ratio
        self.ranking_neg_ratio = ranking_neg_ratio
        self.pointwise_ratio = pointwise_ratio
        self.use_abstract = use_abstract
        self.seed = seed

        self.news = load_news(news_path, use_abstract)
        self.samples = self._load_behaviors(behaviors_path, sample)

        pw_count = sum(1 for s in self.samples if s['task'] == 'pointwise')
        rk_count = len(self.samples) - pw_count
        print(f"[MultiTask] Loaded {len(self.samples)} samples")
        print(f"  Point-wise: {pw_count} ({pw_count/len(self.samples)*100:.1f}%)")
        print(f"  Ranking: {rk_count} ({rk_count/len(self.samples)*100:.1f}%)")

    def _load_behaviors(self, behaviors_path: str, sample_limit: int):
        """Load behaviors and create both point-wise and ranking samples."""
        pointwise_samples = []
        ranking_samples = []
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

                # Create point-wise samples
                for pos_id in positives:
                    pointwise_samples.append({
                        'task': 'pointwise',
                        'history': history,
                        'candidate': self.news[pos_id],
                        'label': 1
                    })

                num_neg_pw = int(len(positives) * self.pointwise_neg_ratio)
                if num_neg_pw > 0 and negatives:
                    sampled = rng.sample(negatives, min(num_neg_pw, len(negatives)))
                    for neg_id in sampled:
                        pointwise_samples.append({
                            'task': 'pointwise',
                            'history': history,
                            'candidate': self.news[neg_id],
                            'label': 0
                        })

                # Create ranking samples
                for pos_id in positives:
                    num_neg_rk = int(self.ranking_neg_ratio)
                    if negatives:
                        sampled_neg = rng.sample(negatives, min(num_neg_rk, len(negatives)))
                        candidates = [pos_id] + sampled_neg
                        rng.shuffle(candidates)
                        correct_idx = candidates.index(pos_id)

                        ranking_samples.append({
                            'task': 'ranking',
                            'history': history,
                            'candidates': [self.news[cid] for cid in candidates],
                            'correct_idx': correct_idx
                        })

        # Balance the datasets based on ratio
        rng.shuffle(pointwise_samples)
        rng.shuffle(ranking_samples)

        # Calculate how many of each to use
        total_target = len(pointwise_samples) + len(ranking_samples)
        if sample_limit > 0:
            total_target = min(total_target, sample_limit)

        num_pointwise = int(total_target * self.pointwise_ratio)
        num_ranking = total_target - num_pointwise

        pointwise_samples = pointwise_samples[:num_pointwise]
        ranking_samples = ranking_samples[:num_ranking]

        # Interleave samples
        all_samples = pointwise_samples + ranking_samples
        rng.shuffle(all_samples)

        return all_samples

    def _build_pointwise_prompt(self, history, candidate, label):
        """Build point-wise Yes/No prompt."""
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

    def _build_ranking_prompt(self, history, candidates, correct_idx):
        """Build ranking selection prompt."""
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

        if sample['task'] == 'pointwise':
            prompt, target = self._build_pointwise_prompt(
                sample['history'], sample['candidate'], sample['label']
            )
        else:  # ranking
            prompt, target = self._build_ranking_prompt(
                sample['history'], sample['candidates'], sample['correct_idx']
            )

        full_text = prompt + target
        input_ids = self.tokenizer.encode(
            full_text, max_length=self.max_len, truncation=True, add_special_tokens=True
        )
        prompt_ids = self.tokenizer.encode(
            prompt, max_length=self.max_len, truncation=True, add_special_tokens=True
        )

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
    # Multi-task settings
    pointwise_ratio: float = 0.5,  # Ratio of point-wise samples
    pointwise_neg_ratio: float = 1.0,
    ranking_neg_ratio: float = 4.0,
    # Training settings
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    """Multi-task training with joint point-wise and ranking objectives."""

    set_seed(seed)

    if not base_model:
        raise ValueError("Please specify --base_model")

    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    print("=" * 60)
    print("Multi-Task Training: Point-wise + Ranking")
    print("=" * 60)
    print(f"Base model: {base_model}")
    print(f"Point-wise ratio: {pointwise_ratio}")
    print(f"Point-wise neg_ratio: {pointwise_neg_ratio}")
    print(f"Ranking neg_ratio: {ranking_neg_ratio}")
    print("=" * 60)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load multi-task datasets
    train_data = MINDMultiTaskDataset(
        behaviors_path=train_behaviors_path,
        news_path=train_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        max_history=max_history,
        pointwise_neg_ratio=pointwise_neg_ratio,
        ranking_neg_ratio=ranking_neg_ratio,
        pointwise_ratio=pointwise_ratio,
        use_abstract=use_abstract,
    )

    val_data = MINDMultiTaskDataset(
        behaviors_path=eval_behaviors_path,
        news_path=eval_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=min(2000, len(train_data) // 10) if sample <= 0 else min(500, sample // 10),
        seed=seed,
        max_history=max_history,
        pointwise_neg_ratio=pointwise_neg_ratio,
        ranking_neg_ratio=ranking_neg_ratio,
        pointwise_ratio=pointwise_ratio,
        use_abstract=use_abstract,
    )

    print(f"\nTraining samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Training arguments
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if wandb_project else "none",
        run_name=wandb_run_name if wandb_run_name else None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Initialize trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False
    trainer.train()

    # Save final model
    final_dir = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("\n" + "=" * 60)
    print("Multi-Task Training Completed!")
    print("=" * 60)
    print(f"Model saved to: {final_dir}")
    print(f"\nTo evaluate (using ranking format):")
    print(f"  bash scripts/eval_mind_ranking.sh {final_dir} dev")
    print(f"\nTo evaluate (using point-wise format):")
    print(f"  bash scripts/eval_mind_pointwise.sh {final_dir} dev")


if __name__ == "__main__":
    fire.Fire(train)
