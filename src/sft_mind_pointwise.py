"""
Train a model on MIND dataset using Point-wise SFT.

This script uses a point-wise approach where each (history, candidate) pair
is scored independently with a Yes/No classification.

Key differences from list-wise (sft_mind_ranking.py):
- Each candidate is evaluated independently
- Binary classification: "Is this article relevant? Yes/No"
- More training signal (every candidate gets a label)
- Shorter context per sample
- No position bias

Training format:
    Prompt: "User History: ... Candidate: [article] Is this relevant? Answer:"
    Target: " Yes" or " No"

Usage:
    torchrun --nproc_per_node 4 src/sft_mind_pointwise.py \
        --base_model Qwen/Qwen3-1.7B \
        --train_behaviors_path ../data/MIND/train/behaviors.tsv \
        --train_news_path ../data/MIND/train/news.tsv \
        --eval_behaviors_path ../data/MIND/dev/behaviors.tsv \
        --eval_news_path ../data/MIND/dev/news.tsv \
        --output_dir output_dir/sft_mind_pointwise
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

# Add parent directory to path
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


class MINDPointwiseSFTDataset:
    """
    MIND dataset for point-wise SFT training.

    Each sample is a (history, single_candidate, label) tuple.
    Model learns to predict Yes/No for relevance.
    """

    def __init__(
        self,
        behaviors_path: str,
        news_path: str,
        tokenizer,
        max_len: int = 2048,
        sample: int = -1,
        seed: int = 42,
        max_history: int = 0,  # 0 = no limit
        neg_ratio: float = 1.0,  # Ratio of negatives to positives per impression
        use_abstract: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_history = max_history if max_history > 0 else None
        self.neg_ratio = neg_ratio
        self.use_abstract = use_abstract
        self.seed = seed

        # Load news articles
        self.news = self._load_news(news_path)

        # Load and expand behaviors into point-wise samples
        self.samples = self._load_and_expand_behaviors(behaviors_path, sample)

        print(f"Loaded {len(self.samples)} point-wise samples")
        pos_count = sum(1 for s in self.samples if s['label'] == 1)
        neg_count = len(self.samples) - pos_count
        print(f"  Positives: {pos_count}, Negatives: {neg_count}, Ratio: {neg_count/max(pos_count,1):.2f}")

    def _load_news(self, news_path: str):
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

                if self.use_abstract and abstract:
                    text = f"{title} {abstract}"
                else:
                    text = title

                news[news_id] = {
                    'text': text,
                    'category': category
                }
        return news

    def _load_and_expand_behaviors(self, behaviors_path: str, sample_limit: int):
        """Load behaviors and expand into point-wise samples."""
        all_samples = []
        rng = random.Random(self.seed)

        with open(behaviors_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue

                impression_id = parts[0]
                history_ids = parts[3].split()
                impressions = parts[4].split()

                # Limit history
                if self.max_history is not None:
                    history_ids = history_ids[-self.max_history:]

                # Build history objects
                history = [self.news[nid] for nid in history_ids if nid in self.news]

                # Parse candidates
                positives = []
                negatives = []
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

                # Skip if no positives
                if not positives:
                    continue

                # Add all positive samples
                for pos_id in positives:
                    all_samples.append({
                        'impression_id': impression_id,
                        'history': history,
                        'candidate_id': pos_id,
                        'candidate': self.news[pos_id],
                        'label': 1
                    })

                # Sample negatives based on ratio
                num_neg_to_sample = int(len(positives) * self.neg_ratio)
                if num_neg_to_sample > 0 and negatives:
                    sampled_negs = rng.sample(negatives, min(num_neg_to_sample, len(negatives)))
                    for neg_id in sampled_negs:
                        all_samples.append({
                            'impression_id': impression_id,
                            'history': history,
                            'candidate_id': neg_id,
                            'candidate': self.news[neg_id],
                            'label': 0
                        })

        # Shuffle all samples
        rng.shuffle(all_samples)

        # Sample if requested
        if sample_limit > 0 and sample_limit < len(all_samples):
            all_samples = all_samples[:sample_limit]

        return all_samples

    def _build_pointwise_prompt(self, history, candidate, label):
        """
        Build point-wise prompt for binary classification.

        Format:
        Role: You are a news recommendation assistant.
        Task: Determine if the candidate article matches the user's interests.

        User History:
        1. [Title] ... (Category)
        2. [Title] ... (Category)
        ...

        Candidate Article:
        [Title] ... (Category)

        Based on the user's reading history, is this article relevant to them?
        Answer with Yes or No.

        Answer:
        """
        prompt = "Role: You are a news recommendation assistant.\n"
        prompt += "Task: Determine if the candidate article matches the user's interests.\n\n"

        # User history
        prompt += "User History:\n"
        if history:
            for i, h in enumerate(history, 1):
                category = f" ({h.get('category', '')})" if h.get('category') else ""
                prompt += f"{i}. [Title] {h['text']}{category}\n"
        else:
            prompt += "(No reading history)\n"

        prompt += "\n"

        # Candidate article
        prompt += "Candidate Article:\n"
        category = f" ({candidate.get('category', '')})" if candidate.get('category') else ""
        prompt += f"[Title] {candidate['text']}{category}\n"

        prompt += "\n"
        prompt += "Based on the user's reading history, is this article relevant to them?\n"
        prompt += "Answer with Yes or No.\n\n"
        prompt += "Answer:"

        # Target
        target = " Yes" if label == 1 else " No"

        return prompt, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Build prompt
        prompt, target = self._build_pointwise_prompt(
            sample['history'],
            sample['candidate'],
            sample['label']
        )

        # Tokenize
        full_text = prompt + target
        input_ids = self.tokenizer.encode(
            full_text,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True
        )

        # Create training labels (mask prompt, only train on Yes/No)
        prompt_ids = self.tokenizer.encode(
            prompt,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True
        )

        train_labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        # Pad if needed
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
    max_history: int = 0,  # 0 = no limit
    neg_ratio: float = 1.0,  # Negatives per positive
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,  # Shorter than list-wise since single candidate
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    train_from_scratch: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    """Train with point-wise SFT format (Yes/No classification)"""

    set_seed(seed)

    if not base_model:
        raise ValueError("Please specify --base_model")

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Load model
    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load point-wise datasets
    train_data = MINDPointwiseSFTDataset(
        behaviors_path=train_behaviors_path,
        news_path=train_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        max_history=max_history,
        neg_ratio=neg_ratio,
        use_abstract=use_abstract,
    )

    val_data = MINDPointwiseSFTDataset(
        behaviors_path=eval_behaviors_path,
        news_path=eval_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=min(5000, len(train_data) // 10) if sample <= 0 else min(1000, sample // 10),
        seed=seed,
        max_history=max_history,
        neg_ratio=neg_ratio,
        use_abstract=use_abstract,
    )

    print(f"\nTraining with Point-wise SFT:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Max history: {'unlimited' if max_history == 0 else max_history}")
    print(f"  Neg ratio: {neg_ratio}")
    print(f"  Cutoff length: {cutoff_len}")
    print(f"  Format: Yes/No classification")

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
        eval_strategy="steps" if val_data else "no",
        save_strategy="steps",
        eval_steps=200 if val_data else None,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if val_data else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if wandb_project else "none",
        run_name=wandb_run_name if wandb_run_name else None,
        metric_for_best_model="eval_loss" if val_data else None,
        greater_is_better=False,
        disable_tqdm=False,
    )

    # Initialize trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if val_data else None,
    )

    model.config.use_cache = False

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final_checkpoint"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_checkpoint"))

    print(f"\n✓ Point-wise SFT training completed!")
    print(f"  Model saved to: {output_dir}/final_checkpoint")
    print(f"\nTo evaluate:")
    print(f"  bash scripts/eval_mind_pointwise.sh {output_dir}/final_checkpoint dev")


if __name__ == "__main__":
    fire.Fire(train)
