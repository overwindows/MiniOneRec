"""
Train a model on MIND dataset using Ranking-Aware SFT.

This script uses a multiple-choice format to teach the model to select
the most relevant news from a list of candidates, aligning training with evaluation.

Key differences from sft_mind.py:
- Training sees ALL candidates (clicked + non-clicked)
- Uses multiple-choice format: "A. Title1\nB. Title2\n..."
- Model outputs option letter (e.g., "C")
- Directly optimizes for ranking/selection task

Expected improvement: +3-6% AUC over standard SFT
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
from datasets import Dataset as HFDataset
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


class MINDRankingSFTDataset:
    """
    MIND dataset for ranking-aware SFT training.

    Uses multiple-choice format where model selects from candidate list.
    """

    def __init__(
        self,
        behaviors_path: str,
        news_path: str,
        tokenizer,
        max_len: int = 1024,
        sample: int = -1,
        seed: int = 42,
        max_history: int = 50,
        max_candidates: int = 20,  # Limit candidates to fit in context
        use_abstract: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_history = max_history
        self.max_candidates = max_candidates
        self.use_abstract = use_abstract

        # Load news articles
        self.news = self._load_news(news_path)

        # Load behaviors
        self.behaviors = self._load_behaviors(behaviors_path)

        # Sample if requested
        if sample > 0 and sample < len(self.behaviors):
            random.seed(seed)
            self.behaviors = random.sample(self.behaviors, sample)

        print(f"Loaded {len(self.behaviors)} behaviors with ranking format")

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
                    news[news_id] = {
                        'title': title,
                        'text': f"{title} {abstract}",
                        'category': category
                    }
                else:
                    news[news_id] = {
                        'title': title,
                        'text': title,
                        'category': category
                    }
        return news

    def _load_behaviors(self, behaviors_path: str):
        """Load user behaviors from behaviors.tsv"""
        behaviors = []
        with open(behaviors_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue

                impression_id = parts[0]
                history_ids = parts[3].split()
                impressions = parts[4].split()

                # Parse impressions
                candidates = []
                labels = []
                for imp in impressions:
                    if '-' not in imp:
                        continue
                    news_id, label = imp.rsplit('-', 1)
                    if news_id in self.news:
                        candidates.append(news_id)
                        labels.append(int(label))

                # Skip if no candidates or no clicks
                if not candidates or sum(labels) == 0:
                    continue

                # Get history
                history = [self.news[nid] for nid in history_ids[-self.max_history:]
                          if nid in self.news]

                behaviors.append({
                    'impression_id': impression_id,
                    'history': history,
                    'candidates': candidates,
                    'labels': labels
                })

        return behaviors

    def _build_multiple_choice_prompt(self, history, candidates, clicked_idx):
        """
        Build multiple-choice ranking prompt.

        Format:
        Role: You are a news recommendation assistant.
        Task: Select the most relevant news article...

        User History:
        1. [Title] ... (Category)
        2. [Title] ... (Category)

        Candidate News Articles:
        A. [Title] ... (Category)
        B. [Title] ... (Category)
        ...

        Output only the option letter.

        Answer:
        """
        prompt = "Role: You are a news recommendation assistant.\n"
        prompt += "Task: Select the most relevant news article for the user based on their reading history.\n\n"

        # User history
        prompt += "User History:\n"
        if history:
            for i, h in enumerate(history, 1):
                category = f" ({h['category']})" if h['category'] else ""
                prompt += f"{i}. [Title] {h['text']}{category}\n"
        else:
            prompt += "(No reading history)\n"

        prompt += "\n"

        # Candidate articles
        prompt += "Candidate News Articles:\n"
        option_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i, cand_id in enumerate(candidates):
            cand = self.news[cand_id]
            letter = option_letters[i]
            category = f" ({cand['category']})" if cand['category'] else ""
            prompt += f"{letter}. [Title] {cand['text']}{category}\n"

        prompt += "\n"
        prompt += "Please analyze the user's interests and select the best article from the candidates above.\n"
        prompt += "Output only the option letter.\n\n"
        prompt += "Answer:"

        # Target is the letter of the clicked article
        target = f" {option_letters[clicked_idx]}"

        return prompt, target

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        behavior = self.behaviors[idx]
        history = behavior['history']
        candidates = behavior['candidates']
        labels = behavior['labels']

        # Limit candidates if too many
        if len(candidates) > self.max_candidates:
            # Sample negatives + keep all positives
            clicked_indices = [i for i, l in enumerate(labels) if l == 1]
            non_clicked_indices = [i for i, l in enumerate(labels) if l == 0]

            # If more clicked than max_candidates, sample from clicked only
            if len(clicked_indices) >= self.max_candidates:
                selected_indices = random.sample(clicked_indices, self.max_candidates)
            else:
                # Keep all clicked + sample negatives to fill up to max_candidates
                num_negatives = self.max_candidates - len(clicked_indices)
                num_negatives = min(num_negatives, len(non_clicked_indices))
                if num_negatives > 0:
                    sampled_neg = random.sample(non_clicked_indices, num_negatives)
                    selected_indices = clicked_indices + sampled_neg
                else:
                    selected_indices = clicked_indices

            random.shuffle(selected_indices)

            candidates = [candidates[i] for i in selected_indices]
            labels = [labels[i] for i in selected_indices]

        # Randomly select one clicked article as target
        clicked_indices = [i for i, l in enumerate(labels) if l == 1]
        target_idx = random.choice(clicked_indices)

        # Build prompt
        prompt, target = self._build_multiple_choice_prompt(
            history, candidates, target_idx
        )

        # Tokenize
        full_text = prompt + target
        input_ids = self.tokenizer.encode(
            full_text,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True
        )

        # Create labels (mask prompt, only train on target letter)
        prompt_ids = self.tokenizer.encode(
            prompt,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True
        )

        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        # Pad if needed
        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        return {
            'input_ids': torch.tensor(input_ids[:self.max_len], dtype=torch.long),
            'labels': torch.tensor(labels[:self.max_len], dtype=torch.long),
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
    max_history: int = 50,
    max_candidates: int = 20,
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    train_from_scratch: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    """Train with ranking-aware SFT format"""

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
            dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load ranking datasets
    train_data = MINDRankingSFTDataset(
        behaviors_path=train_behaviors_path,
        news_path=train_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        max_history=max_history,
        max_candidates=max_candidates,
        use_abstract=use_abstract,
    )

    val_data = MINDRankingSFTDataset(
        behaviors_path=eval_behaviors_path,
        news_path=eval_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=min(2000, len(train_data) // 10) if sample <= 0 else min(500, sample // 10),
        seed=seed,
        max_history=max_history,
        max_candidates=max_candidates,
        use_abstract=use_abstract,
    )

    print(f"\nTraining with Ranking-Aware SFT:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Max candidates per sample: {max_candidates}")
    print(f"  Format: Multiple-choice (A/B/C/...)")

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

    print(f"\n✓ Ranking-aware SFT training completed!")
    print(f"  Model saved to: {output_dir}/final_checkpoint")
    print(f"\nExpected improvement: +3-6% AUC over standard SFT")
    print(f"\nTo evaluate:")
    print(f"  bash scripts/eval_mind.sh {output_dir}/final_checkpoint dev")


if __name__ == "__main__":
    fire.Fire(train)
