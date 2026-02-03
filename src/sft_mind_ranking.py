"""
Train a model on MIND dataset using Ranking-Aware SFT.

This script uses a multiple-choice format to teach the model to select
the most relevant news from a list of candidates, aligning training with evaluation.

Key differences from sft_mind.py:
- Training sees ALL candidates (clicked + non-clicked)
- Uses multiple-choice format: "1. Title1\n2. Title2\n..." (numeric options)
- Model outputs option number (e.g., "3")
- Directly optimizes for ranking/selection task
- Supports unlimited candidates (no 26-option limit like A-Z)

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
        max_len: int = 8192,
        sample: int = -1,
        seed: int = 42,
        max_history: int = 0,  # 0 = no limit (use all history)
        max_candidates: int = 0,  # 0 = no limit (use all candidates)
        neg_ratio: float = 0,  # 0 = no limit, >0 = negatives per positive (e.g., 4.0 = 4 negatives per positive)
        use_abstract: bool = False,
        use_chat_template: bool = False,  # Whether to use chat template for instruction-tuned models
    ):
        self.tokenizer = tokenizer
        self.max_len = int(max_len)  # Ensure it's an integer (fire.Fire may pass as string)
        max_history = int(max_history)
        max_candidates = int(max_candidates)
        neg_ratio = float(neg_ratio)
        self.max_history = max_history if max_history > 0 else None  # None = no limit
        self.max_candidates = max_candidates if max_candidates > 0 else None  # None = no limit
        self.neg_ratio = neg_ratio if neg_ratio > 0 else None  # None = no limit
        self.use_abstract = bool(use_abstract) if isinstance(use_abstract, bool) else str(use_abstract).lower() in ('true', '1', 'yes')
        self.use_chat_template = bool(use_chat_template) if isinstance(use_chat_template, bool) else str(use_chat_template).lower() in ('true', '1', 'yes')
        self.seed = int(seed)

        # Load news articles
        self.news = self._load_news(news_path)

        # Load behaviors
        self.behaviors = self._load_behaviors(behaviors_path)

        # Sample if requested
        sample = int(sample)
        if sample > 0 and sample < len(self.behaviors):
            random.seed(seed)
            self.behaviors = random.sample(self.behaviors, sample)

        self.samples = self._build_samples()

        print(f"Loaded {len(self.behaviors)} behaviors with ranking format")
        print(f"Expanded to {len(self.samples)} samples (one per clicked item)")
        print(f"Chat template: {self.use_chat_template}")

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

                # Get history (use all if max_history is None)
                if self.max_history is not None:
                    history_ids = history_ids[-self.max_history:]
                history = [self.news[nid] for nid in history_ids if nid in self.news]

                behaviors.append({
                    'impression_id': impression_id,
                    'history': history,
                    'candidates': candidates,
                    'labels': labels
                })

        return behaviors

    def _limit_candidates(self, behavior):
        """
        Limit candidates using neg_ratio and/or max_candidates with HARD NEGATIVE SAMPLING.

        Priority:
        1. neg_ratio: Sample negatives based on ratio to positives (e.g., 4.0 = 4 negs per pos)
        2. max_candidates: Hard cap on total candidates

        Hard Negative Sampling:
        - 50% from same category as clicked items (harder)
        - 50% from different categories (easier baselines)
        """
        candidates = behavior['candidates']
        labels = behavior['labels']

        clicked_indices = [i for i, l in enumerate(labels) if l == 1]
        non_clicked_indices = [i for i, l in enumerate(labels) if l == 0]
        rng = random.Random(f"{behavior['impression_id']}-{self.seed}")

        # Determine number of negatives to use
        if self.neg_ratio is not None:
            # Use neg_ratio: sample negatives based on ratio to positives
            num_negatives = int(len(clicked_indices) * self.neg_ratio)
            num_negatives = min(num_negatives, len(non_clicked_indices))
        elif self.max_candidates is not None and len(candidates) > self.max_candidates:
            # Use max_candidates: fill remaining slots with negatives
            num_negatives = self.max_candidates - len(clicked_indices)
            num_negatives = max(0, min(num_negatives, len(non_clicked_indices)))
        else:
            # No limit: use all negatives
            num_negatives = len(non_clicked_indices)

        # IMPROVED: Hard negative sampling (50% same category, 50% different)
        if num_negatives > 0 and num_negatives < len(non_clicked_indices):
            # Get categories of clicked items
            clicked_categories = set()
            for idx in clicked_indices:
                cat = self.news[candidates[idx]].get('category', '')
                clicked_categories.add(cat)

            # Split negatives by category match
            hard_neg_indices = []  # Same category
            easy_neg_indices = []  # Different category

            for idx in non_clicked_indices:
                neg_cat = self.news[candidates[idx]].get('category', '')
                if neg_cat in clicked_categories:
                    hard_neg_indices.append(idx)
                else:
                    easy_neg_indices.append(idx)

            # Sample 50/50 mix
            num_hard = num_negatives // 2
            num_easy = num_negatives - num_hard

            sampled_neg_indices = []
            if hard_neg_indices:
                sampled_neg_indices.extend(rng.sample(hard_neg_indices, min(num_hard, len(hard_neg_indices))))
            # Fill remaining with easy negatives
            if len(sampled_neg_indices) < num_negatives and easy_neg_indices:
                remaining = num_negatives - len(sampled_neg_indices)
                sampled_neg_indices.extend(rng.sample(easy_neg_indices, min(remaining, len(easy_neg_indices))))
            # If still not enough, add more hard negatives
            if len(sampled_neg_indices) < num_negatives and hard_neg_indices:
                remaining = num_negatives - len(sampled_neg_indices)
                available = [idx for idx in hard_neg_indices if idx not in sampled_neg_indices]
                if available:
                    sampled_neg_indices.extend(rng.sample(available, min(remaining, len(available))))
        else:
            sampled_neg_indices = non_clicked_indices[:num_negatives]

        # Combine positives + sampled negatives
        selected_indices = clicked_indices + sampled_neg_indices

        # Apply max_candidates cap if both neg_ratio and max_candidates are set
        if self.max_candidates is not None and len(selected_indices) > self.max_candidates:
            # Keep all positives if possible, otherwise sample
            if len(clicked_indices) >= self.max_candidates:
                selected_indices = rng.sample(clicked_indices, self.max_candidates)
            else:
                # Keep all positives, sample from negatives
                remaining = self.max_candidates - len(clicked_indices)
                selected_indices = clicked_indices + sampled_neg_indices[:remaining]

        rng.shuffle(selected_indices)

        candidates = [candidates[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]
        return candidates, labels

    def _build_samples(self):
        samples = []
        skipped_overlength = 0
        for behavior in self.behaviors:
            candidates, labels = self._limit_candidates(behavior)
            clicked_indices = [i for i, l in enumerate(labels) if l == 1]
            for clicked_idx in clicked_indices:
                content = self._build_prompt_content(behavior['history'], candidates)
                target = str(clicked_idx + 1)
                full_text, _, _ = self._format_for_training(content, target)
                input_ids = self.tokenizer.encode(
                    full_text, add_special_tokens=not self.use_chat_template, truncation=False
                )
                if len(input_ids) > self.max_len:
                    skipped_overlength += 1
                    continue
                samples.append({
                    'history': behavior['history'],
                    'candidates': candidates,
                    'labels': labels,
                    'clicked_idx': clicked_idx,
                })
        if skipped_overlength:
            print(f"Skipped {skipped_overlength} samples over cutoff_len={self.max_len}")
        return samples

    def _sample_hard_negatives(self, positives, negatives, pos_categories):
        """
        Sample hard negatives for ranking (50% same category, 50% different).
        Same-category negatives are harder to distinguish in ranking task.
        """
        hard_negs = []  # Same category as positive
        easy_negs = []  # Different category

        for neg_id in negatives:
            neg_cat = self.news[neg_id].get('category', '')
            if neg_cat in pos_categories:
                hard_negs.append(neg_id)
            else:
                easy_negs.append(neg_id)

        return hard_negs, easy_negs

    def _build_prompt_content(self, history, candidates):
        """
        Build the prompt content (without final formatting).

        Returns the content that can be wrapped in chat template or used directly.
        """
        lines = []
        lines.append("A user read these news articles:")

        # User history - limit to last 30 for token efficiency
        if history:
            recent_history = history[-30:] if len(history) > 30 else history
            for i, h in enumerate(recent_history, 1):
                cat = h.get('category', 'General')
                lines.append(f"{i}. [{cat}] {h['text']}")
        else:
            lines.append("(No reading history)")

        lines.append("")

        # Candidate articles - category first in brackets
        lines.append("Candidate articles:")
        for i, cand_id in enumerate(candidates):
            cand = self.news[cand_id]
            option_num = i + 1  # 1-indexed
            cat = cand.get('category', 'General')
            lines.append(f"{option_num}. [{cat}] {cand['text']}")

        lines.append("")
        lines.append("Which article will this user read? Answer with the number.")

        return "\n".join(lines)

    def _build_multiple_choice_prompt(self, history, candidates, clicked_idx):
        """
        Build multiple-choice ranking prompt with numeric options.

        For backwards compatibility, returns (prompt, target) tuple.
        Use _build_prompt_content() + _format_for_training() for chat template support.
        """
        content = self._build_prompt_content(history, candidates)
        prompt = content + "\n\nAnswer:"
        target = f" {clicked_idx + 1}"
        return prompt, target

    def _format_for_training(self, content, target):
        """
        Format content and target for training.

        If use_chat_template is True, applies chat template.
        Otherwise, uses raw text format.

        Returns (full_text, prompt_text, target_start_char)
        - full_text: complete text for tokenization
        - prompt_text: just the prompt part (for reference)
        - target_start_char: character index where target begins in full_text
        """
        if self.use_chat_template:
            # Apply chat template (chat template already adds its own control tokens)
            messages = [{"role": "user", "content": content}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Record where target starts
            target_start_char = len(prompt)
            # Full text includes target (do not append extra EOS in chat-template mode)
            full_text = prompt + target
        else:
            # Raw text format
            prompt = content + "\n\nAnswer:"
            # Target has a leading space in non-chat mode
            target_start_char = len(prompt)
            full_text = prompt + " " + target

        return full_text, prompt, target_start_char

    def __len__(self):
        return len(self.samples)

    def debug_tokenization(self, idx=0):
        """
        Debug method to verify tokenization and label masking.
        Call this to check if the fix is working correctly.

        Usage:
            dataset = MINDRankingSFTDataset(...)
            dataset.debug_tokenization(0)
        """
        sample = self.samples[idx]
        content = self._build_prompt_content(sample['history'], sample['candidates'])
        target = str(sample['clicked_idx'] + 1)

        full_text, prompt, target_start_char = self._format_for_training(content, target)

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_len,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=not self.use_chat_template
        )

        input_ids = encoding['input_ids']
        offset_mapping = encoding['offset_mapping']

        # Find prompt_len
        prompt_len = len(input_ids)
        for i, (char_start, char_end) in enumerate(offset_mapping):
            if char_start == 0 and char_end == 0 and i > 0:
                continue
            if char_start >= target_start_char:
                prompt_len = i
                break

        print("=" * 70)
        print("TOKENIZATION DEBUG")
        print("=" * 70)
        print(f"Target answer: {target}")
        print(f"Chat template: {self.use_chat_template}")
        print(f"Full text length: {len(full_text)} chars")
        print(f"Target starts at char: {target_start_char}")
        print(f"Total tokens: {len(input_ids)}")
        print(f"Prompt tokens: {prompt_len}")
        print(f"Target tokens: {len(input_ids) - prompt_len}")
        print()
        print("Last 10 tokens of prompt:")
        for i in range(max(0, prompt_len - 10), prompt_len):
            token = self.tokenizer.decode([input_ids[i]])
            char_range = offset_mapping[i]
            print(f"  [{i}] id={input_ids[i]:6d} '{token}' chars={char_range}")
        print()
        print("Target tokens (what model learns to predict):")
        for i in range(prompt_len, len(input_ids)):
            token = self.tokenizer.decode([input_ids[i]])
            char_range = offset_mapping[i]
            print(f"  [{i}] id={input_ids[i]:6d} '{token}' chars={char_range}")
        print()
        print(f"Text around target start (chars {target_start_char-20}:{target_start_char+20}):")
        print(f"  ...{repr(full_text[max(0,target_start_char-20):target_start_char])}|{repr(full_text[target_start_char:target_start_char+20])}...")
        print("=" * 70)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        history = sample['history']
        candidates = sample['candidates']
        clicked_idx = sample['clicked_idx']

        # Build prompt content and target
        content = self._build_prompt_content(history, candidates)
        target = str(clicked_idx + 1)  # 1-indexed

        # Format for training (handles chat template if enabled)
        full_text, prompt, target_start_char = self._format_for_training(content, target)

        # ============================================================
        # ROBUST TOKENIZATION WITH OFFSET MAPPING
        # ============================================================
        # Use offset_mapping to precisely find where target starts in tokens.
        # This avoids all tokenization boundary issues.

        # Tokenize with offset mapping
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_len,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=not self.use_chat_template
        )

        input_ids = encoding['input_ids']
        offset_mapping = encoding['offset_mapping']

        # Find the first token that starts at or after target_start_char
        # This is the first token of the target (what we want to train on)
        prompt_len = len(input_ids)  # default: all tokens are prompt (shouldn't happen)
        for i, (char_start, char_end) in enumerate(offset_mapping):
            # Skip special tokens (they have offset (0, 0))
            if char_start == 0 and char_end == 0 and i > 0:
                continue
            if char_start >= target_start_char:
                prompt_len = i
                break

        # Sanity check: ensure we have at least 1 target token
        if prompt_len >= len(input_ids):
            # Fallback: use last token as target (shouldn't happen with valid data)
            prompt_len = max(1, len(input_ids) - 1)

        # Create training labels: mask prompt tokens with -100, only train on target
        train_labels = [-100] * prompt_len + input_ids[prompt_len:]
        train_labels = train_labels[:len(input_ids)]

        # Verify: count how many tokens we're training on
        num_target_tokens = sum(1 for l in train_labels if l != -100)
        if num_target_tokens == 0:
            # Emergency fallback: train on last token
            train_labels[-1] = input_ids[-1]

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


def verify_tokenization(dataset, num_samples=3, abort_on_error=True):
    """
    Verify tokenization and label masking is correct before training.

    This function checks that:
    1. Target tokens are correctly identified (1-3 tokens for answer number + EOS)
    2. Labels are properly masked (-100 for prompt, actual IDs for target)
    3. No samples have zero target tokens (would cause NaN loss)

    Args:
        dataset: MINDRankingSFTDataset instance
        num_samples: Number of samples to verify
        abort_on_error: If True, raise error on critical issues

    Returns:
        bool: True if verification passed, False if warnings were found
    """
    print("\n" + "=" * 70)
    print("TOKENIZATION VERIFICATION (Pre-training Check)")
    print("=" * 70)
    print(f"Chat template: {dataset.use_chat_template}")
    print(f"Checking {num_samples} samples...")
    print()

    all_good = True
    total_target_tokens = 0
    issues = []

    for i in range(min(num_samples, len(dataset))):
        # Get the item
        item = dataset[i]
        labels = item['labels'].tolist()
        input_ids = item['input_ids'].tolist()

        # Count target tokens (non -100 labels)
        num_target = sum(1 for l in labels if l != -100)
        total_target_tokens += num_target

        # Get expected answer from sample
        sample = dataset.samples[i]
        expected_answer = str(sample['clicked_idx'] + 1)

        # Check for issues
        if num_target == 0:
            issues.append(f"Sample {i}: ERROR - No target tokens! Will cause NaN loss.")
            all_good = False
        elif num_target > 10:
            issues.append(f"Sample {i}: WARNING - {num_target} target tokens (expected 1-3 for answer '{expected_answer}')")
            all_good = False
        else:
            # Decode target tokens to verify
            target_token_ids = [input_ids[j] for j, l in enumerate(zip(input_ids, labels)) if labels[j] != -100]
            target_text = dataset.tokenizer.decode(target_token_ids).strip()

            # Check if expected answer is in target
            if expected_answer not in target_text:
                issues.append(f"Sample {i}: WARNING - Expected '{expected_answer}' but got '{target_text}'")
                all_good = False
            else:
                print(f"  Sample {i}: ✓ {num_target} target tokens, answer='{target_text}'")

    avg_target_tokens = total_target_tokens / num_samples if num_samples > 0 else 0

    print()
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  ❌ {issue}")
        print()

    print(f"Average target tokens per sample: {avg_target_tokens:.1f}")
    print(f"Expected: 1-3 tokens (answer number + optional EOS)")
    print()

    if all_good:
        print("✓ Tokenization verification PASSED!")
        print("  Target tokens are correctly identified.")
        print("  Training should have normal initial loss (3-6 range).")
    else:
        print("⚠ Tokenization verification found issues!")
        print("  Review the warnings above.")
        if abort_on_error and any("ERROR" in issue for issue in issues):
            print()
            print("=" * 70)
            raise ValueError(
                "Critical tokenization error detected! "
                "Training would fail with NaN loss. "
                "Set abort_on_error=False to skip this check."
            )

    print("=" * 70)
    print()

    return all_good


class _TorchStackCollator:
    def __init__(self, debug=False, max_logs=5):
        self.debug = debug
        self.max_logs = max_logs
        self._count = 0

    def _should_log(self):
        if not self.debug or self._count >= self.max_logs:
            return False
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank() == 0
        except Exception:
            return True
        return True

    def __call__(self, batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])

        if self._should_log():
            seq_lens = attention_mask.sum(dim=1).tolist()
            print(
                f"[debug] batch_size={len(batch)} "
                f"seq_len/max={input_ids.shape[1]} "
                f"seq_lens(min/mean/max)={min(seq_lens)}/{sum(seq_lens)/len(seq_lens):.1f}/{max(seq_lens)}"
            )
            self._count += 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def train(
    base_model: str = "",
    train_behaviors_path: str = "",
    train_news_path: str = "",
    eval_behaviors_path: str = "",
    eval_news_path: str = "",
    output_dir: str = "",
    use_abstract: bool = False,
    max_history: int = 30,  # OPTIMIZED: Last 30 items have 90% predictive signal (was 0=unlimited)
    max_candidates: int = 0,  # 0 = no limit (use neg_ratio instead)
    neg_ratio: float = 4.0,  # OPTIMIZED: 4 negatives per positive (was 0=unlimited)
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 256,  # OPTIMIZED: Increased for stability (was 128)
    micro_batch_size: int = 2,  # OPTIMIZED: For 8B model memory (was 4)
    num_epochs: int = 3,  # Reduced to match point-wise (was 5)
    learning_rate: float = 1e-4,  # FIXED: Increased from 2e-5 to match point-wise scale better
    cutoff_len: int = 6144,  # Increased to avoid skipping samples with many candidates (was 4096)
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    train_from_scratch: bool = False,
    use_chat_template: bool = False,  # Whether to use chat template for instruction-tuned models
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    """Train with ranking-aware SFT format (list-wise)"""

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

    # Parse use_chat_template (handle string from fire.Fire)
    use_chat_template = bool(use_chat_template) if isinstance(use_chat_template, bool) else str(use_chat_template).lower() in ('true', '1', 'yes')

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
        neg_ratio=neg_ratio,
        use_abstract=use_abstract,
        use_chat_template=use_chat_template,
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
        neg_ratio=neg_ratio,
        use_abstract=use_abstract,
        use_chat_template=use_chat_template,
    )

    print(f"\nTraining with Ranking-Aware SFT (List-wise):")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Max history: {'unlimited' if max_history == 0 else max_history}")
    print(f"  Max candidates: {'unlimited' if max_candidates == 0 else max_candidates}")
    print(f"  Neg ratio: {'unlimited' if neg_ratio == 0 else neg_ratio}")
    print(f"  Cutoff length: {cutoff_len}")
    print(f"  Chat template: {use_chat_template}")
    print(f"  Format: Multiple-choice (1/2/3/...)")

    # ============================================================
    # AUTOMATIC TOKENIZATION VERIFICATION
    # ============================================================
    # Run verification on train dataset to catch tokenization issues
    # BEFORE training starts. This prevents wasted compute on bad data.
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_main_process:
        verify_tokenization(train_data, num_samples=5, abort_on_error=True)

    # Training arguments
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,  # FIXED: Reduced to match point-wise (was 500)
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="steps" if val_data else "no",
        save_strategy="steps",
        eval_steps=256 if val_data else None,
        save_steps=512,
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
        data_collator=_TorchStackCollator(debug=bool(os.environ.get("DEBUG_SEQ", ""))),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=64)] if val_data else None,
    )

    model.config.use_cache = False

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    final_path = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save training config for evaluation consistency
    import json
    config_path = os.path.join(final_path, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            'max_history': max_history,
            'max_candidates': max_candidates,
            'neg_ratio': neg_ratio,
            'use_abstract': use_abstract,
            'use_chat_template': use_chat_template,
            'seed': seed,
        }, f, indent=2)

    print(f"\n✓ Ranking-aware SFT training completed!")
    print(f"  Model saved to: {final_path}")
    print(f"  Training config saved to: {config_path}")
    print(f"\nTo evaluate with SAME settings:")
    print(f"  python evaluate_mind_ranking.py \\")
    print(f"      --model_path {final_path} \\")
    print(f"      --behaviors_path <dev_behaviors.tsv> \\")
    print(f"      --news_path <dev_news.tsv> \\")
    print(f"      --load_training_config")


if __name__ == "__main__":
    fire.Fire(train)
