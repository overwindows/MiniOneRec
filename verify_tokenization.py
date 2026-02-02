#!/usr/bin/env python3
"""
Quick script to verify tokenization and label masking is correct.

Run this BEFORE training to make sure the fix is working:

    python verify_tokenization.py \
        --model Qwen/Qwen3-1.7B \
        --behaviors_path data/MIND/train/behaviors.tsv \
        --news_path data/MIND/train/news.tsv \
        --use_chat_template

Expected output:
- Target tokens should be 1-3 tokens (the answer number + maybe EOS)
- Initial loss estimate should be around 3-6 (not 15!)
"""

import argparse
import math
import sys
sys.path.insert(0, 'src')

from transformers import AutoTokenizer
from sft_mind_ranking_ds import MINDRankingSFTDataset


def estimate_initial_loss(vocab_size, num_target_tokens):
    """
    Estimate what the initial loss should be for random predictions.

    Initial loss ≈ log(vocab_size) * num_target_tokens / num_target_tokens
                = log(vocab_size)

    For Qwen3 with vocab ~150k: log(150000) ≈ 11.9
    But with a pretrained model, it should be lower (3-6 range).
    """
    return math.log(vocab_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B', help='Model name or path')
    parser.add_argument('--behaviors_path', required=True)
    parser.add_argument('--news_path', required=True)
    parser.add_argument('--use_chat_template', action='store_true')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to check')
    args = parser.parse_args()

    print("=" * 70)
    print("TOKENIZATION VERIFICATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Chat template: {args.use_chat_template}")
    print("=" * 70)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: {repr(tokenizer.eos_token)}")
    print()

    # Create dataset
    print("Loading dataset (small sample)...")
    dataset = MINDRankingSFTDataset(
        behaviors_path=args.behaviors_path,
        news_path=args.news_path,
        tokenizer=tokenizer,
        max_len=4096,
        sample=100,  # Small sample for testing
        seed=42,
        max_history=30,
        neg_ratio=4.0,
        use_chat_template=args.use_chat_template,
    )
    print()

    # Check several samples
    total_target_tokens = 0
    all_good = True

    for i in range(min(args.num_samples, len(dataset))):
        print(f"\n{'='*70}")
        print(f"SAMPLE {i}")
        print(f"{'='*70}")

        # Use debug method
        dataset.debug_tokenization(i)

        # Also get the actual item to verify
        item = dataset[i]
        labels = item['labels'].tolist()

        # Count target tokens
        num_target = sum(1 for l in labels if l != -100)
        total_target_tokens += num_target

        # Verify
        if num_target == 0:
            print("❌ ERROR: No target tokens! This will cause NaN loss.")
            all_good = False
        elif num_target > 10:
            print(f"⚠️  WARNING: {num_target} target tokens seems high for a 1-2 digit answer")
        else:
            print(f"✓ Target tokens: {num_target} (looks good)")

        # Show what tokens we're training on
        input_ids = item['input_ids'].tolist()
        print("\nTokens being trained on:")
        for j, (tok_id, label) in enumerate(zip(input_ids, labels)):
            if label != -100:
                token = tokenizer.decode([tok_id])
                print(f"  Position {j}: id={tok_id} '{repr(token)}'")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_target_tokens = total_target_tokens / args.num_samples
    theoretical_loss = estimate_initial_loss(tokenizer.vocab_size, 1)

    print(f"Average target tokens per sample: {avg_target_tokens:.1f}")
    print(f"Theoretical initial loss (random model): {theoretical_loss:.2f}")
    print(f"Expected initial loss (pretrained model): 3-6")
    print()

    if all_good:
        print("✓ All samples look correct!")
        print("  If you still see loss ~15, check:")
        print("  1. Model loading (is it actually pretrained?)")
        print("  2. Learning rate (try lower, e.g., 1e-5)")
    else:
        print("❌ Some samples have issues. Fix before training!")

    print("=" * 70)


if __name__ == '__main__':
    main()
