"""
Diagnostic script: check how many v8 training samples get their target token
truncated at the max_len boundary (default 4096).

Hypothesis: v8 prompts (with interactions, grouped conversations, 9 rules)
are longer than v7 prompts, causing " Yes"/" No" to be cut off by right-truncation.
When that happens, train_labels is all -100 and the sample contributes zero gradient.

Usage (on AML or any machine with the data):
    python scripts/check_truncation.py --jsonl data/doca_v8/train.jsonl
    python scripts/check_truncation.py --jsonl data/doca_v8/dev.jsonl
    python scripts/check_truncation.py --jsonl data/doca_v8/train.jsonl --max_len 6144
"""

import argparse
import json
import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from data import DOCAPointwiseSFTDataset


def analyze_truncation(jsonl_path, model_name, max_len, neg_ratio, max_interests,
                       max_conversation_msgs, max_shown, use_chat_template, max_feeds):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading dataset: {jsonl_path}")
    ds = DOCAPointwiseSFTDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        max_len=max_len,
        neg_ratio=neg_ratio,
        max_interests=max_interests,
        max_conversation_msgs=max_conversation_msgs,
        max_shown=max_shown,
        use_chat_template=use_chat_template,
        sample=max_feeds if max_feeds > 0 else -1,
    )

    total = len(ds)
    print(f"\nTotal samples: {total}")
    print(f"max_len: {max_len}")
    print(f"use_chat_template: {use_chat_template}")
    print(f"Analyzing truncation...\n")

    truncated_count = 0       # target fully truncated (zero gradient)
    partial_trunc_count = 0   # target partially truncated
    ok_count = 0              # target fully preserved
    prompt_lengths = []
    full_lengths = []
    target_token_counts = []

    for i in range(total):
        sample = ds.samples[i]
        prompt, target = ds._build_prompt(
            sample['user_context'], sample['candidate'], sample['label']
        )

        if use_chat_template:
            messages = [
                {"role": "system", "content": ds.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = formatted_prompt + target
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        else:
            prompt_with_sys = ds.SYSTEM_PROMPT + "\n\n" + prompt
            full_text = prompt_with_sys + target
            full_ids = tokenizer.encode(full_text, add_special_tokens=True)
            prompt_ids = tokenizer.encode(prompt_with_sys, add_special_tokens=True)

        prompt_len = len(prompt_ids)
        full_len = len(full_ids)
        target_len = full_len - prompt_len

        prompt_lengths.append(prompt_len)
        full_lengths.append(full_len)
        target_token_counts.append(target_len)

        # Simulate what __getitem__ does
        truncated_input = full_ids[:max_len]
        truncated_prompt = prompt_ids[:max_len]
        labels = [-100] * len(truncated_prompt) + truncated_input[len(truncated_prompt):]
        labels = labels[:max_len]

        # Count how many non-(-100) labels remain
        actual_target_labels = [l for l in labels if l != -100]

        if len(actual_target_labels) == 0:
            truncated_count += 1
        elif len(actual_target_labels) < target_len:
            partial_trunc_count += 1
        else:
            ok_count += 1

        if i < 3 or (i < total and len(actual_target_labels) == 0 and truncated_count <= 5):
            status = "TRUNCATED (zero gradient)" if len(actual_target_labels) == 0 else "OK"
            print(f"  Sample {i}: prompt={prompt_len} tok, full={full_len} tok, "
                  f"target={target_len} tok, trained_labels={len(actual_target_labels)} → {status}")

    prompt_arr = np.array(prompt_lengths)
    full_arr = np.array(full_lengths)

    print(f"\n{'='*60}")
    print(f"RESULTS (max_len={max_len})")
    print(f"{'='*60}")
    print(f"Total samples:              {total}")
    print(f"Target FULLY truncated:     {truncated_count} ({100*truncated_count/total:.1f}%)  ← zero gradient!")
    print(f"Target partially truncated: {partial_trunc_count} ({100*partial_trunc_count/total:.1f}%)")
    print(f"Target fully preserved:     {ok_count} ({100*ok_count/total:.1f}%)")

    print(f"\nPrompt length (tokens):")
    print(f"  min={prompt_arr.min()}, median={int(np.median(prompt_arr))}, "
          f"mean={prompt_arr.mean():.0f}, p90={int(np.percentile(prompt_arr, 90))}, "
          f"p95={int(np.percentile(prompt_arr, 95))}, p99={int(np.percentile(prompt_arr, 99))}, "
          f"max={prompt_arr.max()}")

    print(f"\nFull sequence length (prompt + target):")
    print(f"  min={full_arr.min()}, median={int(np.median(full_arr))}, "
          f"mean={full_arr.mean():.0f}, p90={int(np.percentile(full_arr, 90))}, "
          f"p95={int(np.percentile(full_arr, 95))}, p99={int(np.percentile(full_arr, 99))}, "
          f"max={full_arr.max()}")

    over = np.sum(full_arr > max_len)
    print(f"\n  Sequences > {max_len} tokens: {over} ({100*over/total:.1f}%)")

    # Show what cutoff would fix it
    for candidate_len in [4096, 5120, 6144, 8192]:
        over_c = np.sum(full_arr > candidate_len)
        print(f"  Sequences > {candidate_len} tokens: {over_c} ({100*over_c/total:.1f}%)")

    # Positive vs negative breakdown
    pos_trunc = 0
    neg_trunc = 0
    pos_total = 0
    neg_total = 0
    for i in range(total):
        sample = ds.samples[i]
        prompt, target = ds._build_prompt(
            sample['user_context'], sample['candidate'], sample['label']
        )
        if use_chat_template:
            messages = [
                {"role": "system", "content": ds.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_ids = tokenizer.encode(formatted_prompt + target, add_special_tokens=False)
            prompt_ids_len = len(tokenizer.encode(formatted_prompt, add_special_tokens=False))
        else:
            prompt_with_sys = ds.SYSTEM_PROMPT + "\n\n" + prompt
            full_ids = tokenizer.encode(prompt_with_sys + target, add_special_tokens=True)
            prompt_ids_len = len(tokenizer.encode(prompt_with_sys, add_special_tokens=True))

        truncated_input = full_ids[:max_len]
        labels = [-100] * min(prompt_ids_len, max_len) + truncated_input[min(prompt_ids_len, max_len):]
        labels = labels[:max_len]
        actual_target = [l for l in labels if l != -100]

        if sample['label'] == 1:
            pos_total += 1
            if len(actual_target) == 0:
                pos_trunc += 1
        else:
            neg_total += 1
            if len(actual_target) == 0:
                neg_trunc += 1

    print(f"\nTruncation by label:")
    print(f"  Positive (Yes): {pos_trunc}/{pos_total} truncated ({100*pos_trunc/max(pos_total,1):.1f}%)")
    print(f"  Negative (No):  {neg_trunc}/{neg_total} truncated ({100*neg_trunc/max(neg_total,1):.1f}%)")

    if truncated_count > 0:
        print(f"\n⚠️  {truncated_count} samples ({100*truncated_count/total:.1f}%) produce ZERO gradient!")
        print(f"   The model learns nothing from these samples.")
        print(f"   Fix: increase max_len, or left-truncate the prompt to preserve the target.")
    else:
        print(f"\n✓ No samples are fully truncated. Truncation is NOT the issue.")


def main():
    parser = argparse.ArgumentParser(description="Check target truncation in DOCA dataset")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to train.jsonl or dev.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Tokenizer model name")
    parser.add_argument("--max_len", type=int, default=4096, help="Max sequence length (same as training)")
    parser.add_argument("--neg_ratio", type=float, default=2.0, help="Negative sampling ratio")
    parser.add_argument("--max_interests", type=int, default=0, help="Max interests (0=unlimited)")
    parser.add_argument("--max_conversation_msgs", type=int, default=15, help="Max conversation messages")
    parser.add_argument("--max_shown", type=int, default=10, help="Max shown articles")
    parser.add_argument("--use_chat_template", action="store_true", help="Use chat template format")
    parser.add_argument("--max_feeds", type=int, default=-1, help="Limit number of samples to check (-1=all)")
    args = parser.parse_args()

    analyze_truncation(
        jsonl_path=args.jsonl,
        model_name=args.model,
        max_len=args.max_len,
        neg_ratio=args.neg_ratio,
        max_interests=args.max_interests,
        max_conversation_msgs=args.max_conversation_msgs,
        max_shown=args.max_shown,
        use_chat_template=args.use_chat_template,
        max_feeds=args.max_feeds,
    )


if __name__ == "__main__":
    main()
