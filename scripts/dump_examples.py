"""
Save 3 example training prompts from the DOCA dataset to inspect.

Usage:
    python scripts/dump_examples.py --jsonl data/doca_v8/train.jsonl
    python scripts/dump_examples.py --jsonl data/doca_v8/train.jsonl --n 5 --output examples.txt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from data import DOCAPointwiseSFTDataset


def main():
    parser = argparse.ArgumentParser(description="Dump example training prompts")
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--neg_ratio", type=float, default=2.0)
    parser.add_argument("--max_interests", type=int, default=0)
    parser.add_argument("--max_conversation_msgs", type=int, default=15)
    parser.add_argument("--max_shown", type=int, default=10)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--n", type=int, default=3, help="Number of examples to dump")
    parser.add_argument("--output", type=str, default="examples.txt", help="Output file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ds = DOCAPointwiseSFTDataset(
        jsonl_path=args.jsonl,
        tokenizer=tokenizer,
        max_len=args.max_len,
        neg_ratio=args.neg_ratio,
        max_interests=args.max_interests,
        max_conversation_msgs=args.max_conversation_msgs,
        max_shown=args.max_shown,
        use_chat_template=args.use_chat_template,
        sample=args.n * 50,  # load enough to pick from
    )

    n = min(args.n, len(ds))
    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(n):
            sample = ds.samples[i]
            prompt, target = ds._build_prompt(
                sample['user_context'], sample['candidate'], sample['label']
            )

            if args.use_chat_template:
                messages = [
                    {"role": "system", "content": ds.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                full_text = formatted + target
                full_ids = tokenizer.encode(full_text, add_special_tokens=False)
                prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
            else:
                full_text = prompt + target
                full_ids = tokenizer.encode(full_text, add_special_tokens=True)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

            label_str = "Yes (clicked)" if sample['label'] == 1 else "No (not clicked)"

            f.write(f"{'='*80}\n")
            f.write(f"EXAMPLE {i+1} / {n}  |  Label: {label_str}\n")
            f.write(f"Prompt tokens: {len(prompt_ids)}  |  Full tokens: {len(full_ids)}  |  max_len: {args.max_len}\n")
            if len(full_ids) > args.max_len:
                f.write(f"⚠️  TRUNCATED! Full={len(full_ids)} > max_len={args.max_len}, "
                        f"target token will be LOST\n")
            f.write(f"{'='*80}\n\n")

            # Always show system prompt
            f.write(f"[SYSTEM PROMPT]\n{ds.SYSTEM_PROMPT}\n\n")
            f.write(f"[USER PROMPT + TARGET]\n")
            f.write(full_text)
            f.write(f"\n\n{'—'*80}\n\n")

    print(f"Saved {n} examples to {args.output}")
    print(f"  File size: {os.path.getsize(args.output):,} bytes")


if __name__ == "__main__":
    main()
