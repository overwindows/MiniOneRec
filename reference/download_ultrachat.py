#!/usr/bin/env python3
"""Download UltraChat-200k and export to JSONL for SFT mixing."""
import argparse
import json
import os
from pathlib import Path


def _extract_prompt_response(example):
    if "messages" in example:
        msgs = example["messages"]
        # Expect system/user/assistant style
        user_parts = [m["content"] for m in msgs if m.get("role") == "user"]
        assistant_parts = [m["content"] for m in msgs if m.get("role") == "assistant"]
        if user_parts and assistant_parts:
            return user_parts[-1], assistant_parts[-1]
    if "prompt" in example and "response" in example:
        return example["prompt"], example["response"]
    if "instruction" in example and "output" in example:
        return example["instruction"], example["output"]
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/general/ultrachat_200k.jsonl")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit("datasets is required: pip install datasets") from exc

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        kept = 0
        for ex in ds:
            prompt, response = _extract_prompt_response(ex)
            if not prompt or not response:
                continue
            rec = {
                "instruction": prompt.strip(),
                "input": "",
                "output": response.strip(),
            }
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            kept += 1

    print(f"Wrote {kept} examples to {out_path}")


if __name__ == "__main__":
    main()
