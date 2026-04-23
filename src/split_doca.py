"""
Split DOCA JSONL file into shards for multi-GPU evaluation.

Usage:
    python src/split_doca.py \
        --input_path data/doca/dev.jsonl \
        --output_path temp_doca/ \
        --cuda_list 0,1,2,3
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to JSONL file")
    parser.add_argument("--output_path", required=True, help="Output directory for shards")
    parser.add_argument("--cuda_list", required=True, help="Comma-separated GPU IDs")
    args = parser.parse_args()

    gpu_ids = [g.strip() for g in args.cuda_list.split(",") if g.strip()]
    num_gpus = len(gpu_ids)

    with open(args.input_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    total = len(lines)
    per_gpu = total // num_gpus

    os.makedirs(args.output_path, exist_ok=True)

    for i, gpu_id in enumerate(gpu_ids):
        start = i * per_gpu
        end = (i + 1) * per_gpu if i < num_gpus - 1 else total
        shard = lines[start:end]

        out_path = os.path.join(args.output_path, f"{gpu_id}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(shard)

        print(f"GPU {gpu_id}: {len(shard)} feeds -> {out_path}")

    print(f"\nTotal: {total} feeds split across {num_gpus} GPUs")


if __name__ == "__main__":
    main()
