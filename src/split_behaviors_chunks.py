#!/usr/bin/env python3
"""Split MIND behaviors.tsv into N sequential chunks (by line ranges)."""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_chunks", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
    n = len(lines)
    per = n // args.n_chunks
    print(f"{n:,} lines -> {args.n_chunks} chunks of ~{per:,}")

    for c in range(args.n_chunks):
        start = c * per
        end = n if c == args.n_chunks - 1 else (c + 1) * per
        out = os.path.join(args.output_dir, f"chunk_{c:02d}.tsv")
        with open(out, "w", encoding="utf-8") as o:
            o.writelines(lines[start:end])
        print(f"chunk_{c:02d}.tsv: lines {start:,}..{end:,} ({end-start:,})", flush=True)
    print("done")


if __name__ == "__main__":
    main()