#!/usr/bin/env python3
"""Per-impression AUC from a raw score file + a behaviors file.

Used by the train-memorization probe: the LLM was trained on MIND_large/train,
so its scores there may be inflated (memorized). If the train-split AUC is
much higher than the dev-split AUC, the train split cannot be used to train
the ranker -- the ranker would learn to over-trust llm_score.

Usage:
    python src/auc_from_scores.py --scores F.tsv --behaviors B.tsv
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--behaviors", required=True)
    args = ap.parse_args()

    # impression_id -> list of scores (candidate order = behaviors order)
    scores = {}
    with open(args.scores) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                scores[p[0]] = [float(v) for v in p[1:]]

    n_imp = 0
    n_valid = 0
    auc_sum = 0.0
    with open(args.behaviors) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            imp = p[0]
            s = scores.get(imp)
            if s is None or len(s) != len(p[4].split()):
                continue
            labels = []
            for tk in p[4].split():
                labels.append(1 if tk.rsplit("-", 1)[-1] == "1" else 0)
            n_imp += 1
            n_pos = sum(labels)
            n_neg = len(labels) - n_pos
            if n_pos == 0 or n_neg == 0:
                continue
            order = np.argsort(s, kind="stable")
            ranks = np.empty(len(s), dtype=np.float64)
            ranks[order] = np.arange(1, len(s) + 1)
            rsum = sum(r for r, l in zip(ranks, labels) if l == 1)
            auc_sum += (rsum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            n_valid += 1

    print(f"impressions scored: {n_imp:,}  valid (both classes): {n_valid:,}")
    if n_valid:
        print(f"per-impression AUC: {auc_sum / n_valid:.4f}")


if __name__ == "__main__":
    main()
