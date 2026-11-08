#!/usr/bin/env python
"""
Paired per-impression AUC comparison of two dev prediction files.

WHY
---
The v6+mh0 pipeline produces a single absolute dev AUC (0.7264). That number
alone does not say whether the change was real or noise, because the two arms
share 376,471 impressions and are highly correlated. The honest statistic is
the PAIRED delta: for every impression, AUC_new - AUC_base, then mean +/- SE
and a t statistic over the impression-level differences.

This is the same estimator scripts/verify_mh0_stack.py uses on the chunk_90
subset; here it runs on the FULL dev split.

Each prediction file is one line per impression:
    <impression_id> <news_id> <news_id> ...
where the news_ids are in RANKED order. The per-impression AUC is computed
from the RANK (not from a raw score), exactly as src/calc_mind_metrics.py
does: the score of the item at rank r is (n - r), so a better ranking gives a
higher AUC.

Usage:
    python src/paired_dev_auc.py --base A.txt --new B.txt \
        --behaviors data/MIND_large/dev/behaviors.tsv
"""

import argparse
import json
import math

import numpy as np


def load_ground_truth(path):
    """impression_id -> {news_id: label}"""
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            imp = parts[0]
            labels = {}
            for item in parts[4].split():
                if "-" not in item:
                    continue
                nid, lab = item.rsplit("-", 1)
                labels[nid] = int(lab)
            gt[imp] = labels
    return gt


def per_impression_auc(labels, scores):
    """AUC from (labels, scores). Returns None for a degenerate impression."""
    if len(labels) == 0:
        return None
    n_pos = int(sum(labels))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rsum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (rsum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def read_preds(path):
    """Yield (impression_id, [news_ids in ranked order])."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            yield parts[0], parts[1:]


def arm_aucs(pred_path, gt):
    """impression_id -> per-impression AUC for one arm."""
    out = {}
    for imp, ranked in read_preds(pred_path):
        g = gt.get(imp)
        if g is None:
            continue
        labels, scores = [], []
        n = len(ranked)
        for rank, nid in enumerate(ranked):
            if nid in g:
                labels.append(g[nid])
                scores.append(n - rank)
        if not labels or sum(labels) == 0:
            continue
        a = per_impression_auc(labels, scores)
        if a is not None:
            out[imp] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--behaviors", required=True)
    ap.add_argument("--out", default=None, help="optional JSON summary path")
    args = ap.parse_args()

    gt = load_ground_truth(args.behaviors)
    print(f"ground truth impressions: {len(gt):,}")

    base = arm_aucs(args.base, gt)
    print(f"base arm: {len(base):,} scored impressions")
    new = arm_aucs(args.new, gt)
    print(f"new  arm: {len(new):,} scored impressions")

    common = sorted(set(base) & set(new))
    print(f"paired on {len(common):,} impressions")

    b = np.array([base[i] for i in common], dtype=np.float64)
    n = np.array([new[i] for i in common], dtype=np.float64)
    d = n - b

    mean_b = float(b.mean())
    mean_n = float(n.mean())
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    se = sd / math.sqrt(len(d)) if len(d) > 1 else 0.0
    t = mean_d / se if se > 0 else 0.0
    ci_lo = mean_d - 1.96 * se
    ci_hi = mean_d + 1.96 * se

    improved = int((d > 1e-12).sum())
    worse = int((d < -1e-12).sum())
    tied = int(len(d) - improved - worse)

    print()
    print("=" * 62)
    print("PAIRED PER-IMPRESSION AUC  (dev split)")
    print("=" * 62)
    print(f"  base ({args.base})")
    print(f"    AUC = {mean_b:.6f}")
    print(f"  new  ({args.new})")
    print(f"    AUC = {mean_n:.6f}")
    print(f"  delta = {mean_d:+.6f}   SE = {se:.6f}   t = {t:+.2f}")
    print(f"  95% CI = [{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print(f"  improved {improved:,} / worse {worse:,} / tied {tied:,}")
    print(f"  verdict: {'BETTER' if abs(t) > 2 else 'NOISE'}")
    print("=" * 62)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "base_file": args.base,
                    "new_file": args.new,
                    "n_impressions": len(common),
                    "base_auc": mean_b,
                    "new_auc": mean_n,
                    "delta": mean_d,
                    "se": se,
                    "t": t,
                    "ci95": [ci_lo, ci_hi],
                    "improved": improved,
                    "worse": worse,
                    "tied": tied,
                },
                f,
                indent=2,
            )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
