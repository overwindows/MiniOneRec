"""Paired per-impression AUC comparison of two MIND dev score files.

WHY PAIRED
----------
The MIND metric is the MEAN of per-impression AUCs. Comparing two models by
the difference of their two mean AUCs throws away the fact that both models
were scored on the SAME impressions: the per-impression AUC values are highly
correlated, so the difference of means has a much larger standard error than
the mean of the per-impression differences.

    independent:  se(AUC_A - AUC_B) ~ sqrt(2) * 0.00043 = 0.00061
    paired (N):   se ~ sd(delta) / sqrt(N)

The paired test is also the honest one for "is this checkpoint better than the
model it was trained from": the two arms share the impressions, the node, the
scoring code and the prompt rendering, so the delta isolates the checkpoint.

Usage:
    python scripts/compare_dev_scores_paired.py --base A.tsv --new B.tsv \
        --behaviors chunk.tsv
    python scripts/compare_dev_scores_paired.py --base-dir DIR_A --new-dir DIR_B \
        --chunk-dir DIR
"""
import argparse
import glob
import os
import sys

import numpy as np


def load_scores(path):
    out = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out[p[0]] = [float(v) for v in p[1:]]
    return out


def per_impression_auc(scores, labels):
    """AUC of one impression, or None if it has only one class."""
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rsum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (rsum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compare(base, new, behaviors_path):
    auc_base, auc_new, ids = [], [], []
    n_missing = 0
    with open(behaviors_path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            imp = p[0]
            sb, sn = base.get(imp), new.get(imp)
            if sb is None or sn is None:
                n_missing += 1
                continue
            labels = [1 if tk.rsplit("-", 1)[-1] == "1" else 0 for tk in p[4].split()]
            if len(sb) != len(labels) or len(sn) != len(labels):
                n_missing += 1
                continue
            ab = per_impression_auc(sb, labels)
            an = per_impression_auc(sn, labels)
            if ab is None or an is None:
                continue
            auc_base.append(ab)
            auc_new.append(an)
            ids.append(imp)
    return np.array(auc_base), np.array(auc_new), ids, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="base score file (tsv)")
    ap.add_argument("--new", help="new score file (tsv)")
    ap.add_argument("--base-dir", help="dir of base score files")
    ap.add_argument("--new-dir", help="dir of new score files")
    ap.add_argument("--behaviors", help="behaviors/chunk file for the ids")
    ap.add_argument("--chunk-dir", help="dir of chunk_XX.tsv (used with --*-dir)")
    ap.add_argument("--label-base", default="base")
    ap.add_argument("--label-new", default="new")
    args = ap.parse_args()

    if args.base_dir and args.new_dir:
        base, new = {}, {}
        for f in sorted(glob.glob(os.path.join(args.base_dir, "*.tsv"))):
            base.update(load_scores(f))
        for f in sorted(glob.glob(os.path.join(args.new_dir, "*.tsv"))):
            new.update(load_scores(f))
        behaviors = args.behaviors or args.chunk_dir
        if os.path.isdir(behaviors):
            # concatenate every chunk, in name order
            rows = []
            for cf in sorted(glob.glob(os.path.join(behaviors, "chunk_*.tsv"))):
                with open(cf) as f:
                    rows.extend(f.readlines())
            tmp = "/tmp/_paired_behaviors.tsv"
            with open(tmp, "w") as f:
                f.writelines(rows)
            behaviors = tmp
    else:
        if not (args.base and args.new and args.behaviors):
            ap.error("need --base/--new/--behaviors, or --base-dir/--new-dir/--chunk-dir")
        base = load_scores(args.base)
        new = load_scores(args.new)
        behaviors = args.behaviors

    ab, an, ids, n_missing = compare(base, new, behaviors)
    if len(ab) < 2:
        print(f"FATAL: only {len(ab)} common impressions")
        sys.exit(1)

    d = an - ab
    se = d.std(ddof=1) / np.sqrt(len(d))
    delta = d.mean()
    t = delta / se if se > 0 else float("nan")
    n_up = int((d > 1e-12).sum())
    n_down = int((d < -1e-12).sum())

    print(f"--- paired comparison: {args.label_new} vs {args.label_base} ---")
    print(f"  impressions compared   : {len(d):,}   (skipped {n_missing:,})")
    print(f"  mean AUC {args.label_base:<14s}: {ab.mean():.5f}")
    print(f"  mean AUC {args.label_new:<14s}: {an.mean():.5f}")
    print(f"  PAIRED delta           : {delta:+.5f}  (SE {se:.5f}, t={t:+.2f})")
    print(f"  improved / worse / tied: {n_up:,} / {n_down:,} / {len(d)-n_up-n_down:,}")
    print(f"  per-impression delta sd: {d.std(ddof=1):.5f}")
    print(f"  95% CI                 : [{delta-1.96*se:+.5f}, {delta+1.96*se:+.5f}]")
    verdict = ("BETTER" if delta > 0 and t > 2 else
               "WORSE" if delta < 0 and t < -2 else
               "NO SIGNIFICANT CHANGE")
    print(f"  VERDICT                : {verdict} (|t| > 2 required)")


if __name__ == "__main__":
    main()
