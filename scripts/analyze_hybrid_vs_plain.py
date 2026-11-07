"""Compare the A/B's arm-OFF (hybrid prefix cache) scores against the existing
plain-path dev scores for the SAME impressions and the SAME checkpoint.

WHY THIS IS WORTH A SEPARATE SCRIPT
-----------------------------------
The A/B's two arms both run with PREFIX_CACHE_MODE=hybrid, because the hybrid
path is 7.1x faster than the plain path on a Qwen3.5 backbone and the A/B
needed that speed. But the existing dev/test score dumps
(results/ranker/lg9b_ckpt45500_dev_scores.tsv and the test_scores/ files)
were produced on the PLAIN path. If hybrid and plain disagree materially,
then (a) the A/B's absolute AUCs are not comparable to our production numbers,
and (b) re-scoring dev/test with hybrid would move every number.

The arm-OFF arm of the A/B is exactly the production prompt (full history), so
comparing it against the plain-path dump isolates ONE variable: the prefix
cache. Same checkpoint, same impressions, same prompts, different batching.

Reports, per part:
  * impressions present in both
  * Spearman rank correlation of the per-impression AUC
  * mean per-impression AUC of each path
  * the paired delta (hybrid - plain)
  * max |score| difference on any candidate

Usage:
    python scripts/analyze_hybrid_vs_plain.py --part 00
    python scripts/analyze_hybrid_vs_plain.py --all
"""
import argparse
import glob
import os
import sys

import numpy as np

SUBSET = "data/MIND_large/dev/chunks/parity"
SCORES = "results/ranker/parity_ab"
PLAIN = "results/ranker/lg9b_ckpt45500_dev_scores.tsv"


def read_scores(path):
    out = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out[p[0]] = [float(v) for v in p[1:]]
    return out


def per_impression_auc(scores, behaviors_path, only=None):
    """impression_id -> (auc, n_candidates) for impressions with both classes."""
    res = {}
    with open(behaviors_path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            imp = p[0]
            if only is not None and imp not in only:
                continue
            s = scores.get(imp)
            if s is None or len(s) != len(p[4].split()):
                continue
            labels = [1 if tk.rsplit("-", 1)[-1] == "1" else 0 for tk in p[4].split()]
            n_pos = sum(labels)
            n_neg = len(labels) - n_pos
            if n_pos == 0 or n_neg == 0:
                continue
            order = np.argsort(s, kind="stable")
            ranks = np.empty(len(s), dtype=np.float64)
            ranks[order] = np.arange(1, len(s) + 1)
            rsum = sum(r for r, l in zip(ranks, labels) if l == 1)
            auc = (rsum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            res[imp] = (auc, len(labels))
    return res


def spearman(a, b):
    """Rank correlation (no ties assumed beyond the average-rank handling)."""
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def analyze(part, plain):
    off_path = f"{SCORES}/p{part}_off.tsv"
    beh = f"{SUBSET}/p{part}.tsv"
    for p in (off_path, beh):
        if not os.path.exists(p):
            print(f"part {part}: MISSING {p}")
            return None

    hybrid = read_scores(off_path)
    a_hyb = per_impression_auc(hybrid, beh)
    a_pln = per_impression_auc(plain, beh)
    common = sorted(set(a_hyb) & set(a_pln))
    if not common:
        print(f"part {part}: no common impressions with the plain-path dump")
        return None

    d = np.array([a_hyb[i][0] - a_pln[i][0] for i in common])
    auc_h = float(np.mean([a_hyb[i][0] for i in common]))
    auc_p = float(np.mean([a_pln[i][0] for i in common]))
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    rho = spearman([a_hyb[i][0] for i in common], [a_pln[i][0] for i in common])
    max_abs = 0.0
    for i in common:
        for x, y in zip(hybrid[i], plain[i]):
            max_abs = max(max_abs, abs(x - y))

    print(f"--- part {part} ---")
    print(f"  impressions in both paths     : {len(common):,}")
    print(f"  AUC plain (no prefix cache)   : {auc_p:.5f}")
    print(f"  AUC hybrid (prefix cache)     : {auc_h:.5f}")
    print(f"  delta (hybrid - plain)        : {d.mean():+.5f}  "
          f"(SE {se:.5f}, t={d.mean()/se if se else float('nan'):+.2f})")
    print(f"  Spearman rho (per-imp AUC)    : {rho:.6f}")
    print(f"  max |score| difference        : {max_abs:.6f}")
    return dict(part=part, n=len(common), auc_plain=auc_p, auc_hybrid=auc_h,
                delta=float(d.mean()), se=se, rho=rho, max_abs=max_abs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--plain", default=PLAIN)
    args = ap.parse_args()

    if not os.path.exists(args.plain):
        sys.exit(f"plain-path score dump not found: {args.plain}")

    parts = []
    if args.all:
        parts = sorted(os.path.basename(p)[1:3]
                       for p in glob.glob(f"{SCORES}/p*_off.tsv"))
    elif args.part:
        parts = [args.part]
    else:
        ap.error("pass --part NN or --all")

    plain = read_scores(args.plain)
    print(f"plain-path dump: {args.plain} ({len(plain):,} impressions)")

    rows = [analyze(p, plain) for p in parts]
    rows = [r for r in rows if r]
    if not rows:
        sys.exit(1)

    tot_n = sum(r["n"] for r in rows)
    auc_p = sum(r["auc_plain"] * r["n"] for r in rows) / tot_n
    auc_h = sum(r["auc_hybrid"] * r["n"] for r in rows) / tot_n
    delta = sum(r["delta"] * r["n"] for r in rows) / tot_n
    se = float(np.sqrt(sum((r["se"] ** 2) * (r["n"] ** 2) for r in rows)) / tot_n)

    print("=" * 62)
    print(f"POOLED over {len(rows)} part(s): {tot_n:,} impressions")
    print(f"  AUC plain  : {auc_p:.5f}")
    print(f"  AUC hybrid : {auc_h:.5f}")
    print(f"  delta      : {delta:+.5f}  (SE {se:.5f}, "
          f"t={delta/se if se else float('nan'):+.2f})")
    print("=" * 62)


if __name__ == "__main__":
    main()
