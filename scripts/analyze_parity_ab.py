"""Compare the two arms of the paired prompt-parity A/B.

Reads the OFF-arm and ON-arm score files for one over-length dev subset part
and reports:
  * how many impressions actually changed (the parity path is a no-op for
    prompts that already fit, so only the over-length ones may move),
  * the per-impression AUC of each arm,
  * the PAIRED delta (per impression, arm ON minus arm OFF) with its standard
    error, which is the number that matters: the two arms ran on the same
    node, on the same GPUs, minutes apart, so the delta is not contaminated by
    node-to-node hardware variation.

EXTRAPOLATION TO THE FULL DEV SET
---------------------------------
The subset is deliberately OVER-INCLUSIVE: it holds every impression whose
prompt comes within 48 tokens of the 2047 limit, so the impressions that
actually fit (and are therefore parity no-ops) act as a control group.

To turn the subset result into a full-dev number we need
    full-dev delta = P(overflow) x E[delta | overflow]
where
  * P(overflow) is measured over the FULL dev split by
    scripts/count_dev_overflow.py (6.51% with the validated-exact estimator),
  * E[delta | overflow] is measured here, restricted to the impressions whose
    prompt actually overflows (the exact per-impression flags come from
    scripts/annotate_subset_overflow.py).

If the flags file is missing the script still works, but the "delta over
CHANGED impressions" line is used instead of the exact overflow subset, and
the extrapolation is skipped.

Usage:
    python scripts/analyze_parity_ab.py --part 00
    python scripts/analyze_parity_ab.py --all
    python scripts/analyze_parity_ab.py --all --dev-overflow-fraction 0.0651
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

SUBSET = "data/MIND_large/dev/chunks/parity"
SCORES = "results/ranker/parity_ab"
FLAGS = "results/ranker/parity_overflow_flags.pkl"


def read_scores(path):
    out = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out[p[0]] = [float(v) for v in p[1:]]
    return out


def per_impression_auc(scores, behaviors_path):
    """impression_id -> (auc, n_candidates, n_pos) for impressions with both classes."""
    res = {}
    with open(behaviors_path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            imp = p[0]
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
            res[imp] = (auc, len(labels), n_pos)
    return res


def load_flags():
    if not os.path.exists(FLAGS):
        return None
    with open(FLAGS, "rb") as f:
        return pickle.load(f)


def analyze(part, flags):
    off_path = f"{SCORES}/p{part}_off.tsv"
    on_path = f"{SCORES}/p{part}_on.tsv"
    beh = f"{SUBSET}/p{part}.tsv"
    for p in (off_path, on_path, beh):
        if not os.path.exists(p):
            print(f"part {part}: MISSING {p}")
            return None

    off = read_scores(off_path)
    on = read_scores(on_path)
    a_off = per_impression_auc(off, beh)
    a_on = per_impression_auc(on, beh)

    common = sorted(set(a_off) & set(a_on))
    if not common:
        print(f"part {part}: no common impressions")
        return None

    # Which impressions had their SCORES changed at all?
    changed = [i for i in common if off.get(i) != on.get(i)]
    n_changed = len(changed)
    n_maxabs = max(
        (max((abs(x - y) for x, y in zip(off[i], on[i])), default=0.0) for i in common),
        default=0.0,
    )

    d = np.array([a_on[i][0] - a_off[i][0] for i in common])
    auc_off = float(np.mean([a_off[i][0] for i in common]))
    auc_on = float(np.mean([a_on[i][0] for i in common]))
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    n_up = int((d > 1e-12).sum())
    n_down = int((d < -1e-12).sum())

    # ---- delta over the impressions the parity path ACTUALLY changed -------
    if changed:
        dch = np.array([a_on[i][0] - a_off[i][0] for i in changed])
        delta_ch = float(dch.mean())
        se_ch = float(dch.std(ddof=1) / np.sqrt(len(dch))) if len(dch) > 1 else float("nan")
    else:
        delta_ch = se_ch = float("nan")

    # ---- delta over the impressions whose prompt genuinely OVERFLOWS -------
    # This is the population the extrapolation is based on.
    delta_ov = se_ov = float("nan")
    n_ov = 0
    if flags is not None and part in flags:
        pf = flags[part]
        ov_ids = [i for i in common if pf.get(i)]
        n_ov = len(ov_ids)
        if n_ov > 1:
            dov = np.array([a_on[i][0] - a_off[i][0] for i in ov_ids])
            delta_ov = float(dov.mean())
            se_ov = float(dov.std(ddof=1) / np.sqrt(n_ov))
        elif n_ov == 1:
            delta_ov = float(a_on[ov_ids[0]][0] - a_off[ov_ids[0]][0])

    print(f"--- part {part} ---")
    print(f"  impressions scored both arms : {len(common):,}")
    print(f"  scores changed by parity      : {n_changed:,} "
          f"({100.0*n_changed/len(common):.1f}%)   max |delta score| = {n_maxabs:.6f}")
    print(f"  AUC (arm OFF, full history)   : {auc_off:.5f}")
    print(f"  AUC (arm ON,  parity shrink)  : {auc_on:.5f}")
    print(f"  PAIRED delta (ON - OFF)       : {d.mean():+.5f}  "
          f"(SE {se:.5f}, t={d.mean()/se if se else float('nan'):+.2f})")
    print(f"  PAIRED delta over CHANGED     : {delta_ch:+.5f}  "
          f"(SE {se_ch:.5f}, n={n_changed:,})")
    if flags is not None:
        print(f"  PAIRED delta over OVERFLOWING : {delta_ov:+.5f}  "
              f"(SE {se_ov:.5f}, n={n_ov:,})")
    print(f"  impressions improved / worse  : {n_up:,} / {n_down:,}")
    return dict(part=part, n=len(common), changed=n_changed, auc_off=auc_off,
                auc_on=auc_on, delta=float(d.mean()), se=se, n_up=n_up, n_down=n_down,
                delta_changed=delta_ch, se_changed=se_ch, n_changed_ids=n_changed,
                delta_overflow=delta_ov, se_overflow=se_ov, n_overflow=n_ov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dev-overflow-fraction", type=float, default=None,
                    help="P(overflow) over the FULL dev split (from "
                         "scripts/count_dev_overflow.py). Enables the extrapolation.")
    args = ap.parse_args()

    parts = []
    if args.all:
        parts = sorted(os.path.basename(p)[1:3]
                       for p in glob.glob(f"{SCORES}/p*_off.tsv"))
    elif args.part:
        parts = [args.part]
    else:
        ap.error("pass --part NN or --all")

    flags = load_flags()
    if flags is None:
        print(f"NOTE: {FLAGS} not found -> the exact overflow-subset delta is "
              f"unavailable (run scripts/annotate_subset_overflow.py)")

    rows = [analyze(p, flags) for p in parts]
    rows = [r for r in rows if r]
    if not rows:
        sys.exit(1)

    tot_n = sum(r["n"] for r in rows)
    tot_ch = sum(r["changed"] for r in rows)
    # Weighted mean of the per-part AUCs (each part has a similar impression
    # count, so this is essentially the pooled AUC over the subset).
    auc_off = sum(r["auc_off"] * r["n"] for r in rows) / tot_n
    auc_on = sum(r["auc_on"] * r["n"] for r in rows) / tot_n
    # Pooled delta: weight each part's mean delta by its impression count.
    delta = sum(r["delta"] * r["n"] for r in rows) / tot_n
    # Pooled SE: parts are independent, so variances add.
    se = float(np.sqrt(sum((r["se"] ** 2) * (r["n"] ** 2) for r in rows)) / tot_n)

    print("=" * 62)
    print(f"POOLED over {len(rows)} part(s): {tot_n:,} impressions")
    print(f"  scores changed by parity : {tot_ch:,} ({100.0*tot_ch/tot_n:.1f}%)")
    print(f"  AUC arm OFF (full history): {auc_off:.5f}")
    print(f"  AUC arm ON  (parity)      : {auc_on:.5f}")
    print(f"  PAIRED delta              : {delta:+.5f}  (SE {se:.5f}, "
          f"t={delta/se if se else float('nan'):+.2f})")

    # ---- pooled overflow-subset delta + extrapolation ----------------------
    ov_rows = [r for r in rows if r["n_overflow"] > 0]
    if ov_rows:
        n_ov = sum(r["n_overflow"] for r in ov_rows)
        d_ov = sum(r["delta_overflow"] * r["n_overflow"] for r in ov_rows) / n_ov
        se_ov = float(np.sqrt(sum((r["se_overflow"] ** 2) * (r["n_overflow"] ** 2)
                                   for r in ov_rows)) / n_ov)
        print(f"  PAIRED delta over OVERFLOWING ({n_ov:,} imps): {d_ov:+.5f} "
              f"(SE {se_ov:.5f}, t={d_ov/se_ov if se_ov else float('nan'):+.2f})")
        if args.dev_overflow_fraction:
            p = args.dev_overflow_fraction
            full = p * d_ov
            full_se = p * se_ov
            print(f"  EXTRAPOLATED to full dev (P(overflow)={p:.4f}): "
                  f"{full:+.5f}  (SE {full_se:.5f})")
    print("=" * 62)


if __name__ == "__main__":
    main()
