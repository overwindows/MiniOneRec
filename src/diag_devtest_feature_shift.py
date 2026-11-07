#!/usr/bin/env python3
"""Compare the dev and TEST feature tables column by column.

WHY
---
The stack ranker claims +0.0575 per-impression AUC over the LLM on dev.
The previous (l13) ranker claimed +0.0105 on dev and transferred as ~0.000
at test. The l13 ranker was trained on the TRAIN split, where the LLM had
MEMORIZED (train probe AUC 0.7692 vs dev 0.7134), so it learned to
over-trust llm_score. The stack trains on dev-80% where the LLM is honest.

But there is a second failure mode that masking does NOT cover: a feature
whose DISTRIBUTION differs between dev and test. Masking only fixes the
SPARSITY (fraction of rows with a non-zero value); it cannot fix a feature
that is present on 100% of rows in both splits but whose VALUES are shifted.
llm_score is exactly such a feature (dev mean -1.2961 vs test -1.3892).

This script measures, for every feature in the production table, the dev vs
test distribution, so any remaining shift is visible before submission.

Usage: python src/diag_devtest_feature_shift.py [--sample 3000000]
"""
import argparse
import os
import sys

import numpy as np

DEV = "results/ranker/large_dev_features_v2.tsv"
TEST = "results/ranker/lg9b_test_features_devpopcat.tsv"

# The 14 base features are stored in BOTH tables, so they can be compared
# directly. The popularity features (dev_pop/cat_pop/sub_pop) are NOT in the
# dev table -- they come from the causal cache -- and were already compared
# by diag_devpop_shift.py.
FEATS = ["llm_score", "cf_score", "cand_pop", "cand_ctr", "cand_log_pop",
         "cat_match_frac", "subcat_match_frac", "hist_len", "hist_uniq_cats",
         "hist_top_cat_frac", "imp_size", "position", "position_frac",
         "hour"]

# llm_score in the dev FEATURE TABLE is stale (the trainer overwrites it with
# the lg9b scores). The comparison that matters uses the score files the
# model actually saw: lg9b_ep082_dev_scores.tsv vs lg9b_ep082_test_scores_all.tsv.
DEV_LLM = "results/ranker/lg9b_ep082_dev_scores.tsv"
TEST_LLM = "results/ranker/lg9b_ep082_test_scores_all.tsv"


def sample_table(path, cols, max_rows, stride=1):
    """Read every `stride`-th row of the table, keeping only `cols`."""
    out = {c: [] for c in cols}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        missing = [c for c in cols if c not in idx]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}")
        take = [idx[c] for c in cols]
        for n, line in enumerate(f):
            if n % stride:
                continue
            p = line.rstrip("\n").split("\t")
            for c, i in zip(cols, take):
                out[c].append(float(p[i]))
            if len(out[cols[0]]) >= max_rows:
                break
    return {c: np.asarray(v, dtype=np.float64) for c, v in out.items()}


def describe(v):
    return (f"mean={v.mean():+.4f} med={np.median(v):+.4f} "
            f"std={v.std():.4f} p1={np.percentile(v,1):+.4f} "
            f"p99={np.percentile(v,99):+.4f} zero={np.mean(v==0):.3f}")


def sample_llm_scores(path, max_rows, stride=1):
    """Sample the raw LLM score files (impression_id -> tab-separated scores).

    These have no header: the first line is already data.
    """
    vals = []
    with open(path) as f:
        for n, line in enumerate(f):
            if n % stride:
                continue
            p = line.rstrip("\n").split("\t")
            for v in p[1:]:
                vals.append(float(v))
            if len(vals) >= max_rows:
                break
    return np.asarray(vals, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=3_000_000)
    args = ap.parse_args()

    # The test table has 93M rows; a stride keeps the scan bounded while
    # still covering the whole file (so the sample is not time-biased).
    print(f"Scanning DEV  {DEV} ...", flush=True)
    dev = sample_table(DEV, FEATS, args.sample, stride=1)
    print(f"  {len(dev[FEATS[0]]):,} rows", flush=True)

    print(f"Scanning TEST {TEST} ...", flush=True)
    test = sample_table(TEST, FEATS, args.sample, stride=31)
    print(f"  {len(test[FEATS[0]]):,} rows", flush=True)

    print(f"\n{'feature':<18} {'dev':<62} {'test':<62}")
    print("-" * 145)
    for c in FEATS:
        if c == "llm_score":
            continue  # stale in the dev table; compared from the score files
        d, t = dev[c], test[c]
        # Standardised mean shift: how many test-sigmas apart the means are.
        sd = max(d.std(), 1e-9)
        shift = (t.mean() - d.mean()) / sd
        flag = "  <== SHIFT" if abs(shift) > 0.25 else ""
        print(f"{c:<18} {describe(d):<62} {describe(t):<62} z={shift:+.3f}{flag}")

    # The single most important check: llm_score is present on 100% of rows
    # in both splits, so masking cannot touch it. Compare the score files the
    # model actually saw.
    print("\n=== llm_score: the unmaskable covariate shift ===")
    print(f"Scanning {DEV_LLM} ...", flush=True)
    d = sample_llm_scores(DEV_LLM, args.sample, stride=1)
    print(f"Scanning {TEST_LLM} ...", flush=True)
    t = sample_llm_scores(TEST_LLM, args.sample, stride=31)
    for name, v in (("dev", d), ("test", t)):
        print(f"  {name:<5} n={len(v):>10,}  mean={v.mean():+.4f}  "
              f"med={np.median(v):+.4f}  std={v.std():.4f}  "
              f"p10={np.percentile(v,10):+.4f}  p90={np.percentile(v,90):+.4f}")
    shift = t.mean() - d.mean()
    print(f"  mean shift = {shift:+.4f} ({shift/d.std():+.3f} dev-sigma)")
    print("  NOTE: llm_score is present on 100% of rows in both splits, so the")
    print("        test-regime masking cannot correct this shift.")


if __name__ == "__main__":
    main()
