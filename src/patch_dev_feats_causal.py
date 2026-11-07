#!/usr/bin/env python3
"""Rebuild the dev eval table's popularity columns PRIOR-ONLY (leak-free).

WHY
---
The first v6 dev eval used append_devpop_cat_test.py, which computes
dev_pop/dev_ctr/... from the FULL dev split. That is the correct
construction for TEST rows (all of dev precedes test), but on DEV rows it
folds each row's own impression -- and its own click -- into its own
features (self-inclusion leak), plus uses future dev labels
(transductive leak). Single-feature LOO probe on 2M rows:
dev_ctr 0.7411->0.7358, cat_ctr 0.6120->0.5505, sub_ctr 0.6738->0.6562.

This script replaces the 7 popularity columns with the CAUSAL DEV walk
(each row sees only PRIOR dev impressions), the exact construction the
v6 trainer used on train rows (causal train walk). No self, no future.

Usage:
    python src/patch_dev_feats_causal.py \
        --base results/ranker/ck45500_dev_features.tsv \
        --out results/ranker/ck45500_dev_features_causalpop.tsv
"""
import argparse

import numpy as np

CACHE = "results/ranker/causal_dev_arrays.npz"
CACHE2 = "results/ranker/causal_subcat_arrays.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/ranker/ck45500_dev_features.tsv")
    ap.add_argument("--out", default="results/ranker/ck45500_dev_features_causalpop.tsv")
    args = ap.parse_args()

    z = np.load(CACHE)
    C = {k: z[k] for k in z.files}
    z2 = np.load(CACHE2)
    S = {k: z2[k] for k in z2.files}

    mp = np.maximum(C["asof_max_pop"], 1.0)
    mc = np.maximum(C["asof_max_cat"], 1.0)
    ms = np.maximum(S["sub_asof"], 1.0)
    dev_pop = C["causal_pop"] / mp
    dev_ctr = np.where(C["causal_pop"] > 0,
                       C["causal_clicks"] / np.maximum(C["causal_pop"], 1), 0.0)
    dev_log = np.log1p(C["causal_pop"]) / np.maximum(np.log1p(mp), 1e-6)
    cat_pop = C["causal_cat_pop"] / mc
    cat_ctr = np.where(C["causal_cat_pop"] > 0,
                       C["causal_cat_clicks"] / np.maximum(C["causal_cat_pop"], 1), 0.0)
    sub_pop = S["sub_pop"] / ms
    sub_ctr = np.where(S["sub_pop"] > 0,
                       S["sub_clk"] / np.maximum(S["sub_pop"], 1), 0.0)

    n = 0
    with open(args.base) as f, open(args.out, "w") as out:
        header = f.readline().rstrip("\n").split("\t")
        out.write("\t".join(header + ["dev_pop", "dev_ctr", "dev_log_pop",
                                      "cat_pop", "cat_ctr",
                                      "sub_pop", "sub_ctr"]) + "\n")
        for line in f:
            p = line.rstrip("\n").split("\t")
            p.append(f"{dev_pop[n]:.6f}")
            p.append(f"{dev_ctr[n]:.6f}")
            p.append(f"{dev_log[n]:.6f}")
            p.append(f"{cat_pop[n]:.6f}")
            p.append(f"{cat_ctr[n]:.6f}")
            p.append(f"{sub_pop[n]:.6f}")
            p.append(f"{sub_ctr[n]:.6f}")
            out.write("\t".join(p) + "\n")
            n += 1
            if n % 2_000_000 == 0:
                print(f"  {n:,} rows", flush=True)
    assert n == len(dev_pop), f"row mismatch {n} vs {len(dev_pop)}"
    print(f"Wrote {n:,} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()