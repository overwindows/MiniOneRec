#!/usr/bin/env python3
"""Measure the PRODUCTION-EXACT stack recipe on a held-out dev split.

WHY
---
`exp_ranker_stack.py` found that the best arm was

    G  +joint+rd+wit  bin/2000      mean=0.7663  (+0.0070 +/- 0.0024 vs A)

but arm G's within-impression transform used the POSITIONAL rank
(`argsort().argsort()`), while BOTH the production trainer
(`train_ranker_causal_final.py`) and the serving path
(`apply_ranker_devpop.py`) use the tie-safe AVERAGE rank
(`_avg_rank_grouped`).

That is not a cosmetic difference. After test-regime masking, cf_score is 0
for 88.7% of rows, so most candidates in an impression tie at zero:
  * positional rank gives them an arbitrary 0..m-1 ordering (pure noise),
  * average rank gives them all the same value.

So the "+0.0070" number was measured on a transform that the shipped model
does NOT use. The production-exact arm (binary + joint + rounds + WIT with
average rank) was never measured at all.

This script measures the three arms that matter, on the SAME joint-masked
held-out validation set, with the SAME feature set the production trainer
builds (base 14 + devpop 3 + catpop 2 + subpop 2 [+ WIT 4]):

    C  prod-noWIT      binary + joint + 2000 rounds, no WIT
    G  prod-WITpos     same + WIT with POSITIONAL rank  (the old "winner")
    N  prod-WITavg     same + WIT with AVERAGE rank    (PRODUCTION-EXACT)

Arm N is the one that decides what we ship.

Usage: python src/exp_ranker_wit_exact.py [--seeds 3] [--threads 32]
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups  # noqa: E402
from exp_ranker_stack import (  # noqa: E402
    load_llm, imp_ids, per_imp, mask_news_to_coverage, mask_rows_to_coverage,
    joint_mask, _avg_rank_grouped, _zscore_grouped,
)

import lightgbm as lgb  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features.tsv"
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"
CACHE = "results/ranker/causal_dev_arrays.npz"
CACHE2 = "results/ranker/causal_subcat_arrays.npz"

COV_CF = 0.113

BASE_PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05,
                   num_leaves=63, min_child_samples=200, feature_fraction=0.9,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   num_threads=16, verbose=-1)


def within_imp_transforms(X, g, avg_rank=False):
    """Per-impression rank + z-score of llm_score and cf_score.

    avg_rank=False -> POSITIONAL rank (argsort().argsort()), the transform
    used by exp_ranker_stack.py arm G.
    avg_rank=True  -> tie-safe AVERAGE rank, the transform used by the
    production trainer and the serving path.
    """
    n = X.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    llm = X[:, 0]
    cf = X[:, 1]
    if avg_rank:
        out[:, 0] = _avg_rank_grouped(llm, g)
        out[:, 2] = _avg_rank_grouped(cf, g)
    else:
        order = np.argsort(g, kind="stable")
        g_s = g[order]
        llm_s = llm[order]
        cf_s = cf[order]
        ug = np.unique(g_s)
        starts = np.searchsorted(g_s, ug, side="left")
        ends = np.searchsorted(g_s, ug, side="right")
        for s0, e0 in zip(starts, ends):
            m = e0 - s0
            if m <= 1:
                continue
            l = llm_s[s0:e0]
            c = cf_s[s0:e0]
            out[order[s0:e0], 0] = l.argsort().argsort() / (m - 1)
            out[order[s0:e0], 2] = c.argsort().argsort() / (m - 1)
    out[:, 1] = _zscore_grouped(llm, g)
    out[:, 3] = _zscore_grouped(cf, g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    BASE_PARAMS["num_threads"] = args.threads

    print("Loading dev features ...", flush=True)
    Xs, ys, gs = [], [], []
    for gi, (Xb, y) in enumerate(stream_groups(DEV_FEATS)):
        Xs.append(Xb)
        ys.append(y)
        gs.append(np.full(len(y), gi, dtype=np.int64))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    del Xs, ys
    ids = imp_ids(DEV_FEATS)
    llm9 = load_llm(LG9B_DEV)
    off = 0
    for imp, n in ids:
        s = llm9.get(imp)
        if s and len(s) == n:
            X[off:off + n, 0] = s
        off += n
    n_imp = g.max() + 1
    cut = int(n_imp * 0.8)
    print(f"  {X.shape[0]:,} rows, {n_imp:,} imps, cut={cut}", flush=True)

    z = np.load(CACHE)
    C = {k: z[k] for k in z.files}
    z2 = np.load(CACHE2)
    S = {k: z2[k] for k in z2.files}

    with open(DEV_FEATS) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_nid = header.index("news_id")
    nids = []
    with open(DEV_FEATS) as f:
        f.readline()
        for line in f:
            nids.append(line.rstrip("\n").split("\t")[i_nid])

    va = g >= cut
    tr = g < cut
    yv, gv = y[va], g[va]

    # ---- production feature blocks (identical to train_ranker_causal_final) ----
    mp = np.maximum(C["asof_max_pop"], 1.0)
    mc = np.maximum(C["asof_max_cat"], 1.0)
    ms = np.maximum(S["sub_asof"], 1.0)
    DEVPOP = np.column_stack([
        C["causal_pop"] / mp,
        np.where(C["causal_pop"] > 0,
                 C["causal_clicks"] / np.maximum(C["causal_pop"], 1), 0.0),
        np.log1p(C["causal_pop"]) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
    CAT = np.column_stack([
        C["causal_cat_pop"] / mc,
        np.where(C["causal_cat_pop"] > 0,
                 C["causal_cat_clicks"] / np.maximum(C["causal_cat_pop"], 1), 0.0)
    ]).astype(np.float32)
    SUBCAT = np.column_stack([
        S["sub_pop"] / ms,
        np.where(S["sub_pop"] > 0,
                 S["sub_clk"] / np.maximum(S["sub_pop"], 1), 0.0)]).astype(np.float32)

    trainpop_hit = X[:, 2] > 0
    cf_hit = X[:, 1] != 0
    devpop_hit = C["causal_pop"] > 0

    results = {}
    for seed in range(args.seeds):
        # ---- EVAL mask: joint (test-like), independent of the train mask ----
        rng = np.random.default_rng(2000 + seed)
        m_tr_eval, m_dv_eval = joint_mask(nids, trainpop_hit, devpop_hit, rng)
        m_cf_eval = mask_rows_to_coverage(cf_hit, COV_CF, rng)
        Xeval = X.copy()
        Xeval[~m_tr_eval, 2:5] = 0.0
        Xeval[~m_cf_eval, 1] = 0.0
        dp_eval = DEVPOP.copy()
        dp_eval[~m_dv_eval] = 0.0
        # WIT from the MASKED matrix (test-time sparsity)
        WIT_eval_pos = within_imp_transforms(Xeval, g, avg_rank=False)
        WIT_eval_avg = within_imp_transforms(Xeval, g, avg_rank=True)
        Xeval_base = np.hstack([Xeval, dp_eval, CAT, SUBCAT])
        Xeval_pos = np.hstack([Xeval_base, WIT_eval_pos])
        Xeval_avg = np.hstack([Xeval_base, WIT_eval_avg])

        # ---- TRAIN mask: joint, DIFFERENT seed from the eval mask ----
        rng_j = np.random.default_rng(4000 + seed)
        m_tr_j, m_dv_j = joint_mask(nids, trainpop_hit, devpop_hit, rng_j)
        m_cf_j = mask_rows_to_coverage(cf_hit, COV_CF, rng_j)
        Xtr = X.copy()
        Xtr[~m_tr_j, 2:5] = 0.0
        Xtr[~m_cf_j, 1] = 0.0
        dptr = DEVPOP.copy()
        dptr[~m_dv_j] = 0.0
        WIT_tr_pos = within_imp_transforms(Xtr, g, avg_rank=False)
        WIT_tr_avg = within_imp_transforms(Xtr, g, avg_rank=True)
        Xtr_base = np.hstack([Xtr, dptr, CAT, SUBCAT])
        Xtr_pos = np.hstack([Xtr_base, WIT_tr_pos])
        Xtr_avg = np.hstack([Xtr_base, WIT_tr_avg])
        del Xtr, WIT_tr_pos, WIT_tr_avg

        arms = [
            ("C  prod-noWIT   bin/2000", Xtr_base, Xeval_base),
            ("G  prod-WITpos  bin/2000", Xtr_pos, Xeval_pos),
            ("N  prod-WITavg  bin/2000", Xtr_avg, Xeval_avg),
        ]
        for name, Xf, Xe in arms:
            dtr = lgb.Dataset(Xf[tr], label=y[tr])
            bst = lgb.train(BASE_PARAMS, dtr, num_boost_round=2000)
            p = bst.predict(Xe[va])
            a = per_imp(yv, p, gv)
            results.setdefault(name, []).append(a)
            print(f"  seed {seed}  {name:<28} val={a:.4f}", flush=True)
            del dtr, bst
        del Xtr_base, Xtr_pos, Xtr_avg, Xeval_base, Xeval_pos, Xeval_avg

    print("\n=== production-exact stack comparison "
          "(joint-masked held-out val) ===", flush=True)
    for name, vals in results.items():
        v = np.array(vals)
        print(f"{name:<28}  mean={v.mean():.4f}  std={v.std():.4f}  "
              f"vals={['%.4f' % x for x in v]}", flush=True)
    ref = np.array(results["C  prod-noWIT   bin/2000"])
    print("", flush=True)
    for name, vals in results.items():
        if name.startswith("C "):
            continue
        d = np.array(vals) - ref
        print(f"{name:<28}  - C = {d.mean():+.4f} +/- {d.std():.4f}", flush=True)
    # the decisive comparison: production-exact vs the old "winner"
    d = np.array(results["N  prod-WITavg  bin/2000"]) - \
        np.array(results["G  prod-WITpos  bin/2000"])
    print(f"\nN (production-exact) - G (positional) = {d.mean():+.4f} "
          f"+/- {d.std():.4f}", flush=True)


if __name__ == "__main__":
    main()
