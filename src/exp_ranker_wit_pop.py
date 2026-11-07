#!/usr/bin/env python3
"""Does a within-impression transform on the POPULARITY features help?

WHY
---
`exp_ranker_wit_exact.py` settled the WIT question for llm_score/cf_score:
the production-exact arm (average rank) is the best of the three, but only
+0.0007 +/- 0.0009 over no-WIT. The old "+0.0070" was an artifact of the
older feature set.

But WIT is currently applied ONLY to llm_score and cf_score. The two
strongest always-available features -- category popularity and subcategory
popularity -- have 100% test coverage (vs 15.5% for news-level devpop) and
carry NO within-impression transform.

That is a real gap. A tree splits on ABSOLUTE values, so it can learn
"cat_pop > 0.3 is good" but it CANNOT learn "this candidate is the most
popular category WITHIN this impression". The latter is a different
question, and it is exactly the question the metric asks: the score is
per-impression AUC, so only the ORDER WITHIN an impression matters.

Concretely, an impression with candidates from categories {sport: 0.9,
politics: 0.2, sport: 0.8} has a very different best candidate than one
with {sport: 0.9, politics: 0.8, sport: 0.7} -- but the absolute cat_pop
values are nearly identical. Only a within-impression rank/z-score can
tell them apart.

Arms (all production-exact: binary + joint mask + 2000 rounds + average-rank
WIT on llm/cf, the config that `train_ranker_causal_final.py --wit` builds):

    N  prod-WITavg            llm/cf WIT only  (CURRENT PRODUCTION)
    P  N + WIT(cat_pop,sub_pop)          +4 cols
    Q  N + WIT(cat,sub,cat_ctr,sub_ctr)  +8 cols

Usage: python src/exp_ranker_wit_pop.py [--seeds 3] [--threads 32]
"""
import argparse
import os
import sys

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


def wit_cols(v, g):
    """Average rank + z-score of one column, within each impression."""
    return np.column_stack([_avg_rank_grouped(v, g), _zscore_grouped(v, g)]
                            ).astype(np.float32)


def within_imp_transforms(X, g):
    """Production-exact WIT: average rank + z-score of llm_score, cf_score."""
    out = np.zeros((X.shape[0], 4), dtype=np.float32)
    out[:, 0] = _avg_rank_grouped(X[:, 0], g)
    out[:, 1] = _zscore_grouped(X[:, 0], g)
    out[:, 2] = _avg_rank_grouped(X[:, 1], g)
    out[:, 3] = _zscore_grouped(X[:, 1], g)
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

    # popularity columns to transform: cat_pop, cat_ctr, sub_pop, sub_ctr
    POP4 = np.hstack([CAT, SUBCAT])          # [cat_pop, cat_ctr, sub_pop, sub_ctr]
    POP2 = np.hstack([CAT[:, :1], SUBCAT[:, :1]])   # [cat_pop, sub_pop]

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
        WIT_eval = within_imp_transforms(Xeval, g)
        Xeval_base = np.hstack([Xeval, dp_eval, CAT, SUBCAT])
        Xeval_N = np.hstack([Xeval_base, WIT_eval])
        Xeval_P = np.hstack([Xeval_N, wit_cols(POP2[:, 0], g),
                             wit_cols(POP2[:, 1], g)])
        Xeval_Q = np.hstack([Xeval_N, wit_cols(POP4[:, 0], g),
                             wit_cols(POP4[:, 1], g),
                             wit_cols(POP4[:, 2], g),
                             wit_cols(POP4[:, 3], g)])

        # ---- TRAIN mask: joint, DIFFERENT seed from the eval mask ----
        rng_j = np.random.default_rng(4000 + seed)
        m_tr_j, m_dv_j = joint_mask(nids, trainpop_hit, devpop_hit, rng_j)
        m_cf_j = mask_rows_to_coverage(cf_hit, COV_CF, rng_j)
        Xtr = X.copy()
        Xtr[~m_tr_j, 2:5] = 0.0
        Xtr[~m_cf_j, 1] = 0.0
        dptr = DEVPOP.copy()
        dptr[~m_dv_j] = 0.0
        WIT_tr = within_imp_transforms(Xtr, g)
        Xtr_base = np.hstack([Xtr, dptr, CAT, SUBCAT])
        Xtr_N = np.hstack([Xtr_base, WIT_tr])
        Xtr_P = np.hstack([Xtr_N, wit_cols(POP2[:, 0], g),
                           wit_cols(POP2[:, 1], g)])
        Xtr_Q = np.hstack([Xtr_N, wit_cols(POP4[:, 0], g),
                           wit_cols(POP4[:, 1], g),
                           wit_cols(POP4[:, 2], g),
                           wit_cols(POP4[:, 3], g)])
        del Xtr, WIT_tr

        arms = [
            ("N  prod-WITavg          bin/2000", Xtr_N, Xeval_N),
            ("P  N+WIT(cat_pop,sub_pop)      ", Xtr_P, Xeval_P),
            ("Q  N+WIT(cat,sub,ctr,subctr)  ", Xtr_Q, Xeval_Q),
        ]
        for name, Xf, Xe in arms:
            dtr = lgb.Dataset(Xf[tr], label=y[tr])
            bst = lgb.train(BASE_PARAMS, dtr, num_boost_round=2000)
            p = bst.predict(Xe[va])
            a = per_imp(yv, p, gv)
            results.setdefault(name, []).append(a)
            print(f"  seed {seed}  {name}  val={a:.4f}", flush=True)
            del dtr, bst
        del Xtr_base, Xtr_N, Xtr_P, Xtr_Q, Xeval_base, Xeval_N, Xeval_P, Xeval_Q

    print("\n=== popularity-WIT comparison "
          "(joint-masked held-out val) ===", flush=True)
    for name, vals in results.items():
        v = np.array(vals)
        print(f"{name}  mean={v.mean():.4f}  std={v.std():.4f}  "
              f"vals={['%.4f' % x for x in v]}", flush=True)
    ref = np.array(results["N  prod-WITavg          bin/2000"])
    print("", flush=True)
    for name, vals in results.items():
        if name.startswith("N "):
            continue
        d = np.array(vals) - ref
        print(f"{name}  - N = {d.mean():+.4f} +/- {d.std():.4f}", flush=True)


if __name__ == "__main__":
    main()
