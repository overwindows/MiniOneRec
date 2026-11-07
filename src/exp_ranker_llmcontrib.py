#!/usr/bin/env python3
"""CORRECTED LLM ablation: how much does the LLM actually contribute?

WHY THIS SCRIPT EXISTS
----------------------
src/exp_ranker_ablate.py reported that dropping `llm_score` costs only
0.0013 AUC, and we concluded "the LLM is nearly irrelevant to the final
score". That conclusion was WRONG: the ablation dropped only the raw
`llm_score` column but KEPT `llm_rank` and `llm_z`, which are the
within-impression average-rank and z-score of `llm_score` -- monotone
transforms of the SAME signal. The LLM information was never removed.

This script removes the LLM signal COMPLETELY:
    llm_all = llm_score, llm_rank, llm_z
and also reports the "no-LLM-at-all" stack, i.e. what the popularity +
profile features alone can do.

The answer decides where to spend the remaining compute:
  - if the LLM contributes a lot -> improving the LLM (better checkpoint,
    DPO, new prompt variants) is worth a full test re-score;
  - if it contributes ~nothing -> the LLM work is done and the remaining
    leverage is entirely in ranker features.

Usage: python src/exp_ranker_llmcontrib.py [--rounds 2000] [--threads 16]
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups  # noqa: E402

import lightgbm as lgb  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features_v2.tsv"
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"
CACHE = "results/ranker/causal_dev_arrays.npz"
CACHE2 = "results/ranker/causal_subcat_arrays.npz"

TEST_BOTH = 0.11599
TEST_TR_ONLY = 0.01163
TEST_DV_ONLY = 0.03928
TEST_NEITHER = 0.83310
COV_CF = 0.113

ALL_FEATS = ["llm_score", "cf_score", "cand_pop", "cand_ctr", "cand_log_pop",
             "cat_match_frac", "subcat_match_frac", "hist_len",
             "hist_uniq_cats", "hist_top_cat_frac", "imp_size", "position",
             "position_frac", "hour", "dev_pop", "dev_ctr", "dev_log_pop",
             "cat_pop", "cat_ctr", "sub_pop", "sub_ctr", "llm_rank", "llm_z",
             "cf_rank", "cf_z"]

# The LLM signal, in full.
LLM_ALL = ["llm_score", "llm_rank", "llm_z"]
# The CF signal, in full (cf_score + its within-impression transforms).
CF_ALL = ["cf_score", "cf_rank", "cf_z"]


def load_llm(path):
    out = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out[p[0]] = [float(v) for v in p[1:]]
    return out


def imp_ids(path):
    out = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_imp = header.index("impression_id")
        cur = None
        n = 0
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[i_imp] != cur:
                if cur is not None:
                    out.append((cur, n))
                cur = p[i_imp]
                n = 0
            n += 1
    out.append((cur, n))
    return out


def per_imp(y_t, s, gg):
    order = np.argsort(gg, kind="stable")
    g_s, p_s, y_s = gg[order], s[order], y_t[order]
    ug = np.unique(g_s)
    starts = np.searchsorted(g_s, ug, side="left")
    ends = np.searchsorted(g_s, ug, side="right")
    ranks = np.empty(len(p_s))
    for s0, e0 in zip(starts, ends):
        o = np.argsort(p_s[s0:e0], kind="stable")
        ranks[s0:e0] = o.argsort() + 1
    pos = y_s == 1
    n_grp = int(g_s.max()) + 1
    n_pos = np.bincount(g_s[pos], minlength=n_grp).astype(float)
    n_neg = np.bincount(g_s[~pos], minlength=n_grp).astype(float)
    rsum = np.bincount(g_s[pos], weights=ranks[pos], minlength=n_grp)
    valid = (n_pos > 0) & (n_neg > 0)
    return float(((rsum[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2) /
                  (n_pos[valid] * n_neg[valid])).mean()), int(valid.sum())


def mask_rows_to_coverage(hit, target, rng):
    total = len(hit)
    want = int(target * total)
    idx = np.where(hit)[0]
    rng.shuffle(idx)
    m = np.zeros(total, dtype=bool)
    m[idx[:want]] = True
    return m


def joint_mask(nids, train_hit, dev_hit, rng):
    total = len(nids)
    wmap = Counter(nids)
    has_train, has_dev = {}, {}
    for n, t, d in zip(nids, train_hit, dev_hit):
        if t:
            has_train[n] = True
        if d:
            has_dev[n] = True
    quotas = {"both": int(TEST_BOTH * total),
              "tr_only": int(TEST_TR_ONLY * total),
              "dv_only": int(TEST_DV_ONLY * total),
              "neither": int(TEST_NEITHER * total)}
    remaining = dict(quotas)
    keys = list(wmap.keys())
    order = rng.permutation(len(keys))
    assign = {}
    for j in order:
        n = keys[j]
        t = has_train.get(n, False)
        d = has_dev.get(n, False)
        if t and d:
            feasible = ["both", "tr_only", "dv_only", "neither"]
        elif t:
            feasible = ["tr_only", "neither"]
        elif d:
            feasible = ["dv_only", "neither"]
        else:
            feasible = ["neither"]
        best = max(feasible, key=lambda c: (remaining[c], rng.random()))
        assign[n] = best
        remaining[best] -= wmap[n]
    m_train = np.zeros(total, dtype=bool)
    m_dev = np.zeros(total, dtype=bool)
    cell = np.empty(total, dtype=np.int8)
    cid = {"both": 0, "tr_only": 1, "dv_only": 2, "neither": 3}
    for i, n in enumerate(nids):
        c = assign[n]
        if c in ("both", "tr_only"):
            m_train[i] = True
        if c in ("both", "dv_only"):
            m_dev[i] = True
        cell[i] = cid[c]
    return m_train, m_dev, cell


def _avg_rank_grouped(v, g):
    n = len(v)
    o = np.lexsort((v, g))
    gs, vs = g[o], v[o]
    new_g = np.empty(n, dtype=bool)
    new_g[0] = True
    new_g[1:] = gs[1:] != gs[:-1]
    grp_start = np.flatnonzero(new_g)
    grp_end = np.append(grp_start[1:], n)
    grp_size = grp_end - grp_start
    tie = np.empty(n, dtype=bool)
    tie[0] = True
    tie[1:] = (gs[1:] != gs[:-1]) | (vs[1:] != vs[:-1])
    run_start = np.flatnonzero(tie)
    run_end = np.append(run_start[1:], n)
    run_len = run_end - run_start
    gidx = np.searchsorted(grp_start, run_start, side="right") - 1
    avg0 = (run_start - grp_start[gidx]) + (run_len - 1) / 2.0
    denom = np.maximum(grp_size[gidx] - 1, 1)
    out = np.empty(n, dtype=np.float32)
    out[o] = np.repeat(avg0 / denom, run_len)
    return out


def _zscore_grouped(v, g):
    cnt = np.bincount(g).astype(np.float64)
    s1 = np.bincount(g, weights=v.astype(np.float64))
    s2 = np.bincount(g, weights=(v.astype(np.float64) ** 2))
    mean = s1 / np.maximum(cnt, 1)
    var = np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0.0)
    return ((v - mean[g]) / (np.sqrt(var[g]) + 1e-6)).astype(np.float32)


def within_imp_transforms(X, g):
    out = np.zeros((X.shape[0], 4), dtype=np.float32)
    out[:, 0] = _avg_rank_grouped(X[:, 0], g)
    out[:, 1] = _zscore_grouped(X[:, 0], g)
    out[:, 2] = _avg_rank_grouped(X[:, 1], g)
    out[:, 3] = _zscore_grouped(X[:, 1], g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

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
    va = g >= cut
    tr = g < cut
    yv, gv = y[va], g[va]
    print(f"  {X.shape[0]:,} rows, {n_imp:,} imps, cut={cut}", flush=True)

    with open(DEV_FEATS) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_nid = header.index("news_id")
    nids = []
    with open(DEV_FEATS) as f:
        f.readline()
        for line in f:
            nids.append(line.rstrip("\n").split("\t")[i_nid])

    z = np.load(CACHE)
    C = {"pop": z["causal_pop"], "clicks": z["causal_clicks"],
         "cat_pop": z["causal_cat_pop"], "cat_clicks": z["causal_cat_clicks"],
         "asof_pop": z["asof_max_pop"], "asof_cat": z["asof_max_cat"]}
    z2 = np.load(CACHE2)
    C["sub_pop"] = z2["sub_pop"]
    C["sub_clk"] = z2["sub_clk"]
    C["sub_asof"] = z2["sub_asof"]

    train_hit = X[:, 2] > 0
    cf_hit = X[:, 1] != 0
    dev_hit = C["pop"] > 0

    rng_j = np.random.default_rng(4000)
    m_tr_j, m_dv_j, _ = joint_mask(nids, train_hit, dev_hit, rng_j)
    m_cf_j = mask_rows_to_coverage(cf_hit, COV_CF, rng_j)
    rng_e = np.random.default_rng(2000)
    m_tr_e, m_dv_e, cell_e = joint_mask(nids, train_hit, dev_hit, rng_e)
    m_cf_e = mask_rows_to_coverage(cf_hit, COV_CF, rng_e)

    def blocks(m_dev, m_train, m_cf):
        Xm = X.copy()
        Xm[~m_train, 2:5] = 0.0
        Xm[~m_cf, 1] = 0.0
        mp = np.maximum(C["asof_pop"], 1.0)
        mc = np.maximum(C["asof_cat"], 1.0)
        dp = np.column_stack([
            C["pop"] / mp,
            np.where(C["pop"] > 0, C["clicks"] / np.maximum(C["pop"], 1), 0.0),
            np.log1p(C["pop"]) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
        dp[~m_dev] = 0.0
        cp = np.column_stack([
            C["cat_pop"] / mc,
            np.where(C["cat_pop"] > 0,
                     C["cat_clicks"] / np.maximum(C["cat_pop"], 1), 0.0)
        ]).astype(np.float32)
        ms = np.maximum(C["sub_asof"], 1.0)
        sp = np.column_stack([
            C["sub_pop"] / ms,
            np.where(C["sub_pop"] > 0,
                     C["sub_clk"] / np.maximum(C["sub_pop"], 1), 0.0)
        ]).astype(np.float32)
        return np.hstack([Xm, dp, cp, sp, within_imp_transforms(Xm, g)])

    Xf = blocks(m_dv_j, m_tr_j, m_cf_j)
    Xe = blocks(m_dv_e, m_tr_e, m_cf_e)
    assert Xf.shape[1] == len(ALL_FEATS), (Xf.shape, len(ALL_FEATS))

    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=63, min_child_samples=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=args.threads, verbose=-1)

    def fit_eval(cols, name):
        idx = [ALL_FEATS.index(c) for c in cols]
        dtr = lgb.Dataset(Xf[tr][:, idx], label=y[tr])
        bst = lgb.train(params, dtr, num_boost_round=args.rounds)
        Xva = Xe[va][:, idx]
        p = bst.predict(Xva)
        a, n = per_imp(yv, p, gv)
        al, _ = per_imp(yv, Xva[:, 0], gv)
        # the test-regime slice: no popularity features at all
        nopop = (Xva[:, 2] == 0) & (Xva[:, 1] == 0) & (Xva[:, 14] == 0)
        an, _ = per_imp(yv[nopop], p[nopop], gv[nopop])
        aln, _ = per_imp(yv[nopop], Xva[nopop, 0], gv[nopop])
        print(f"  {name:<30} val={a:.4f}  nopop={an:.4f}  "
              f"(llm-only={al:.4f})  n={n:,}", flush=True)
        return a, an, p

    print("\n=== CORRECTED LLM ABLATION ===", flush=True)
    results = {}
    a_full, an_full, p_full = fit_eval(ALL_FEATS, "FULL (25)")
    results["FULL"] = [a_full, an_full]

    # the OLD (wrong) ablation: drop only the raw llm_score
    keep = [c for c in ALL_FEATS if c != "llm_score"]
    a_old, an_old, _ = fit_eval(keep, "-llm_score only (OLD)")
    results["-llm_score_only"] = [a_old, an_old]

    # the CORRECT ablation: drop the whole LLM signal
    keep = [c for c in ALL_FEATS if c not in LLM_ALL]
    a_nollm, an_nollm, p_nollm = fit_eval(keep, "-llm_score,llm_rank,llm_z")
    results["-llm_all"] = [a_nollm, an_nollm]

    # and the CF signal, for symmetry
    keep = [c for c in ALL_FEATS if c not in CF_ALL]
    a_nocf, an_nocf, _ = fit_eval(keep, "-cf_score,cf_rank,cf_z")
    results["-cf_all"] = [a_nocf, an_nocf]

    # neither LLM nor CF
    keep = [c for c in ALL_FEATS if c not in LLM_ALL and c not in CF_ALL]
    a_none, an_none, _ = fit_eval(keep, "-llm_all -cf_all")
    results["-llm_all-cf_all"] = [a_none, an_none]

    print("\n=== VERDICT ===")
    print(f"  FULL                       val={a_full:.4f}  nopop={an_full:.4f}")
    print(f"  -llm_score only (OLD)      val={a_old:.4f}  nopop={an_old:.4f}"
          f"   drop={a_full-a_old:+.4f}")
    print(f"  -llm_score,llm_rank,llm_z  val={a_nollm:.4f}  nopop={an_nollm:.4f}"
          f"   drop={a_full-a_nollm:+.4f}")
    print(f"  -cf_score,cf_rank,cf_z     val={a_nocf:.4f}  nopop={an_nocf:.4f}"
          f"   drop={a_full-a_nocf:+.4f}")
    print(f"  -llm_all -cf_all           val={a_none:.4f}  nopop={an_none:.4f}"
          f"   drop={a_full-a_none:+.4f}")
    d = a_full - a_nollm
    if d > 0.02:
        print(f"  => the LLM contributes {d:.4f} AUC: LLM improvements DO "
              f"matter, a better checkpoint is worth a test re-score.")
    else:
        print(f"  => the LLM contributes only {d:.4f} AUC: the remaining "
              f"leverage is in ranker features, not the LLM.")

    with open("logs/exp_ranker_llmcontrib.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
