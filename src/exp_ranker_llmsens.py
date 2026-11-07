#!/usr/bin/env python3
"""How much does the RANKER care about LLM quality?

WHY
---
The LLM is the single most expensive component (a full test re-score costs
~13 GPU-hours on 8 nodes) and the ranker spends ~49% of its gain on
llm_score. Before spending that, measure the sensitivity:

    ranker(dev, llm=ckpt-32500, AUC 0.7134)  vs
    ranker(dev, llm=ckpt-38500, AUC 0.7092)

If the ranker AUC barely moves for a 0.004 LLM drop, then LLM quality is NOT
the bottleneck and re-scoring the test set is wasted GPU time. If it moves a
lot, re-scoring with the best checkpoint (ckpt-30500, AUC 0.7162) is worth it.

Also tests whether `imp_size` -- which is CONSTANT within an impression and
therefore cannot change within-impression ordering -- is worth its 30% of
gain, or whether the model is burning capacity on global calibration.

Usage: python src/exp_ranker_llmsens.py [--seeds 3] [--threads 16]
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups  # noqa: E402

import lightgbm as lgb  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features.tsv"
LLM_FILES = {
    "ckpt32500": "results/ranker/lg9b_ep082_dev_scores.tsv",
    "ckpt38500": "results/ranker/lg9b_ep099_dev_scores.tsv",
}
CACHE = "results/ranker/causal_dev_arrays.npz"
CACHE2 = "results/ranker/causal_subcat_arrays.npz"

COV_TRAINPOP = 0.128
COV_CF = 0.113
COV_DEVPOP = 0.155

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05,
              num_leaves=63, min_child_samples=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=16, verbose=-1)


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


def per_imp(y_t, s, gg, mask=None):
    yy, ss, ggg = (y_t, s, gg) if mask is None else (y_t[mask], s[mask], gg[mask])
    order = np.argsort(ggg, kind="stable")
    g_s, p_s, y_s = ggg[order], ss[order], yy[order]
    ug = np.unique(g_s)
    starts = np.searchsorted(g_s, ug, side="left")
    ends = np.searchsorted(g_s, ug, side="right")
    ranks = np.empty(len(p_s))
    for s0, e0 in zip(starts, ends):
        o = np.argsort(p_s[s0:e0], kind="stable")
        ranks[s0:e0] = o.argsort() + 1
    pos = y_s == 1
    n_pos = np.bincount(g_s[pos], minlength=len(ug)).astype(float)
    n_neg = np.bincount(g_s[~pos], minlength=len(ug)).astype(float)
    rsum = np.bincount(g_s[pos], weights=ranks[pos], minlength=len(ug))
    valid = (n_pos > 0) & (n_neg > 0)
    return float(((rsum[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2) /
                  (n_pos[valid] * n_neg[valid])).mean()), int(valid.sum())


def mask_news_to_coverage(nids, hit, target, rng):
    total = len(nids)
    want = int(target * total)
    wmap = Counter()
    for n, h in zip(nids, hit):
        if h:
            wmap[n] += 1
    keys = list(wmap.keys())
    order = rng.permutation(len(keys))
    keep = set()
    cov = 0
    for j in order:
        n = keys[j]
        keep.add(n)
        cov += wmap[n]
        if cov >= want:
            break
    return np.array([n in keep for n in nids], dtype=bool)


def mask_rows_to_coverage(hit, target, rng):
    total = len(hit)
    want = int(target * total)
    idx = np.where(hit)[0]
    rng.shuffle(idx)
    m = np.zeros(total, dtype=bool)
    m[idx[:want]] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    PARAMS["num_threads"] = args.threads

    print("Loading dev features ...", flush=True)
    Xs, ys, gs = [], [], []
    for gi, (Xb, y) in enumerate(stream_groups(DEV_FEATS)):
        Xs.append(Xb)
        ys.append(y)
        gs.append(np.full(len(y), gi, dtype=np.int64))
    X0 = np.vstack(Xs)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    del Xs, ys
    ids = imp_ids(DEV_FEATS)
    n_imp = g.max() + 1
    cut = int(n_imp * 0.8)
    print(f"  {X0.shape[0]:,} rows, {n_imp:,} imps", flush=True)

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

    mp = np.maximum(C["asof_max_pop"], 1.0)
    mc = np.maximum(C["asof_max_cat"], 1.0)
    ms = np.maximum(S["sub_asof"], 1.0)
    DEVPOP = np.column_stack([
        C["causal_pop"] / mp,
        np.where(C["causal_pop"] > 0, C["causal_clicks"] / np.maximum(C["causal_pop"], 1), 0.0),
        np.log1p(C["causal_pop"]) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
    CAT = np.column_stack([
        C["causal_cat_pop"] / mc,
        np.where(C["causal_cat_pop"] > 0,
                 C["causal_cat_clicks"] / np.maximum(C["causal_cat_pop"], 1), 0.0)]).astype(np.float32)
    SUBCAT = np.column_stack([
        S["sub_pop"] / ms,
        np.where(S["sub_pop"] > 0, S["sub_clk"] / np.maximum(S["sub_pop"], 1), 0.0)]).astype(np.float32)

    trainpop_hit = X0[:, 2] > 0
    cf_hit = X0[:, 1] != 0
    devpop_hit = C["causal_pop"] > 0

    # imp_size is column 10 in BASE_FEATS
    IMP_SIZE_COL = 10

    results = {}
    for llm_name, llm_path in LLM_FILES.items():
        llm = load_llm(llm_path)
        X = X0.copy()
        off = 0
        n_miss = 0
        for imp, n in ids:
            s = llm.get(imp)
            if s and len(s) == n:
                X[off:off + n, 0] = s
            else:
                n_miss += 1
            off += n
        print(f"  {llm_name}: missing llm for {n_miss} imps", flush=True)

        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            m_trainpop = mask_news_to_coverage(nids, trainpop_hit, COV_TRAINPOP, rng)
            m_trainpop &= trainpop_hit
            m_cf = mask_rows_to_coverage(cf_hit, COV_CF, rng)
            m_devpop = mask_news_to_coverage(nids, devpop_hit, COV_DEVPOP, rng)
            m_devpop &= devpop_hit

            Xm = X.copy()
            Xm[~m_trainpop, 2:5] = 0.0
            Xm[~m_cf, 1] = 0.0
            dp = DEVPOP.copy()
            dp[~m_devpop] = 0.0

            full = np.hstack([Xm, dp, CAT, SUBCAT])
            # drop imp_size (col 10 of Xm)
            keep = [i for i in range(Xm.shape[1]) if i != IMP_SIZE_COL]
            no_impsize = np.hstack([Xm[:, keep], dp, CAT, SUBCAT])

            for fname, Xf in [("full", full), ("no_impsize", no_impsize)]:
                dtr = lgb.Dataset(Xf[tr], label=y[tr])
                bst = lgb.train(PARAMS, dtr, num_boost_round=300)
                p = bst.predict(Xf[va])
                a, n = per_imp(yv, p, gv)
                key = f"{llm_name} | {fname}"
                results.setdefault(key, []).append(a)
                print(f"  seed {seed}  {key:>28}  val={a:.4f}", flush=True)
            del Xm, dp, full, no_impsize
        del X

    print("\n=== summary over %d mask seeds ===" % args.seeds, flush=True)
    for key, vals in results.items():
        v = np.array(vals)
        print(f"{key:>30}  mean={v.mean():.4f}  std={v.std():.4f}", flush=True)
    a = np.array(results["ckpt32500 | full"])
    b = np.array(results["ckpt38500 | full"])
    print(f"\nLLM sensitivity: ckpt32500 - ckpt38500 = {(a-b).mean():+.4f} "
          f"(LLM AUC gap 0.7134-0.7092 = 0.0042)", flush=True)


if __name__ == "__main__":
    main()
