#!/usr/bin/env python3
"""FAIR comparison of independent vs joint test-regime masking.

WHY THIS SCRIPT EXISTS
-----------------------
exp_ranker_jointmask.py evaluates each model on its OWN masked val set:

    indep: base            val=0.7459   (val masked independently)
    joint: base            val=0.7379   (val masked jointly)

Those two numbers are NOT comparable. The joint-masked val set is genuinely
harder (it reproduces the test sparsity: "both trainpop+devpop" is 11.6% of
rows instead of 3.1%), so a lower score there does not mean a worse model.

The only fair comparison is: train both models, then score BOTH on the SAME
val set -- and that val set must be masked to the TEST joint distribution,
because that is the regime the model will actually face at inference.

    model_indep  -> scored on joint-masked val
    model_joint  -> scored on joint-masked val

The higher number is the model that will do better on test.

Usage: python src/exp_ranker_jointfair.py [--seeds 3] [--threads 16]
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
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"
CACHE = "results/ranker/causal_dev_arrays.npz"
CACHE2 = "results/ranker/causal_subcat_arrays.npz"

# Measured TEST joint distribution (train_pop>0, dev_pop>0), over 93,115,001 rows
TEST_BOTH = 0.11599
TEST_TR_ONLY = 0.01163
TEST_DV_ONLY = 0.03928
TEST_NEITHER = 0.83310

COV_CF = 0.113

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


def joint_mask(nids, train_hit, dev_hit, rng):
    """Assign each news to one of 4 cells so the ROW joint distribution
    matches the measured test distribution. Quotas are in row counts."""
    total = len(nids)
    wmap = Counter(nids)
    has_train = {}
    has_dev = {}
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
    for i, n in enumerate(nids):
        c = assign[n]
        if c in ("both", "tr_only"):
            m_train[i] = True
        if c in ("both", "dv_only"):
            m_dev[i] = True
    return m_train, m_dev


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
    print(f"  {X.shape[0]:,} rows, {n_imp:,} imps", flush=True)

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

    train_hit = X[:, 2] > 0
    cf_hit = X[:, 1] != 0
    dev_hit = C["causal_pop"] > 0

    results = {}
    for seed in range(args.seeds):
        rng = np.random.default_rng(2000 + seed)

        # --- the EVALUATION mask: joint (test-like). Both models are scored
        #     on this same val set, so the comparison is apples-to-apples.
        m_tr_eval, m_dv_eval = joint_mask(nids, train_hit, dev_hit, rng)
        m_cf_eval = mask_rows_to_coverage(cf_hit, COV_CF, rng)

        Xeval = X.copy()
        Xeval[~m_tr_eval, 2:5] = 0.0
        Xeval[~m_cf_eval, 1] = 0.0
        dp_eval = DEVPOP.copy()
        dp_eval[~m_dv_eval] = 0.0
        Xeval_full = np.hstack([Xeval, dp_eval, CAT, SUBCAT])

        # --- the TRAINING masks
        # independent: each family masked to its own marginal test coverage
        rng_i = np.random.default_rng(3000 + seed)
        m_tr_i = mask_news_to_coverage(nids, train_hit, 0.128, rng_i)
        m_tr_i &= train_hit
        m_dv_i = mask_news_to_coverage(nids, dev_hit, 0.155, rng_i)
        m_dv_i &= dev_hit
        m_cf_i = mask_rows_to_coverage(cf_hit, COV_CF, rng_i)

        # joint: same 4-way assignment used for the eval mask
        rng_j = np.random.default_rng(2000 + seed)
        m_tr_j, m_dv_j = joint_mask(nids, train_hit, dev_hit, rng_j)
        m_cf_j = mask_rows_to_coverage(cf_hit, COV_CF, rng_j)

        for mname, (mt, md, mc_) in [("indep", (m_tr_i, m_dv_i, m_cf_i)),
                                     ("joint", (m_tr_j, m_dv_j, m_cf_j))]:
            Xtr = X.copy()
            Xtr[~mt, 2:5] = 0.0
            Xtr[~mc_, 1] = 0.0
            dptr = DEVPOP.copy()
            dptr[~md] = 0.0
            Xf = np.hstack([Xtr, dptr, CAT, SUBCAT])
            dtr = lgb.Dataset(Xf[tr], label=y[tr])
            bst = lgb.train(PARAMS, dtr, num_boost_round=300)
            p = bst.predict(Xeval_full[va])
            a, n = per_imp(yv, p, gv)
            key = f"train={mname}"
            results.setdefault(key, []).append(a)
            print(f"  seed {seed}  {key:>16}  scored on JOINT-masked val: "
                  f"val={a:.4f}", flush=True)
            del Xtr, dptr, Xf

    print("\n=== FAIR comparison: both models scored on the joint-masked val ===",
          flush=True)
    for key, vals in results.items():
        v = np.array(vals)
        print(f"{key:>20}  mean={v.mean():.4f}  std={v.std():.4f}  "
              f"vals={['%.4f' % x for x in v]}", flush=True)
    if "train=indep" in results and "train=joint" in results:
        d = np.array(results["train=joint"]) - np.array(results["train=indep"])
        print(f"\njoint - indep = {d.mean():+.4f} +/- {d.std():.4f}  "
              f"(positive => joint masking wins)", flush=True)


if __name__ == "__main__":
    main()
