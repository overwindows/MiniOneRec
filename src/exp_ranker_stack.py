#!/usr/bin/env python3
"""Stack every confirmed win into one ranker and measure the total.

WHY
---
Four independent levers have each been measured on the honest (test-regime)
validation set:

  1. JOINT masking      +0.0044 +/- 0.0030   (exp_ranker_jointfair2.py:
     train_pop and dev_pop are masked to the measured TEST JOINT
     distribution, not their independent marginals)
  2. MORE ROUNDS        +0.0052              (exp_ranker_capacity.py:
     300 -> 2000 boosting rounds, still climbing)
  3. LAMBDARANK         +0.0011..0.0017      (exp_ranker_objective.py:
     the metric is per-impression AUC, so a within-group ranking objective
     matches the metric better than a global pointwise logloss)
  4. WITHIN-IMPRESSION   +0.0001..0.0025     (exp_ranker_objective.py:
     per-impression rank/z-score of llm_score and cf_score)

Each was measured ALONE, so the total is not simply their sum -- the levers
may overlap (e.g. lambdarank and within-impression transforms both exploit
within-impression structure). This script measures the stack directly.

Every arm is scored on the SAME joint-masked validation set, so the numbers
are directly comparable and reflect the test regime.

Usage: python src/exp_ranker_stack.py [--seeds 3] [--threads 12]
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

BASE_PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05,
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
    n_pos = np.bincount(g_s[pos], minlength=len(ug)).astype(float)
    n_neg = np.bincount(g_s[~pos], minlength=len(ug)).astype(float)
    rsum = np.bincount(g_s[pos], weights=ranks[pos], minlength=len(ug))
    valid = (n_pos > 0) & (n_neg > 0)
    return float(((rsum[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2) /
                  (n_pos[valid] * n_neg[valid])).mean())


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


def _avg_rank_grouped(v, g):
    """Average (fractional) rank of v within each group g, normalized to 0..1.

    Plain `argsort().argsort()` breaks ties by POSITION, which injects pure
    noise: after test-regime masking, cf_score is 0 for 88.7% of rows, so
    most candidates in an impression tie at zero and would get an arbitrary
    0..m-1 ordering. Average rank gives every tied candidate the same value.

    Fully vectorized: sort by (group, value), find tie runs, assign each run
    the mean of the ranks it spans.
    """
    n = len(v)
    o = np.lexsort((v, g))
    gs = g[o]
    vs = v[o]

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
    avg_norm = avg0 / denom

    out = np.empty(n, dtype=np.float32)
    out[o] = np.repeat(avg_norm, run_len)
    return out


def _zscore_grouped(v, g):
    """Per-group z-score of v (population std), vectorized."""
    cnt = np.bincount(g).astype(np.float64)
    s1 = np.bincount(g, weights=v.astype(np.float64))
    s2 = np.bincount(g, weights=(v.astype(np.float64) ** 2))
    mean = s1 / np.maximum(cnt, 1)
    var = np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0.0)
    return ((v - mean[g]) / (np.sqrt(var[g]) + 1e-6)).astype(np.float32)


def within_imp_transforms(X, g, avg_rank=False):
    """Per-impression rank + z-score of llm_score and cf_score."""
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
    ap.add_argument("--threads", type=int, default=12)
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
        np.where(C["causal_pop"] > 0,
                 C["causal_clicks"] / np.maximum(C["causal_pop"], 1), 0.0),
        np.log1p(C["causal_pop"]) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
    CAT = np.column_stack([
        C["causal_cat_pop"] / mc,
        np.where(C["causal_cat_pop"] > 0,
                 C["causal_cat_clicks"] / np.maximum(C["causal_cat_pop"], 1), 0.0)]).astype(np.float32)
    SUBCAT = np.column_stack([
        S["sub_pop"] / ms,
        np.where(S["sub_pop"] > 0,
                 S["sub_clk"] / np.maximum(S["sub_pop"], 1), 0.0)]).astype(np.float32)

    train_hit = X[:, 2] > 0
    cf_hit = X[:, 1] != 0
    dev_hit = C["causal_pop"] > 0

    # group SIZES for LightGBM's lambdarank `group` argument
    gtr_sizes = np.bincount(g[tr]).astype(np.int32)

    results = {}
    for seed in range(args.seeds):
        # --- EVAL mask: joint (test-like). Every arm is scored on this. ---
        rng = np.random.default_rng(2000 + seed)
        m_tr_eval, m_dv_eval = joint_mask(nids, train_hit, dev_hit, rng)
        m_cf_eval = mask_rows_to_coverage(cf_hit, COV_CF, rng)
        Xeval = X.copy()
        Xeval[~m_tr_eval, 2:5] = 0.0
        Xeval[~m_cf_eval, 1] = 0.0
        dp_eval = DEVPOP.copy()
        dp_eval[~m_dv_eval] = 0.0
        # WIT must be computed from the MASKED matrix: at test time cf_score is
        # genuinely absent for 88.7% of rows, so the within-impression z-score
        # is taken over a mostly-zero column. Computing it from the unmasked
        # X would train the model on a distribution it never sees at test.
        WIT_eval = within_imp_transforms(Xeval, g)
        WIT_eval_ar = within_imp_transforms(Xeval, g, avg_rank=True)
        Xeval_full = np.hstack([Xeval, dp_eval, CAT, SUBCAT])
        Xeval_full_wit = np.hstack([Xeval_full, WIT_eval])
        Xeval_full_wit_ar = np.hstack([Xeval_full, WIT_eval_ar])

        # --- TRAIN masks ---
        # indep: each family masked to its own marginal test coverage
        rng_i = np.random.default_rng(3000 + seed)
        m_tr_i = mask_news_to_coverage(nids, train_hit, 0.128, rng_i) & train_hit
        m_dv_i = mask_news_to_coverage(nids, dev_hit, 0.155, rng_i) & dev_hit
        m_cf_i = mask_rows_to_coverage(cf_hit, COV_CF, rng_i)
        # joint: joint distribution, DIFFERENT seed from the eval mask
        rng_j = np.random.default_rng(4000 + seed)
        m_tr_j, m_dv_j = joint_mask(nids, train_hit, dev_hit, rng_j)
        m_cf_j = mask_rows_to_coverage(cf_hit, COV_CF, rng_j)

        def build(mt, md, mc_, with_wit, wit=None):
            Xtr = X.copy()
            Xtr[~mt, 2:5] = 0.0
            Xtr[~mc_, 1] = 0.0
            dptr = DEVPOP.copy()
            dptr[~md] = 0.0
            blocks = [Xtr, dptr, CAT, SUBCAT]
            if with_wit:
                # WIT from the MASKED train matrix, for the same reason as the
                # eval matrix: the transform must see the sparsity the model
                # will face, not the dense dev regime.
                blocks.append(wit)
            return np.hstack(blocks)

        # Arms F/G/H share the joint train mask, so their masked-WIT matrices
        # are identical -- compute them once (the transform loops over 376k
        # impressions, so recomputing per arm is minutes of wasted CPU).
        Xtr_joint = X.copy()
        Xtr_joint[~m_tr_j, 2:5] = 0.0
        Xtr_joint[~m_cf_j, 1] = 0.0
        WIT_train_joint = within_imp_transforms(Xtr_joint, g)
        WIT_train_joint_ar = within_imp_transforms(Xtr_joint, g, avg_rank=True)
        del Xtr_joint

        # (name, train mask, with_wit, objective, rounds)
        # with_wit: None = no WIT block, "joint" = WIT from the joint train mask
        #           (positional rank), "joint_ar" = same but average (tie-safe) rank
        arms = [
            ("A base            bin/300", (m_tr_i, m_dv_i, m_cf_i), None, "binary", 300),
            ("B +joint          bin/300", (m_tr_j, m_dv_j, m_cf_j), None, "binary", 300),
            ("C +joint+rounds   bin/2000", (m_tr_j, m_dv_j, m_cf_j), None, "binary", 2000),
            ("D +joint+lr       lr/300", (m_tr_j, m_dv_j, m_cf_j), None, "lambdarank", 300),
            ("E +joint+lr+rd    lr/2000", (m_tr_j, m_dv_j, m_cf_j), None, "lambdarank", 2000),
            ("F +joint+lr+rd+wit lr/2000", (m_tr_j, m_dv_j, m_cf_j), "joint", "lambdarank", 2000),
            ("G +joint+rd+wit   bin/2000", (m_tr_j, m_dv_j, m_cf_j), "joint", "binary", 2000),
            ("H +joint+lr+rd+witAR lr/2000", (m_tr_j, m_dv_j, m_cf_j), "joint_ar", "lambdarank", 2000),
        ]

        for name, (mt, md, mc_), wit_kind, obj, rounds in arms:
            if wit_kind == "joint":
                Xf = build(mt, md, mc_, True, WIT_train_joint)
                Xe = Xeval_full_wit
            elif wit_kind == "joint_ar":
                Xf = build(mt, md, mc_, True, WIT_train_joint_ar)
                Xe = Xeval_full_wit_ar
            else:
                Xf = build(mt, md, mc_, False)
                Xe = Xeval_full
            params = dict(BASE_PARAMS)
            if obj == "lambdarank":
                params.update(objective="lambdarank", metric="ndcg",
                              label_gain=[0, 1], lambdarank_truncation_level=30)
            dtr = lgb.Dataset(Xf[tr], label=y[tr], group=gtr_sizes)
            bst = lgb.train(params, dtr, num_boost_round=rounds)
            p = bst.predict(Xe[va])
            a = per_imp(yv, p, gv)
            results.setdefault(name, []).append(a)
            print(f"  seed {seed}  {name:<30} val={a:.4f}", flush=True)
            del Xf, dtr, bst

    print("\n=== stacked comparison (all arms scored on the joint-masked val) ===",
          flush=True)
    for name, vals in results.items():
        v = np.array(vals)
        print(f"{name:<30}  mean={v.mean():.4f}  std={v.std():.4f}  "
              f"vals={['%.4f' % x for x in v]}", flush=True)
    ref = np.array(results["A base            bin/300"])
    print("", flush=True)
    for name, vals in results.items():
        if name.startswith("A "):
            continue
        d = np.array(vals) - ref
        print(f"{name:<30}  - A = {d.mean():+.4f} +/- {d.std():.4f}", flush=True)


if __name__ == "__main__":
    main()
