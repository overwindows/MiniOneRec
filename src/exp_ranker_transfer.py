#!/usr/bin/env python3
"""Does the stack's dev gain survive the TEST regime?

WHY
---
The stack ranker claims +0.0575 per-impression AUC over the LLM on dev.
The previous (l13) ranker claimed +0.0105 on dev and transferred as ~0.000
at test. Before submitting the stack we must know whether its gain is real.

Two candidate explanations for the l13 failure:
  (a) the l13 ranker trained on the TRAIN split, where the LLM MEMORIZED
      (train probe AUC 0.7692 vs dev 0.7134), so it learned to just trust
      llm_score (79% of its gain is llm_score + imp_size) and ignored the
      popularity features that are absent for 83.3% of test rows;
  (b) the stack's gain is a dev artifact that will not transfer.

This script distinguishes them with two measurements on the SAME stack model
(trained on dev-80%, arm G recipe):

  1. TIME-BLOCK STABILITY: split the val block (dev-20%) into 4 consecutive
     time blocks and report the gain over llm_score in each. Test is the
     period AFTER dev, so if the gain decays toward the end of dev it will
     not transfer.

  2. COLD SLICE: 83.3% of test rows reference a news that appears in NEITHER
     train NOR dev, so every popularity feature is absent for them. On dev
     the news are warm. This measures the gain on the dev rows whose news is
     genuinely cold -- the closest available proxy for the test regime.

Usage: python src/exp_ranker_transfer.py [--rounds 2000] [--threads 32]
"""
import argparse
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
TRAIN_BEHAVIORS = "data/MIND_large/train/behaviors.tsv"
DEV_BEHAVIORS = "data/MIND_large/dev/behaviors.tsv"

TEST_BOTH = 0.11599
TEST_TR_ONLY = 0.01163
TEST_DV_ONLY = 0.03928
TEST_NEITHER = 0.83310
COV_CF = 0.113
SEED = 13


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
    """Mean per-impression AUC over the groups in gg."""
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
    for i, n in enumerate(nids):
        c = assign[n]
        if c in ("both", "tr_only"):
            m_train[i] = True
        if c in ("both", "dv_only"):
            m_dev[i] = True
    return m_train, m_dev


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


def news_set(path):
    s = set()
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            for tk in parts[4].split():
                if "-" in tk:
                    s.add(tk.rsplit("-", 1)[0])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=32)
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
                 C["causal_cat_clicks"] / np.maximum(C["causal_cat_pop"], 1), 0.0)
    ]).astype(np.float32)
    SUBCAT = np.column_stack([
        S["sub_pop"] / ms,
        np.where(S["sub_pop"] > 0, S["sub_clk"] / np.maximum(S["sub_pop"], 1), 0.0)
    ]).astype(np.float32)

    train_hit = X[:, 2] > 0
    cf_hit = X[:, 1] != 0
    dev_hit = C["causal_pop"] > 0

    # ---- masks (arm G recipe: joint mask, different seed for train vs eval)
    rng_j = np.random.default_rng(4000)
    m_tr_j, m_dv_j = joint_mask(nids, train_hit, dev_hit, rng_j)
    m_cf_j = mask_rows_to_coverage(cf_hit, COV_CF, rng_j)
    rng_e = np.random.default_rng(2000)
    m_tr_e, m_dv_e = joint_mask(nids, train_hit, dev_hit, rng_e)
    m_cf_e = mask_rows_to_coverage(cf_hit, COV_CF, rng_e)

    Xtr = X.copy()
    Xtr[~m_tr_j, 2:5] = 0.0
    Xtr[~m_cf_j, 1] = 0.0
    dptr = DEVPOP.copy()
    dptr[~m_dv_j] = 0.0
    WIT_train = within_imp_transforms(Xtr, g)
    Xf = np.hstack([Xtr, dptr, CAT, SUBCAT, WIT_train])
    del Xtr, WIT_train

    Xev = X.copy()
    Xev[~m_tr_e, 2:5] = 0.0
    Xev[~m_cf_e, 1] = 0.0
    dpe = DEVPOP.copy()
    dpe[~m_dv_e] = 0.0
    WIT_eval = within_imp_transforms(Xev, g)
    Xe = np.hstack([Xev, dpe, CAT, SUBCAT, WIT_eval])
    del Xev, WIT_eval

    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=63, min_child_samples=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=args.threads, verbose=-1)
    print(f"Training arm G (joint mask + WIT, binary/{args.rounds}) ...", flush=True)
    dtr = lgb.Dataset(Xf[tr], label=y[tr])
    bst = lgb.train(params, dtr, num_boost_round=args.rounds)
    Xva = Xe[va]
    p = bst.predict(Xva)
    llm_p = Xva[:, 0]

    a_all, n_all = per_imp(yv, p, gv)
    a_llm, _ = per_imp(yv, llm_p, gv)
    print(f"\n=== overall val (dev-20%) ===")
    print(f"  llm_score only   AUC={a_llm:.4f}")
    print(f"  stack (arm G)    AUC={a_all:.4f}   gain={a_all-a_llm:+.4f}  (n={n_all})")

    # ---- 1. time-block stability ----
    print(f"\n=== gain by val time block (block 4 is closest to test) ===")
    print(f"{'block':>6} {'imps':>8} {'llm AUC':>9} {'stack AUC':>10} {'gain':>8}")
    ug = np.unique(gv)
    nb = 4
    edges = np.linspace(0, len(ug), nb + 1).astype(int)
    for b in range(nb):
        lo, hi = edges[b], edges[b + 1]
        sel = (gv >= ug[lo]) & (gv <= ug[hi - 1])
        if not sel.any():
            continue
        ab, _ = per_imp(yv[sel], p[sel], gv[sel])
        al, _ = per_imp(yv[sel], llm_p[sel], gv[sel])
        print(f"{b+1:>6} {hi-lo:>8,} {al:>9.4f} {ab:>10.4f} {ab-al:>+8.4f}")

    # ---- 2. cold slice ----
    print(f"\n=== cold slice (news in NEITHER train NOR dev) ===", flush=True)
    tr_news = news_set(TRAIN_BEHAVIORS)
    dv_news = news_set(DEV_BEHAVIORS)
    cold = np.array([n not in tr_news and n not in dv_news for n in nids])
    print(f"  dev rows with cold news: {cold.mean():.1%} "
          f"(test: 83.3%)", flush=True)
    for name, sel in (("cold rows", va & cold), ("warm rows", va & ~cold)):
        if not sel.any():
            continue
        ab, nb_ = per_imp(y[sel], p[sel], g[sel])
        al, _ = per_imp(y[sel], llm_p[sel], g[sel])
        print(f"  {name:<10} n={nb_:>7,}  llm={al:.4f}  stack={ab:.4f}  "
              f"gain={ab-al:+.4f}")

    # ---- 3. per-impression gain distribution on the cold slice ----
    # A mean gain can hide a few impressions where the stack is much worse.
    print(f"\n=== per-impression gain distribution (cold rows) ===", flush=True)
    sel = va & cold
    order = np.argsort(g[sel], kind="stable")
    ys, ps, ls, gs2 = y[sel][order], p[sel][order], llm_p[sel][order], g[sel][order]
    ug2 = np.unique(gs2)
    st = np.searchsorted(gs2, ug2, side="left")
    en = np.searchsorted(gs2, ug2, side="right")
    gains = []
    for s0, e0 in zip(st, en):
        yy = ys[s0:e0]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        gains.append(_auc(yy, ps[s0:e0]) - _auc(yy, ls[s0:e0]))
    gains = np.array(gains)
    print(f"  n={len(gains):,}  mean={gains.mean():+.4f}  med={np.median(gains):+.4f}")
    for q in (1, 5, 25, 50, 75, 95, 99):
        print(f"    p{q:<3} = {np.percentile(gains, q):+.4f}")
    print(f"  fraction of cold impressions where the stack is WORSE: "
          f"{np.mean(gains < 0):.1%}")


def _auc(y, s):
    """Plain AUC for one impression."""
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    npos = y.sum()
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


if __name__ == "__main__":
    main()
