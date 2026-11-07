#!/usr/bin/env python3
"""Remap the train dev_pop values onto the TEST dev_pop distribution.

WHY
---
diag_devpop_masked_match.py measured the largest remaining train/test
mismatch in the whole pipeline:

    masked TRAIN dev_pop: mean=0.1820  p25=0.0937  med=0.1560  p75=0.2810
    TEST          dev_pop: mean=0.0610  p25=0.0000  med=0.0000  p75=0.0000
    34.4% of the TEST dev_pop mass falls OUTSIDE the train p1..p99 range

Masking fixes the SPARSITY (84.5% of test rows have dev_pop==0) but not the
VALUES of the surviving rows. The model therefore learns "dev_pop=0.15 =>
clicked" from a regime where 0.15 is common, and at test time 0.15 is in the
top decile -- so it over-trusts the feature exactly where it is wrong.

FIX: quantile mapping (histogram matching). The train dev_pop values are
remapped through their own empirical CDF onto the TEST empirical CDF, so the
model sees the SAME marginal distribution it will face at inference. The
mapping is monotone, so it preserves the ordering information (the part the
model actually uses) while fixing the scale.

The mapping is derived from the TEST feature table (already built) and applied
to BOTH the train and the val rows, so the val measurement is honest.

Usage: python src/exp_ranker_devpop_remap.py
"""
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
TEST_FEATS = "results/ranker/lg9b_test_features_devpopcat.tsv"

COV_TRAINPOP = 0.128
COV_CF = 0.113
COV_DEVPOP = 0.155

SEED = 13
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
    if mask is not None:
        _, ggg = np.unique(ggg, return_inverse=True)
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


def test_devpop_distribution(path, max_rows=20_000_000):
    """Empirical CDF of the TEST dev_pop values on the rows where it is > 0.

    The test feature table has 93M rows; the nonzero ones are ~15.5% of them
    (~14M), so a 20M-row cap is plenty and keeps the scan bounded.
    """
    print(f"Scanning {path} for the TEST dev_pop distribution ...", flush=True)
    vals = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_dp = header.index("dev_pop")
        for n, line in enumerate(f):
            if n >= max_rows:
                break
            p = line.rstrip("\n").split("\t")
            v = float(p[i_dp])
            if v > 0:
                vals.append(v)
    v = np.sort(np.array(vals, dtype=np.float64))
    print(f"  {len(v):,} nonzero test dev_pop values "
          f"(mean={v.mean():.4f} med={np.median(v):.4f})", flush=True)
    return v


def quantile_map(src, ref_sorted):
    """Map each value in `src` through its own CDF onto `ref_sorted`'s CDF.

    Monotone, so the ordering (what the model uses) is preserved; only the
    marginal distribution changes.
    """
    s = np.sort(src)
    q = np.searchsorted(s, src, side="right") / max(len(s), 1)
    pos = np.clip((q * len(ref_sorted)).astype(int), 0, len(ref_sorted) - 1)
    return ref_sorted[pos].astype(np.float32)


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

rng = np.random.default_rng(SEED)
trainpop_hit = X[:, 2] > 0
m_trainpop = mask_news_to_coverage(nids, trainpop_hit, COV_TRAINPOP, rng)
m_trainpop &= trainpop_hit
cf_hit = X[:, 1] != 0
m_cf = mask_rows_to_coverage(cf_hit, COV_CF, rng)
devpop_hit = C["causal_pop"] > 0
m_devpop = mask_news_to_coverage(nids, devpop_hit, COV_DEVPOP, rng)
m_devpop &= devpop_hit
print(f"  masks: trainpop={m_trainpop.mean():.1%}  cf={m_cf.mean():.1%}  "
      f"devpop={m_devpop.mean():.1%}", flush=True)

Xm = X.copy()
Xm[~m_trainpop, 2:5] = 0.0
Xm[~m_cf, 1] = 0.0

mp = np.maximum(C["asof_max_pop"], 1.0)
mc = np.maximum(C["asof_max_cat"], 1.0)
ms = np.maximum(S["sub_asof"], 1.0)
DEVPOP = np.column_stack([
    C["causal_pop"] / mp,
    np.where(C["causal_pop"] > 0,
             C["causal_clicks"] / np.maximum(C["causal_pop"], 1), 0.0),
    np.log1p(C["causal_pop"]) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
DEVPOP_M = DEVPOP.copy()
DEVPOP_M[~m_devpop] = 0.0
CAT = np.column_stack([
    C["causal_cat_pop"] / mc,
    np.where(C["causal_cat_pop"] > 0,
             C["causal_cat_clicks"] / np.maximum(C["causal_cat_pop"], 1), 0.0)]).astype(np.float32)
SUBCAT = np.column_stack([
    S["sub_pop"] / ms,
    np.where(S["sub_pop"] > 0, S["sub_clk"] / np.maximum(S["sub_pop"], 1), 0.0)]).astype(np.float32)

va = g >= cut
tr = g < cut
yv, gv = y[va], g[va]

# ---- the remap ----
test_vals = test_devpop_distribution(TEST_FEATS)
# The train values that survive masking are the ones the model actually sees.
src = DEVPOP_M[m_devpop, 0]
print(f"  train masked dev_pop: n={len(src):,} mean={src.mean():.4f} "
      f"med={np.median(src):.4f}", flush=True)
remapped = quantile_map(src, test_vals)
DEVPOP_R = DEVPOP_M.copy()
DEVPOP_R[m_devpop, 0] = remapped
# the log-pop feature is monotone in the same quantity, so remap it too
src_log = DEVPOP_M[m_devpop, 2]
DEVPOP_R[m_devpop, 2] = quantile_map(src_log, test_vals)
print(f"  remapped dev_pop: mean={DEVPOP_R[m_devpop,0].mean():.4f} "
      f"med={np.median(DEVPOP_R[m_devpop,0]):.4f}", flush=True)


def fit_eval(Xf, name):
    dtr = lgb.Dataset(Xf[tr], label=y[tr])
    bst = lgb.train(PARAMS, dtr, num_boost_round=300)
    p = bst.predict(Xf[va])
    a, n = per_imp(yv, p, gv)
    print(f"{name:>44}  val={a:.4f} (n={n})", flush=True)
    return a


print("\n=== dev_pop quantile remap ===", flush=True)
BASE = np.hstack([Xm, CAT, SUBCAT])
fit_eval(BASE, "base + cat + subcat (no devpop)")
fit_eval(np.hstack([BASE, DEVPOP_M]), "base + cat + subcat + devpop (as-is)")
fit_eval(np.hstack([BASE, DEVPOP_R]), "base + cat + subcat + devpop (remapped)")
fit_eval(np.hstack([Xm, DEVPOP_M, CAT, SUBCAT]), "base + devpop + cat + subcat")
fit_eval(np.hstack([Xm, DEVPOP_R, CAT, SUBCAT]), "base + devpop REMAP + cat + subcat")
