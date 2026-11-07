#!/usr/bin/env python3
"""Final production-recipe selection for the causal-devpop ranker.

Established so far (exp_ranker_causal.py, exp_ranker_causal2.py):
    masked base + causal devpop + cat  ->  val 0.7618 (honest test regime)

Two more always-available levers are tested here, because 83.3% of test rows
reference a news that appears in NEITHER train NOR dev, so news-level
popularity is missing for the large majority of test rows:

  (a) within-impression transforms of llm_score / cf_score (rank + z-score
      inside the impression). These are computable for EVERY row at test
      time and are scale-free, so they transfer perfectly.
  (b) category popularity (cat_pop, cat_ctr) -- 100% test coverage.

Usage: python src/exp_ranker_final.py
"""
import math
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


def within_imp_transforms(X, g):
    """Per-impression rank + z-score of llm_score and cf_score.

    Computed on the RAW (unmasked) values so the transform is identical at
    test time: the rank of a candidate among its impression-mates does not
    depend on which other candidates happen to have a popularity feature.
    """
    n = X.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    llm = X[:, 0]
    cf = X[:, 1]
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
        lr = l.argsort().argsort() / (m - 1)
        cr = c.argsort().argsort() / (m - 1)
        lz = (l - l.mean()) / (l.std() + 1e-6)
        cz = (c - c.mean()) / (c.std() + 1e-6)
        out[order[s0:e0], 0] = lr
        out[order[s0:e0], 1] = lz
        out[order[s0:e0], 2] = cr
        out[order[s0:e0], 3] = cz
    return out


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
max_pop = float(C["max_pop"])
max_cat = float(C["max_cat"])
log_max = math.log1p(max_pop)

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

cp = C["causal_pop"]
cc = C["causal_clicks"]
ccp = C["causal_cat_pop"]
ccc = C["causal_cat_clicks"]
DEVPOP = np.column_stack([
    cp / max_pop,
    np.where(cp > 0, cc / np.maximum(cp, 1), 0.0),
    np.log1p(cp) / log_max]).astype(np.float32)
DEVPOP_M = DEVPOP.copy()
DEVPOP_M[~m_devpop] = 0.0
CAT = np.column_stack([
    ccp / max_cat,
    np.where(ccp > 0, ccc / np.maximum(ccp, 1), 0.0)]).astype(np.float32)

print("Computing within-impression transforms ...", flush=True)
WIT = within_imp_transforms(X, g)

cf0 = X[:, 1] == 0
pop0 = X[:, 2] == 0.0
fully_cold_row = cf0 & pop0
frac = np.bincount(g, weights=fully_cold_row.astype(float)) / np.bincount(g)
fc_imps = np.where(frac == 1.0)[0]
fc_mask = np.isin(g, fc_imps)
va = g >= cut
fc_va = fc_mask[va]
yv, gv = y[va], g[va]
tr = g < cut


def fit_eval(Xf, name):
    dtr = lgb.Dataset(Xf[tr], label=y[tr])
    bst = lgb.train(PARAMS, dtr, num_boost_round=300)
    p = bst.predict(Xf[va])
    a, n = per_imp(yv, p, gv)
    ac, nc = per_imp(yv, p, gv, fc_va)
    print(f"{name:>44}  val={a:.4f} (n={n})  cold={ac:.4f} (n={nc})", flush=True)
    return a


print("\n=== within-impression transforms ===", flush=True)
fit_eval(Xm, "masked base")
fit_eval(np.hstack([Xm, WIT]), "masked base + within-imp transforms")
fit_eval(np.hstack([Xm, DEVPOP_M, CAT]), "masked base + devpop + cat")
fit_eval(np.hstack([Xm, DEVPOP_M, CAT, WIT]), "masked base + devpop + cat + within-imp")
fit_eval(np.hstack([Xm, WIT, CAT]), "masked base + within-imp + cat")

print("\n=== does cat help the 83% no-popularity rows? ===", flush=True)
# rows whose news is in neither train nor dev -> no devpop, no trainpop
no_pop = (~m_devpop) & (~m_trainpop)
print(f"  rows with no news-level popularity: {no_pop.mean():.1%}", flush=True)
for nm, Xf in [("base", Xm),
               ("base+cat", np.hstack([Xm, CAT])),
               ("base+devpop+cat", np.hstack([Xm, DEVPOP_M, CAT])),
               ("base+devpop+cat+within", np.hstack([Xm, DEVPOP_M, CAT, WIT]))]:
    dtr = lgb.Dataset(Xf[tr], label=y[tr])
    bst = lgb.train(PARAMS, dtr, num_boost_round=300)
    p = bst.predict(Xf[va])
    sub = no_pop[va]
    a, n = per_imp(yv, p, gv, sub)
    print(f"{nm:>44}  no-pop rows val={a:.4f} (n={n})", flush=True)

print("\n=== reference ===", flush=True)
a0, _ = per_imp(yv, X[va, 0], gv)
print(f"{'llm9 alone':>44}  val={a0:.4f}", flush=True)
