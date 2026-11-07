#!/usr/bin/env python3
"""Variant comparison for the dev-trained ranker (all CPU, honest val = last 20% dev):
  1. dev-only vs train+dev (does old train data help or hurt?)
  2. BASE_FEATS vs +within-impression augment (v2-style features)
"""
import sys
import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups, BASE_FEATS, augment  # noqa: E402

import lightgbm as lgb  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features.tsv"
TRAIN_FEATS = "results/ranker/large_train_features.tsv"
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"
# l13 train scores exist for the train table; lg9b train scores do NOT,
# so for train rows we keep l13 llm (mismatch) -> skip train mixing on llm9;
# instead test augment features dev-only, and train-mix with l13 dev table.

def load_dev():
    Xs, ys, gs = [], [], []
    for gi, (Xb, y) in enumerate(stream_groups(DEV_FEATS)):
        Xs.append(Xb); ys.append(y)
        gs.append(np.full(len(y), gi, dtype=np.int64))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)

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
        cur = None; n = 0
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[i_imp] != cur:
                if cur is not None:
                    out.append((cur, n))
                cur = p[i_imp]; n = 0
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
    return float(((rsum[valid] - n_pos[valid]*(n_pos[valid]+1)/2) /
                  (n_pos[valid]*n_neg[valid])).mean()), int(valid.sum())

print("Loading dev ...", flush=True)
X, y, g = load_dev()
ids = imp_ids(DEV_FEATS)
llm9 = load_llm(LG9B_DEV)
off = 0
for gi, (imp, n) in enumerate(ids):
    s = llm9.get(imp)
    if s and len(s) == n:
        X[off:off+n, 0] = s
    off += n
print(f"  {X.shape[0]:,} rows", flush=True)

cold_row = X[:, 1] == 0
imp_cold_frac = np.bincount(g, weights=cold_row.astype(float)) / np.bincount(g)
cold_imps = np.where(imp_cold_frac == 1.0)[0]
imp_mask = np.isin(g, cold_imps)

n_imp = g.max() + 1
cut = int(n_imp * 0.8)
tr = g < cut; va = ~tr
yv, gv, va_mask = y[va], g[va], imp_mask[va]

params = dict(objective="binary", metric="auc", learning_rate=0.05,
              num_leaves=63, min_child_samples=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=32, verbose=-1)

def fit_eval(Xtr_full, name):
    dtr = lgb.Dataset(Xtr_full[tr], label=y[tr])
    bst = lgb.train(params, dtr, num_boost_round=300)
    p = bst.predict(Xtr_full[va])
    a, n = per_imp(yv, p, gv)
    ac, nc = per_imp(yv, p, gv, va_mask)
    print(f"{name:>16}  per-imp={a:.4f} (n={n})  cold={ac:.4f} (n={nc})", flush=True)
    return bst

print("\n=== VAL slice (last 20% dev) ===", flush=True)
fit_eval(X, "base_devonly")
Xa = np.vstack([augment(Xb) for Xb in
                [X[np.searchsorted(g, gi):np.searchsorted(g, gi+1)]
                 for gi in range(n_imp)]]) if False else None
# augment per impression efficiently: group rows are contiguous
Xaug = np.empty((X.shape[0], 16), dtype=np.float32)
off = 0
for gi in range(n_imp):
    e = np.searchsorted(g, gi+1)
    Xaug[off:e] = augment(X[off:e])
    off = e
fit_eval(Xaug, "aug_devonly")

# train+dev mixing: train table has l13 llm scores (col 0) — different model.
# Substitute is impossible (no lg9b train scores); test mixing with l13 dev
# scores instead to isolate the "recent labels" effect.
print("\n(note: train-table llm col is l13; mixing tests label-recency effect)", flush=True)
Xs, ys2, gs2 = [], [], []
for gi, (Xb, yy) in enumerate(stream_groups(TRAIN_FEATS, 400000)):
    Xs.append(Xb); ys2.append(yy)
    gs2.append(np.full(len(yy), gi, dtype=np.int64))
Xtr_train = np.vstack(Xs); ytr_train = np.concatenate(ys2)
print(f"  train subsample: {Xtr_train.shape[0]:,} rows", flush=True)
Xmix = np.vstack([Xtr_train, X[tr]])
ymix = np.concatenate([ytr_train, y[tr]])
dtr = lgb.Dataset(Xmix, label=ymix)
bst = lgb.train(params, dtr, num_boost_round=300)
p = bst.predict(X[va])
a, n = per_imp(yv, p, gv)
ac, nc = per_imp(yv, p, gv, va_mask)
print(f"{'train400k+dev':>16}  per-imp={a:.4f} (n={n})  cold={ac:.4f} (n={nc})", flush=True)