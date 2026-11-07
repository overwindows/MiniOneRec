#!/usr/bin/env python3
"""Ranker-on-lg9b experiment, all on dev (no GPU needed).

Questions:
  A) Does the EXISTING v1 ranker (trained on l13 train scores) still work when
     llm_score is swapped for lg_9b ep082 scores?  (distribution-shift check)
  B) Does a v1-style binary LightGBM RETRAINED on dev impressions with lg_9b
     scores beat plain llm order on a held-out (chronological last 20%) slice,
     especially cold-only impressions?

Feature set = v1 BASE_FEATS (llm, cf, cand_pop, cand_ctr, cand_log_pop,
cat/subcat match, hist stats, imp_size, position, position_frac, hour).
"""
import sys
import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups, BASE_FEATS  # noqa: E402

import lightgbm as lgb  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features.tsv"
LG9B = "results/ranker/lg9b_ep082_dev_scores.tsv"
V1 = "results/ranker/ranker_l13_large.txt"

# ---------- load dev features (l13 llm_score) ----------
print("Streaming dev features ...", flush=True)
Xs, ys, gs, imp_ids = [], [], [], []
for gi, (Xb, y) in enumerate(stream_groups(DEV_FEATS)):
    Xs.append(Xb)
    ys.append(y)
    gs.append(np.full(len(y), gi, dtype=np.int64))
X = np.vstack(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
del Xs, ys
print(f"  {X.shape[0]:,} rows, {g.max()+1:,} impressions", flush=True)

# ---------- load lg9b scores, swap into col 0 ----------
print("Loading lg9b dev scores ...", flush=True)
llm9 = {}
with open(LG9B) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            llm9[p[0]] = [float(v) for v in p[1:]]
print(f"  {len(llm9):,} impressions", flush=True)

# need impression ids per group: re-stream just ids in same order
ids_per_group = []
with open(DEV_FEATS) as f:
    header = f.readline().rstrip("\n").split("\t")
    i_imp = header.index("impression_id")
    i_nid = header.index("news_id")
    cur = None
    cur_n = 0
    for line in f:
        p = line.rstrip("\n").split("\t")
        if p[i_imp] != cur:
            if cur is not None:
                ids_per_group.append((cur, cur_n))
            cur = p[i_imp]; cur_n = 0
        cur_n += 1
    ids_per_group.append((cur, cur_n))
assert len(ids_per_group) == g.max() + 1

X9 = X.copy()
n_miss = 0
off = 0
for gi, (imp, n) in enumerate(ids_per_group):
    s = llm9.get(imp)
    if s is None or len(s) != n:
        n_miss += 1
        off += n
        continue
    X9[off:off+n, 0] = s
    off += n
print(f"  swapped llm col; missing/mismatched imps: {n_miss}", flush=True)

# ---------- per-imp AUC (vectorized rank-sum) ----------
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

cold_row = X9[:, 1] == 0
imp_cold_frac = np.bincount(g, weights=cold_row.astype(float)) / np.bincount(g)
cold_imps = np.where(imp_cold_frac == 1.0)[0]
imp_mask = np.isin(g, cold_imps)
print(f"cold-only impressions: {len(cold_imps):,} ({len(cold_imps)/(g.max()+1):.1%})", flush=True)

# ---------- A) existing v1 ranker, llm swapped ----------
v1 = lgb.Booster(model_file=V1)
p_v1_on_9 = v1.predict(X9)
p_v1_on_l13 = v1.predict(X)
llm9_s = X9[:, 0]

# ---------- B) retrain v1-style on first 80% of dev (chronological), val last 20% ----------
n_imp = g.max() + 1
cut = int(n_imp * 0.8)
tr = g < cut
va = ~tr
print(f"train imps {cut:,}, val imps {n_imp-cut:,}", flush=True)

params = dict(objective="binary", metric="auc", learning_rate=0.05,
              num_leaves=63, min_child_samples=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=32, verbose=-1)
dtr = lgb.Dataset(X9[tr], label=y[tr])
bst = lgb.train(params, dtr, num_boost_round=300)
p_b = bst.predict(X9[va])

# reference on val slice
llm_va = llm9_s[va]; yv = y[va]; gv = g[va]
p_v1_va = p_v1_on_9[va]
va_mask = imp_mask[va]

print("\n=== VAL (last 20% dev) ===")
for name, s in [("llm9", llm_va), ("v1_on_9", p_v1_va), ("retrain", p_b)]:
    a, n = per_imp(yv, s, gv)
    ac, nc = per_imp(yv, s, gv, va_mask)
    print(f"{name:>8}  per-imp={a:.4f} (n={n})  cold={ac:.4f} (n={nc})", flush=True)

print("\n=== FULL DEV (train-slice numbers are fit; val-only is honest) ===")
for name, s in [("llm9", llm9_s), ("v1_on_9", p_v1_on_9)]:
    a, n = per_imp(y, s, g)
    ac, nc = per_imp(y, s, g, imp_mask)
    print(f"{name:>8}  per-imp={a:.4f} (n={n})  cold={ac:.4f} (n={nc})", flush=True)

bst.save_model("results/ranker/ranker_v1_lg9b_dev80.txt")
print("\nsaved results/ranker/ranker_v1_lg9b_dev80.txt")