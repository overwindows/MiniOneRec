#!/usr/bin/env python3
"""Leave-one-impression-out devpop: fixes a train/test leakage in the devpop feature.

The devpop feature for a TRAINING row was computed from dev-80 INCLUDING that
row's own impression. So if news X was clicked in impression i, devpop(X) was
inflated by that click, and the model learned "high devpop => clicked".
At test time devpop comes from dev only and the test impression is NOT in it,
so the feature is a clean prior and the model over-trusts it.

Fix: for every row, subtract that row's own impression contribution:
    pop_loo(nid)   = pop_total(nid)   - 1
    clicks_loo(nid)= clicks_total(nid) - label
For TEST rows no subtraction applies (test impressions are not in dev).

This script measures the honest gain of devpop under leave-one-out, both
unmasked and masked to test coverage.

Usage: python src/exp_ranker_loo_devpop.py
"""
import math
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups  # noqa: E402

import lightgbm as lgb  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features.tsv"
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"
DEV_BEHAVIORS = "data/MIND_large/dev/behaviors.tsv"
NEWS = "data/MIND_large/test/news.tsv"

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
        # Remap the group ids to 0..k-1. After masking, ggg still holds the
        # ORIGINAL impression ids (e.g. 301176..376470), so
        # np.bincount(g_s, minlength=len(ug)) would allocate len(ug) buckets
        # while g_s indexes far beyond them -> broadcast error.
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
tr = g < cut
va = ~tr
print(f"  {X.shape[0]:,} rows, {n_imp:,} imps", flush=True)

news_cat = {}
with open(NEWS) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            news_cat[p[0]] = p[1]

# ---- dev-80 stats (pop/clicks per news, cat pop/clicks) ----
print("Computing dev-80 statistics ...", flush=True)
pop = Counter()
clicks = Counter()
cat_pop = Counter()
cat_clicks = Counter()
n_dev = 0
with open(DEV_BEHAVIORS) as f:
    for line in f:
        if n_dev >= cut:
            break
        parts = line.strip().split("\t")
        if len(parts) < 5:
            n_dev += 1
            continue
        for tk in parts[4].split():
            if "-" not in tk:
                continue
            nid, lab = tk.rsplit("-", 1)
            pop[nid] += 1
            cat = news_cat.get(nid, "")
            cat_pop[cat] += 1
            if lab == "1":
                clicks[nid] += 1
                cat_clicks[cat] += 1
        n_dev += 1
max_pop = max(pop.values()) if pop else 1
log_max = math.log1p(max_pop)
max_cat_pop = max(cat_pop.values()) if cat_pop else 1

# ---- row metadata: news_id + label per row ----
with open(DEV_FEATS) as f:
    header = f.readline().rstrip("\n").split("\t")
    i_nid = header.index("news_id")
nids = []
labels = []
with open(DEV_FEATS) as f:
    f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t")
        nids.append(p[i_nid])
        labels.append(int(p[header.index("label")]))
labels = np.array(labels, dtype=np.float32)

# ---- leave-one-impression-out devpop ----
# every row's own impression contributed exactly 1 to pop[nid] and label to clicks[nid]
dp_total = np.array([pop.get(nid, 0) for nid in nids], dtype=np.float32)
dc_total = np.array([clicks.get(nid, 0) for nid in nids], dtype=np.float32)
dp_loo = np.maximum(dp_total - 1.0, 0.0)
dc_loo = np.maximum(dc_total - labels, 0.0)

DEVPOP_LEAK = np.column_stack([
    dp_total / max_pop,
    np.where(dp_total > 0, dc_total / np.maximum(dp_total, 1), 0.0),
    np.log1p(dp_total) / log_max]).astype(np.float32)
DEVPOP_LOO = np.column_stack([
    dp_loo / max_pop,
    np.where(dp_loo > 0, dc_loo / np.maximum(dp_loo, 1), 0.0),
    np.log1p(dp_loo) / log_max]).astype(np.float32)

# ---- category block (leave-one-out too) ----
c_pop_total = np.array([cat_pop.get(news_cat.get(nid, ""), 0) for nid in nids], dtype=np.float32)
c_clk_total = np.array([cat_clicks.get(news_cat.get(nid, ""), 0) for nid in nids], dtype=np.float32)
c_pop_loo = np.maximum(c_pop_total - 1.0, 0.0)
c_clk_loo = np.maximum(c_clk_total - labels, 0.0)
CAT_LEAK = np.column_stack([c_pop_total / max_cat_pop,
                             np.where(c_pop_total > 0, c_clk_total / np.maximum(c_pop_total, 1), 0.0)]).astype(np.float32)
CAT_LOO = np.column_stack([c_pop_loo / max_cat_pop,
                            np.where(c_pop_loo > 0, c_clk_loo / np.maximum(c_pop_loo, 1), 0.0)]).astype(np.float32)

# ---- mask devpop to test coverage ----
rng = np.random.default_rng(SEED)
all_nids = sorted(pop.keys())
val_rows = np.where(va)[0]
val_counts = Counter(np.array([nids[i] for i in val_rows]).tolist())
total_val_rows = len(val_rows)
order = rng.permutation(len(all_nids))
known = set()
cov = 0
for j in order:
    nid = all_nids[j]
    known.add(nid)
    cov += val_counts.get(nid, 0)
    if cov / total_val_rows >= COV_DEVPOP:
        break
m_devpop = np.array([nid in known for nid in nids], dtype=bool)
print(f"  devpop mask: {m_devpop.mean():.1%} rows kept", flush=True)

DEVPOP_LOO_M = DEVPOP_LOO.copy()
DEVPOP_LOO_M[~m_devpop] = 0.0
DEVPOP_LEAK_M = DEVPOP_LEAK.copy()
DEVPOP_LEAK_M[~m_devpop] = 0.0

# cold slice
cf0 = X[:, 1] == 0
pop0 = X[:, 2] == 0.0
fully_cold_row = cf0 & pop0
frac = np.bincount(g, weights=fully_cold_row.astype(float)) / np.bincount(g)
fc_imps = np.where(frac == 1.0)[0]
fc_mask = np.isin(g, fc_imps)
fc_va = fc_mask[va]
yv, gv = y[va], g[va]


def fit_eval(Xf, name):
    dtr = lgb.Dataset(Xf[tr], label=y[tr])
    bst = lgb.train(PARAMS, dtr, num_boost_round=300)
    p = bst.predict(Xf[va])
    a, n = per_imp(yv, p, gv)
    ac, nc = per_imp(yv, p, gv, fc_va)
    print(f"{name:>36}  val={a:.4f} (n={n})  cold={ac:.4f} (n={nc})", flush=True)
    return bst


print("\n=== leave-one-out devpop (train rows exclude own impression) ===", flush=True)
fit_eval(X, "base")
fit_eval(np.hstack([X, DEVPOP_LEAK]), "+devpop LEAK (own imp included)")
fit_eval(np.hstack([X, DEVPOP_LOO]), "+devpop LOO (own imp excluded)")
fit_eval(np.hstack([X, DEVPOP_LEAK_M]), "+devpop LEAK masked")
fit_eval(np.hstack([X, DEVPOP_LOO_M]), "+devpop LOO masked")
fit_eval(np.hstack([X, DEVPOP_LOO_M, CAT_LOO]), "+devpop LOO masked + cat LOO")
fit_eval(np.hstack([X, DEVPOP_LOO, CAT_LOO]), "+devpop LOO + cat LOO")

a0, _ = per_imp(yv, X[va, 0], gv)
ac0, _ = per_imp(yv, X[va, 0], gv, fc_va)
print(f"{'llm9 alone':>36}  val={a0:.4f}            cold={ac0:.4f}", flush=True)
