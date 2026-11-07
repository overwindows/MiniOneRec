#!/usr/bin/env python3
"""Ranker feature-family search on the honest dev-80 -> dev-20 val slice.

Baseline (validated): BASE_FEATS + devpop  -> val 0.7829 / fully-cold 0.8584

New feature families tested here (all computed from dev-80% ONLY, so the
dev-20 val slice stays honest):
  A) user-level dev history: per-user click count / CTR / category affinity
     (31% of test users appear in dev -> transferable)
  B) category-level dev popularity: cold news in a hot category is likelier
  C) recency-weighted news popularity: later dev impressions matter more for
     the test period (dev ends 11/15, test is 11/18-11/19)

Usage: python src/exp_ranker_feat_families.py
"""
import math
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups  # noqa: E402

import lightgbm as lgb  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features.tsv"
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"
DEV_BEHAVIORS = "data/MIND_large/dev/behaviors.tsv"
NEWS = "data/MIND_large/test/news.tsv"

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


# ---------------- load dev features ----------------
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
print(f"  {X.shape[0]:,} rows, {n_imp:,} imps (train {cut:,} / val {n_imp-cut:,})", flush=True)

# ---------------- news metadata ----------------
news_cat = {}
news_subcat = {}
with open(NEWS) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            news_cat[p[0]] = p[1]
            news_subcat[p[0]] = p[2]

# ---------------- dev-80 statistics ----------------
print("Computing dev-80 statistics ...", flush=True)
pop = Counter()
clicks = Counter()
cat_pop = Counter()
cat_clicks = Counter()
user_imps = Counter()
user_clicks = Counter()
user_cat_clicks = defaultdict(Counter)
user_subcat_clicks = defaultdict(Counter)
rec_pop = Counter()          # recency-weighted (linear in position)
rec_clicks = Counter()
n_dev = 0
with open(DEV_BEHAVIORS) as f:
    for line in f:
        if n_dev >= cut:
            break
        parts = line.strip().split("\t")
        if len(parts) < 5:
            n_dev += 1
            continue
        uid = parts[1]
        user_imps[uid] += 1
        w = 1.0 + n_dev / max(cut, 1)   # 1.0 -> 2.0 across dev-80
        for tk in parts[4].split():
            if "-" not in tk:
                continue
            nid, lab = tk.rsplit("-", 1)
            pop[nid] += 1
            rec_pop[nid] += w
            cat = news_cat.get(nid, "")
            cat_pop[cat] += 1
            if lab == "1":
                clicks[nid] += 1
                rec_clicks[nid] += w
                cat_clicks[cat] += 1
                user_clicks[uid] += 1
                user_cat_clicks[uid][cat] += 1
                user_subcat_clicks[uid][news_subcat.get(nid, "")] += 1
        n_dev += 1
max_pop = max(pop.values()) if pop else 1
log_max = math.log1p(max_pop)
max_cat_pop = max(cat_pop.values()) if cat_pop else 1
max_rec = max(rec_pop.values()) if rec_pop else 1
print(f"  {len(pop):,} nids, {len(user_imps):,} users, {len(cat_pop):,} cats", flush=True)

# ---------------- map rows -> news_id / user_id ----------------
with open(DEV_FEATS) as f:
    header = f.readline().rstrip("\n").split("\t")
    i_nid = header.index("news_id")
    i_uid = header.index("user_id")
nids = []
uids = []
with open(DEV_FEATS) as f:
    f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t")
        nids.append(p[i_nid])
        uids.append(p[i_uid])
assert len(nids) == X.shape[0]

# ---------------- feature blocks ----------------
dp = np.array([pop.get(nid, 0) for nid in nids], dtype=np.float32)
dc = np.array([clicks.get(nid, 0) for nid in nids], dtype=np.float32)
dpop = dp / max_pop
dctr = np.where(dp > 0, dc / np.maximum(dp, 1), 0.0)
dlog = np.log1p(dp) / log_max
BASE_DEVPOP = np.column_stack([dpop, dctr, dlog])

# A) user-level
u_imp = np.array([user_imps.get(u, 0) for u in uids], dtype=np.float32)
u_clk = np.array([user_clicks.get(u, 0) for u in uids], dtype=np.float32)
u_ctr = np.where(u_imp > 0, u_clk / np.maximum(u_imp, 1), 0.0)
u_seen = (u_imp > 0).astype(np.float32)
u_cat_aff = np.zeros(len(nids), dtype=np.float32)
u_sub_aff = np.zeros(len(nids), dtype=np.float32)
for i, (nid, u) in enumerate(zip(nids, uids)):
    cc = user_cat_clicks.get(u)
    if cc:
        u_cat_aff[i] = cc.get(news_cat.get(nid, ""), 0) / max(u_clk[i], 1)
    sc = user_subcat_clicks.get(u)
    if sc:
        u_sub_aff[i] = sc.get(news_subcat.get(nid, ""), 0) / max(u_clk[i], 1)
USER_BLOCK = np.column_stack([u_imp, u_clk, u_ctr, u_seen, u_cat_aff, u_sub_aff])

# B) category-level
c_pop = np.array([cat_pop.get(news_cat.get(nid, ""), 0) for nid in nids], dtype=np.float32)
c_clk = np.array([cat_clicks.get(news_cat.get(nid, ""), 0) for nid in nids], dtype=np.float32)
c_pop_n = c_pop / max_cat_pop
c_ctr = np.where(c_pop > 0, c_clk / np.maximum(c_pop, 1), 0.0)
CAT_BLOCK = np.column_stack([c_pop_n, c_ctr])

# C) recency-weighted
r_pop = np.array([rec_pop.get(nid, 0) for nid in nids], dtype=np.float32)
r_clk = np.array([rec_clicks.get(nid, 0) for nid in nids], dtype=np.float32)
r_pop_n = r_pop / max_rec
r_ctr = np.where(r_pop > 0, r_clk / np.maximum(r_pop, 1), 0.0)
REC_BLOCK = np.column_stack([r_pop_n, r_ctr])

# ---------------- cold slice ----------------
cf0 = X[:, 1] == 0
pop0 = X[:, 2] == 0.0
fully_cold_row = cf0 & pop0
frac = np.bincount(g, weights=fully_cold_row.astype(float)) / np.bincount(g)
fc_imps = np.where(frac == 1.0)[0]
fc_mask = np.isin(g, fc_imps)
fc_va = fc_mask[va]
yv, gv = y[va], g[va]
print(f"fully-cold val imps: {fc_va.sum():,}", flush=True)


def fit_eval(Xf, name):
    dtr = lgb.Dataset(Xf[tr], label=y[tr])
    bst = lgb.train(PARAMS, dtr, num_boost_round=300)
    p = bst.predict(Xf[va])
    a, n = per_imp(yv, p, gv)
    ac, nc = per_imp(yv, p, gv, fc_va)
    print(f"{name:>22}  val={a:.4f} (n={n})  cold={ac:.4f} (n={nc})", flush=True)
    return bst, a, ac


print("\n=== feature family search (train dev-80, eval dev-20) ===", flush=True)
X_base = np.hstack([X, BASE_DEVPOP])
fit_eval(X_base, "base+devpop")
fit_eval(np.hstack([X_base, USER_BLOCK]), "+user")
fit_eval(np.hstack([X_base, CAT_BLOCK]), "+cat")
fit_eval(np.hstack([X_base, REC_BLOCK]), "+recency")
fit_eval(np.hstack([X_base, USER_BLOCK, CAT_BLOCK]), "+user+cat")
fit_eval(np.hstack([X_base, USER_BLOCK, CAT_BLOCK, REC_BLOCK]), "+user+cat+rec")
fit_eval(np.hstack([X_base, USER_BLOCK, REC_BLOCK]), "+user+rec")

# reference: llm alone
a0, _ = per_imp(yv, X[va, 0], gv)
ac0, _ = per_imp(yv, X[va, 0], gv, fc_va)
print(f"{'llm9 alone':>22}  val={a0:.4f}            cold={ac0:.4f}", flush=True)
