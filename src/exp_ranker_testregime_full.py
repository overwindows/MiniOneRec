#!/usr/bin/env python3
"""Full test-regime ranker validation.

The dev-80 -> dev-20 val slice is NOT representative of the test split:
every feature family that is derived from a *previous* period has much lower
coverage on test than on the val slice. Measured test coverage:

  cand_pop / cand_ctr / cand_log_pop (train-derived news pop)  ~12.5%
  cf_score (train-derived CF)                                  ~11.3%
  devpop (dev-derived news pop)                                ~15.5%
  user-level dev history (per impression)                      ~43.8%
  category-level dev popularity                                100%

On the val slice, devpop covers 100% of rows and train-pop covers ~68%, so a
model trained there never learns what to do when those features are missing.

This script masks each feature family down to its TEST coverage (applied to
both train and val rows) so the model is trained and evaluated in the same
sparsity regime it will face on test. It then compares feature sets under
that regime.

Usage: python src/exp_ranker_testregime_full.py
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

# measured test coverage (row level)
COV_TRAINPOP = 0.125
COV_CF = 0.113
COV_DEVPOP = 0.155
COV_USER = 0.438

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


def mask_to_coverage(keys, weights, target, rng):
    """Choose a subset of keys so that the weighted coverage hits `target`."""
    uniq = sorted(set(keys))
    order = rng.permutation(len(uniq))
    wmap = {}
    for k, w in zip(keys, weights):
        wmap[k] = wmap.get(k, 0) + w
    total = sum(wmap.values()) or 1
    keep = set()
    cov = 0
    for j in order:
        k = uniq[j]
        keep.add(k)
        cov += wmap.get(k, 0)
        if cov / total >= target:
            break
    return keep


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

# ---- news metadata ----
news_cat = {}
with open(NEWS) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            news_cat[p[0]] = p[1]

# ---- dev-80 stats ----
print("Computing dev-80 statistics ...", flush=True)
pop = Counter()
clicks = Counter()
cat_pop = Counter()
cat_clicks = Counter()
user_imps = Counter()
user_clicks = Counter()
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
                user_clicks[uid] += 1
        n_dev += 1
max_pop = max(pop.values()) if pop else 1
log_max = math.log1p(max_pop)
max_cat_pop = max(cat_pop.values()) if cat_pop else 1

# ---- row metadata ----
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

# ---- masks to test coverage ----
rng = np.random.default_rng(SEED)
row_w = np.ones(len(nids), dtype=np.float64)

# devpop mask (news-level)
keep_devpop = mask_to_coverage(nids, row_w, COV_DEVPOP, rng)
m_devpop = np.array([nid in keep_devpop for nid in nids], dtype=bool)
print(f"  devpop mask: {m_devpop.mean():.1%} rows kept (target {COV_DEVPOP:.1%})", flush=True)

# train-pop mask (news-level, uses the train-derived cand_pop>0 signal)
trainpop_hit = X[:, 2] > 0
keep_trainpop = mask_to_coverage(
    [n for n, h in zip(nids, trainpop_hit) if h], row_w[trainpop_hit], COV_TRAINPOP, rng)
m_trainpop = np.array([(nid in keep_trainpop) for nid in nids], dtype=bool)
print(f"  trainpop mask: {m_trainpop.mean():.1%} rows kept (target {COV_TRAINPOP:.1%})", flush=True)

# cf mask (row-level: cf_score != 0)
cf_hit = X[:, 1] != 0
keep_cf = mask_to_coverage(
    [i for i, h in enumerate(cf_hit) if h], row_w[cf_hit], COV_CF, rng)
m_cf = np.zeros(len(nids), dtype=bool)
m_cf[list(keep_cf)] = True
print(f"  cf mask: {m_cf.mean():.1%} rows kept (target {COV_CF:.1%})", flush=True)

# user mask (impression-level: user seen in dev-80)
keep_users = mask_to_coverage(
    list(uids), row_w, COV_USER, rng)
m_user = np.array([u in keep_users for u in uids], dtype=bool)
print(f"  user mask: {m_user.mean():.1%} rows kept (target {COV_USER:.1%})", flush=True)

# ---- build feature blocks ----
dp = np.array([pop.get(nid, 0) for nid in nids], dtype=np.float32)
dc = np.array([clicks.get(nid, 0) for nid in nids], dtype=np.float32)
dpop = dp / max_pop
dctr = np.where(dp > 0, dc / np.maximum(dp, 1), 0.0)
dlog = np.log1p(dp) / log_max
DEVPOP = np.column_stack([dpop, dctr, dlog]).astype(np.float32)
DEVPOP_M = DEVPOP.copy()
DEVPOP_M[~m_devpop] = 0.0

c_pop = np.array([cat_pop.get(news_cat.get(nid, ""), 0) for nid in nids], dtype=np.float32)
c_clk = np.array([cat_clicks.get(news_cat.get(nid, ""), 0) for nid in nids], dtype=np.float32)
CAT = np.column_stack([c_pop / max_cat_pop,
                       np.where(c_pop > 0, c_clk / np.maximum(c_pop, 1), 0.0)]).astype(np.float32)

u_imp = np.array([user_imps.get(u, 0) for u in uids], dtype=np.float32)
u_clk = np.array([user_clicks.get(u, 0) for u in uids], dtype=np.float32)
USER = np.column_stack([u_imp, u_clk,
                        np.where(u_imp > 0, u_clk / np.maximum(u_imp, 1), 0.0)]).astype(np.float32)
USER_M = USER.copy()
USER_M[~m_user] = 0.0

# ---- masked BASE ----
Xm = X.copy()
Xm[~m_trainpop, 2:5] = 0.0     # cand_pop, cand_ctr, cand_log_pop
Xm[~m_cf, 1] = 0.0              # cf_score

# ---- cold slice (train-derived coldness) ----
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
    print(f"{name:>34}  val={a:.4f} (n={n})  cold={ac:.4f} (n={nc})", flush=True)
    return bst


print("\n=== TEST-REGIME (all families masked to test coverage) ===", flush=True)
fit_eval(Xm, "masked base")
fit_eval(np.hstack([Xm, DEVPOP_M]), "masked base + devpop")
fit_eval(np.hstack([Xm, DEVPOP_M, CAT]), "masked base + devpop + cat")
fit_eval(np.hstack([Xm, DEVPOP_M, CAT, USER_M]), "masked base + devpop + cat + user")
fit_eval(np.hstack([Xm, CAT]), "masked base + cat")
fit_eval(np.hstack([Xm, USER_M]), "masked base + user")

print("\n=== UNMASKED reference (optimistic, for contrast) ===", flush=True)
fit_eval(X, "unmasked base")
fit_eval(np.hstack([X, DEVPOP]), "unmasked base + devpop")
fit_eval(np.hstack([X, DEVPOP, CAT]), "unmasked base + devpop + cat")

a0, _ = per_imp(yv, X[va, 0], gv)
ac0, _ = per_imp(yv, X[va, 0], gv, fc_va)
print(f"{'llm9 alone':>34}  val={a0:.4f}            cold={ac0:.4f}", flush=True)
