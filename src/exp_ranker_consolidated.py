#!/usr/bin/env python3
"""Consolidated test-regime ranker experiment (correct masking + LOO devpop).

Why this exists
---------------
The dev-80 -> dev-20 val slice is NOT representative of the test split.
Feature families derived from a *previous* period are far sparser on test:

  family                        dev val slice   test
  cand_pop/ctr/log_pop (train)      ~68%        12.8%
  cf_score (train)                  ~49%        11.3%
  devpop (dev)                      100%        15.5%
  user-level dev history            100%        43.8% (per impression)
  category-level dev popularity     100%       100.0%

A ranker trained on the val slice never learns what to do when those features
are missing, so its val number is optimistic. This script masks each family
down to its measured TEST coverage (applied to train AND val rows) so the
model is trained and evaluated in the sparsity regime it will face on test.

It also fixes a leakage in the devpop feature: for a training row, devpop was
computed from dev-80 INCLUDING that row's own impression, so a clicked news
had its own click inflating the feature. At test time the test impression is
not in dev, so the feature is a clean prior. The LOO variant subtracts the
row's own impression contribution.

Usage: python src/exp_ranker_consolidated.py
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

# measured TEST row coverage
COV_TRAINPOP = 0.128
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


def mask_news_to_coverage(nids, hit, target, rng):
    """Keep a random subset of news so that (rows whose news is kept)/all rows
    == target. `hit` = rows where the feature is currently nonzero."""
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
    """Keep a random subset of the hit ROWS so that kept/all == target."""
    total = len(hit)
    want = int(target * total)
    idx = np.where(hit)[0]
    rng.shuffle(idx)
    m = np.zeros(total, dtype=bool)
    m[idx[:want]] = True
    return m


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
    i_lab = header.index("label")
nids = []
uids = []
labels = []
with open(DEV_FEATS) as f:
    f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t")
        nids.append(p[i_nid])
        uids.append(p[i_uid])
        labels.append(int(p[i_lab]))
labels = np.array(labels, dtype=np.float32)

# ---- masks (targets are fractions of ALL rows) ----
rng = np.random.default_rng(SEED)
trainpop_hit = X[:, 2] > 0
m_trainpop = mask_news_to_coverage(nids, trainpop_hit, COV_TRAINPOP, rng)
m_trainpop &= trainpop_hit
cf_hit = X[:, 1] != 0
m_cf = mask_rows_to_coverage(cf_hit, COV_CF, rng)
devpop_hit = np.array([nid in pop for nid in nids], dtype=bool)
m_devpop = mask_news_to_coverage(nids, devpop_hit, COV_DEVPOP, rng)
m_devpop &= devpop_hit
m_user = mask_rows_to_coverage(np.ones(len(nids), dtype=bool), COV_USER, rng)
# user features are per-IMPRESSION: mask whole impressions, not individual rows
_imp_users = [uids[0]]
for i in range(1, len(nids)):
    if g[i] != g[i - 1]:
        _imp_users.append(uids[i])
_imp_users = np.array(_imp_users)
_keep_u = mask_rows_to_coverage(np.ones(len(_imp_users), dtype=bool), COV_USER, rng)
_keep_u_set = set(_imp_users[_keep_u].tolist())
m_user = np.array([u in _keep_u_set for u in uids], dtype=bool)
print(f"  masks: trainpop={m_trainpop.mean():.1%} (t {COV_TRAINPOP:.1%})  "
      f"cf={m_cf.mean():.1%} (t {COV_CF:.1%})  "
      f"devpop={m_devpop.mean():.1%} (t {COV_DEVPOP:.1%})  "
      f"user={m_user.mean():.1%} (t {COV_USER:.1%})", flush=True)

# ---- feature blocks ----
dp_total = np.array([pop.get(nid, 0) for nid in nids], dtype=np.float32)
dc_total = np.array([clicks.get(nid, 0) for nid in nids], dtype=np.float32)
dp_loo = np.maximum(dp_total - 1.0, 0.0)
dc_loo = np.maximum(dc_total - labels, 0.0)


def devpop_block(dp, dc):
    return np.column_stack([
        dp / max_pop,
        np.where(dp > 0, dc / np.maximum(dp, 1), 0.0),
        np.log1p(dp) / log_max]).astype(np.float32)


DEVPOP_LEAK = devpop_block(dp_total, dc_total)
DEVPOP_LOO = devpop_block(dp_loo, dc_loo)
DEVPOP_LEAK_M = DEVPOP_LEAK.copy(); DEVPOP_LEAK_M[~m_devpop] = 0.0
DEVPOP_LOO_M = DEVPOP_LOO.copy(); DEVPOP_LOO_M[~m_devpop] = 0.0

c_pop_total = np.array([cat_pop.get(news_cat.get(nid, ""), 0) for nid in nids], dtype=np.float32)
c_clk_total = np.array([cat_clicks.get(news_cat.get(nid, ""), 0) for nid in nids], dtype=np.float32)
c_pop_loo = np.maximum(c_pop_total - 1.0, 0.0)
c_clk_loo = np.maximum(c_clk_total - labels, 0.0)
CAT_LEAK = np.column_stack([c_pop_total / max_cat_pop,
                             np.where(c_pop_total > 0, c_clk_total / np.maximum(c_pop_total, 1), 0.0)]).astype(np.float32)
CAT_LOO = np.column_stack([c_pop_loo / max_cat_pop,
                            np.where(c_pop_loo > 0, c_clk_loo / np.maximum(c_pop_loo, 1), 0.0)]).astype(np.float32)

u_imp = np.array([user_imps.get(u, 0) for u in uids], dtype=np.float32)
u_clk = np.array([user_clicks.get(u, 0) for u in uids], dtype=np.float32)
USER = np.column_stack([u_imp, u_clk,
                        np.where(u_imp > 0, u_clk / np.maximum(u_imp, 1), 0.0)]).astype(np.float32)
USER_M = USER.copy(); USER_M[~m_user] = 0.0

# ---- masked BASE ----
Xm = X.copy()
Xm[~m_trainpop, 2:5] = 0.0
Xm[~m_cf, 1] = 0.0

# ---- cold slice ----
cf0 = X[:, 1] == 0
pop0 = X[:, 2] == 0.0
fully_cold_row = cf0 & pop0
frac = np.bincount(g, weights=fully_cold_row.astype(float)) / np.bincount(g)
fc_imps = np.where(frac == 1.0)[0]
fc_mask = np.isin(g, fc_imps)
fc_va = fc_mask[va]
yv, gv = y[va], g[va]


def fit_eval(Xf, name, save=None):
    dtr = lgb.Dataset(Xf[tr], label=y[tr])
    bst = lgb.train(PARAMS, dtr, num_boost_round=300)
    p = bst.predict(Xf[va])
    a, n = per_imp(yv, p, gv)
    ac, nc = per_imp(yv, p, gv, fc_va)
    print(f"{name:>38}  val={a:.4f} (n={n})  cold={ac:.4f} (n={nc})", flush=True)
    if save:
        bst.save_model(save)
        print(f"{'':>38}  saved {save}", flush=True)
    return bst, a, ac


print("\n=== TEST-REGIME (families masked to measured test coverage) ===", flush=True)
fit_eval(Xm, "masked base")
fit_eval(np.hstack([Xm, DEVPOP_LOO_M]), "masked base + devpop LOO")
fit_eval(np.hstack([Xm, DEVPOP_LEAK_M]), "masked base + devpop LEAK")
fit_eval(np.hstack([Xm, DEVPOP_LOO_M, CAT_LOO]), "masked base + devpop LOO + cat LOO")
fit_eval(np.hstack([Xm, DEVPOP_LOO_M, CAT_LOO, USER_M]), "masked base + devpop LOO + cat + user")
fit_eval(np.hstack([Xm, CAT_LOO]), "masked base + cat LOO")
fit_eval(np.hstack([Xm, USER_M]), "masked base + user")

print("\n=== DROP dead train features entirely (no masking needed) ===", flush=True)
Xd = X.copy()
Xd[:, 1] = 0.0
Xd[:, 2:5] = 0.0
fit_eval(Xd, "drop cf+trainpop")
fit_eval(np.hstack([Xd, DEVPOP_LOO]), "drop cf+trainpop + devpop LOO")
fit_eval(np.hstack([Xd, DEVPOP_LOO, CAT_LOO]), "drop cf+trainpop + devpop LOO + cat LOO")
fit_eval(np.hstack([Xd, DEVPOP_LOO, CAT_LOO, USER_M]), "drop cf+trainpop + devpop + cat + user")

print("\n=== UNMASKED reference (optimistic) ===", flush=True)
fit_eval(X, "unmasked base")
fit_eval(np.hstack([X, DEVPOP_LEAK]), "unmasked base + devpop LEAK")

a0, _ = per_imp(yv, X[va, 0], gv)
ac0, _ = per_imp(yv, X[va, 0], gv, fc_va)
print(f"{'llm9 alone':>38}  val={a0:.4f}            cold={ac0:.4f}", flush=True)
