#!/usr/bin/env python3
"""Causal (past-only) popularity features + honest test-regime validation.

Two bugs found in earlier devpop variants:

1. SELF-INCLUSION: for a training row, devpop was computed from dev-80
   INCLUDING that row's own impression, so a clicked news had its own click
   inflating the feature. At test time the test impression is not in dev, so
   the feature is a clean prior -> the model over-trusts it.

2. LOO IS STILL LEAKY: subtracting the row's own click gives
   dc_loo = clicks_total - label, which is a deterministic function of the
   label within each news (positives get clicks_total-1, negatives get
   clicks_total). That drove val AUC to 1.0000.

The correct construction is CAUSAL: for each impression, popularity is
computed from the impressions that came BEFORE it. This matches test-time
semantics exactly (test impressions come after all of dev), and it is
leak-free by construction.

This script:
  - builds causal news-pop / news-ctr / cat-pop / cat-ctr features
  - masks each family to its measured TEST coverage
  - compares feature sets on the dev-80 -> dev-20 val slice

Usage: python src/exp_ranker_causal.py
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
print(f"  {X.shape[0]:,} rows, {n_imp:,} imps", flush=True)

news_cat = {}
with open(NEWS) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            news_cat[p[0]] = p[1]

# ---------------- causal features ----------------
# Walk dev behaviors in order; for each impression record the popularity
# accumulated from PRIOR impressions only, then fold this impression in.
print("Building causal popularity features ...", flush=True)
run_pop = Counter()
run_clicks = Counter()
run_cat_pop = Counter()
run_cat_clicks = Counter()
run_user_imps = Counter()
run_user_clicks = Counter()

causal_pop = np.zeros(X.shape[0], dtype=np.float32)
causal_clicks = np.zeros(X.shape[0], dtype=np.float32)
causal_cat_pop = np.zeros(X.shape[0], dtype=np.float32)
causal_cat_clicks = np.zeros(X.shape[0], dtype=np.float32)
causal_u_imps = np.zeros(X.shape[0], dtype=np.float32)
causal_u_clicks = np.zeros(X.shape[0], dtype=np.float32)

row = 0
n_dev = 0
with open(DEV_BEHAVIORS) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            n_dev += 1
            continue
        uid = parts[1]
        cands = []
        for tk in parts[4].split():
            if "-" not in tk:
                continue
            nid, lab = tk.rsplit("-", 1)
            cands.append((nid, lab))
        # record PRIOR state for this impression's candidates
        for nid, lab in cands:
            causal_pop[row] = run_pop.get(nid, 0)
            causal_clicks[row] = run_clicks.get(nid, 0)
            cat = news_cat.get(nid, "")
            causal_cat_pop[row] = run_cat_pop.get(cat, 0)
            causal_cat_clicks[row] = run_cat_clicks.get(cat, 0)
            causal_u_imps[row] = run_user_imps.get(uid, 0)
            causal_u_clicks[row] = run_user_clicks.get(uid, 0)
            row += 1
        # fold this impression into the running state
        run_user_imps[uid] += 1
        for nid, lab in cands:
            run_pop[nid] += 1
            cat = news_cat.get(nid, "")
            run_cat_pop[cat] += 1
            if lab == "1":
                run_clicks[nid] += 1
                run_cat_clicks[cat] += 1
                run_user_clicks[uid] += 1
        n_dev += 1
assert row == X.shape[0], f"row mismatch {row} vs {X.shape[0]}"
print(f"  {n_dev:,} dev imps processed", flush=True)

# final (full-dev) popularity for normalisation scale
final_pop = Counter()
final_clicks = Counter()
final_cat_pop = Counter()
final_cat_clicks = Counter()
with open(DEV_BEHAVIORS) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        for tk in parts[4].split():
            if "-" not in tk:
                continue
            nid, lab = tk.rsplit("-", 1)
            final_pop[nid] += 1
            cat = news_cat.get(nid, "")
            final_cat_pop[cat] += 1
            if lab == "1":
                final_clicks[nid] += 1
                final_cat_clicks[cat] += 1
max_pop = max(final_pop.values()) if final_pop else 1
log_max = math.log1p(max_pop)
max_cat_pop = max(final_cat_pop.values()) if final_cat_pop else 1

# ---------------- row metadata ----------------
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

# ---------------- masks ----------------
rng = np.random.default_rng(SEED)
trainpop_hit = X[:, 2] > 0
m_trainpop = mask_news_to_coverage(nids, trainpop_hit, COV_TRAINPOP, rng)
m_trainpop &= trainpop_hit
cf_hit = X[:, 1] != 0
m_cf = mask_rows_to_coverage(cf_hit, COV_CF, rng)
devpop_hit = causal_pop > 0
m_devpop = mask_news_to_coverage(nids, devpop_hit, COV_DEVPOP, rng)
m_devpop &= devpop_hit
# user: per-impression mask
_imp_users = [uids[0]]
for i in range(1, len(nids)):
    if g[i] != g[i - 1]:
        _imp_users.append(uids[i])
_imp_users = np.array(_imp_users)
_keep_u = mask_rows_to_coverage(np.ones(len(_imp_users), dtype=bool), COV_USER, rng)
_keep_u_set = set(_imp_users[_keep_u].tolist())
m_user = np.array([u in _keep_u_set for u in uids], dtype=bool)
print(f"  masks: trainpop={m_trainpop.mean():.1%}  cf={m_cf.mean():.1%}  "
      f"devpop={m_devpop.mean():.1%}  user={m_user.mean():.1%}", flush=True)

# ---------------- feature blocks ----------------
CAUSAL = np.column_stack([
    causal_pop / max_pop,
    np.where(causal_pop > 0, causal_clicks / np.maximum(causal_pop, 1), 0.0),
    np.log1p(causal_pop) / log_max]).astype(np.float32)
CAUSAL_M = CAUSAL.copy()
CAUSAL_M[~m_devpop] = 0.0

CAT_CAUSAL = np.column_stack([
    causal_cat_pop / max_cat_pop,
    np.where(causal_cat_pop > 0, causal_cat_clicks / np.maximum(causal_cat_pop, 1), 0.0)]).astype(np.float32)

USER_CAUSAL = np.column_stack([
    causal_u_imps,
    causal_u_clicks,
    np.where(causal_u_imps > 0, causal_u_clicks / np.maximum(causal_u_imps, 1), 0.0)]).astype(np.float32)
USER_CAUSAL_M = USER_CAUSAL.copy()
USER_CAUSAL_M[~m_user] = 0.0

Xm = X.copy()
Xm[~m_trainpop, 2:5] = 0.0
Xm[~m_cf, 1] = 0.0

# ---------------- cold slice ----------------
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
    print(f"{name:>40}  val={a:.4f} (n={n})  cold={ac:.4f} (n={nc})", flush=True)
    return bst


print("\n=== TEST-REGIME with CAUSAL features ===", flush=True)
fit_eval(Xm, "masked base")
fit_eval(np.hstack([Xm, CAUSAL_M]), "masked base + causal devpop")
fit_eval(np.hstack([Xm, CAUSAL_M, CAT_CAUSAL]), "masked base + causal devpop + cat")
fit_eval(np.hstack([Xm, CAUSAL_M, CAT_CAUSAL, USER_CAUSAL_M]), "masked base + causal devpop + cat + user")
fit_eval(np.hstack([Xm, CAT_CAUSAL]), "masked base + cat")
fit_eval(np.hstack([Xm, USER_CAUSAL_M]), "masked base + user")

print("\n=== drop dead train features entirely ===", flush=True)
Xd = X.copy()
Xd[:, 1] = 0.0
Xd[:, 2:5] = 0.0
fit_eval(Xd, "drop cf+trainpop")
fit_eval(np.hstack([Xd, CAUSAL_M]), "drop cf+trainpop + causal devpop")
fit_eval(np.hstack([Xd, CAUSAL_M, CAT_CAUSAL]), "drop cf+trainpop + causal devpop + cat")
fit_eval(np.hstack([Xd, CAUSAL_M, CAT_CAUSAL, USER_CAUSAL_M]), "drop cf+trainpop + causal devpop + cat + user")

print("\n=== UNMASKED reference (optimistic) ===", flush=True)
fit_eval(X, "unmasked base")
fit_eval(np.hstack([X, CAUSAL]), "unmasked base + causal devpop")

a0, _ = per_imp(yv, X[va, 0], gv)
ac0, _ = per_imp(yv, X[va, 0], gv, fc_va)
print(f"{'llm9 alone':>40}  val={a0:.4f}            cold={ac0:.4f}", flush=True)
