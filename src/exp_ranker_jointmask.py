#!/usr/bin/env python3
"""Joint test-regime masking: match the 4-way (trainpop, devpop) distribution.

The current masking applies the trainpop and devpop masks INDEPENDENTLY.
That is wrong, because the two masks are applied to the SAME news: a news
that is popular in train is usually also popular in dev. Independent masks
therefore produce a joint distribution that does not match the test split:

    cell            train (independent)   test (measured)
    both                 2.022%              11.599%
    train only          10.778%               1.163%
    dev only            13.778%               3.928%
    neither             73.422%              83.310%

The model trained under the independent regime sees "both" only 2% of the
time but will meet it 11.6% of the time at test time, and sees "train only"
10.8% of the time but will meet it 1.2% of the time. That is a large
train/test mismatch in exactly the regime the popularity features live in.

This script masks the two families JOINTLY so that the 4-way distribution
matches the measured test distribution, and compares against the
independent masking.

Usage: python src/exp_ranker_jointmask.py
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

# Measured TEST joint distribution (train_pop>0, dev_pop>0)
TEST_BOTH = 0.11599
TEST_TR_ONLY = 0.01163
TEST_DV_ONLY = 0.03928
TEST_NEITHER = 0.83310

COV_CF = 0.113

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


def joint_mask(nids, train_hit, dev_hit, rng):
    """Mask the two families so the 4-way joint distribution matches test.

    The mask is applied at the NEWS level (a news either has its train-pop
    feature or it does not), matching test semantics where presence is a
    property of the news.

    Each news is assigned to one of four cells:
        both      - keep trainpop AND devpop
        tr_only   - keep trainpop, mask devpop
        dv_only   - mask trainpop, keep devpop
        neither   - mask both

    A news can only be assigned to a cell whose features it actually has.
    Quotas are in ROW counts (the test distribution is measured over rows).
    """
    total = len(nids)
    # row count per news
    wmap = Counter(nids)
    # which families each news has
    has_train = {}
    has_dev = {}
    for n, t, d in zip(nids, train_hit, dev_hit):
        if t:
            has_train[n] = True
        if d:
            has_dev[n] = True

    quotas = {
        "both": int(TEST_BOTH * total),
        "tr_only": int(TEST_TR_ONLY * total),
        "dv_only": int(TEST_DV_ONLY * total),
        "neither": int(TEST_NEITHER * total),
    }
    remaining = dict(quotas)

    keys = list(wmap.keys())
    order = rng.permutation(len(keys))

    assign = {}
    for j in order:
        n = keys[j]
        t = has_train.get(n, False)
        d = has_dev.get(n, False)
        feasible = []
        if t and d:
            feasible = ["both", "tr_only", "dv_only", "neither"]
        elif t:
            feasible = ["tr_only", "neither"]
        elif d:
            feasible = ["dv_only", "neither"]
        else:
            feasible = ["neither"]
        # pick the feasible cell with the largest remaining quota
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
train_hit = X[:, 2] > 0
dev_hit = C["causal_pop"] > 0
cf_hit = X[:, 1] != 0
m_cf = mask_rows_to_coverage(cf_hit, COV_CF, rng)

# ---- independent masking (current production) ----
m_trainpop_i = mask_news_to_coverage(nids, train_hit, 0.128, rng)
m_trainpop_i &= train_hit
m_devpop_i = mask_news_to_coverage(nids, dev_hit, 0.155, rng)
m_devpop_i &= dev_hit

# ---- joint masking (new) ----
m_trainpop_j, m_devpop_j = joint_mask(nids, train_hit, dev_hit, rng)


def report(name, mt, md):
    both = (mt & md).mean()
    to = (mt & ~md).mean()
    do = (~mt & md).mean()
    ne = (~mt & ~md).mean()
    print(f"  {name:>22}: both={both:.3%} tr_only={to:.3%} "
          f"dv_only={do:.3%} neither={ne:.3%}", flush=True)


print("\n=== joint distribution ===", flush=True)
report("TEST (target)", np.ones(len(nids), bool), np.ones(len(nids), bool))
report("independent (current)", m_trainpop_i, m_devpop_i)
report("joint (new)", m_trainpop_j, m_devpop_j)

# ---- feature blocks ----
mp = np.maximum(C["asof_max_pop"], 1.0)
mc = np.maximum(C["asof_max_cat"], 1.0)
ms = np.maximum(S["sub_asof"], 1.0)
DEVPOP = np.column_stack([
    C["causal_pop"] / mp,
    np.where(C["causal_pop"] > 0, C["causal_clicks"] / np.maximum(C["causal_pop"], 1), 0.0),
    np.log1p(C["causal_pop"]) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
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


def fit_eval(Xf, name):
    dtr = lgb.Dataset(Xf[tr], label=y[tr])
    bst = lgb.train(PARAMS, dtr, num_boost_round=300)
    p = bst.predict(Xf[va])
    a, n = per_imp(yv, p, gv)
    print(f"{name:>52}  val={a:.4f} (n={n})", flush=True)
    return a


def build(mt, md):
    Xm = X.copy()
    Xm[~mt, 2:5] = 0.0
    Xm[~m_cf, 1] = 0.0
    dp = DEVPOP.copy()
    dp[~md] = 0.0
    return Xm, dp


print("\n=== independent vs joint masking ===", flush=True)
Xi, dpi = build(m_trainpop_i, m_devpop_i)
Xj, dpj = build(m_trainpop_j, m_devpop_j)

fit_eval(Xi, "indep: base")
fit_eval(np.hstack([Xi, dpi, CAT]), "indep: base + devpop + cat")
fit_eval(np.hstack([Xi, dpi, SUBCAT]), "indep: base + devpop + subcat")
fit_eval(np.hstack([Xi, dpi, CAT, SUBCAT]), "indep: base + devpop + cat + subcat")

fit_eval(Xj, "joint: base")
fit_eval(np.hstack([Xj, dpj, CAT]), "joint: base + devpop + cat")
fit_eval(np.hstack([Xj, dpj, SUBCAT]), "joint: base + devpop + subcat")
fit_eval(np.hstack([Xj, dpj, CAT, SUBCAT]), "joint: base + devpop + cat + subcat")
fit_eval(np.hstack([Xj, CAT, SUBCAT]), "joint: base + cat + subcat")
