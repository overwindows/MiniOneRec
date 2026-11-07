#!/usr/bin/env python3
"""Pick the PRODUCTION training recipe for the causal-devpop ranker.

exp_ranker_causal.py established the leak-free feature set:
    masked base + causal devpop + cat  ->  val 0.7618 (honest test regime)

But it left one question open: SCALE. Causal devpop is read from the
impressions that came BEFORE the row, so early dev rows see tiny counts
while late dev rows (and every test row) see large ones. The model is
trained on a mixture of scales but applied at full-dev scale.

Two candidate fixes:
  (a) normalise by the AS-OF max (max count observed so far) instead of the
      global max -> scale-free, and at test time the as-of max IS the
      full-dev max, so the formula is identical.
  (b) train only on rows whose history is mature (late dev), so the training
      scale matches the test scale.

This script measures both, under the same test-regime masking, and caches
the causal arrays so later iterations are fast.

Usage: python src/exp_ranker_causal2.py
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
DEV_BEHAVIORS = "data/MIND_large/dev/behaviors.tsv"
NEWS = "data/MIND_large/test/news.tsv"
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


# ---------------- causal arrays (cached) ----------------
def build_causal(n_rows):
    news_cat = {}
    with open(NEWS) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                news_cat[p[0]] = p[1]

    run_pop = Counter()
    run_clicks = Counter()
    run_cat_pop = Counter()
    run_cat_clicks = Counter()

    causal_pop = np.zeros(n_rows, dtype=np.float32)
    causal_clicks = np.zeros(n_rows, dtype=np.float32)
    causal_cat_pop = np.zeros(n_rows, dtype=np.float32)
    causal_cat_clicks = np.zeros(n_rows, dtype=np.float32)
    asof_max_pop = np.zeros(n_rows, dtype=np.float32)
    asof_max_cat = np.zeros(n_rows, dtype=np.float32)

    row = 0
    max_pop = 0
    max_cat = 0
    with open(DEV_BEHAVIORS) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            cands = []
            for tk in parts[4].split():
                if "-" not in tk:
                    continue
                nid, lab = tk.rsplit("-", 1)
                cands.append((nid, lab))
            for nid, lab in cands:
                causal_pop[row] = run_pop.get(nid, 0)
                causal_clicks[row] = run_clicks.get(nid, 0)
                cat = news_cat.get(nid, "")
                causal_cat_pop[row] = run_cat_pop.get(cat, 0)
                causal_cat_clicks[row] = run_cat_clicks.get(cat, 0)
                asof_max_pop[row] = max_pop
                asof_max_cat[row] = max_cat
                row += 1
            for nid, lab in cands:
                run_pop[nid] += 1
                if run_pop[nid] > max_pop:
                    max_pop = run_pop[nid]
                cat = news_cat.get(nid, "")
                run_cat_pop[cat] += 1
                if run_cat_pop[cat] > max_cat:
                    max_cat = run_cat_pop[cat]
                if lab == "1":
                    run_clicks[nid] += 1
                    run_cat_clicks[cat] += 1
    assert row == n_rows, f"row mismatch {row} vs {n_rows}"
    return dict(causal_pop=causal_pop, causal_clicks=causal_clicks,
                causal_cat_pop=causal_cat_pop, causal_cat_clicks=causal_cat_clicks,
                asof_max_pop=asof_max_pop, asof_max_cat=asof_max_cat,
                max_pop=np.float32(max_pop), max_cat=np.float32(max_cat))


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
print(f"  {X.shape[0]:,} rows, {n_imp:,} imps", flush=True)

if os.path.exists(CACHE):
    print(f"Loading cached causal arrays from {CACHE} ...", flush=True)
    z = np.load(CACHE)
    C = {k: z[k] for k in z.files}
else:
    print("Building causal popularity features ...", flush=True)
    C = build_causal(X.shape[0])
    np.savez(CACHE, **C)
    print(f"  cached -> {CACHE}", flush=True)

max_pop = float(C["max_pop"])
max_cat = float(C["max_cat"])
log_max = math.log1p(max_pop)
log_max_cat = math.log1p(max_cat)

# ---------------- row metadata ----------------
with open(DEV_FEATS) as f:
    header = f.readline().rstrip("\n").split("\t")
    i_nid = header.index("news_id")
nids = []
with open(DEV_FEATS) as f:
    f.readline()
    for line in f:
        nids.append(line.rstrip("\n").split("\t")[i_nid])

# ---------------- masks ----------------
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


def blocks(norm):
    """Return (devpop_block, cat_block) for a normalisation mode.

    norm='full': divide by the global max over all dev (what the current
                 production append_devpop_test.py does).
    norm='run' : divide by the as-of max (max count observed so far). At
                 test time the as-of max IS the full-dev max, so the formula
                 is identical to production.
    """
    cp = C["causal_pop"]
    cc = C["causal_clicks"]
    ccp = C["causal_cat_pop"]
    ccc = C["causal_cat_clicks"]
    if norm == "full":
        mp = np.full_like(cp, max_pop)
        mc = np.full_like(ccp, max_cat)
    else:
        mp = np.maximum(C["asof_max_pop"], 1.0)
        mc = np.maximum(C["asof_max_cat"], 1.0)
    devpop = np.column_stack([
        cp / mp,
        np.where(cp > 0, cc / np.maximum(cp, 1), 0.0),
        np.log1p(cp) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
    cat = np.column_stack([
        ccp / mc,
        np.where(ccp > 0, ccc / np.maximum(ccp, 1), 0.0)]).astype(np.float32)
    return devpop, cat


# ---------------- cold slice ----------------
cf0 = X[:, 1] == 0
pop0 = X[:, 2] == 0.0
fully_cold_row = cf0 & pop0
frac = np.bincount(g, weights=fully_cold_row.astype(float)) / np.bincount(g)
fc_imps = np.where(frac == 1.0)[0]
fc_mask = np.isin(g, fc_imps)
va = g >= cut
fc_va = fc_mask[va]
yv, gv = y[va], g[va]


def fit_eval(Xf, name, tr):
    dtr = lgb.Dataset(Xf[tr], label=y[tr])
    bst = lgb.train(PARAMS, dtr, num_boost_round=300)
    p = bst.predict(Xf[va])
    a, n = per_imp(yv, p, gv)
    ac, nc = per_imp(yv, p, gv, fc_va)
    print(f"{name:>46}  val={a:.4f} (n={n})  cold={ac:.4f} (n={nc})", flush=True)
    return a


print("\n=== scale-normalisation x training-window ===", flush=True)
results = {}
for norm in ("full", "run"):
    devpop, cat = blocks(norm)
    devpop_m = devpop.copy()
    devpop_m[~m_devpop] = 0.0
    for frac_tr in (1.0, 0.5, 0.25):
        lo = int(cut * (1.0 - frac_tr))
        tr = (g < cut) & (g >= lo)
        n_tr = int(tr.sum())
        Xf = np.hstack([Xm, devpop_m, cat])
        nm = f"norm={norm} train=last{int(frac_tr*100)}% ({n_tr:,} rows)"
        results[nm] = fit_eval(Xf, nm, tr)

print("\n=== best-norm detail: does cat still help? ===", flush=True)
for norm in ("full", "run"):
    devpop, cat = blocks(norm)
    devpop_m = devpop.copy()
    devpop_m[~m_devpop] = 0.0
    tr = g < cut
    fit_eval(np.hstack([Xm, devpop_m]), f"norm={norm} base+devpop (no cat)", tr)
    fit_eval(np.hstack([Xm, cat]), f"norm={norm} base+cat (no devpop)", tr)
    fit_eval(np.hstack([Xm, devpop_m, cat]), f"norm={norm} base+devpop+cat", tr)

print("\n=== reference ===", flush=True)
fit_eval(Xm, "masked base", g < cut)
a0, _ = per_imp(yv, X[va, 0], gv)
print(f"{'llm9 alone':>46}  val={a0:.4f}", flush=True)
