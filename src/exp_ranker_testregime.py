#!/usr/bin/env python3
"""Test-regime validation: does devpop help when it is MISSING for 84.5% of rows?

Problem: on the dev-80 -> dev-20 val slice, dev-derived popularity covers
100% of candidate rows (MIND dev has a small, stable news pool of ~7k ids).
On the real test split it covers only 15.5% of rows (test news is much
fresher). So the val number for the devpop ranker is optimistic: the model
never saw a devpop==0 row during training.

Fix: mask news IDs at random so that val coverage matches the test coverage
(15.5%). A masked news behaves exactly like a test news that never appeared
in dev: devpop=0, devctr=0, devlog=0.

Compares, all trained on dev-80 and evaluated on dev-20:
  base            : BASE_FEATS only (no devpop)
  +devpop-full    : devpop unmasked (the optimistic number)
  +devpop-masked  : devpop masked to test-like sparsity
  +devpop-mask+flag: masked devpop + a "devpop_seen" indicator so the model
                     can learn to fall back to other signals when it is absent
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

TARGET_COVERAGE = 0.155   # measured test coverage of devpop>0
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

# dev-80 popularity
print("Computing dev-80 popularity ...", flush=True)
pop = Counter()
clicks = Counter()
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
            if lab == "1":
                clicks[nid] += 1
        n_dev += 1
max_pop = max(pop.values()) if pop else 1
log_max = math.log1p(max_pop)

with open(DEV_FEATS) as f:
    header = f.readline().rstrip("\n").split("\t")
    i_nid = header.index("news_id")
nids = []
with open(DEV_FEATS) as f:
    f.readline()
    for line in f:
        nids.append(line.rstrip("\n").split("\t")[i_nid])

dp = np.array([pop.get(nid, 0) for nid in nids], dtype=np.float32)
dc = np.array([clicks.get(nid, 0) for nid in nids], dtype=np.float32)
dpop = dp / max_pop
dctr = np.where(dp > 0, dc / np.maximum(dp, 1), 0.0)
dlog = np.log1p(dp) / log_max
DEVPOP = np.column_stack([dpop, dctr, dlog]).astype(np.float32)

# ---- mask news so val coverage == test coverage ----
rng = np.random.default_rng(SEED)
all_nids = sorted(pop.keys())
# keep a random subset of news "known"; size chosen so that the fraction of
# val candidate ROWS with devpop>0 hits TARGET_COVERAGE
val_rows = np.where(va)[0]
val_nid_arr = np.array([nids[i] for i in val_rows])
val_counts = Counter(val_nid_arr.tolist())
total_val_rows = len(val_rows)
# greedy: shuffle news, add until coverage target reached
order = rng.permutation(len(all_nids))
known = set()
cov = 0
for j in order:
    nid = all_nids[j]
    known.add(nid)
    cov += val_counts.get(nid, 0)
    if cov / total_val_rows >= TARGET_COVERAGE:
        break
print(f"  masked: {len(known):,}/{len(all_nids):,} news kept -> "
      f"val row coverage {cov/total_val_rows:.1%} (target {TARGET_COVERAGE:.1%})", flush=True)

known_mask = np.array([nid in known for nid in nids], dtype=bool)
DEVPOP_MASKED = DEVPOP.copy()
DEVPOP_MASKED[~known_mask] = 0.0
SEEN = known_mask.astype(np.float32)[:, None]

# cold slice (train-derived coldness, independent of the mask)
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
    print(f"{name:>26}  val={a:.4f} (n={n})  cold={ac:.4f} (n={nc})", flush=True)
    return bst


print("\n=== test-regime validation (train dev-80, eval dev-20) ===", flush=True)
fit_eval(X, "base (no devpop)")
fit_eval(np.hstack([X, DEVPOP]), "+devpop FULL (optimistic)")
fit_eval(np.hstack([X, DEVPOP_MASKED]), "+devpop MASKED (test-like)")
fit_eval(np.hstack([X, DEVPOP_MASKED, SEEN]), "+devpop MASKED +seen flag")

# also: train on masked, evaluate on masked, but ALSO report the fully-cold
# slice restricted to masked-out news (the true test-like cold population)
cold_unseen = fc_va & (~known_mask[va])
a, n = per_imp(yv, np.zeros(len(yv)), gv, cold_unseen)
print(f"\nfully-cold AND devpop-unseen val imps: {cold_unseen.sum():,}", flush=True)

# reference
a0, _ = per_imp(yv, X[va, 0], gv)
ac0, _ = per_imp(yv, X[va, 0], gv, fc_va)
print(f"{'llm9 alone':>26}  val={a0:.4f}            cold={ac0:.4f}", flush=True)
