#!/usr/bin/env python3
"""Fully-cold slice check + dev-derived popularity features.

1. Evaluate the dev80-retrained ranker on val impressions where EVERY candidate
   has cf==0 AND train-pop==0 (the test-like regime: 87.5% of test candidates
   are train-cold).
2. Test adding dev-derived popularity (computed from dev-80% only, honest for
   the val slice) as extra features.
"""
import sys
import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups, BASE_FEATS  # noqa: E402

import lightgbm as lgb  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features.tsv"
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"
DEV_BEHAVIORS = "data/MIND_large/dev/behaviors.tsv"

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
    return float(((rsum[valid] - n_pos[valid]*(n_pos[valid]+1)/2) /
                  (n_pos[valid]*n_neg[valid])).mean()), int(valid.sum())

print("Loading dev features ...", flush=True)
Xs, ys, gs = [], [], []
for gi, (Xb, y) in enumerate(stream_groups(DEV_FEATS)):
    Xs.append(Xb); ys.append(y)
    gs.append(np.full(len(y), gi, dtype=np.int64))
X = np.vstack(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
del Xs, ys
ids = imp_ids(DEV_FEATS)
llm9 = load_llm(LG9B_DEV)
off = 0
for gi, (imp, n) in enumerate(ids):
    s = llm9.get(imp)
    if s and len(s) == n:
        X[off:off+n, 0] = s
    off += n
print(f"  {X.shape[0]:,} rows, {g.max()+1:,} imps", flush=True)

n_imp = g.max() + 1
cut = int(n_imp * 0.8)
tr = g < cut; va = ~tr
yv, gv = y[va], g[va]

# fully-cold: every candidate cf==0 AND cand_pop==0 (BASE cols: cf=1, pop=2)
cf0 = X[:, 1] == 0
pop0 = X[:, 2] == 0.0
fully_cold_row = cf0 & pop0
frac = np.bincount(g, weights=fully_cold_row.astype(float)) / np.bincount(g)
fully_cold_imps = np.where(frac == 1.0)[0]
fc_mask = np.isin(g, fully_cold_imps)
print(f"fully-cold imps (cf=0 & pop=0 all cands): {len(fully_cold_imps):,} "
      f"({len(fully_cold_imps)/n_imp:.1%})", flush=True)
fc_va = fc_mask[va]

params = dict(objective="binary", metric="auc", learning_rate=0.05,
              num_leaves=63, min_child_samples=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=32, verbose=-1)

# --- baseline retrain (same as exp1) ---
dtr = lgb.Dataset(X[tr], label=y[tr])
bst = lgb.train(params, dtr, num_boost_round=300)
p = bst.predict(X[va])
a, n = per_imp(yv, p, gv)
ac, nc = per_imp(yv, p, gv, fc_va)
print(f"\n{'base':>12}  val per-imp={a:.4f} (n={n})  fully-cold={ac:.4f} (n={nc})", flush=True)
llm_va = X[va, 0]
a0, _ = per_imp(yv, llm_va, gv)
ac0, _ = per_imp(yv, llm_va, gv, fc_va)
print(f"{'llm9':>12}  val per-imp={a0:.4f}            fully-cold={ac0:.4f}", flush=True)

# --- dev-pop features (from dev-80% only) ---
print("\nComputing dev-80 popularity ...", flush=True)
from collections import Counter
pop = Counter(); clicks = Counter()
imp_count = 0
with open(DEV_BEHAVIORS) as f:
    for line in f:
        if imp_count >= cut:
            break
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        for tk in parts[4].split():
            if "-" not in tk:
                continue
            nid, lab = tk.rsplit("-", 1)
            pop[nid] += 1
            if lab == "1":
                clicks[nid] += 1
        imp_count += 1
max_pop = max(pop.values()) if pop else 1
import math
# map rows -> dev pop/ctr/log_pop (3 new cols appended)
nid_col_idx = None
with open(DEV_FEATS) as f:
    header = f.readline().rstrip("\n").split("\t")
    i_nid = header.index("news_id")
nids = []
with open(DEV_FEATS) as f:
    f.readline()
    for line in f:
        nids.append(line.rstrip("\n").split("\t")[i_nid])
print(f"  {len(nids):,} candidate nids mapped", flush=True)
dp = np.array([pop.get(nid, 0) for nid in nids], dtype=np.float32)
dc = np.array([clicks.get(nid, 0) for nid in nids], dtype=np.float32)
dpop = dp / max_pop
dctr = np.where(dp > 0, dc / np.maximum(dp, 1), 0.0)
dlog = np.array([math.log1p(v) / math.log1p(max_pop) for v in dp], dtype=np.float32)
X2 = np.hstack([X, dpop[:, None], dctr[:, None], dlog[:, None]])
del nids, dp, dc, dpop, dctr, dlog

dtr = lgb.Dataset(X2[tr], label=y[tr])
bst2 = lgb.train(params, dtr, num_boost_round=300)
p2 = bst2.predict(X2[va])
a2, n2 = per_imp(yv, p2, gv)
ac2, nc2 = per_imp(yv, p2, gv, fc_va)
print(f"{'+devpop':>12}  val per-imp={a2:.4f} (n={n2})  fully-cold={ac2:.4f} (n={nc2})", flush=True)
bst2.save_model("results/ranker/ranker_v1_lg9b_dev80_devpop.txt")
print("saved results/ranker/ranker_v1_lg9b_dev80_devpop.txt")