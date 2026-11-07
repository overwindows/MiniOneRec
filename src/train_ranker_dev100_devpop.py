#!/usr/bin/env python3
"""Final ranker for the test submission: train on 100% of dev.

Recipe is IDENTICAL to the validated exp_ranker_cold.py winner (val slice
0.7829 per-imp AUC / 0.8584 fully-cold): binary LightGBM, 300 rounds,
BASE_FEATS + 3 dev-derived popularity columns. Only the data extent changes:
  - train on ALL dev impressions (not just first 80%)
  - devpop computed from FULL dev behaviors (matches inference on test,
    where devpop comes from full dev)
"""
import sys
from collections import Counter
import math

import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups  # noqa: E402

import lightgbm as lgb  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features.tsv"
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"
DEV_BEHAVIORS = "data/MIND_large/dev/behaviors.tsv"
OUT = "results/ranker/ranker_v1_lg9b_dev100_devpop.txt"


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


print("Loading dev features ...", flush=True)
Xs, ys = [], []
for Xb, y in stream_groups(DEV_FEATS):
    Xs.append(Xb)
    ys.append(y)
X = np.vstack(Xs)
y = np.concatenate(ys)
del Xs
ids = imp_ids(DEV_FEATS)
llm9 = load_llm(LG9B_DEV)
off = 0
n_miss = 0
for imp, n in ids:
    s = llm9.get(imp)
    if s and len(s) == n:
        X[off:off + n, 0] = s
    else:
        n_miss += 1
    off += n
print(f"  {X.shape[0]:,} rows, {len(ids):,} imps, missing llm: {n_miss}", flush=True)

print("Computing FULL-dev popularity ...", flush=True)
pop = Counter()
clicks = Counter()
with open(DEV_BEHAVIORS) as f:
    for line in f:
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
max_pop = max(pop.values()) if pop else 1
log_max = math.log1p(max_pop)
print(f"  {len(pop):,} unique nids, max_pop={max_pop}", flush=True)

# map feature rows -> news_id
with open(DEV_FEATS) as f:
    header = f.readline().rstrip("\n").split("\t")
    i_nid = header.index("news_id")
nids = []
with open(DEV_FEATS) as f:
    f.readline()
    for line in f:
        nids.append(line.rstrip("\n").split("\t")[i_nid])
assert len(nids) == X.shape[0]
dp = np.array([pop.get(nid, 0) for nid in nids], dtype=np.float32)
dc = np.array([clicks.get(nid, 0) for nid in nids], dtype=np.float32)
dpop = dp / max_pop
dctr = np.where(dp > 0, dc / np.maximum(dp, 1), 0.0)
dlog = np.log1p(dp) / log_max
X2 = np.hstack([X, dpop[:, None], dctr[:, None], dlog[:, None]])
del nids, dp, dc, dpop, dctr, dlog
cov = (X2[:, 14] > 0).mean()
print(f"  devpop coverage on dev rows: {cov:.1%}", flush=True)

params = dict(objective="binary", metric="auc", learning_rate=0.05,
              num_leaves=63, min_child_samples=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=32, verbose=-1)
dtr = lgb.Dataset(X2, label=y)
bst = lgb.train(params, dtr, num_boost_round=300)

# in-sample sanity only (NOT a validation number — just confirms the model fit)
p = bst.predict(X2)
print(f"  in-sample mean pred: {p.mean():.4f} (sanity)", flush=True)

bst.save_model(OUT)
print(f"saved {OUT}", flush=True)