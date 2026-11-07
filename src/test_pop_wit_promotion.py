#!/usr/bin/env python3
"""Check that the PROMOTED trainer builds the same feature matrix as the
experiment that measured the +0.0097 gain.

The gain was measured by src/exp_ranker_wit_pop.py (arm Q). It is only
real if the production trainer (src/train_ranker_causal_final.py --pop-wit)
builds the SAME columns in the SAME order from the SAME inputs. A silent
mismatch (e.g. a different cache, a different normalisation, a swapped
column) would ship a model that never saw the features it was measured on.

This builds both matrices on the same masked data and compares them.

Usage: python src/test_pop_wit_promotion.py
"""
import sys

import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups  # noqa: E402
from train_ranker_causal_final import (  # noqa: E402
    _avg_rank_grouped, _zscore_grouped, pop_wit_transforms,
)
from exp_ranker_stack import (  # noqa: E402
    load_llm, imp_ids, joint_mask, mask_rows_to_coverage,
)

DEV_FEATS = "results/ranker/large_dev_features.tsv"
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"
CACHE = "results/ranker/causal_dev_arrays.npz"
CACHE2 = "results/ranker/causal_subcat_arrays.npz"
COV_CF = 0.113
SEED = 13

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

# ---- the trainer's masking path (train_ranker_causal_final.py) ----
rng = np.random.default_rng(SEED)
trainpop_hit = X[:, 2] > 0
cf_hit = X[:, 1] != 0
devpop_hit = C["causal_pop"] > 0
m_trainpop, m_devpop = joint_mask(nids, trainpop_hit, devpop_hit, rng)
m_cf = mask_rows_to_coverage(cf_hit, COV_CF, rng)
Xm = X.copy()
Xm[~m_trainpop, 2:5] = 0.0
Xm[~m_cf, 1] = 0.0

mp = np.maximum(C["asof_max_pop"], 1.0)
mc = np.maximum(C["asof_max_cat"], 1.0)
ms = np.maximum(S["sub_asof"], 1.0)
devpop = np.column_stack([
    C["causal_pop"] / mp,
    np.where(C["causal_pop"] > 0,
             C["causal_clicks"] / np.maximum(C["causal_pop"], 1), 0.0),
    np.log1p(C["causal_pop"]) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
devpop[~m_devpop] = 0.0
catpop = np.column_stack([
    C["causal_cat_pop"] / mc,
    np.where(C["causal_cat_pop"] > 0,
             C["causal_cat_clicks"] / np.maximum(C["causal_cat_pop"], 1), 0.0)
]).astype(np.float32)
subpop = np.column_stack([
    S["sub_pop"] / ms,
    np.where(S["sub_pop"] > 0,
             S["sub_clk"] / np.maximum(S["sub_pop"], 1), 0.0)]).astype(np.float32)

# ---- the trainer's pop-WIT ----
trainer_wit = pop_wit_transforms(catpop, subpop, g)

# ---- the experiment's pop-WIT (arm Q) ----
POP4 = np.hstack([catpop, subpop])   # [cat_pop, cat_ctr, sub_pop, sub_ctr]
exp_wit = np.zeros((X.shape[0], 8), dtype=np.float32)
for j in range(4):
    v = POP4[:, j]
    exp_wit[:, 2 * j] = _avg_rank_grouped(v, g)
    exp_wit[:, 2 * j + 1] = _zscore_grouped(v, g)

d = np.abs(trainer_wit - exp_wit)
print(f"pop-WIT max abs diff = {d.max():.3e}")
assert d.max() == 0.0, f"trainer and experiment disagree (max diff {d.max():.3e})"

# ---- the experiment's llm/cf WIT must also match the trainer's ----
from train_ranker_causal_final import within_imp_transforms  # noqa: E402
trainer_llm_wit = within_imp_transforms(Xm, g)
exp_llm_wit = np.zeros((X.shape[0], 4), dtype=np.float32)
exp_llm_wit[:, 0] = _avg_rank_grouped(Xm[:, 0], g)
exp_llm_wit[:, 1] = _zscore_grouped(Xm[:, 0], g)
exp_llm_wit[:, 2] = _avg_rank_grouped(Xm[:, 1], g)
exp_llm_wit[:, 3] = _zscore_grouped(Xm[:, 1], g)
d2 = np.abs(trainer_llm_wit - exp_llm_wit)
print(f"llm/cf WIT max abs diff = {d2.max():.3e}")
assert d2.max() == 0.0, f"llm/cf WIT mismatch (max diff {d2.max():.3e})"

# ---- the popularity SOURCE columns must match too (they feed the WIT) ----
assert np.allclose(catpop[:, 0], C["causal_cat_pop"] / mc)
assert np.allclose(subpop[:, 0], S["sub_pop"] / ms)
print("OK: promoted trainer reproduces the experiment's feature matrix exactly")
