#!/usr/bin/env python3
"""Check that the trainer and the serving path compute the SAME pop-WIT.

The trainer (train_ranker_causal_final.py) and the serving path
(apply_ranker_devpop.py) each implement the popularity within-impression
transforms. If they disagree by even one ULP the model sees a different
feature distribution at test time than it was trained on, which silently
costs AUC. This compares them on random data.

Also checks the column ORDER: the trainer emits
    [cat_pop_rank, cat_pop_z, cat_ctr_rank, cat_ctr_z,
     sub_pop_rank, sub_pop_z, sub_ctr_rank, sub_ctr_z]
and the serving path must place each value in the matching model slot.

Usage: python src/test_pop_wit_consistency.py
"""
import sys

import numpy as np

sys.path.insert(0, "src")
from train_ranker_causal_final import (  # noqa: E402
    POP_WIT_FEATS, pop_wit_transforms as train_pop_wit,
)
from apply_ranker_devpop import (  # noqa: E402
    POP_WIT_FEATS as SERV_POP_WIT_FEATS,
    POP_WIT_SRC, pop_wit_transforms as serv_pop_wit,
)

rng = np.random.default_rng(0)
N = 5000
G = 200
g = np.repeat(np.arange(G), N // G)

# 4 popularity source columns: cat_pop, cat_ctr, sub_pop, sub_ctr.
# Ties are deliberate -- cf_score-like columns tie at 0 in production and
# the average-rank must handle that identically on both sides.
src = np.column_stack([
    rng.random(N).astype(np.float32),
    np.where(rng.random(N) < 0.7, 0.0, rng.random(N)).astype(np.float32),
    rng.random(N).astype(np.float32),
    np.where(rng.random(N) < 0.5, 0.0, rng.random(N)).astype(np.float32),
])

catpop = src[:, 0:2]
subpop = src[:, 2:4]

a = train_pop_wit(catpop, subpop, g)

# The serving path gets ONE matrix in the MODEL's feature order and a list
# of source positions. Emulate that: build a matrix whose model slots are
# [cat_pop, cat_ctr, sub_pop, sub_ctr] at positions 0..3.
X = np.zeros((N, len(SERV_POP_WIT_FEATS)), dtype=np.float32)
for j, c in enumerate(POP_WIT_SRC):
    X[:, j] = src[:, j]
b = serv_pop_wit(X, g, [0, 1, 2, 3])

assert POP_WIT_FEATS == SERV_POP_WIT_FEATS, \
    f"feature name/order mismatch:\n  trainer: {POP_WIT_FEATS}\n  serving: {SERV_POP_WIT_FEATS}"

d = np.abs(a - b)
print(f"max abs diff = {d.max():.3e}")
assert d.max() == 0.0, f"trainer and serving disagree (max diff {d.max():.3e})"

# The rank columns must be in [0,1] and the z-scores must be ~0 mean / 1 std
# within each impression.
for j in range(0, 8, 2):
    r = a[:, j]
    assert r.min() >= -1e-6 and r.max() <= 1 + 1e-6, f"rank out of range: {r.min()} {r.max()}"
for j in range(1, 8, 2):
    z = a[:, j]
    for gi in range(G):
        zz = z[g == gi]
        assert abs(zz.mean()) < 1e-3, f"z-score mean {zz.mean()} != 0 in group {gi}"
        assert abs(zz.std() - 1.0) < 1e-2, f"z-score std {zz.std()} != 1 in group {gi}"

print("OK: trainer and serving produce identical pop-WIT features")
print(f"   features: {POP_WIT_FEATS}")
