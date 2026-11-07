#!/usr/bin/env python3
"""Compare v1 vs v2 rankers on dev: per-imp AUC (leaderboard proxy), global,
and cold-only-impression slices. One dev load, all models."""
import sys
import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import load_all, BASE_FEATS  # noqa: E402

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

DEV = "results/ranker/large_dev_features.tsv"
V1 = "results/ranker/ranker_l13_large.txt"
V2 = "results/ranker/ranker_v2_large.txt"

print("Loading dev ...", flush=True)
# load_all returns augmented X (v2 order). Rebuild raw base from stream for v1.
Xdv, ydv, gdv = load_all(DEV)
print(f"  {Xdv.shape[0]:,} rows, {len(np.unique(gdv)):,} impressions", flush=True)

# raw base features: inverse of augment is easiest by re-streaming, but we can
# reconstruct: augment maps base -> [llm, llm_rank, llm_z, cf, cf_rank, cf_z,
# cf_present, base[5:]]. So base = [X[:,0], X[:,3], ???]. cand_pop/ctr/log_pop
# (base cols 2..4) are NOT recoverable from augmented. Re-stream for v1.
def load_base(path):
    Xs, ys, gs = [], [], []
    import train_ranker_v2 as t
    for gi, (Xb, y) in enumerate(t.stream_groups(path)):
        Xs.append(Xb); ys.append(y)
        gs.append(np.full(len(y), gi, dtype=np.int64))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)

print("Loading dev raw (for v1) ...", flush=True)
Xb, yb, gb = load_base(DEV)
assert np.array_equal(ydv, yb) and np.array_equal(gdv, gb)

v1 = lgb.Booster(model_file=V1)
v2 = lgb.Booster(model_file=V2)
print("Predicting v1 / v2 ...", flush=True)
p1 = v1.predict(Xb)
p2 = v2.predict(Xdv)
llm = Xb[:, 0]
cold_row = Xb[:, 1] == 0  # cf absent (raw base col 1)

def per_imp(y, s, g, mask=None):
    yy, ss, gg = (y, s, g) if mask is None else (y[mask], s[mask], g[mask])
    order = np.argsort(gg, kind="stable")
    g_s, p_s, y_s = gg[order], ss[order], yy[order]
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

# cold-only impressions: every candidate cold
imp_cold_frac = np.bincount(gdv, weights=cold_row.astype(float)) / np.bincount(gdv)
cold_imps = np.where(imp_cold_frac == 1.0)[0]
imp_mask = np.isin(gdv, cold_imps)
print(f"\ncold-only impressions: {len(cold_imps):,} ({len(cold_imps)/len(np.unique(gdv)):.1%}), "
      f"rows {imp_mask.sum():,}")

rows = []
for name, s in [("llm", llm), ("v1", p1), ("v2", p2)]:
    g_auc = roc_auc_score(ydv, s)
    pia, n = per_imp(ydv, s, gdv)
    pia_c, nc = per_imp(ydv[imp_mask], s[imp_mask], gdv[imp_mask])
    rows.append((name, g_auc, pia, pia_c))
    print(f"{name:>4}  global={g_auc:.4f}  per-imp={pia:.4f} (n={n})  "
          f"cold-imp per-imp={pia_c:.4f} (n={nc})", flush=True)

print("\ndeltas vs llm (per-imp):")
for name, g_auc, pia, pia_c in rows:
    print(f"{name:>4}  all {pia-rows[0][2]:+.4f}   cold-only {pia_c-rows[0][3]:+.4f}")