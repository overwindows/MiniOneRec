#!/usr/bin/env python3
"""Fast dev eval for ranker v2 (vectorized per-impression AUC)."""
import sys
import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import load_all, FEATS  # noqa: E402

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "results/ranker/ranker_v2_large.txt"
DEV = sys.argv[2] if len(sys.argv) > 2 else "results/ranker/large_dev_features.tsv"

print("Loading dev ...", flush=True)
Xdv, ydv, gdv = load_all(DEV)
print(f"  {Xdv.shape[0]:,} rows, {len(np.unique(gdv)):,} impressions", flush=True)

booster = lgb.Booster(model_file=MODEL)
print("Predicting ...", flush=True)
pred = booster.predict(Xdv)

cold = Xdv[:, 6] == 0  # cf_present
llm = Xdv[:, 0]

auc = roc_auc_score(ydv, pred)
auc_llm = roc_auc_score(ydv, llm)
auc_cold = roc_auc_score(ydv[cold], pred[cold])
auc_cold_llm = roc_auc_score(ydv[cold], llm[cold])

# vectorized per-impression AUC via rank formula
order = np.argsort(gdv, kind="stable")
g_s = gdv[order]
p_s = pred[order]
y_s = ydv[order]
# ranks within group (1-based, ascending by pred)
starts = np.searchsorted(g_s, np.unique(g_s), side="left")
ends = np.searchsorted(g_s, np.unique(g_s), side="right")
ranks = np.empty(len(p_s), dtype=np.float64)
for s, e in zip(starts, ends):
    o = np.argsort(p_s[s:e], kind="stable")
    ranks[s:e] = o.argsort() + 1  # 1..n
pos = y_s == 1
neg = y_s == 0
g_pos_sum = np.bincount(g_s[pos], weights=ranks[pos], minlength=len(starts))
n_pos = np.bincount(g_s[pos], minlength=len(starts)).astype(float)
n_neg = np.bincount(g_s[neg], minlength=len(starts)).astype(float)
valid = (n_pos > 0) & (n_neg > 0)
aucs = (g_pos_sum[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2) / (n_pos[valid] * n_neg[valid])
pia = aucs.mean()

# llm per-imp AUC
p_s2 = llm[order]
ranks2 = np.empty(len(p_s2), dtype=np.float64)
for s, e in zip(starts, ends):
    o = np.argsort(p_s2[s:e], kind="stable")
    ranks2[s:e] = o.argsort() + 1
g_pos_sum2 = np.bincount(g_s[pos], weights=ranks2[pos], minlength=len(starts))
aucs2 = (g_pos_sum2[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2) / (n_pos[valid] * n_neg[valid])
pia_llm = aucs2.mean()

print(f"\n== dev results (ranker v2) ==")
print(f"  global AUC:  ranker={auc:.4f}  llm={auc_llm:.4f}  delta={auc-auc_llm:+.4f}")
print(f"  COLD  AUC:   ranker={auc_cold:.4f}  llm={auc_cold_llm:.4f}  delta={auc_cold-auc_cold_llm:+.4f}  (cold frac {cold.mean():.1%})")
print(f"  per-imp AUC: ranker={pia:.4f}  llm={pia_llm:.4f}  delta={pia-pia_llm:+.4f}")