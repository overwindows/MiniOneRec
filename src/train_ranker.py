#!/usr/bin/env python3
"""
Stage-2 ranker probe on MIND_small dev features (from build_ranker_features.py).

Trains LightGBM (if available) or a small MLP with user-grouped CV on dev,
and reports per-fold + mean PER-IMPRESSION AUC (MIND's actual metric) against
baselines:
  - llm_score alone (must reproduce the LLM's dev AUC)
  - cf_score alone
  - llm + cf rank-blend reference

Pooled (global) AUC is printed only as a reference: impression-constant
features (imp_size, hist_len, hour, ...) can raise pooled AUC via
cross-impression calibration without changing within-impression ranking,
so pooled AUC must never be the gate.

Decision gate: mean CV per-impression AUC >= llm-only + 0.002 -> worth
engineering for large/test.

Usage:
    python src/train_ranker.py --features results/ranker/dev_features.tsv
"""

import argparse
import math

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
except ImportError:
    raise SystemExit("scikit-learn required: pip install scikit-learn")

FEATURES = [
    "llm_score", "cf_score",
    "cand_pop", "cand_ctr", "cand_log_pop",
    "cat_match_frac", "subcat_match_frac",
    "hist_len", "hist_uniq_cats", "hist_top_cat_frac",
    "imp_size", "position", "position_frac",
    "hour",
]


def load_features(path):
    import csv
    X, y, groups, imps = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            X.append([float(row[k]) for k in FEATURES])
            y.append(int(row["label"]))
            groups.append(row["user_id"])
            imps.append(row["impression_id"])
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.int8),
            np.array(groups), np.array(imps))


def per_impression_auc(y, p, imps):
    """MIND's metric: mean ROC-AUC over impressions holding both classes."""
    from collections import defaultdict
    by_imp = defaultdict(list)
    for i, imp in enumerate(imps):
        by_imp[imp].append(i)
    aucs = []
    for idx in by_imp.values():
        yy = y[idx]
        if 0 < yy.sum() < len(yy):
            aucs.append(roc_auc_score(yy, p[idx]))
    return float(np.mean(aucs)) if aucs else float("nan")


def eval_cv(X, y, groups, imps, model_fn, name, n_splits=5, seed=42):
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        m = model_fn(seed + fold)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        a = per_impression_auc(y[va], p, imps[va])
        aucs.append(a)
        print(f"  [{name}] fold {fold}: per-imp AUC={a:.4f} "
              f"(pooled {roc_auc_score(y[va], p):.4f})")
    mean = float(np.mean(aucs))
    print(f"  [{name}] mean per-impression AUC = {mean:.4f}")
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    X, y, groups, imps = load_features(args.features)
    print(f"Loaded {len(y):,} rows, {X.shape[1]} features, "
          f"{len(np.unique(groups)):,} users, {len(np.unique(imps)):,} impressions, "
          f"pos rate {y.mean():.4f}")
    print("NOTE: if llm_score came from a checkpoint selected on this same dev split "
          "(early stopping / best-ckpt on dev loss), all numbers below are optimistic; "
          "dump scores from a fixed-epoch checkpoint for the go/no-go gate.")

    i_llm = FEATURES.index("llm_score")
    i_cf = FEATURES.index("cf_score")

    # Single-feature baselines (no CV needed — monotone in the feature)
    print("\n== Baselines ==")
    print(f"  llm_score alone: per-imp AUC {per_impression_auc(y, X[:, i_llm], imps):.4f} "
          f"(pooled {roc_auc_score(y, X[:, i_llm]):.4f})")
    print(f"  cf_score  alone: per-imp AUC {per_impression_auc(y, X[:, i_cf], imps):.4f} "
          f"(pooled {roc_auc_score(y, X[:, i_cf]):.4f})")

    # CV baselines with constant models (sanity: CV AUC of a single feature)
    print(f"\n== GroupKFold({args.n_splits}) by user ==")
    const_llm = lambda seed: _Const(i_llm)
    const_cf = lambda seed: _Const(i_cf)
    eval_cv(X, y, groups, imps, const_llm, "llm-only", args.n_splits)
    eval_cv(X, y, groups, imps, const_cf, "cf-only", args.n_splits)

    # LightGBM
    try:
        import lightgbm as lgb

        def lgb_fn(seed):
            return lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=100,
                subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=32,
                verbose=-1,
            )

        print("\n== LightGBM (all features) ==")
        eval_cv(X, y, groups, imps, lgb_fn, "lgbm", args.n_splits)

        # Ablation: without llm_score (how much signal do dense features carry alone?)
        keep = [i for i in range(len(FEATURES)) if i != i_llm]
        X2 = X[:, keep]
        print("\n== LightGBM (dense features only, no llm_score) ==")
        eval_cv(X2, y, groups, imps, lgb_fn, "lgbm-dense", args.n_splits)
    except ImportError:
        print("\nLightGBM not available; trying sklearn MLP")

        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        def mlp_fn(seed):
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=30,
                              early_stopping=True, random_state=seed),
            )

        print("\n== MLP (all features) ==")
        eval_cv(X, y, groups, imps, mlp_fn, "mlp", args.n_splits)


class _Const:
    """Wraps a single feature column as a 'model' for CV bookkeeping."""
    def __init__(self, idx):
        self.idx = idx
    def fit(self, X, y):
        pass
    def predict_proba(self, X):
        col = X[:, self.idx]
        return np.column_stack([1 - _sigmoid(col), _sigmoid(col)])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


if __name__ == "__main__":
    main()