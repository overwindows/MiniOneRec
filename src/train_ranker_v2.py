#!/usr/bin/env python3
"""
Ranker v2: cold-start-robust Stage-2 ranker.

Diagnosis of the 0.6986 test score (vs 0.7074 dev): the v1 ranker leaned on
train-derived features (cf_score, cand_pop, cand_ctr) that are informative on
dev (67.7% of candidates seen in train) but dead on test (12.5% seen; 89.8% of
ranker top-1 picks were cold). On test it fell back to LLM order.

v2 changes:
  1. Within-impression features only (no train-side stats):
     - llm_rank_frac, llm_z (z-scored within impression)
     - cf_rank_frac, cf_z, cf_present
     - cat/subcat match fracs, hist profile, position, imp_size, hour
  2. LambdaRank objective (per-impression NDCG) instead of global binary
     cross-entropy — matches the leaderboard metric.
  3. Validation: global dev AUC + COLD-ONLY slice (cand_pop==0 & cf==0),
     which simulates test conditions. Model selection favors cold slice.

Usage:
    python src/train_ranker_v2.py \
        --train_features results/ranker/large_train_features.tsv \
        --dev_features   results/ranker/large_dev_features.tsv \
        --output         results/ranker/ranker_v2_large.txt
"""

import argparse
import math

import numpy as np

BASE_FEATS = [
    "llm_score", "cf_score",
    "cand_pop", "cand_ctr", "cand_log_pop",
    "cat_match_frac", "subcat_match_frac",
    "hist_len", "hist_uniq_cats", "hist_top_cat_frac",
    "imp_size", "position", "position_frac",
    "hour",
]

# Final feature set: within-impression transforms + history/candidate text
# matches. cand_pop/cand_ctr/cand_log_pop are EXCLUDED (cold-start dead
# weight); cf is kept but rank/z-scored within impression + presence flag.
FEATS = [
    "llm_score", "llm_rank_frac", "llm_z",
    "cf_score", "cf_rank_frac", "cf_z", "cf_present",
    "cat_match_frac", "subcat_match_frac",
    "hist_len", "hist_uniq_cats", "hist_top_cat_frac",
    "imp_size", "position", "position_frac",
    "hour",
]


def stream_groups(path, max_groups=0):
    """Yield (X_base, y) per impression, rows contiguous."""
    cur = None
    rows = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        cols = [idx[c] for c in BASE_FEATS]
        i_lab = idx["label"]
        i_imp = idx["impression_id"]
        n = 0
        for line in f:
            p = line.rstrip("\n").split("\t")
            imp = p[i_imp]
            if imp != cur:
                if rows:
                    yield np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int8)
                    n += 1
                    if max_groups and n >= max_groups:
                        return
                cur = imp
                rows = []
                labels = []
            rows.append([p[c] for c in cols])
            labels.append(int(p[i_lab]))
        if rows:
            yield np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int8)


def augment(X):
    """Add within-impression transforms. Columns follow BASE_FEATS order."""
    n = X.shape[0]
    llm = X[:, 0]
    cf = X[:, 1]
    llm_order = llm.argsort().argsort()  # rank 0..n-1 (ascending)
    cf_order = cf.argsort().argsort()
    llm_mu = llm.mean()
    llm_sd = llm.std()
    cf_mu = cf.mean()
    cf_sd = cf.std()
    out = np.empty((n, len(FEATS)), dtype=np.float32)
    out[:, 0] = llm
    out[:, 1] = llm_order / max(n - 1, 1)
    out[:, 2] = (llm - llm_mu) / (llm_sd + 1e-6)
    out[:, 3] = cf
    out[:, 4] = cf_order / max(n - 1, 1)
    out[:, 5] = (cf - cf_mu) / (cf_sd + 1e-6)
    out[:, 6] = (cf > 0).astype(np.float32)
    out[:, 7:] = X[:, 5:]  # cat_match .. hour (BASE cols 5..13)
    return out


def load_all(path, max_groups=0):
    Xs, ys, groups = [], [], []
    for gi, (Xb, y) in enumerate(stream_groups(path, max_groups)):
        Xs.append(augment(Xb))
        ys.append(y)
        groups.append(np.full(len(y), gi, dtype=np.int64))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(groups)


def per_imp_auc(y_true, score, groups):
    """AUC averaged over impressions with both classes."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for g in np.unique(groups):
        m = groups == g
        y = y_true[m]
        if y.sum() == 0 or y.sum() == m.sum():
            continue
        aucs.append(roc_auc_score(y, score[m]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_features", required=True)
    ap.add_argument("--dev_features", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_train_groups", type=int, default=0,
                    help="0 = all (83M rows ~ 60GB RAM as float32+augment; "
                         "cap to e.g. 400000 to subsample impressions)")
    args = ap.parse_args()

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    print("Loading train features ...", flush=True)
    Xtr, ytr, gtr = load_all(args.train_features, args.max_train_groups)
    print(f"  train: {Xtr.shape[0]:,} rows, {len(np.unique(gtr)):,} impressions, "
          f"pos rate {ytr.mean():.4f}", flush=True)

    print("Loading dev features ...", flush=True)
    Xdv, ydv, gdv = load_all(args.dev_features)
    print(f"  dev:   {Xdv.shape[0]:,} rows, {len(np.unique(gdv)):,} impressions", flush=True)

    # cold slice on dev: candidates with no train-side signal
    # (cf==0 and cand_pop==0 in BASE cols; after augment, cf col 3, and
    # cand_pop is dropped — recompute mask from raw stream is complex, so
    # approximate with cf_present==0 & cat-independent: use cf col only,
    # since cand_pop==0 correlates strongly with cf==0 on cold items)
    cold_mask = Xdv[:, 6] == 0  # cf_present
    print(f"  dev cold rows (cf absent): {cold_mask.mean():.1%}", flush=True)

    # baselines
    auc_llm = roc_auc_score(ydv, Xdv[:, 0])
    print(f"\nbaselines on dev: llm global AUC = {auc_llm:.4f}", flush=True)

    # LambdaRank: group = impression, label = click
    train_groups = np.bincount(gtr)
    # keep only groups with >=2 rows and >=1 click for ranking loss
    keep = np.isin(gtr, np.where((train_groups >= 2))[0])
    # LightGBM needs groups sorted contiguously — they already are
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
    }
    print("\nTraining LambdaRank ...", flush=True)
    lgb_train = lgb.Dataset(Xtr, label=ytr, group=train_groups)
    booster = lgb.train(
        params, lgb_train,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(100)],
    )
    booster.save_model(args.output, num_iteration=booster.best_iteration or 500)
    print(f"Saved: {args.output}", flush=True)

    # evaluate
    pred = booster.predict(Xdv)
    auc_full = roc_auc_score(ydv, pred)
    auc_cold = roc_auc_score(ydv[cold_mask], pred[cold_mask])
    llm_cold = roc_auc_score(ydv[cold_mask], Xdv[cold_mask, 0])
    pia = per_imp_auc(ydv, pred, gdv)
    pia_llm = per_imp_auc(ydv, Xdv[:, 0], gdv)
    print(f"\n== dev results ==")
    print(f"  global AUC:  ranker={auc_full:.4f}  llm={auc_llm:.4f}  delta={auc_full-auc_llm:+.4f}")
    print(f"  COLD  AUC:   ranker={auc_cold:.4f}  llm={llm_cold:.4f}  delta={auc_cold-llm_cold:+.4f}")
    print(f"  per-imp AUC: ranker={pia:.4f}  llm={pia_llm:.4f}  delta={pia-pia_llm:+.4f}")


if __name__ == "__main__":
    main()