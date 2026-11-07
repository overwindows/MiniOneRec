#!/usr/bin/env python3
"""
Diagnose the dev->test transfer drop (dev 0.7074 -> test 0.6986).

Splits the large dev feature table by impression order (time proxy) and
scores the saved ranker on each block separately. If the ranker's gain over
llm_score shrinks in the LATE block (closest in time to test), the ranker
features are drifting over time and the dev number was optimistic.

Usage:
    python src/diag_ranker_timesplit.py \
        --features results/ranker/large_dev_features.tsv \
        --model    results/ranker/ranker_l13_large.txt
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_blocks", type=int, default=5)
    args = ap.parse_args()

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    booster = lgb.Booster(model_file=args.model)

    feat_names = [
        "llm_score", "cf_score",
        "cand_pop", "cand_ctr", "cand_log_pop",
        "cat_match_frac", "subcat_match_frac",
        "hist_len", "hist_uniq_cats", "hist_top_cat_frac",
        "imp_size", "position", "position_frac",
        "hour",
    ]

    # Stream: rows are in behaviors (time) order; impression_id increases.
    blocks = [[] for _ in range(args.n_blocks)]
    block_imps = [set() for _ in range(args.n_blocks)]
    cur_imp = None
    cur_rows = []
    n_rows = 0
    all_imps = []

    def flush():
        nonlocal n_rows, cur_rows
        if not cur_rows:
            return
        # assign whole impressions to blocks by running order
        idx = min(int(len(all_imps) * args.n_blocks / 376471), args.n_blocks - 1)
        blocks[idx].extend(cur_rows)
        block_imps[idx].add(cur_imp)
        n_rows += len(cur_rows)
        cur_rows = []

    with open(args.features, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        cols = [idx[c] for c in feat_names]
        i_lab = idx["label"]
        i_imp = idx["impression_id"]
        for line in f:
            p = line.rstrip("\n").split("\t")
            imp = p[i_imp]
            if imp != cur_imp:
                flush()
                cur_imp = imp
                all_imps.append(imp)
            cur_rows.append(([p[c] for c in cols], p[i_lab]))
    flush()

    print(f"Loaded {n_rows:,} rows, {len(all_imps):,} impressions, "
          f"{args.n_blocks} time blocks")
    print(f"{'block':>5} {'imps':>8} {'rows':>10} {'llm AUC':>8} {'ranker AUC':>10} {'delta':>7}")
    for b in range(args.n_blocks):
        if not blocks[b]:
            continue
        X = np.array([r[0] for r in blocks[b]], dtype=np.float32)
        y = np.array([int(r[1]) for r in blocks[b]], dtype=np.int8)
        if y.sum() == 0 or y.sum() == len(y):
            print(f"{b:>5} {len(block_imps[b]):>8} {len(y):>10}  (no positives)")
            continue
        llm_auc = roc_auc_score(y, X[:, 0])
        pred = booster.predict(X)
        rk_auc = roc_auc_score(y, pred)
        print(f"{b:>5} {len(block_imps[b]):>8} {len(y):>10} {llm_auc:>8.4f} "
              f"{rk_auc:>10.4f} {rk_auc - llm_auc:>+7.4f}")


if __name__ == "__main__":
    main()