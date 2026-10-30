#!/usr/bin/env python3
"""
Apply the saved Stage-2 LightGBM ranker to a feature table and write
ranked predictions in the internal format (one line per impression):

    impression_id N12345 N67890 ...   <- news IDs in ranked order (best first)

Memory-safe: processes the table impression-group by impression-group
(rows for one impression are contiguous), never materializing all 93M rows.

Usage:
    python src/apply_ranker.py \
        --features results/ranker/large_test_features.tsv \
        --model    results/ranker/ranker_l13_large.txt \
        --output   results/ranker/test_ranker_predictions.txt
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import lightgbm as lgb
    booster = lgb.Booster(model_file=args.model)
    print(f"Loaded model: {args.model}", flush=True)

    # Feature order used at training time (train_ranker.FEATURES). The booster
    # was fit on a raw numpy array, so its stored names are Column_0..N —
    # we must supply columns in that same order.
    feat_names = [
        "llm_score", "cf_score",
        "cand_pop", "cand_ctr", "cand_log_pop",
        "cat_match_frac", "subcat_match_frac",
        "hist_len", "hist_uniq_cats", "hist_top_cat_frac",
        "imp_size", "position", "position_frac",
        "hour",
    ]

    n_impressions = 0
    n_rows = 0
    with open(args.features, "r", encoding="utf-8") as f, \
         open(args.output, "w", encoding="utf-8") as out:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        try:
            cols = [idx[c] for c in feat_names]
        except KeyError as e:
            raise SystemExit(f"Feature table missing column: {e}")

        cur_imp = None
        cur_nids = []
        cur_feats = []

        def flush():
            nonlocal n_impressions
            if not cur_nids:
                return
            X = np.array(cur_feats, dtype=np.float32)
            scores = booster.predict(X)
            order = sorted(range(len(cur_nids)), key=lambda i: -scores[i])
            out.write(cur_imp + " " + " ".join(cur_nids[i] for i in order) + "\n")
            n_impressions += 1

        for line in f:
            p = line.rstrip("\n").split("\t")
            imp = p[0]
            if imp != cur_imp:
                flush()
                cur_imp = imp
                cur_nids = []
                cur_feats = []
            cur_nids.append(p[1])
            cur_feats.append([p[c] for c in cols])
            n_rows += 1
        flush()

    print(f"Scored {n_rows:,} rows across {n_impressions:,} impressions -> {args.output}")


if __name__ == "__main__":
    main()