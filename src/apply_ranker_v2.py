#!/usr/bin/env python3
"""
Apply ranker v2 (within-impression features) to a feature table and write
ranked predictions (one line per impression: imp_id nid1 nid2 ...).

Streams impression-group by impression-group; applies the same augment()
transform as train_ranker_v2.py before predicting.

Usage:
    python src/apply_ranker_v2.py \
        --features results/ranker/large_test_features.tsv \
        --model    results/ranker/ranker_v2_large.txt \
        --output   results/ranker/test_ranker_v2_predictions.txt
"""

import argparse

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_ranker_v2 import BASE_FEATS, augment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import lightgbm as lgb
    booster = lgb.Booster(model_file=args.model)
    print(f"Loaded model: {args.model}", flush=True)

    n_impressions = 0
    n_rows = 0
    with open(args.features, "r", encoding="utf-8") as f, \
         open(args.output, "w", encoding="utf-8") as out:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        cols = [idx[c] for c in BASE_FEATS]

        cur_imp = None
        cur_nids = []
        cur_feats = []

        def flush():
            nonlocal n_impressions
            if not cur_nids:
                return
            X = np.array(cur_feats, dtype=np.float32)
            Xa = augment(X)
            scores = booster.predict(Xa)
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