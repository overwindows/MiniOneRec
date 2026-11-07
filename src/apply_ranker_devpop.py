#!/usr/bin/env python3
"""Apply a LightGBM ranker to a test feature table.

LightGBM saves generic feature names (Column_0, Column_1, ...), so the
feature list is read from a sidecar JSON file next to the model:
    <model>.feats.json  ->  {"features": ["llm_score", ...]}

If the sidecar is missing, the model's own feature_name() is used (works
for models trained with explicit names).

Streams impression-by-impression and writes the internal format:
    impression_id N12345 N67890 ...   (news IDs in ranked order)

Usage:
    python src/apply_ranker_devpop.py \
        --features results/ranker/lg9b_test_features_devpopcat.tsv \
        --model    results/ranker/ranker_v2_lg9b_causal_devpop_cat.txt \
        --output   results/ranker/lg9b_causal_test_predictions.txt
"""
import argparse
import json
import os

import numpy as np


def _avg_rank_grouped(v, g):
    """Average (fractional) rank of v within each group g, normalized to 0..1.

    Plain `argsort().argsort()` breaks ties by POSITION, which injects pure
    noise: cf_score is 0 for 88.7% of test rows, so most candidates in an
    impression tie at zero and would get an arbitrary 0..m-1 ordering.
    Average rank gives every tied candidate the same value.
    """
    n = len(v)
    o = np.lexsort((v, g))
    gs = g[o]
    vs = v[o]

    new_g = np.empty(n, dtype=bool)
    new_g[0] = True
    new_g[1:] = gs[1:] != gs[:-1]
    grp_start = np.flatnonzero(new_g)
    grp_end = np.append(grp_start[1:], n)
    grp_size = grp_end - grp_start

    tie = np.empty(n, dtype=bool)
    tie[0] = True
    tie[1:] = (gs[1:] != gs[:-1]) | (vs[1:] != vs[:-1])
    run_start = np.flatnonzero(tie)
    run_end = np.append(run_start[1:], n)
    run_len = run_end - run_start

    gidx = np.searchsorted(grp_start, run_start, side="right") - 1
    avg0 = (run_start - grp_start[gidx]) + (run_len - 1) / 2.0
    denom = np.maximum(grp_size[gidx] - 1, 1)
    avg_norm = avg0 / denom

    out = np.empty(n, dtype=np.float32)
    out[o] = np.repeat(avg_norm, run_len)
    return out


def _zscore_grouped(v, g):
    """Per-group z-score of v (population std), vectorized."""
    cnt = np.bincount(g).astype(np.float64)
    s1 = np.bincount(g, weights=v.astype(np.float64))
    s2 = np.bincount(g, weights=(v.astype(np.float64) ** 2))
    mean = s1 / np.maximum(cnt, 1)
    var = np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0.0)
    return ((v - mean[g]) / (np.sqrt(var[g]) + 1e-6)).astype(np.float32)


def within_imp_transforms(X, g):
    """Per-impression average-rank + z-score of llm_score and cf_score.

    Returns 4 columns: [llm_rank, llm_z, cf_rank, cf_z].
    """
    out = np.zeros((X.shape[0], 4), dtype=np.float32)
    out[:, 0] = _avg_rank_grouped(X[:, 0], g)
    out[:, 1] = _zscore_grouped(X[:, 0], g)
    out[:, 2] = _avg_rank_grouped(X[:, 1], g)
    out[:, 3] = _zscore_grouped(X[:, 1], g)
    return out


POP_WIT_FEATS = ["cat_pop_rank", "cat_pop_z", "cat_ctr_rank", "cat_ctr_z",
                 "sub_pop_rank", "sub_pop_z", "sub_ctr_rank", "sub_ctr_z"]
# the source columns the pop WIT is derived from, in the same order
POP_WIT_SRC = ["cat_pop", "cat_ctr", "sub_pop", "sub_ctr"]


def pop_wit_transforms(X, g, src_pos):
    """Per-impression average-rank + z-score of the popularity features.

    Returns 8 columns in POP_WIT_FEATS order. `src_pos` gives the column
    index in X of each of POP_WIT_SRC.
    """
    out = np.zeros((X.shape[0], 8), dtype=np.float32)
    for j, p in enumerate(src_pos):
        v = X[:, p]
        out[:, 2 * j] = _avg_rank_grouped(v, g)
        out[:, 2 * j + 1] = _zscore_grouped(v, g)
    return out


def load_feats(model_path, booster):
    side = model_path + ".feats.json"
    if os.path.exists(side):
        with open(side) as f:
            return json.load(f)["features"]
    names = booster.feature_name()
    if names and not names[0].startswith("Column_"):
        return names
    raise SystemExit(
        f"no feature list for {model_path}: expected {side}. "
        "Retrain with train_ranker_causal_final.py or write the sidecar.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import lightgbm as lgb
    booster = lgb.Booster(model_file=args.model)
    feats = load_feats(args.model, booster)
    print(f"Loaded model: {args.model}", flush=True)
    print(f"  {len(feats)} features: {feats}", flush=True)

    # The within-impression transforms are DERIVED, not stored in the feature
    # table: they are the per-impression rank/z-score of llm_score and
    # cf_score. They must be computed from the SAME (masked) columns the
    # model was trained on, so they are rebuilt here per impression.
    WIT_FEATS = ["llm_rank", "llm_z", "cf_rank", "cf_z"]
    use_wit = all(w in feats for w in WIT_FEATS)
    if use_wit:
        print("  model uses within-impression transforms; computing them "
              "per impression", flush=True)

    # The popularity WIT is derived the same way: per-impression average-rank
    # + z-score of cat_pop / cat_ctr / sub_pop / sub_ctr. Those four source
    # columns ARE stored in the feature table, so they are read normally and
    # only the 8 derived columns are rebuilt here.
    use_pop_wit = all(w in feats for w in POP_WIT_FEATS)
    if use_pop_wit:
        print("  model uses popularity within-impression transforms; "
              "computing them per impression", flush=True)

    n_impressions = 0
    n_rows = 0
    # LightGBM's predict() has a fixed per-call overhead, so calling it once
    # per impression (2.37M calls for the test split) dominates the runtime.
    # Buffer whole impressions and predict in large batches instead.
    BATCH_ROWS = 200_000
    with open(args.features) as f, open(args.output, "w") as out:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        derived = set(WIT_FEATS) | set(POP_WIT_FEATS)
        missing = [c for c in feats if c not in idx and c not in derived]
        if missing:
            raise SystemExit(f"feature table is missing columns: {missing}")
        cols = [idx[c] for c in feats if c in idx]
        col_feats = [c for c in feats if c in idx]
        # position of each stored column in the model's feature order
        col_pos = [feats.index(c) for c in col_feats]
        wit_pos = [feats.index(w) for w in WIT_FEATS] if use_wit else []
        pop_wit_pos = ([feats.index(w) for w in POP_WIT_FEATS]
                       if use_pop_wit else [])
        # where the 4 popularity source columns live in the model's order
        pop_src_pos = ([feats.index(c) for c in POP_WIT_SRC]
                       if use_pop_wit else [])

        buf_imps = []          # list of (imp_id, [nids], start_row, n_rows)
        buf_X = []             # list of per-impression feature arrays
        buf_rows = 0

        def flush_batch():
            nonlocal n_impressions, buf_imps, buf_X, buf_rows
            if not buf_imps:
                return
            X = np.vstack(buf_X)
            if use_wit:
                # WIT is per impression: each impression is its own group.
                wit = np.zeros((len(X), 4), dtype=np.float32)
                off = 0
                for _imp, _nids, m in buf_imps:
                    wit[off:off + m] = within_imp_transforms(
                        X[off:off + m], np.zeros(m, dtype=np.int64))
                    off += m
                for j, pos in enumerate(wit_pos):
                    X[:, pos] = wit[:, j]
            if use_pop_wit:
                # Same, for the popularity features. The source columns are
                # already in X at their model positions.
                pw = np.zeros((len(X), 8), dtype=np.float32)
                off = 0
                for _imp, _nids, m in buf_imps:
                    pw[off:off + m] = pop_wit_transforms(
                        X[off:off + m], np.zeros(m, dtype=np.int64), pop_src_pos)
                    off += m
                for j, pos in enumerate(pop_wit_pos):
                    X[:, pos] = pw[:, j]
            scores = booster.predict(X)
            off = 0
            for imp, nids, m in buf_imps:
                s = scores[off:off + m]
                order = np.argsort(-s, kind="stable")
                out.write(imp + " " + " ".join(nids[i] for i in order) + "\n")
                off += m
                n_impressions += 1
            buf_imps = []
            buf_X = []
            buf_rows = 0

        cur_imp = None
        cur_nids = []
        cur_feats = []

        def close_impression():
            nonlocal cur_imp, cur_nids, cur_feats, buf_rows
            if not cur_nids:
                return
            X = np.array(cur_feats, dtype=np.float32)
            if use_wit or use_pop_wit:
                # place the stored columns at their positions in the feature list
                full = np.zeros((len(X), len(feats)), dtype=np.float32)
                for j, pos in enumerate(col_pos):
                    full[:, pos] = X[:, j]
                X = full
            buf_imps.append((cur_imp, cur_nids, len(cur_nids)))
            buf_X.append(X)
            buf_rows += len(cur_nids)
            cur_imp = None
            cur_nids = []
            cur_feats = []

        for line in f:
            p = line.rstrip("\n").split("\t")
            imp = p[0]
            if imp != cur_imp:
                close_impression()
                cur_imp = imp
            cur_nids.append(p[1])
            cur_feats.append([p[c] for c in cols])
            n_rows += 1
            if buf_rows >= BATCH_ROWS:
                flush_batch()
            if n_rows % 10_000_000 == 0:
                print(f"  {n_rows:,} rows", flush=True)
        close_impression()
        flush_batch()

    print(f"Scored {n_rows:,} rows across {n_impressions:,} impressions -> {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
