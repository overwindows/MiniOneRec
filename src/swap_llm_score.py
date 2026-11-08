#!/usr/bin/env python3
"""Replace the llm_score column of a ranker feature table with new scores.

WHY
---
Re-scoring a split with a different LLM configuration changes ONLY the
llm_score column of the feature table. Every other column (cf_score, the
popularity families, the history-derived features, the position features) is
a function of the DATA, not of the LLM.

The feature tables are huge -- the test table is 93,115,002 rows x 24
columns (17 GB) -- so rebuilding them from scratch just to swap one column
wastes an hour of CPU and 17 GB of scratch disk. This tool streams both
files and rewrites the one column in place.

ALIGNMENT
---------
The score file is one line per impression, holding the candidate scores in
CANDIDATE ORDER. The feature table holds one row per (impression, candidate)
in the same candidate order (both come from the same behaviors file). The
tool therefore walks the score file and the feature table in lockstep and
verifies the impression_id of every row group against the score file. A
mismatch is fatal -- a silent misalignment would produce a plausible but
wrong table.

Usage:
    python src/swap_llm_score.py \
        --features results/ranker/ck45500_test_features_devpopcat.tsv \
        --scores results/ranker/lg9b_ckpt45500_test_scores_mh0.tsv \
        --output results/ranker/ck45500_test_features_mh0.tsv
"""
import argparse
import sys


def iter_scores(path):
    """Yield (impression_id, [scores]) in file order."""
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            yield p[0], p[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    scores = iter_scores(args.scores)
    cur_id, cur_scores = None, None
    pos = 0
    n_rows = 0
    n_imps = 0

    with open(args.features) as fin, open(args.output, "w") as fout:
        header = fin.readline().rstrip("\n").split("\t")
        try:
            llm_i = header.index("llm_score")
            imp_i = header.index("impression_id")
        except ValueError:
            sys.exit(f"FATAL: feature table lacks llm_score/impression_id: {header}")
        fout.write("\t".join(header) + "\n")

        for line in fin:
            p = line.rstrip("\n").split("\t")
            imp = p[imp_i]

            if imp != cur_id:
                # Advance to this impression's score line.
                if cur_id is not None:
                    if pos != len(cur_scores):
                        sys.exit(
                            f"FATAL: impression {cur_id} has {len(cur_scores)} "
                            f"scores but {pos} feature rows -- tables do not align")
                    n_imps += 1
                for sid, sc in scores:
                    if sid == imp:
                        cur_id, cur_scores, pos = sid, sc, 0
                        break
                else:
                    sys.exit(f"FATAL: no scores for impression {imp}")
                if pos >= len(cur_scores):
                    sys.exit(f"FATAL: no score at position {pos} for impression {imp}")

            if pos >= len(cur_scores):
                sys.exit(f"FATAL: too many feature rows for impression {imp}")

            p[llm_i] = f"{float(cur_scores[pos]):.6f}"
            fout.write("\t".join(p) + "\n")
            pos += 1
            n_rows += 1
            if not args.quiet and n_rows % 5_000_000 == 0:
                print(f"  {n_rows:,} rows", flush=True)

    if cur_id is not None:
        if pos != len(cur_scores):
            sys.exit(
                f"FATAL: last impression {cur_id} has {len(cur_scores)} scores "
                f"but {pos} feature rows -- tables do not align")
        n_imps += 1

    print(f"Wrote {n_rows:,} rows for {n_imps:,} impressions -> {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
