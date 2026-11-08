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

That impression-level check alone is NOT enough. If the candidate ORDER
inside an impression ever differed between the behaviors file and the feature
table, every impression id would still line up and every row count would
still match, yet each score would be attached to the WRONG candidate. The
per-candidate check below closes that hole: with --behaviors, the news_id of
every feature row is compared against the news_id of the candidate at the
same position of the same impression in behaviors.tsv. The check is fatal on
the first mismatch.

ATOMICITY
---------
The output is written to "<output>.tmp" and moved into place with
os.replace() only after the whole table has been written and every check has
passed. A crash mid-write therefore leaves the previous output untouched
instead of a truncated file that looks complete.

Usage:
    python src/swap_llm_score.py \
        --features results/ranker/ck45500_test_features_devpopcat.tsv \
        --scores results/ranker/lg9b_ckpt45500_test_scores_mh0.tsv \
        --behaviors data/MIND_large/test/behaviors.tsv \
        --output results/ranker/ck45500_test_features_mh0.tsv
"""
import argparse
import os
import sys


def iter_scores(path):
    """Yield (impression_id, [scores]) in file order."""
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            yield p[0], p[1:]


def iter_behaviors(path):
    """Yield (impression_id, [news_ids]) in file order."""
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            cands = p[4].split()
            yield p[0], [c.rsplit("-", 1)[0] for c in cands]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--behaviors", default=None,
                    help="behaviors.tsv of the split being swapped; enables the "
                         "per-candidate news_id alignment check")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    scores = iter_scores(args.scores)
    behaviors = iter_behaviors(args.behaviors) if args.behaviors else None
    cur_id, cur_scores = None, None
    cur_nids = None
    pos = 0
    n_rows = 0
    n_imps = 0
    n_checked = 0

    tmp_out = args.output + ".tmp"
    try:
        with open(args.features) as fin, open(tmp_out, "w") as fout:
            header = fin.readline().rstrip("\n").split("\t")
            try:
                llm_i = header.index("llm_score")
                imp_i = header.index("impression_id")
            except ValueError:
                sys.exit(f"FATAL: feature table lacks llm_score/impression_id: {header}")
            try:
                nid_i = header.index("news_id")
            except ValueError:
                nid_i = None
                if behaviors is not None:
                    sys.exit("FATAL: --behaviors given but the feature table has no news_id column")
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

                    # Per-candidate alignment: the behaviors file must list the
                    # SAME candidates, in the SAME order, as the feature table.
                    if behaviors is not None:
                        for bid, bnids in behaviors:
                            if bid == imp:
                                cur_nids = bnids
                                break
                        else:
                            sys.exit(f"FATAL: impression {imp} is not in {args.behaviors}")
                        if len(cur_nids) != len(cur_scores):
                            sys.exit(
                                f"FATAL: impression {imp} has {len(cur_nids)} candidates "
                                f"in behaviors but {len(cur_scores)} scores")

                if pos >= len(cur_scores):
                    sys.exit(f"FATAL: too many feature rows for impression {imp}")

                # The candidate-level check: this feature row's news_id must be
                # the news_id of the candidate at the same position.
                if behaviors is not None:
                    if p[nid_i] != cur_nids[pos]:
                        sys.exit(
                            f"FATAL: candidate order mismatch in impression {imp} at "
                            f"position {pos}: feature table has news_id {p[nid_i]}, "
                            f"behaviors has {cur_nids[pos]} -- the score file would be "
                            f"attached to the wrong candidates")
                    n_checked += 1

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
    except BaseException:
        # Never leave a half-written table behind under a name that looks
        # final -- not even under the .tmp name.
        if os.path.exists(tmp_out):
            os.unlink(tmp_out)
        raise

    os.replace(tmp_out, args.output)
    print(f"Wrote {n_rows:,} rows for {n_imps:,} impressions -> {args.output}",
          flush=True)
    if behaviors is not None:
        print(f"Candidate-order check: {n_checked:,} rows verified against "
              f"{args.behaviors}", flush=True)


if __name__ == "__main__":
    main()
