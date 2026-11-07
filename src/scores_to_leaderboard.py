#!/usr/bin/env python3
"""Convert raw per-candidate LLM scores straight to MIND leaderboard format.

WHY A SEPARATE SCRIPT
---------------------
src/convert_to_leaderboard_format.py expects the INTERNAL format
(impression_id + ranked news IDs), which requires a full ranking pass first.
For the LLM-only fallback we already HAVE the scores, so we can go straight
to the leaderboard format and skip the intermediate 685 MB file.

The leaderboard format is:
    impression_id [r1,r2,...,rn]
where ri is the 1-indexed RANK of the i-th candidate in its ORIGINAL order
(1 = best). Ties are broken by original position (stable sort), which is
what the internal pipeline does too.

Both files are streamed in parallel: the score file is in impression order
and the behaviors file is in impression order, so they stay aligned.

Usage:
    python src/scores_to_leaderboard.py \
        --scores results/ranker/lg9b_ep082_test_scores_all.tsv \
        --behaviors data/MIND_large/test/behaviors.tsv \
        --output results/ranker/submission_lg9b_llm_only.txt
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--behaviors", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    n_out = 0
    n_mismatch = 0
    with open(args.scores) as sf, open(args.behaviors) as bf, \
            open(args.output, "w") as out:
        for sline, bline in zip(sf, bf):
            sp = sline.rstrip("\n").split("\t")
            bp = bline.rstrip("\n").split("\t")
            if len(sp) < 2 or len(bp) < 5:
                continue
            imp = sp[0]
            if imp != bp[0]:
                print(f"FATAL: impression mismatch at line {n_out+1}: "
                      f"scores={imp} behaviors={bp[0]}", file=sys.stderr)
                sys.exit(1)
            scores = [float(v) for v in sp[1:]]
            cands = bp[4].split()
            if len(scores) != len(cands):
                n_mismatch += 1
                if n_mismatch <= 5:
                    print(f"WARN: imp {imp}: {len(scores)} scores vs "
                          f"{len(cands)} candidates", file=sys.stderr)
                continue
            # rank of each candidate: sort by score DESC, stable so ties keep
            # the original order.
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            rank = [0] * len(scores)
            for r, i in enumerate(order):
                rank[i] = r + 1
            out.write(f"{imp} [{','.join(map(str, rank))}]\n")
            n_out += 1
            if n_out % 500_000 == 0:
                print(f"  {n_out:,} impressions", flush=True)

    print(f"wrote {n_out:,} impressions -> {args.output}")
    if n_mismatch:
        print(f"WARNING: {n_mismatch} impressions had a score/candidate "
              f"count mismatch and were SKIPPED")


if __name__ == "__main__":
    main()
