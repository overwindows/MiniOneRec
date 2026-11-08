#!/usr/bin/env python3
"""Verify a Codabench submission zip before upload.

The MIND leaderboard format is:
    <impression_id> [r1,r2,...,rn]
where ri is the 1-based RANK of the i-th candidate in its ORIGINAL behaviors
order (1 = best). So the list must be a permutation of 1..n.

Checks (all must pass):
  1. the zip contains EXACTLY ONE member, named `prediction.txt`
     (the official msnews/MIND evaluate.py hardcodes that name)
  2. prediction.txt has exactly one line per test impression
  3. the impression ids, IN ORDER, match data/MIND_large/test/behaviors.tsv
  4. every rank list is a permutation of 1..n for that impression's
     candidate count (no duplicates, no out-of-range, no gaps)
  5. the rank list is not constant (i.e. the ranking is not degenerate)

Usage:
    python3 src/verify_submission_zip.py results/submissions/MINDlarge_test_mh0_v6.zip
"""

import argparse
import os
import sys
import zipfile

BEHAVIORS = "data/MIND_large/test/behaviors.tsv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--behaviors", default=BEHAVIORS)
    args = ap.parse_args()

    ok = True

    # ---- 1. zip structure -------------------------------------------------
    with zipfile.ZipFile(args.zip_path) as z:
        names = z.namelist()
        print(f"zip members: {names}")
        if names != ["prediction.txt"]:
            print(f"FAIL: expected exactly one member named 'prediction.txt', got {names}")
            ok = False
        else:
            print("PASS: single member named prediction.txt")
        payload = z.read("prediction.txt").decode("utf-8")

    lines = payload.splitlines()
    print(f"prediction.txt lines: {len(lines):,}")

    # ---- 2. line count vs behaviors --------------------------------------
    with open(args.behaviors, "r", encoding="utf-8") as f:
        beh = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    print(f"behaviors impressions: {len(beh):,}")
    if len(lines) != len(beh):
        print(f"FAIL: line count {len(lines):,} != impression count {len(beh):,}")
        ok = False
    else:
        print("PASS: line count == impression count")

    # ---- 3/4/5. per-impression alignment ---------------------------------
    n_id_mismatch = 0
    n_len_mismatch = 0
    n_not_perm = 0
    n_constant = 0
    first_bad = None

    for i, (pline, b) in enumerate(zip(lines, beh)):
        imp_id = b[0]
        n_cand = len(b[4].split())

        try:
            pid, rest = pline.split(" ", 1)
        except ValueError:
            n_len_mismatch += 1
            if first_bad is None:
                first_bad = (i, pline[:120])
            continue

        if pid != imp_id:
            n_id_mismatch += 1
            if first_bad is None:
                first_bad = (i, f"id {pid} != {imp_id}")
            continue

        ranks = rest.strip()
        if not (ranks.startswith("[") and ranks.endswith("]")):
            n_not_perm += 1
            if first_bad is None:
                first_bad = (i, f"{pid}: not bracketed: {ranks[:60]}")
            continue
        vals = ranks[1:-1].split(",")

        if len(vals) != n_cand:
            n_len_mismatch += 1
            if first_bad is None:
                first_bad = (i, f"{pid}: {len(vals)} ranks vs {n_cand} candidates")
            continue

        try:
            iv = [int(v) for v in vals]
        except ValueError:
            n_not_perm += 1
            if first_bad is None:
                first_bad = (i, f"{pid}: non-integer rank")
            continue

        if sorted(iv) != list(range(1, n_cand + 1)):
            n_not_perm += 1
            if first_bad is None:
                first_bad = (i, f"{pid}: ranks are not a permutation of 1..{n_cand}")
            continue

        if n_cand > 1 and len(set(iv)) == 1:
            n_constant += 1

    if n_id_mismatch:
        print(f"FAIL: {n_id_mismatch:,} impression-id mismatches")
        ok = False
    else:
        print("PASS: every impression id matches the behaviors file, in order")

    if n_len_mismatch:
        print(f"FAIL: {n_len_mismatch:,} candidate-count mismatches")
        ok = False
    else:
        print("PASS: every rank list length == candidate count")

    if n_not_perm:
        print(f"FAIL: {n_not_perm:,} rank lists are not permutations of 1..n")
        ok = False
    else:
        print("PASS: every rank list is a permutation of 1..n")

    if n_constant:
        print(f"WARN: {n_constant:,} impressions have a constant rank list (degenerate ranking)")
    else:
        print("PASS: no degenerate (constant) rank lists")

    if first_bad:
        print(f"first bad row: {first_bad}")

    size = os.path.getsize(args.zip_path)
    print(f"zip size: {size:,} bytes")
    print("RESULT:", "ALL CHECKS PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
