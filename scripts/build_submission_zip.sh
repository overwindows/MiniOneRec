#!/bin/bash
# Build a Codabench-ready submission zip.
#
# WHY THE INNER FILE MUST BE NAMED prediction.txt
# ------------------------------------------------
# The MIND competition's official evaluator (msnews/MIND evaluate.py) does:
#
#     submission_answer_file = open(os.path.join(submit_dir, "prediction.txt"), 'r')
#
# and the official sample archive (msnews/MIND prediction.zip) contains
# exactly one member, `prediction.txt`. The competition's Submission
# Guidelines page states the zip must contain "nothing but a json file
# named prediction.txt".
#
# Our earlier zips used `submission_lg9b_*.txt` / `submission_ranker_l13.txt`
# as the inner name, which the evaluator cannot find -> the submission would
# fail to score (or be scored as an empty ranking).
#
# The CONTENT format is already correct and is NOT changed here:
#     <impression_id> [r1,r2,...,rn]
# where ri is the 1-based rank of the i-th candidate in its original order.
#
# Usage: bash scripts/build_submission_zip.sh <txt_file> <out_zip>
set -euo pipefail
cd /import/ml-sc-scratch5/chenw/MiniOneRec

TXT="${1:?usage: build_submission_zip.sh <txt_file> <out_zip>}"
OUT="${2:?usage: build_submission_zip.sh <txt_file> <out_zip>}"

[ -f "$TXT" ] || { echo "ERROR: missing $TXT" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"
TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT

# zip requires the member to be named prediction.txt, so stage it under that
# name in a temp dir and zip from there.
cp "$TXT" "$TMPD/prediction.txt"

# The zip target must be an ABSOLUTE path, because zip is run from inside
# $TMPD. "$OLDPWD/$OUT" breaks in two ways: it resolves against the CALLER's
# directory rather than this script's, and an already-absolute $OUT becomes
# "/path//abs/path.zip".
case "$OUT" in
  /*) OUT_ABS="$OUT" ;;
  *)  OUT_ABS="$PWD/$OUT" ;;
esac

rm -f "$OUT_ABS"
( cd "$TMPD" && zip -q -X "$OUT_ABS" prediction.txt )

# ---- verify: exactly one member, named prediction.txt, same byte count ----
python3 - "$OUT_ABS" "$TXT" <<'PY'
import sys, zipfile, os
zpath, txt = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zpath) as z:
    names = z.namelist()
    assert names == ["prediction.txt"], f"bad members: {names}"
    inner = z.getinfo("prediction.txt")
    assert inner.file_size == os.path.getsize(txt), \
        f"size mismatch {inner.file_size} != {os.path.getsize(txt)}"
    with z.open("prediction.txt") as f:
        first = f.readline().decode()
print(f"OK {zpath}: 1 member 'prediction.txt' "
      f"({inner.file_size:,} B)  first={first.strip()[:60]}")
PY
