#!/bin/bash
# Compare every COMPLETED screen arm against the ck45500 base, paired.
#
# Only arms whose score file is complete (40,000 lines) are compared: a
# partially-written file would silently produce a wrong delta. The base arm
# is the reference for every comparison.
#
# Usage: bash scripts/analyze_screen90.sh
cd /import/ml-sc-scratch5/chenw/MiniOneRec
BASE=results/ranker/dev_scores/screen90/base/hs_hybrid_base_90.tsv
BEH=data/MIND_large/dev/chunks/sub/d00/chunk_90.tsv
WANT=40000

if [ ! -f "${BASE}" ]; then
  echo "FATAL: base arm missing (${BASE})"
  exit 1
fi
nb=$(wc -l < "${BASE}")
if [ "${nb}" -ne "${WANT}" ]; then
  echo "FATAL: base arm has ${nb} lines, expected ${WANT}"
  exit 1
fi

for d in results/ranker/dev_scores/screen90/*/; do
  arm=$(basename "${d}")
  [ "${arm}" = "base" ] && continue
  f=$(ls "${d}"*.tsv 2>/dev/null | head -1)
  [ -z "${f}" ] && { echo "== ${arm}: no score file yet"; continue; }
  n=$(wc -l < "${f}")
  if [ "${n}" -ne "${WANT}" ]; then
    echo "== ${arm}: INCOMPLETE (${n}/${WANT} lines) -- skipped"
    continue
  fi
  echo "== ${arm} ($(basename "${f}"))"
  python3 scripts/compare_dev_scores_paired.py \
    --base "${BASE}" --new "${f}" --behaviors "${BEH}" \
    --label-base ck45500 --label-new "${arm}" 2>&1 | sed 's/^/   /'
  echo
done
