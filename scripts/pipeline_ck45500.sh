#!/bin/bash
# Rebuild the test submission from the ckpt-45500 test scores.
#
# WHY ckpt-45500
# --------------
# Fast checkpoint ranking on the SAME 60k dev impressions:
#     ckpt-45500  AUC 0.7185   <- peak
#     ckpt-47000  AUC 0.7165
#     ckpt-32500  AUC 0.7136   (the checkpoint the current scores use)
# ckpt-45500 is +0.005 over ckpt-32500 on the LLM signal alone.
#
# The 36 sub-chunks (6 chunks x 6 parts) are scored by
# scripts/eval_test_subchunk_large.sh into results/ranker/test_scores/ck45500/.
#
# Outputs:
#   results/ranker/lg9b_ck45500_test_scores_all.tsv   (merged, 2,370,727 lines)
#   results/ranker/submission_lg9b_ck45500_llm_only.txt
#   results/submissions/MINDlarge_test_lg9b_ck45500_llmonly.zip
set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"

LOG=logs/pipeline_ck45500.log
exec >> $LOG 2>&1
echo "=== pipeline start $(date) ==="

SCOREDIR=results/ranker/test_scores/ck45500
MERGED=results/ranker/lg9b_ck45500_test_scores_all.tsv
TOTAL=$(wc -l < data/MIND_large/test/behaviors.tsv)

# ---- 1. wait for all 36 sub-chunks ----
wait_sub () {
  local sub=$1 part=$2
  local f=${SCOREDIR}/sub_${sub}_${part}.tsv
  local want=$(wc -l < data/MIND_large/test/chunks/sub/${sub}/chunk_${part}.tsv)
  while true; do
    if [ -f "$f" ]; then
      n=$(wc -l < "$f")
      if [ "$n" -ge "$want" ]; then
        echo "sub_${sub}_${part} OK: n=$n (want $want)"
        return 0
      fi
      echo "sub_${sub}_${part} partial: n=$n (want $want)"
    else
      echo "sub_${sub}_${part} missing"
    fi
    sleep 600
  done
}
for sub in c00 c01 c02 c03 c04 c05; do
  for part in 00 01 02 03 04 05; do
    wait_sub $sub $part
  done
done

# ---- 2. merge in TEST IMPRESSION ORDER ----
# The sub-chunks are sequential slices of the test behaviors file, so
# concatenating c00..c05 x 00..05 reproduces the original order exactly.
echo "--- merging scores ---"
cat ${SCOREDIR}/sub_c0{0,1,2,3,4,5}_0{0,1,2,3,4,5}.tsv > ${MERGED}
MERGED_N=$(wc -l < ${MERGED})
echo "merged lines: $MERGED_N  test impressions: $TOTAL"
if [ "$MERGED_N" -ne "$TOTAL" ]; then
  echo "FATAL: merged score count != test impression count"
  exit 1
fi
echo "first: $(head -1 ${MERGED} | cut -f1)  last: $(tail -1 ${MERGED} | cut -f1)"

# ---- 3. LLM-only leaderboard submission (the floor) ----
echo "--- LLM-only leaderboard format ---"
python src/scores_to_leaderboard.py \
  --scores ${MERGED} \
  --behaviors data/MIND_large/test/behaviors.tsv \
  --output results/ranker/submission_lg9b_ck45500_llm_only.txt
SUB_LINES=$(wc -l < results/ranker/submission_lg9b_ck45500_llm_only.txt)
echo "submission lines: $SUB_LINES (want $TOTAL)"
if [ "$SUB_LINES" -ne "$TOTAL" ]; then
  echo "FATAL: submission line count != test impression count"
  exit 1
fi
mkdir -p results/submissions
# The zip MUST contain a single member named `prediction.txt` -- the official
# msnews/MIND evaluate.py hardcodes that name.
bash scripts/build_submission_zip.sh \
  results/ranker/submission_lg9b_ck45500_llm_only.txt \
  results/submissions/MINDlarge_test_lg9b_ck45500_llmonly.zip
ls -la results/submissions/MINDlarge_test_lg9b_ck45500_llmonly.zip

echo "=== pipeline DONE $(date) ==="
