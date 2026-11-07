#!/bin/bash
# Rebuild the STACK submission from the ckpt-45500 test scores.
#
# Chain:
#   1. wait for the 36 ckpt-45500 sub-chunk scores
#   2. merge -> lg9b_ck45500_test_scores_all.tsv (2,370,727 lines)
#   3. rebuild the test feature table with the NEW llm_score
#   4. append devpop + catpop + subcatpop
#   5. apply the stack ranker (retrained on the ckpt-45500 dev scores)
#   6. leaderboard format -> zip
#
# The ranker must be retrained on the ckpt-45500 DEV scores, otherwise the
# model is calibrated to a different LLM's score distribution.
set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"

LOG=logs/pipeline_ck45500_stack.log
exec >> $LOG 2>&1
echo "=== stack pipeline start $(date) ==="

SCOREDIR=results/ranker/test_scores/ck45500
MERGED=results/ranker/lg9b_ck45500_test_scores_all.tsv
TOTAL=$(wc -l < data/MIND_large/test/behaviors.tsv)
MODEL=results/ranker/ranker_v5_ck45500_joint_wit.txt

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

# ---- 2. merge in test impression order ----
echo "--- merging scores ---"
cat ${SCOREDIR}/sub_c0{0,1,2,3,4,5}_0{0,1,2,3,4,5}.tsv > ${MERGED}
MERGED_N=$(wc -l < ${MERGED})
echo "merged lines: $MERGED_N  test impressions: $TOTAL"
if [ "$MERGED_N" -ne "$TOTAL" ]; then
  echo "FATAL: merged score count != test impression count"
  exit 1
fi

# ---- 3. rebuild the test feature table with the new llm_score ----
echo "--- building test features ---"
python src/build_ranker_features.py \
  --behaviors data/MIND_large/test/behaviors.tsv \
  --news data/MIND_large/test/news.tsv \
  --train_behaviors data/MIND_large/train/behaviors.tsv \
  --llm_scores ${MERGED} \
  --cf_scores data/MIND_large/test/cf_scores.tsv \
  --output results/ranker/ck45500_test_features.tsv

# ---- 4. append dev-derived popularity ----
echo "--- appending devpop + catpop + subcatpop ---"
python src/append_devpop_cat_test.py \
  --features results/ranker/ck45500_test_features.tsv \
  --dev_behaviors data/MIND_large/dev/behaviors.tsv \
  --news data/MIND_large/test/news.tsv \
  --output results/ranker/ck45500_test_features_devpopcat.tsv

# ---- 5. wait for the retrained ranker ----
echo "--- waiting for model $MODEL ---"
for i in $(seq 1 480); do
  if [ -f "${MODEL}.feats.json" ]; then break; fi
  echo "model not ready ($i)"
  sleep 60
done
if [ ! -f "${MODEL}.feats.json" ]; then
  echo "FATAL: model ${MODEL}.feats.json never appeared"
  exit 1
fi

# ---- 6. apply the ranker ----
echo "--- applying ranker ($MODEL) ---"
python src/apply_ranker_devpop.py \
  --features results/ranker/ck45500_test_features_devpopcat.tsv \
  --model ${MODEL} \
  --output results/ranker/ck45500_test_predictions.txt

# ---- 7. leaderboard format + zip ----
echo "--- converting to leaderboard format ---"
python src/convert_to_leaderboard_format.py \
  --predictions results/ranker/ck45500_test_predictions.txt \
  --behaviors data/MIND_large/test/behaviors.tsv \
  --output results/ranker/submission_ck45500_stack.txt

SUB_LINES=$(wc -l < results/ranker/submission_ck45500_stack.txt)
echo "submission lines: $SUB_LINES (want $TOTAL)"
if [ "$SUB_LINES" -ne "$TOTAL" ]; then
  echo "FATAL: submission line count != test impression count"
  exit 1
fi
mkdir -p results/submissions
# The zip MUST contain a single member named `prediction.txt` -- the official
# msnews/MIND evaluate.py hardcodes that name.
bash scripts/build_submission_zip.sh \
  results/ranker/submission_ck45500_stack.txt \
  results/submissions/MINDlarge_test_ck45500_stack.zip
ls -la results/submissions/MINDlarge_test_ck45500_stack.zip

echo "=== stack pipeline DONE $(date) ==="
