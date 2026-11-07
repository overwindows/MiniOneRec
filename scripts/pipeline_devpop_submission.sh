#!/bin/bash
# Full test-submission pipeline for the causal-devpop + cat + subcat ranker.
#
# Waits for chunk_01/chunk_02 re-scores, then:
#   merge -> features -> devpop+cat+subcat -> ranker -> leaderboard format -> zip
#
# The ranker is the leak-free one: devpop is CAUSAL (past-only) during
# training, and category/subcategory popularity is included because 83.3% of
# test rows reference a news that appears in neither train nor dev, so
# news-level popularity is missing for the large majority of test rows.
set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"

LOG=logs/pipeline_devpop.log
MODEL=results/ranker/ranker_v3_lg9b_joint_wit.txt
exec >> $LOG 2>&1
echo "=== pipeline start $(date) ==="

# ---- 1. wait for the 12 sub-chunk scores (c01/c02 x 6 parts) ----
wait_sub () {
  local sub=$1 part=$2
  local f=results/ranker/test_scores/sub_${sub}_${part}.tsv
  local want=$(( $(wc -l < data/MIND_large/test/chunks/sub/${sub}/chunk_${part}.tsv) ))
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
for sub in c01 c02; do
  for part in 00 01 02 03 04 05; do
    wait_sub $sub $part
  done
done

# ---- 2. merge all 6 chunks (00,03,04,05 original + 01,02 from sub-chunks) ----
echo "--- merging scores ---"
cat results/ranker/test_scores/chunk_00.tsv \
    results/ranker/test_scores/sub_c01_0{0,1,2,3,4,5}.tsv \
    results/ranker/test_scores/sub_c02_0{0,1,2,3,4,5}.tsv \
    results/ranker/test_scores/chunk_0{3,4,5}.tsv \
  > results/ranker/lg9b_ep082_test_scores_all.tsv
MERGED=$(wc -l < results/ranker/lg9b_ep082_test_scores_all.tsv)
TOTAL=$(wc -l < data/MIND_large/test/behaviors.tsv)
echo "merged lines: $MERGED  test impressions: $TOTAL"
if [ "$MERGED" -ne "$TOTAL" ]; then
  echo "FATAL: merged score count != test impression count"
  exit 1
fi
echo "first: $(head -1 results/ranker/lg9b_ep082_test_scores_all.tsv | cut -f1)  last: $(tail -1 results/ranker/lg9b_ep082_test_scores_all.tsv | cut -f1)"

# ---- 3. rebuild test feature table ----
echo "--- building test features ---"
python src/build_ranker_features.py \
  --behaviors data/MIND_large/test/behaviors.tsv \
  --news data/MIND_large/test/news.tsv \
  --train_behaviors data/MIND_large/train/behaviors.tsv \
  --llm_scores results/ranker/lg9b_ep082_test_scores_all.tsv \
  --cf_scores data/MIND_large/test/cf_scores.tsv \
  --output results/ranker/lg9b_test_features.tsv

# ---- 4. append dev-derived popularity + category + subcategory popularity ----
echo "--- appending devpop + catpop + subcatpop ---"
python src/append_devpop_cat_test.py \
  --features results/ranker/lg9b_test_features.tsv \
  --dev_behaviors data/MIND_large/dev/behaviors.tsv \
  --news data/MIND_large/test/news.tsv \
  --output results/ranker/lg9b_test_features_devpopcat.tsv

# ---- 5. apply ranker (wait for the trainer to finish if still running) ----
echo "--- waiting for model $MODEL ---"
for i in $(seq 1 240); do
  if [ -f "${MODEL}.feats.json" ]; then
    break
  fi
  echo "model not ready ($i)"
  sleep 60
done
if [ ! -f "${MODEL}.feats.json" ]; then
  echo "FATAL: model ${MODEL}.feats.json never appeared"
  exit 1
fi
echo "--- applying ranker ($MODEL) ---"
python src/apply_ranker_devpop.py \
  --features results/ranker/lg9b_test_features_devpopcat.tsv \
  --model ${MODEL} \
  --output results/ranker/lg9b_causal_test_predictions.txt

# ---- 6. leaderboard format ----
echo "--- converting to leaderboard format ---"
python src/convert_to_leaderboard_format.py \
  --predictions results/ranker/lg9b_causal_test_predictions.txt \
  --behaviors data/MIND_large/test/behaviors.tsv \
  --output results/ranker/submission_lg9b_causal.txt

# ---- 7. verify + zip ----
SUB_LINES=$(wc -l < results/ranker/submission_lg9b_causal.txt)
echo "submission lines: $SUB_LINES (want $TOTAL)"
if [ "$SUB_LINES" -ne "$TOTAL" ]; then
  echo "FATAL: submission line count != test impression count"
  exit 1
fi
mkdir -p results/submissions
cd results/ranker
zip -f ../submissions/MINDlarge_test_lg9b_causal.zip submission_lg9b_causal.txt
cd /import/ml-sc-scratch5/chenw/MiniOneRec
echo "=== pipeline DONE $(date) ==="
ls -la results/submissions/MINDlarge_test_lg9b_causal.zip
