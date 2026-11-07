#!/bin/bash
# Train the stack ranker on the TRAIN split (dev = evaluation only) and
# evaluate ONCE on dev.
#
# WHY
# ---
# The previous stack rankers (v3/v4/v5) were trained on dev labels, which
# violates the evaluation protocol: dev is touched exactly once, as the
# final held-out eval. This pipeline retrains on MIND_large/train labels
# with the promoted pop-WIT feature set, then scores dev a single time.
#
# Chain:
#   1. wait for the 34 train sub-chunk scores (lg_9b ckpt-45500)
#   2. merge -> results/ranker/lg9b_ckpt45500_train_scores.tsv (2,232,748 lines)
#   3. train on train labels -> results/ranker/ranker_v6_train_popwit.txt
#   4. apply to dev features -> per-impression AUC (the ONE dev number)
#
# The 34 sub-chunks are exact sequential line ranges of
# data/MIND_large/train/behaviors.tsv, so the merged file aligns
# row-for-row with results/ranker/large_train_features.tsv.
set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"

LOG=logs/pipeline_train_ranker.log
exec >> $LOG 2>&1
echo "=== train-ranker pipeline start $(date) ==="

SCOREDIR=results/ranker/train_scores/ck45500
MERGED=results/ranker/lg9b_ckpt45500_train_scores.tsv
TOTAL=$(wc -l < data/MIND_large/train/behaviors.tsv)
MODEL=results/ranker/ranker_v6_train_popwit.txt

# ---- 1. wait for all 34 train sub-chunks ----
wait_sub () {
  local part=$1
  local f=${SCOREDIR}/sub_tr00_${part}.tsv
  local want=$(wc -l < data/MIND_large/train/chunks/sub/tr00/chunk_${part}.tsv)
  while true; do
    if [ -f "$f" ]; then
      n=$(wc -l < "$f")
      if [ "$n" -ge "$want" ]; then
        echo "sub_tr00_${part} OK: n=$n (want $want)"
        return 0
      fi
      echo "sub_tr00_${part} partial: n=$n (want $want)"
    else
      echo "sub_tr00_${part} missing"
    fi
    sleep 300
  done
}
for part in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33; do
  wait_sub $part
done

# ---- 2. merge in train impression order ----
echo "--- merging train scores ---"
cat ${SCOREDIR}/sub_tr00_{00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33}.tsv > ${MERGED}
MERGED_N=$(wc -l < ${MERGED})
echo "merged lines: $MERGED_N  train impressions: $TOTAL"
if [ "$MERGED_N" -ne "$TOTAL" ]; then
  echo "FATAL: merged score count != train impression count"
  exit 1
fi
echo "first: $(head -1 ${MERGED} | cut -f1)  last: $(tail -1 ${MERGED} | cut -f1)"

# ---- 3. train on TRAIN labels ----
echo "--- training ranker v6 on train split ---"
python src/train_ranker_train_split.py \
  --train-feats results/ranker/large_train_features.tsv \
  --llm-train ${MERGED} \
  --out ${MODEL} \
  --joint-mask --wit --subcat --pop-wit \
  --rounds 2000 --threads 64

if [ ! -f "${MODEL}.feats.json" ]; then
  echo "FATAL: model ${MODEL}.feats.json never appeared"
  exit 1
fi
ls -la ${MODEL} ${MODEL}.feats.json

# ---- 4. evaluate ONCE on dev ----
echo "--- applying ranker v6 to dev (single honest eval) ---"
python src/apply_ranker_devpop.py \
  --features results/ranker/lg9b_test_features_devpopcat_v2.tsv \
  --model ${MODEL} \
  --output results/ranker/v6_dev_predictions.txt 2>&1 | tail -5 || true

echo "=== train-ranker pipeline DONE $(date) ==="