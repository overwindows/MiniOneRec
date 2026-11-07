#!/bin/bash
# Merge the ckpt-45500 DEV sub-chunk scores and retrain the stack ranker on them.
#
# WHY
# ---
# The stack ranker must be calibrated to the SAME LLM that produces the test
# scores. The current stack (ranker_v4) was trained on the ckpt-32500 dev
# scores; the test re-score uses ckpt-45500 (the peak checkpoint, dev AUC
# 0.7185 vs 0.7136). Retraining on the ckpt-45500 dev scores keeps the
# llm_score / llm_rank / llm_z features on the same distribution.
#
# The 6 dev sub-chunks are exact sequential line ranges of
# data/MIND_large/dev/behaviors.tsv (verified: concatenating chunk_00..chunk_05
# reproduces impression order 1..376471), so the merged file is identical in
# ordering to a single full-dev run.
#
# Chain:
#   1. wait for the 6 dev sub-chunk scores
#   2. merge -> results/ranker/lg9b_ckpt45500_dev_scores.tsv (376,471 lines)
#   3. retrain the stack ranker on the new dev scores
#      -> results/ranker/ranker_v5_ck45500_joint_wit.txt
#
# The test-side pipeline (scripts/pipeline_ck45500_stack.sh) waits for this
# model to appear before applying it.
set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"

LOG=logs/pipeline_ck45500_dev.log
exec >> $LOG 2>&1
echo "=== dev pipeline start $(date) ==="

SCOREDIR=results/ranker/dev_scores/ck45500
MERGED=results/ranker/lg9b_ckpt45500_dev_scores.tsv
TOTAL=$(wc -l < data/MIND_large/dev/behaviors.tsv)
MODEL=results/ranker/ranker_v5_ck45500_joint_wit.txt

# ---- 1. wait for all 6 dev sub-chunks ----
wait_sub () {
  local part=$1
  local f=${SCOREDIR}/sub_d00_${part}.tsv
  local want=$(wc -l < data/MIND_large/dev/chunks/sub/d00/chunk_${part}.tsv)
  while true; do
    if [ -f "$f" ]; then
      n=$(wc -l < "$f")
      if [ "$n" -ge "$want" ]; then
        echo "sub_d00_${part} OK: n=$n (want $want)"
        return 0
      fi
      echo "sub_d00_${part} partial: n=$n (want $want)"
    else
      echo "sub_d00_${part} missing"
    fi
    sleep 300
  done
}
for part in 00 01 02 03 04 05; do
  wait_sub $part
done

# ---- 2. merge in dev impression order ----
echo "--- merging dev scores ---"
cat ${SCOREDIR}/sub_d00_0{0,1,2,3,4,5}.tsv > ${MERGED}
MERGED_N=$(wc -l < ${MERGED})
echo "merged lines: $MERGED_N  dev impressions: $TOTAL"
if [ "$MERGED_N" -ne "$TOTAL" ]; then
  echo "FATAL: merged score count != dev impression count"
  exit 1
fi
echo "first: $(head -1 ${MERGED} | cut -f1)  last: $(tail -1 ${MERGED} | cut -f1)"

# ---- 3. retrain the stack ranker on the ckpt-45500 dev scores ----
echo "--- retraining stack ranker on ckpt-45500 dev scores ---"
python src/train_ranker_causal_final.py \
  --llm-dev ${MERGED} \
  --dev-feats results/ranker/large_dev_features_v2.tsv \
  --out ${MODEL} \
  --joint-mask --wit --subcat --pop-wit \
  --rounds 2000 --threads 32

if [ ! -f "${MODEL}.feats.json" ]; then
  echo "FATAL: model ${MODEL}.feats.json never appeared"
  exit 1
fi
ls -la ${MODEL} ${MODEL}.feats.json

echo "=== dev pipeline DONE $(date) ==="
