#!/bin/bash
# Score-dump eval: run dev eval with --output_scores_file for ensemble building.
# Usage: sbatch scripts/eval_scores_small.sh <RUN_TAG> <CKPT_PATH> [TE_USE_RECENCY] [TE_USE_ENTITY] [TE_USE_IMPRESSION_TIMESTAMP]
# Writes results/scores/<RUN_TAG>_dev_scores.txt (impression_id \t score per candidate).
#SBATCH --job-name=sm_score
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:b200m180:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400000
#SBATCH --time=04:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

RUN_TAG=$1
CKPT=$2
USE_REC=${3:-0}
USE_ENT=${4:-0}
USE_TS=${5:-0}
# Remaining prompt flags are env-overridable and MUST match the checkpoint's
# training config (defaults = the baseline training config):
USE_SUBCAT=${USE_SUBCATEGORY:-0}
USE_PROF=${USE_PROFILE_SUMMARY:-0}
HIST_SEL=${HISTORY_SELECTION:-recent}
HIST_FMT=${HISTORY_FORMAT:-flat}
HIST_MAX=${MAX_HISTORY:-30}

mkdir -p results/scores
SCORES=results/scores/${RUN_TAG}_dev_scores.txt

echo "=== [${RUN_TAG}] SCORE DUMP (ckpt=${CKPT}, recency=${USE_REC}, entity=${USE_ENT}, ts=${USE_TS}, subcat=${USE_SUBCAT}, profile=${USE_PROF}, selection=${HIST_SEL}, format=${HIST_FMT}, max_history=${HIST_MAX}) ==="
MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_small \
USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=${HIST_MAX} \
USE_RECENCY=${USE_REC} USE_ENTITY=${USE_ENT} USE_IMPRESSION_TIMESTAMP=${USE_TS} \
USE_SUBCATEGORY=${USE_SUBCAT} USE_PROFILE_SUMMARY=${USE_PROF} \
HISTORY_SELECTION=${HIST_SEL} HISTORY_FORMAT=${HIST_FMT} \
OUTPUT_SCORES_FILE=${SCORES} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/eval_mind_pointwise.sh ${CKPT} dev 2>&1 | tee logs/eval_scores_${RUN_TAG}.log

# `set -e` alone cannot catch a failed eval behind `| tee`, so verify output.
EXPECTED=$(wc -l < data/MIND_small/dev/behaviors.tsv)
GOT=$(wc -l < "${SCORES}" 2>/dev/null || echo 0)
if [[ ${GOT} -eq 0 ]]; then
  echo "=== [${RUN_TAG}] SCORES FAILED: no scores written — see logs/eval_scores_${RUN_TAG}.log ==="
  exit 1
fi
if [[ ${GOT} -lt ${EXPECTED} ]]; then
  echo "WARNING: [${RUN_TAG}] scores cover ${GOT}/${EXPECTED} impressions (partial — did a GPU shard fail?)"
fi
echo "=== [${RUN_TAG}] SCORES DONE: ${SCORES} (${GOT}/${EXPECTED} impressions) ==="