#!/bin/bash
# Eval-only job: MIND_large dev AUC for a trained checkpoint (submit via sngpu).
# Usage: bash scripts/eval_only_large.sh <RUN_TAG> <CKPT_PATH> [USE_CHAT_TEMPLATE]
# Appends "RUN_TAG <ts> <AUC>" to results/large_results.tsv.
#SBATCH --job-name=lg_eval
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400000
#SBATCH --time=12:00:00
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
CHAT=${3:-1}
SPLIT=${TE_SPLIT:-dev}

echo "=== [${RUN_TAG}] EVAL large ${SPLIT} AUC (ckpt=${CKPT}, chat=${CHAT}) ==="
MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large \
USE_ABSTRACT=1 USE_CHAT_TEMPLATE=${CHAT} FLASH_ATTN=0 MAX_HISTORY=30 \
# USE_RECENCY must match the TRAINING config of the checkpoint and the
# TEST-side scoring (scripts/eval_test_subchunk_large.sh hardcodes 1).
# A mismatch silently shifts the llm_score distribution between dev and
# test, which breaks the ranker's llm_rank / llm_z calibration.
USE_RECENCY=${TE_USE_RECENCY:-1} \
HISTORY_SELECTION=${TE_HISTORY_SELECTION:-recent} \
HISTORY_FORMAT=flat \
OUTPUT_SCORES_FILE=${TE_OUTPUT_SCORES:-} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/eval_mind_pointwise.sh ${CKPT} ${SPLIT} 2>&1 | tee logs/eval_${RUN_TAG}.log

AUC=$(grep -aoE "AUC: +[0-9.]+" logs/eval_${RUN_TAG}.log | tail -1 | grep -oE "[0-9.]+$")
mkdir -p results
echo -e "${RUN_TAG}\t$(date +%F\ %T)\t${AUC:-FAILED}" >> results/large_results.tsv
echo "=== [${RUN_TAG}] DONE AUC=${AUC:-FAILED} ==="