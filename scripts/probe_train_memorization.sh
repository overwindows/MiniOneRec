#!/bin/bash
# Probe: did the LLM memorize MIND_large/train?
#
# WHY
# ---
# lg_9b was trained on MIND_large/train/behaviors.tsv. If its scores on that
# split are inflated (memorized), the train split CANNOT be used to train the
# ranker: the ranker would learn "llm_score is always right" and over-trust
# it, which is exactly the failure that made the previous test submission
# score 0.6986.
#
# If the train-split AUC is close to the dev-split AUC (0.7134), the LLM
# generalizes and the train split is 6x more ranker training data.
#
# Scores 20,000 train impressions (a 1% sample) and computes per-impression
# AUC. Also scores the SAME 20,000 impressions' dev counterpart for a
# like-for-like comparison.
#
#SBATCH --job-name=lg_memprobe
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
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

CKPT=results/ranker/lg_9b_ckpt32500_snapshot
OUT=results/ranker/train_probe_scores.tsv

echo "=== train memorization probe: scoring 20k TRAIN impressions ==="
MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large \
USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=30 \
USE_RECENCY=1 \
HISTORY_SELECTION=recent \
HISTORY_FORMAT=flat \
OUTPUT_SCORES_FILE=${OUT} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/eval_mind_pointwise.sh ${CKPT} train/subset20k 2>&1 | tail -40

echo "=== AUC on the TRAIN probe subset ==="
python src/auc_from_scores.py \
  --scores ${OUT} \
  --behaviors data/MIND_large/train/subset20k/behaviors.tsv

echo "=== reference: AUC on the full DEV split (same ckpt) ==="
python src/auc_from_scores.py \
  --scores results/ranker/lg9b_ep082_dev_scores.tsv \
  --behaviors data/MIND_large/dev/behaviors.tsv

echo "=== probe done $(date) ==="
