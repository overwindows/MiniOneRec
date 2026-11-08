#!/bin/bash
# Score ONE chunk of the MIND_large test split with a checkpoint (8 GPUs).
# Usage: TE_CHUNK=NN [TE_CKPT=path] sbatch scripts/eval_test_chunk_large.sh
# Writes raw scores to results/ranker/test_scores/chunk_NN.tsv
#SBATCH --job-name=lg_tchunk
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400000
#SBATCH --time=24:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# ---- CUDA driver compatibility -------------------------------------------
# The cu130 torch build needs driver >= 595.71.05. Some A100/H100 nodes still
# carry 565.57.01, where torch.cuda.is_available() is False and the model would
# load on CPU (days for a 9B). The CUDA forward-compat package ships a newer
# libcuda that works against the old kernel driver, so preload it when the
# installed driver is too old.
#
# The shared preflight also resolves the PHYSICAL device indices Slurm
# actually allocated (a shared node hands out e.g. 2,3,4,5, not 0..3) and
# refuses to start if one of OUR GPUs is busy.
# shellcheck source=scripts/_gpu_preflight.sh
source scripts/_gpu_preflight.sh
mkgpu_preflight "${TE_NGPU:-0}" || exit 1

CHUNK=$(printf "%02d" ${TE_CHUNK})
CKPT=${TE_CKPT:-results/ranker/lg_9b_ckpt32500_snapshot}
# TE_TAG namespaces the output file per checkpoint; the chunk number is
# ALWAYS part of the name so parallel chunk jobs never overwrite each other.
OUT=results/ranker/test_scores/${TE_TAG:-chunk}_${CHUNK}.tsv

echo "=== test chunk ${CHUNK} ckpt=${CKPT} ngpu=${NGPU} gpus=${GPUS} ==="
mkdir -p results/ranker/test_scores

# Build a mini split dir: news.tsv + chunk as behaviors.tsv
MINI=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/chunks/split_${CHUNK}
mkdir -p ${MINI}
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/news.tsv ${MINI}/news.tsv
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/chunks/chunk_${CHUNK}.tsv ${MINI}/behaviors.tsv

# HISTORY_SELECTION must match the checkpoint's TRAINING config. It used to be
# hardcoded to "recent", which silently mis-scored any checkpoint trained with
# another history policy (e.g. the dedup_backfill 1.7B model). Override with
# EVAL_HISTORY_SELECTION -- the SAME env var the dev-side scripts read, so a
# dev/test pair can never drift apart.
# EVAL_MAX_HISTORY: how many history items the prompt may carry (0 = unlimited).
#
# THIS USED TO BE HARDCODED TO 30 -- the same budget TRAINING used -- so eval
# silently cut the real user history of ~43% of test impressions down to their
# last 30 items. That cap is an INFERENCE-TIME choice, not a model property:
# raising it to unlimited at eval time, with the SAME weights, was measured
# paired on 40,000 dev impressions and gained
#   +0.00392 AUC (SE 0.00036, t=+10.77)
# with the delta bit-identically 0.00000 on the 65% of impressions that carry
# <=30 items, and growing monotonically with history length
# (31-50: +0.00551, 51-100: +0.01317, >100: +0.02091).
# Cost: 1.19x wall-clock. Keep the dev and test caps identical.
EVAL_MAX_HISTORY=${EVAL_MAX_HISTORY:-0}
MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/chunks \
USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=${EVAL_MAX_HISTORY} \
USE_RECENCY=${EVAL_USE_RECENCY:-1} \
HISTORY_SELECTION=${EVAL_HISTORY_SELECTION:-recent} \
HISTORY_FORMAT=flat \
PREFIX_CACHE_MODE=${EVAL_PREFIX_CACHE_MODE:-auto} \
OUTPUT_SCORES_FILE=${OUT} \
CUDA_VISIBLE_DEVICES=${GPUS} \
bash scripts/eval_mind_pointwise.sh ${CKPT} split_${CHUNK} 2>&1 | tee logs/test_chunk_${TE_TAG:-chunk}_${CHUNK}.log

echo "=== chunk ${CHUNK} DONE ==="
