#!/bin/bash
# Multi-Task Training: Joint Point-wise + Ranking
#
# Trains a single model on both tasks simultaneously:
# - Point-wise: Yes/No classification (learns absolute relevance)
# - Ranking: Select best candidate (learns relative ordering)
#
# Usage:
#   bash scripts/sft_mind_multitask.sh
#   POINTWISE_RATIO=0.3 bash scripts/sft_mind_multitask.sh

set -euo pipefail

export NCCL_IB_DISABLE=1
export WANDB_API_KEY="${WANDB_API_KEY:-fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b}"

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# Detect number of GPUs
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    PROCESS_NUM=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        PROCESS_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        PROCESS_NUM=1
    fi
fi

# ========================================
# Configuration
# ========================================

# Model
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B-Base"}

# MIND dataset
MIND_SIZE=${MIND_SIZE:-"small"}
if [[ -z "${MIND_ROOT:-}" ]]; then
    if [[ "${MIND_SIZE}" == "large" && -d "../data/MIND_large" ]]; then
        MIND_ROOT="../data/MIND_large"
    elif [[ "${MIND_SIZE}" == "small" && -d "../data/MIND_small" ]]; then
        MIND_ROOT="../data/MIND_small"
    else
        MIND_ROOT="../data/MIND"
    fi
fi

# Data paths
TRAIN_BEHAVIORS="${MIND_ROOT}/train/behaviors.tsv"
TRAIN_NEWS="${MIND_ROOT}/train/news.tsv"
DEV_BEHAVIORS="${MIND_ROOT}/dev/behaviors.tsv"
DEV_NEWS="${MIND_ROOT}/dev/news.tsv"

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-1024}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
CUTOFF_LEN=${CUTOFF_LEN:-4096}

# Multi-task settings
POINTWISE_RATIO=${POINTWISE_RATIO:-0.5}  # 0.5 = 50% pointwise, 50% ranking
POINTWISE_NEG_RATIO=${POINTWISE_NEG_RATIO:-1.0}
RANKING_NEG_RATIO=${RANKING_NEG_RATIO:-4.0}

# Other settings
MAX_HISTORY=${MAX_HISTORY:-0}
USE_ABSTRACT=${USE_ABSTRACT:-0}
SAMPLE=${SAMPLE:--1}

# Output
MODEL_BASENAME=$(basename ${MODEL_PATH})
OUTPUT_NAME="sft_mind_multitask_${MIND_SIZE}_${MODEL_BASENAME}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_pw${POINTWISE_RATIO}"
OUTPUT_DIR="output_dir/${OUTPUT_NAME}"

# Wandb
WANDB_PROJECT=${WANDB_PROJECT:-"MiniOneRec"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${OUTPUT_NAME}"}

echo "========================================="
echo "Multi-Task MIND Training"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "MIND: ${MIND_ROOT} (${MIND_SIZE})"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Multi-Task Settings:"
echo "  Point-wise ratio: ${POINTWISE_RATIO}"
echo "  Point-wise neg_ratio: ${POINTWISE_NEG_RATIO}"
echo "  Ranking neg_ratio: ${RANKING_NEG_RATIO}"
echo ""
echo "Training Settings:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Cutoff length: ${CUTOFF_LEN}"
echo ""
echo "GPUs: ${PROCESS_NUM}"
echo "========================================="
echo ""

# Check MIND data
if [[ ! -f "${TRAIN_BEHAVIORS}" ]] || [[ ! -f "${TRAIN_NEWS}" ]]; then
    echo "ERROR: MIND data not found!" >&2
    exit 1
fi

# Build abstract flag
if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
    ABSTRACT_FLAG="--use_abstract"
else
    ABSTRACT_FLAG=""
fi

# Run training
torchrun --nproc_per_node ${PROCESS_NUM} \
    src/sft_mind_multitask.py \
    --base_model ${MODEL_PATH} \
    --train_behaviors_path ${TRAIN_BEHAVIORS} \
    --train_news_path ${TRAIN_NEWS} \
    --eval_behaviors_path ${DEV_BEHAVIORS} \
    --eval_news_path ${DEV_NEWS} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --pointwise_ratio ${POINTWISE_RATIO} \
    --pointwise_neg_ratio ${POINTWISE_NEG_RATIO} \
    --ranking_neg_ratio ${RANKING_NEG_RATIO} \
    --max_history ${MAX_HISTORY} \
    --sample ${SAMPLE} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME} \
    --seed 42 \
    ${ABSTRACT_FLAG}

echo ""
echo "========================================="
echo "Multi-Task Training Completed!"
echo "========================================="
echo "Model saved to: ${OUTPUT_DIR}/final_checkpoint"
echo ""
echo "To evaluate (ranking):"
echo "  bash scripts/eval_mind_ranking.sh ${OUTPUT_DIR}/final_checkpoint dev"
echo ""
echo "To evaluate (point-wise):"
echo "  bash scripts/eval_mind_pointwise.sh ${OUTPUT_DIR}/final_checkpoint dev"
