#!/bin/bash
# Two-Stage Training: Point-wise Pre-training -> Ranking Fine-tuning
#
# Stage 1: Point-wise SFT - Learn absolute relevance (Yes/No)
# Stage 2: Ranking SFT - Learn relative ordering (select best)
#
# Usage:
#   bash scripts/sft_mind_twostage.sh
#   MODEL_PATH=Qwen/Qwen3-4B bash scripts/sft_mind_twostage.sh

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
MIND_ROOT=${MIND_ROOT:-"../data/MIND"}
MIND_SIZE=${MIND_SIZE:-"small"}

# Data paths
TRAIN_BEHAVIORS="${MIND_ROOT}/train/behaviors.tsv"
TRAIN_NEWS="${MIND_ROOT}/train/news.tsv"
DEV_BEHAVIORS="${MIND_ROOT}/dev/behaviors.tsv"
DEV_NEWS="${MIND_ROOT}/dev/news.tsv"

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-1024}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-3e-4}

# Stage 1 settings
STAGE1_EPOCHS=${STAGE1_EPOCHS:-1}
STAGE1_NEG_RATIO=${STAGE1_NEG_RATIO:-1.0}
STAGE1_CUTOFF=${STAGE1_CUTOFF:-2048}

# Stage 2 settings
STAGE2_EPOCHS=${STAGE2_EPOCHS:-2}
STAGE2_NEG_RATIO=${STAGE2_NEG_RATIO:-4.0}
STAGE2_CUTOFF=${STAGE2_CUTOFF:-4096}

# Other settings
MAX_HISTORY=${MAX_HISTORY:-0}
USE_ABSTRACT=${USE_ABSTRACT:-0}
SAMPLE=${SAMPLE:--1}

# Output
MODEL_BASENAME=$(basename ${MODEL_PATH})
OUTPUT_NAME="sft_mind_twostage_${MIND_SIZE}_${MODEL_BASENAME}_bs${BATCH_SIZE}"
OUTPUT_DIR="output_dir/${OUTPUT_NAME}"

# Wandb
WANDB_PROJECT=${WANDB_PROJECT:-"MiniOneRec"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${OUTPUT_NAME}"}

echo "========================================="
echo "Two-Stage MIND Training"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "MIND: ${MIND_ROOT} (${MIND_SIZE})"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Stage 1 (Point-wise):"
echo "  Epochs: ${STAGE1_EPOCHS}"
echo "  Neg ratio: ${STAGE1_NEG_RATIO}"
echo "  Cutoff: ${STAGE1_CUTOFF}"
echo ""
echo "Stage 2 (Ranking):"
echo "  Epochs: ${STAGE2_EPOCHS}"
echo "  Neg ratio: ${STAGE2_NEG_RATIO}"
echo "  Cutoff: ${STAGE2_CUTOFF}"
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
    src/sft_mind_twostage.py \
    --base_model ${MODEL_PATH} \
    --train_behaviors_path ${TRAIN_BEHAVIORS} \
    --train_news_path ${TRAIN_NEWS} \
    --eval_behaviors_path ${DEV_BEHAVIORS} \
    --eval_news_path ${DEV_NEWS} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --stage1_epochs ${STAGE1_EPOCHS} \
    --stage1_neg_ratio ${STAGE1_NEG_RATIO} \
    --stage1_cutoff_len ${STAGE1_CUTOFF} \
    --stage2_epochs ${STAGE2_EPOCHS} \
    --stage2_neg_ratio ${STAGE2_NEG_RATIO} \
    --stage2_cutoff_len ${STAGE2_CUTOFF} \
    --max_history ${MAX_HISTORY} \
    --sample ${SAMPLE} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME} \
    --seed 42 \
    ${ABSTRACT_FLAG}

echo ""
echo "========================================="
echo "Two-Stage Training Completed!"
echo "========================================="
echo "Stage 1 checkpoint: ${OUTPUT_DIR}/stage1_pointwise/checkpoint"
echo "Final model: ${OUTPUT_DIR}/final_checkpoint"
echo ""
echo "To evaluate:"
echo "  bash scripts/eval_mind_ranking.sh ${OUTPUT_DIR}/final_checkpoint dev"
