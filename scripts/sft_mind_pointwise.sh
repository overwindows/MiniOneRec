#!/bin/bash
# Train MIND with Point-wise SFT (Yes/No Classification)
#
# This script trains using a point-wise approach where each candidate
# is evaluated independently with Yes/No classification.
#
# Key advantages over list-wise:
# - More training signal (every candidate gets a label)
# - Shorter context per sample
# - No position bias
# - Can leverage more negatives efficiently
#
# Usage:
#   bash scripts/sft_mind_pointwise.sh                    # Train on MINDsmall
#   MIND_SIZE=large bash scripts/sft_mind_pointwise.sh   # Train on MINDlarge

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

# Model - UPGRADED TO 8B FOR SOTA PERFORMANCE
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B-Instruct"}
# Alternatives: Qwen/Qwen3-1.7B (for quick testing), meta-llama/Llama-3.3-8B

# MIND dataset - USE LARGE FOR SOTA
MIND_SIZE=${MIND_SIZE:-"large"}  # small or large (large recommended for SOTA)
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

# Training hyperparameters - OPTIMIZED FOR 8B MODEL
BATCH_SIZE=${BATCH_SIZE:-256}  # Increased for stability
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}  # Reduced for 8B model (was 16 for 1.7B)
NUM_EPOCHS=${NUM_EPOCHS:-5}  # Increased for MIND-large (was 3)
LEARNING_RATE=${LEARNING_RATE:-2e-5}  # CRITICAL: Lower for fine-tuning pretrained 8B (was 3e-4)
CUTOFF_LEN=${CUTOFF_LEN:-2048}  # Shorter than list-wise
MAX_HISTORY=${MAX_HISTORY:-30}  # Optimized: last 30 items have 90% predictive signal
NEG_RATIO=${NEG_RATIO:-2.0}  # Increased for harder training (was 1.0)
USE_ABSTRACT=${USE_ABSTRACT:-0}
SAMPLE=${SAMPLE:--1}

# Output
MODEL_BASENAME=$(basename ${MODEL_PATH})
# Build output dir name with key settings
OUTPUT_NAME="sft_mind_pointwise_${MIND_SIZE}_${MODEL_BASENAME}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_neg${NEG_RATIO}"
if [[ "${MAX_HISTORY}" != "0" ]]; then
    OUTPUT_NAME="${OUTPUT_NAME}_hist${MAX_HISTORY}"
fi
if [[ "${USE_ABSTRACT}" == "1" ]]; then
    OUTPUT_NAME="${OUTPUT_NAME}_abs"
fi
OUTPUT_DIR="output_dir/${OUTPUT_NAME}"

# Wandb
WANDB_PROJECT=${WANDB_PROJECT:-"MiniOneRec"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${OUTPUT_NAME}"}

echo "========================================="
echo "MIND Point-wise SFT Training"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "MIND: ${MIND_ROOT} (${MIND_SIZE})"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Training Settings:"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Micro batch: ${MICRO_BATCH_SIZE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Cutoff length: ${CUTOFF_LEN}"
echo "  Max history: ${MAX_HISTORY:-unlimited}"
echo "  Neg ratio: ${NEG_RATIO}"
echo "  Use abstract: ${USE_ABSTRACT}"
echo "  Format: Yes/No classification"
echo ""
echo "GPUs: ${PROCESS_NUM}"
echo "========================================="
echo ""

# ========================================
# Check MIND data
# ========================================

if [[ ! -f "${TRAIN_BEHAVIORS}" ]] || [[ ! -f "${TRAIN_NEWS}" ]]; then
    echo "ERROR: MIND data not found!" >&2
    echo "  Behaviors: ${TRAIN_BEHAVIORS}" >&2
    echo "  News: ${TRAIN_NEWS}" >&2
    echo "" >&2
    echo "Please prepare MIND data first:" >&2
    echo "  See README.md MIND section for setup instructions" >&2
    exit 1
fi

echo "✓ MIND data found"
echo "  Train behaviors: ${TRAIN_BEHAVIORS} ($(wc -l < ${TRAIN_BEHAVIORS}) lines)"
echo "  Train news: ${TRAIN_NEWS} ($(wc -l < ${TRAIN_NEWS}) articles)"
echo ""

# ========================================
# Run training
# ========================================

if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
    ABSTRACT_FLAG="--use_abstract"
else
    ABSTRACT_FLAG=""
fi

# Run training
torchrun --nproc_per_node ${PROCESS_NUM} \
        src/sft_mind_pointwise.py \
        --base_model ${MODEL_PATH} \
        --batch_size ${BATCH_SIZE} \
        --micro_batch_size ${MICRO_BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --cutoff_len ${CUTOFF_LEN} \
        --max_history ${MAX_HISTORY} \
        --neg_ratio ${NEG_RATIO} \
        --train_behaviors_path ${TRAIN_BEHAVIORS} \
        --train_news_path ${TRAIN_NEWS} \
        --eval_behaviors_path ${DEV_BEHAVIORS} \
        --eval_news_path ${DEV_NEWS} \
        --output_dir ${OUTPUT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${WANDB_RUN_NAME} \
        --sample ${SAMPLE} \
        --seed 42 \
        ${ABSTRACT_FLAG}

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
echo "Model: ${OUTPUT_DIR}/final_checkpoint"
echo ""
echo "To evaluate (point-wise evaluation):"
echo "  bash scripts/eval_mind_pointwise.sh ${OUTPUT_DIR}/final_checkpoint dev"
echo ""
echo "Comparison with list-wise:"
echo "  Point-wise: More training signal, no position bias"
echo "  List-wise:  Direct candidate comparison"
echo ""
