#!/bin/bash
# Train MIND with Ranking-Aware SFT (Multiple-Choice Format)
#
# This script trains using a multiple-choice format where the model
# selects from a list of candidates (1/2/3/...), aligning training with evaluation.
#
# Expected improvement: +3-6% AUC over standard SFT
#
# Usage:
#   bash scripts/sft_mind_ranking.sh                    # Train on MINDsmall
#   MIND_SIZE=large bash scripts/sft_mind_ranking.sh   # Train on MINDlarge

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
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B"}

# MIND dataset
MIND_ROOT=${MIND_ROOT:-"../data/MIND"}
MIND_SIZE=${MIND_SIZE:-"small"}  # small or large

# Data paths
TRAIN_BEHAVIORS="${MIND_ROOT}/train/behaviors.tsv"
TRAIN_NEWS="${MIND_ROOT}/train/news.tsv"
DEV_BEHAVIORS="${MIND_ROOT}/dev/behaviors.tsv"
DEV_NEWS="${MIND_ROOT}/dev/news.tsv"

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-1024}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
NUM_EPOCHS=${NUM_EPOCHS:-8}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
CUTOFF_LEN=${CUTOFF_LEN:-8192}  # Increased for 128K context models
MAX_HISTORY=${MAX_HISTORY:-0}  # 0 = no limit (use all history)
MAX_CANDIDATES=${MAX_CANDIDATES:-0}  # 0 = no limit (use all candidates)
USE_ABSTRACT=${USE_ABSTRACT:-0}
SAMPLE=${SAMPLE:--1}

# Output
MODEL_BASENAME=$(basename ${MODEL_PATH})
OUTPUT_DIR="output_dir/sft_mind_ranking_${MIND_SIZE}_${MODEL_BASENAME}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_${USE_ABSTRACT}"

# Wandb
WANDB_PROJECT=${WANDB_PROJECT:-"MiniOneRec"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"sft_mind_ranking_${MIND_SIZE}_${MODEL_BASENAME}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_${USE_ABSTRACT}"}
echo "========================================="
echo "MIND Ranking-Aware SFT Training"
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
echo "  Max candidates: ${MAX_CANDIDATES:-unlimited}"
echo "  Use abstract: ${USE_ABSTRACT}"
echo "  Format: Multiple-choice (1/2/3/...)"
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
        src/sft_mind_ranking.py \
        --base_model ${MODEL_PATH} \
        --batch_size ${BATCH_SIZE} \
        --micro_batch_size ${MICRO_BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --cutoff_len ${CUTOFF_LEN} \
        --max_history ${MAX_HISTORY} \
        --max_candidates ${MAX_CANDIDATES} \
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
echo "Expected improvement: +3-6% AUC over standard SFT"
echo ""
echo "To evaluate (use ranking-aware evaluation):"
echo "  bash scripts/eval_mind_ranking.sh ${OUTPUT_DIR}/final_checkpoint dev"
echo ""
echo "To run RL on top (for additional +3-5% AUC):"
echo "  SFT_MODEL_PATH=${OUTPUT_DIR}/final_checkpoint bash scripts/rl_mind.sh"
echo ""
