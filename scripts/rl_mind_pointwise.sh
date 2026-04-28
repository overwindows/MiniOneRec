#!/bin/bash
# MIND Dataset RL Training Script - POINTWISE FORMAT
# Uses VERL framework with pointwise Yes/No prediction reward
#
# This script is for models trained with sft_mind_pointwise_ds.py
# For listwise models, use rl_mind.sh instead.
#
# USAGE EXAMPLES:
# ===============
#
# 1. Basic usage (uses defaults):
#    bash scripts/rl_mind_pointwise.sh
#
# 2. Specify SFT model path:
#    SFT_MODEL_PATH=output_dir/mind_pointwise_ds/final_checkpoint \
#    bash scripts/rl_mind_pointwise.sh
#
# 3. Custom data paths:
#    MIND_ROOT=../data/MIND_large \
#    MIND_SIZE=large \
#    bash scripts/rl_mind_pointwise.sh
#
# 4. Change reward type:
#    REWARD_TYPE=pointwise_weighted bash scripts/rl_mind_pointwise.sh
#
# ENVIRONMENT VARIABLES:
# ======================
#   SFT_MODEL_PATH: Path to pointwise SFT checkpoint (required)
#   MIND_ROOT: Root directory containing MIND data (default: ../data/MIND)
#   MIND_SIZE: Dataset size - small or large (default: small)
#   OUTPUT_DIR: Output directory for checkpoints (auto-generated if not set)
#   REWARD_TYPE: pointwise_binary, pointwise_weighted, pointwise_margin (default: pointwise_weighted)
#   TOTAL_EPOCHS: Number of epochs (default: 1)
#   LEARNING_RATE: Learning rate (default: 1e-7)
#   TRAIN_BATCH_SIZE: Batch size (default: 128)
#   KL_LOSS_COEF: KL penalty coefficient (default: 0.5)
#   MAX_HISTORY: Max history items (default: 30)
#   NEG_RATIO: Negative to positive ratio (default: 1.0)
#   USE_ABSTRACT: Use news abstracts 0/1 (default: 0)
#   REGENERATE_DATA: Force regenerate parquet files 0/1 (default: 0)

set -euo pipefail

# Activate MiniOneRec conda environment (matches sft_mind_pointwise_ds.sh)
export PATH="$HOME/.conda/envs/MiniOneRec/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate MiniOneRec 2>/dev/null || true

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

# Model configuration - POINTWISE SFT MODEL
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"output_dir/mind_pointwise_ds/final_checkpoint"}

# MIND data configuration
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
TRAIN_BEHAVIORS="${TRAIN_BEHAVIORS:-${MIND_ROOT}/train/behaviors.tsv}"
TRAIN_NEWS="${TRAIN_NEWS:-${MIND_ROOT}/train/news.tsv}"
DEV_BEHAVIORS="${DEV_BEHAVIORS:-${MIND_ROOT}/dev/behaviors.tsv}"
DEV_NEWS="${DEV_NEWS:-${MIND_ROOT}/dev/news.tsv}"

# NEG_RATIO must be set before parquet paths (which encode it in filename)
NEG_RATIO=${NEG_RATIO:-3.0}  # 3:1 neg:pos ratio prevents mode collapse to always-Yes

# RL data paths - POINTWISE FORMAT (neg_ratio encoded in filename to prevent stale data)
NEG_RATIO_TAG=$(echo "${NEG_RATIO}" | tr '.' 'p')
TRAIN_PARQUET="${TRAIN_PARQUET:-${MIND_ROOT}/train/rl_pointwise_neg${NEG_RATIO_TAG}_train.parquet}"
DEV_PARQUET="${DEV_PARQUET:-${MIND_ROOT}/dev/rl_pointwise_neg${NEG_RATIO_TAG}_dev.parquet}"

# Training parameters - POINTWISE SPECIFIC
REWARD_TYPE=${REWARD_TYPE:-"pointwise_asymmetric"}  # pointwise_asymmetric (recommended), pointwise_weighted, pointwise_margin, pointwise_binary
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-7}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}  # Use 128 for 8 GPUs or 252 for 7 GPUs (batch_size*8 must be divisible by n_gpus)
KL_LOSS_COEF=${KL_LOSS_COEF:-0.1}  # Lower KL ok with asymmetric reward (reward itself prevents collapse)
MAX_HISTORY=${MAX_HISTORY:-30}
NUM_GENERATIONS=${NUM_GENERATIONS:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8}
USE_ABSTRACT=${USE_ABSTRACT:-0}
REGENERATE_DATA=${REGENERATE_DATA:-0}

# Output directory
MODEL_BASENAME=$(basename "${SFT_MODEL_PATH}")
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/rl_mind_pointwise_${MIND_SIZE}_${MODEL_BASENAME}_${REWARD_TYPE}}"

# Wandb configuration
WANDB_PROJECT=${WANDB_PROJECT:-"MiniOneRec"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"rl_mind_pointwise_${MIND_SIZE}_${REWARD_TYPE}"}

echo "========================================="
echo "MIND Pointwise RL Training Configuration"
echo "========================================="
echo "SFT Model: ${SFT_MODEL_PATH}"
echo "MIND Root: ${MIND_ROOT} (${MIND_SIZE})"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Data Paths:"
echo "  Train behaviors: ${TRAIN_BEHAVIORS}"
echo "  Train news: ${TRAIN_NEWS}"
echo "  Dev behaviors: ${DEV_BEHAVIORS}"
echo "  Dev news: ${DEV_NEWS}"
echo "  Train parquet: ${TRAIN_PARQUET}"
echo "  Dev parquet: ${DEV_PARQUET}"
echo ""
echo "RL Settings (Pointwise):"
echo "  Reward type: ${REWARD_TYPE}"
echo "  Epochs: ${TOTAL_EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Batch size: ${TRAIN_BATCH_SIZE}"
echo "  KL coefficient: ${KL_LOSS_COEF}"
echo "  Max history: ${MAX_HISTORY}"
echo "  Neg ratio: ${NEG_RATIO}"
echo "  Use abstracts: ${USE_ABSTRACT}"
echo "  Regenerate data: ${REGENERATE_DATA}"
echo ""
echo "GPUs: ${PROCESS_NUM}"
echo "========================================="
echo ""

# ========================================
# Step 1: Check if SFT model exists
# ========================================

if [[ ! -d "${SFT_MODEL_PATH}" ]]; then
    echo "ERROR: Pointwise SFT model not found: ${SFT_MODEL_PATH}" >&2
    echo "" >&2
    echo "Please train a pointwise SFT model first:" >&2
    echo "  bash scripts/sft_mind_pointwise_ds.sh" >&2
    echo "" >&2
    echo "Or specify a different model path:" >&2
    echo "  SFT_MODEL_PATH=/path/to/model bash scripts/rl_mind_pointwise.sh" >&2
    exit 1
fi

echo "SFT model found: ${SFT_MODEL_PATH}"
echo ""

# ========================================
# Step 2: Check MIND data
# ========================================

if [[ ! -f "${TRAIN_BEHAVIORS}" ]] || [[ ! -f "${TRAIN_NEWS}" ]]; then
    echo "ERROR: MIND training data not found" >&2
    echo "  Behaviors: ${TRAIN_BEHAVIORS}" >&2
    echo "  News: ${TRAIN_NEWS}" >&2
    exit 1
fi

echo "MIND data found"
echo "  Train behaviors: ${TRAIN_BEHAVIORS} ($(wc -l < ${TRAIN_BEHAVIORS}) impressions)"
echo "  Train news: ${TRAIN_NEWS} ($(wc -l < ${TRAIN_NEWS}) articles)"
echo ""

# ========================================
# Step 3: Prepare RL data (if not exists)
# ========================================

ABSTRACT_FLAG=""
if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
    ABSTRACT_FLAG="--use_abstract"
fi

# Delete existing parquet files if REGENERATE_DATA is set
if [[ "${REGENERATE_DATA}" -eq 1 ]]; then
    echo "REGENERATE_DATA=1: Removing existing parquet files..."
    rm -f "${TRAIN_PARQUET}" "${DEV_PARQUET}"
    echo ""
fi

# Prepare training data with POINTWISE format
if [[ ! -f "${TRAIN_PARQUET}" ]]; then
    echo "Preparing POINTWISE training data for RL..."
    python src/prepare_mind_rl_pointwise.py \
        --behaviors_path "${TRAIN_BEHAVIORS}" \
        --news_path "${TRAIN_NEWS}" \
        --output_parquet "${TRAIN_PARQUET}" \
        --max_history ${MAX_HISTORY} \
        --neg_ratio ${NEG_RATIO} \
        ${ABSTRACT_FLAG}
    echo ""
else
    echo "Training parquet already exists: ${TRAIN_PARQUET}"
    echo "  (Set REGENERATE_DATA=1 to regenerate with new settings)"
    echo ""
fi

# Prepare dev data with POINTWISE format (always regenerate)
echo "Preparing POINTWISE dev data for RL..."
python src/prepare_mind_rl_pointwise.py \
    --behaviors_path "${DEV_BEHAVIORS}" \
    --news_path "${DEV_NEWS}" \
    --output_parquet "${DEV_PARQUET}" \
    --max_history ${MAX_HISTORY} \
    --neg_ratio ${NEG_RATIO} \
    ${ABSTRACT_FLAG}
echo ""

# ========================================
# Step 4: Run RL training
# ========================================

echo "========================================="
echo "Starting Pointwise RL Training..."
echo "========================================="
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

if [[ "${VERL_PRETTY_LOG:-1}" == "1" ]]; then
    python src/rl_mind_verl.py \
        --model_path "${SFT_MODEL_PATH}" \
        --train_parquet "${TRAIN_PARQUET}" \
        --eval_parquet "${DEV_PARQUET}" \
        --output_dir "${OUTPUT_DIR}" \
        --reward_type "${REWARD_TYPE}" \
        --num_generations ${NUM_GENERATIONS} \
        --max_prompt_length ${MAX_PROMPT_LENGTH} \
        --max_response_length ${MAX_RESPONSE_LENGTH} \
        --total_epochs ${TOTAL_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --train_batch_size ${TRAIN_BATCH_SIZE} \
        --ppo_mini_batch_size 32 \
        --kl_loss_coef ${KL_LOSS_COEF} \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_run_name "${WANDB_RUN_NAME}" \
        --n_gpus_per_node ${PROCESS_NUM} \
        2>&1 | python scripts/verl_log_filter.py
else
    python src/rl_mind_verl.py \
        --model_path "${SFT_MODEL_PATH}" \
        --train_parquet "${TRAIN_PARQUET}" \
        --eval_parquet "${DEV_PARQUET}" \
        --output_dir "${OUTPUT_DIR}" \
        --reward_type "${REWARD_TYPE}" \
        --num_generations ${NUM_GENERATIONS} \
        --max_prompt_length ${MAX_PROMPT_LENGTH} \
        --max_response_length ${MAX_RESPONSE_LENGTH} \
        --total_epochs ${TOTAL_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --train_batch_size ${TRAIN_BATCH_SIZE} \
        --ppo_mini_batch_size 32 \
        --kl_loss_coef ${KL_LOSS_COEF} \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_run_name "${WANDB_RUN_NAME}" \
        --n_gpus_per_node ${PROCESS_NUM}
fi

echo ""
echo "========================================="
echo "Pointwise RL Training Completed!"
echo "========================================="
echo "Model saved to: ${OUTPUT_DIR}"
echo ""
echo "To evaluate the RL model (pointwise format):"
echo "  bash scripts/eval_mind_pointwise.sh ${OUTPUT_DIR}/final_checkpoint dev"
echo ""
echo "Example evaluation commands:"
echo "  # Quick test (100 impressions)"
echo "  bash scripts/eval_mind_pointwise.sh ${OUTPUT_DIR}/final_checkpoint dev 100"
echo ""
echo "  # Full evaluation"
echo "  bash scripts/eval_mind_pointwise.sh ${OUTPUT_DIR}/final_checkpoint dev"
echo ""
