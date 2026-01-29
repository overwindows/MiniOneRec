#!/bin/bash
# MIND Dataset RL Training Script
# Uses VERL framework with AUC-based reward for news ranking optimization
#
# USAGE EXAMPLES:
# ===============
#
# 1. Basic usage (uses defaults):
#    bash scripts/rl_mind.sh
#
# 2. Specify SFT model path:
#    SFT_MODEL_PATH=output_dir/sft_mind_ranking_small_Qwen3-1.7B-Base_bs1024_ep3/final_checkpoint \
#    bash scripts/rl_mind.sh
#
# 3. Custom data paths:
#    MIND_ROOT=../data/MIND_large \
#    MIND_SIZE=large \
#    bash scripts/rl_mind.sh
#
# 4. Custom output directory:
#    OUTPUT_DIR=output_dir/my_rl_experiment \
#    bash scripts/rl_mind.sh
#
# 5. Change reward type:
#    REWARD_TYPE=mind_auc_rank bash scripts/rl_mind.sh
#
# 6. Adjust hyperparameters:
#    LEARNING_RATE=5e-8 KL_LOSS_COEF=1.0 TRAIN_BATCH_SIZE=32 \
#    bash scripts/rl_mind.sh
#
# 7. Force regenerate RL data (if changing format):
#    REGENERATE_DATA=1 bash scripts/rl_mind.sh
#
# 8. Full custom example:
#    SFT_MODEL_PATH=/path/to/model \
#    MIND_ROOT=/path/to/data \
#    OUTPUT_DIR=/path/to/output \
#    REWARD_TYPE=mind_auc \
#    bash scripts/rl_mind.sh
#
# ENVIRONMENT VARIABLES:
# ======================
#   SFT_MODEL_PATH: Path to SFT checkpoint (required)
#   MIND_ROOT: Root directory containing MIND data (default: ../data/MIND)
#   MIND_SIZE: Dataset size - small or large (default: small)
#   OUTPUT_DIR: Output directory for checkpoints (auto-generated if not set)
#   TRAIN_PARQUET: Custom path for training parquet (optional)
#   DEV_PARQUET: Custom path for dev parquet (optional)
#   REWARD_TYPE: mind_auc, mind_auc_rank, mind_ndcg, mind_mrr (default: mind_auc)
#   TOTAL_EPOCHS: Number of epochs (default: 1)
#   LEARNING_RATE: Learning rate (default: 1e-7)
#   TRAIN_BATCH_SIZE: Batch size (default: 64)
#   KL_LOSS_COEF: KL penalty coefficient (default: 0.5)
#   MAX_HISTORY: Max history items (default: 50)
#   USE_ABSTRACT: Use news abstracts 0/1 (default: 0)
#   USE_NUMERIC: Use numeric indices 0/1 (default: 1)
#   REGENERATE_DATA: Force regenerate parquet files 0/1 (default: 0)
#   WANDB_PROJECT: Wandb project name (default: MiniOneRec)
#   WANDB_RUN_NAME: Wandb run name (auto-generated if not set)

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

# Model configuration
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"output_dir/sft_mind_ranking_small_Qwen3-1.7B-Base_bs1024_ep3/final_checkpoint"}

# MIND data configuration
MIND_SIZE=${MIND_SIZE:-"small"}  # small or large
if [[ -z "${MIND_ROOT:-}" ]]; then
    if [[ "${MIND_SIZE}" == "large" && -d "../data/MIND_large" ]]; then
        MIND_ROOT="../data/MIND_large"
    elif [[ "${MIND_SIZE}" == "small" && -d "../data/MIND_small" ]]; then
        MIND_ROOT="../data/MIND_small"
    else
        MIND_ROOT="../data/MIND"
    fi
fi

# Data paths (can be overridden)
TRAIN_BEHAVIORS="${TRAIN_BEHAVIORS:-${MIND_ROOT}/train/behaviors.tsv}"
TRAIN_NEWS="${TRAIN_NEWS:-${MIND_ROOT}/train/news.tsv}"
DEV_BEHAVIORS="${DEV_BEHAVIORS:-${MIND_ROOT}/dev/behaviors.tsv}"
DEV_NEWS="${DEV_NEWS:-${MIND_ROOT}/dev/news.tsv}"

# RL data paths (can be overridden or auto-generated)
TRAIN_PARQUET="${TRAIN_PARQUET:-${MIND_ROOT}/train/rl_train.parquet}"
DEV_PARQUET="${DEV_PARQUET:-${MIND_ROOT}/dev/rl_dev.parquet}"

# Training parameters
REWARD_TYPE=${REWARD_TYPE:-"mind_auc"}  # mind_auc, mind_auc_rank, mind_ndcg, mind_mrr
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-7}  # Very conservative for stability
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}  # Reduced for memory with larger prompts
KL_LOSS_COEF=${KL_LOSS_COEF:-0.5}  # High KL penalty to stay close to SFT
MAX_HISTORY=${MAX_HISTORY:-50}
USE_ABSTRACT=${USE_ABSTRACT:-0}
# IMPORTANT: For ranking SFT models, use numeric format (1,2,3...) to match training/eval
USE_NUMERIC=${USE_NUMERIC:-1}
REGENERATE_DATA=${REGENERATE_DATA:-0}  # Force regenerate parquet files

# Output directory (can be overridden or auto-generated)
MODEL_BASENAME=$(basename "${SFT_MODEL_PATH}")
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/rl_mind_${MIND_SIZE}_${MODEL_BASENAME}_${REWARD_TYPE}}"

# Wandb configuration
WANDB_PROJECT=${WANDB_PROJECT:-"MiniOneRec"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"rl_mind_${MIND_SIZE}_${REWARD_TYPE}"}

echo "========================================="
echo "MIND RL Training Configuration"
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
echo "RL Settings:"
echo "  Reward type: ${REWARD_TYPE}"
echo "  Epochs: ${TOTAL_EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Batch size: ${TRAIN_BATCH_SIZE}"
echo "  KL coefficient: ${KL_LOSS_COEF}"
echo "  Max history: ${MAX_HISTORY}"
echo "  Use abstracts: ${USE_ABSTRACT}"
echo "  Use numeric indices: ${USE_NUMERIC}"
echo "  Regenerate data: ${REGENERATE_DATA}"
echo ""
echo "GPUs: ${PROCESS_NUM}"
echo "========================================="
echo ""

# ========================================
# Step 1: Check if SFT model exists
# ========================================

if [[ ! -d "${SFT_MODEL_PATH}" ]]; then
    echo "ERROR: SFT model not found: ${SFT_MODEL_PATH}" >&2
    echo "" >&2
    echo "Please train an SFT model first:" >&2
    echo "  bash scripts/sft_mind.sh" >&2
    echo "" >&2
    echo "Or specify a different model path:" >&2
    echo "  SFT_MODEL_PATH=/path/to/model bash scripts/rl_mind.sh" >&2
    exit 1
fi

echo "✓ SFT model found: ${SFT_MODEL_PATH}"
echo ""

# ========================================
# Step 2: Check MIND data
# ========================================

if [[ ! -f "${TRAIN_BEHAVIORS}" ]] || [[ ! -f "${TRAIN_NEWS}" ]]; then
    echo "ERROR: MIND training data not found" >&2
    echo "  Behaviors: ${TRAIN_BEHAVIORS}" >&2
    echo "  News: ${TRAIN_NEWS}" >&2
    echo "" >&2
    echo "Please prepare MIND data first:" >&2
    echo "  python prepare_mind.py --root ${MIND_ROOT} --size ${MIND_SIZE} --splits train,dev" >&2
    exit 1
fi

echo "✓ MIND data found"
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

NUMERIC_FLAG=""
if [[ "${USE_NUMERIC}" -eq 0 ]]; then
    NUMERIC_FLAG="--use_letters"
fi

# Delete existing parquet files if REGENERATE_DATA is set
if [[ "${REGENERATE_DATA}" -eq 1 ]]; then
    echo "REGENERATE_DATA=1: Removing existing parquet files..."
    rm -f "${TRAIN_PARQUET}" "${DEV_PARQUET}"
    echo ""
fi

# Prepare training data
if [[ ! -f "${TRAIN_PARQUET}" ]]; then
    echo "Preparing training data for RL..."
    python prepare_mind_rl.py \
        --behaviors_path "${TRAIN_BEHAVIORS}" \
        --news_path "${TRAIN_NEWS}" \
        --output_parquet "${TRAIN_PARQUET}" \
        --max_history ${MAX_HISTORY} \
        ${ABSTRACT_FLAG} ${NUMERIC_FLAG}
    echo ""
else
    echo "✓ Training parquet already exists: ${TRAIN_PARQUET}"
    echo "  (Set REGENERATE_DATA=1 to regenerate with new settings)"
    echo ""
fi

# Prepare dev data
if [[ ! -f "${DEV_PARQUET}" ]]; then
    echo "Preparing dev data for RL..."
    python prepare_mind_rl.py \
        --behaviors_path "${DEV_BEHAVIORS}" \
        --news_path "${DEV_NEWS}" \
        --output_parquet "${DEV_PARQUET}" \
        --max_history ${MAX_HISTORY} \
        ${ABSTRACT_FLAG} ${NUMERIC_FLAG}
    echo ""
else
    echo "✓ Dev parquet already exists: ${DEV_PARQUET}"
    echo "  (Set REGENERATE_DATA=1 to regenerate with new settings)"
    echo ""
fi

# ========================================
# Step 4: Run RL training
# ========================================

echo "========================================="
echo "Starting RL Training..."
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
        --total_epochs ${TOTAL_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --train_batch_size ${TRAIN_BATCH_SIZE} \
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
        --total_epochs ${TOTAL_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --train_batch_size ${TRAIN_BATCH_SIZE} \
        --kl_loss_coef ${KL_LOSS_COEF} \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_run_name "${WANDB_RUN_NAME}" \
        --n_gpus_per_node ${PROCESS_NUM}
fi

echo ""
echo "========================================="
echo "RL Training Completed!"
echo "========================================="
echo "Model saved to: ${OUTPUT_DIR}"
echo ""
echo "To evaluate the RL model (ranking-aware format):"
echo "  bash scripts/eval_mind_ranking.sh ${OUTPUT_DIR}/final_checkpoint dev"
echo ""
echo "Example evaluation commands:"
echo "  # Quick test (100 impressions) - ranking-aware format"
echo "  bash scripts/eval_mind_ranking.sh ${OUTPUT_DIR}/final_checkpoint dev 100"
echo ""
echo "  # Full evaluation (ranking-aware format)"
echo "  bash scripts/eval_mind_ranking.sh ${OUTPUT_DIR}/final_checkpoint dev"
echo ""
echo "  # Alternative: Standard evaluation (if using numeric format)"
echo "  # CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_mind.sh ${OUTPUT_DIR}/final_checkpoint dev"
echo ""
