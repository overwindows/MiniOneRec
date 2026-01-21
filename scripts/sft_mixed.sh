#!/usr/bin/env bash
#
# Mixed SFT: Combines recommendation data with general instruction data
# to prevent catastrophic forgetting of general LLM capabilities
#
# USAGE:
#   bash sft_mixed.sh
#
# ENVIRONMENT VARIABLES:
#   GENERAL_DATA_RATIO: Ratio of general data (default: 0.3 for 30%)
#   GENERAL_DATA_SAMPLE: Max general data examples (default: 50000)
#   GENERAL_DATA_PATH: Path to JSONL file (default: data/general/ultrachat_200k.jsonl)

set -euo pipefail

# ==========================================
# Multi-GPU Configuration (same as sft.sh)
# ==========================================
export NCCL_IB_DISABLE=1        # Disable IB/RoCE
export WANDB_API_KEY="${WANDB_API_KEY:-fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b}"

PROCESS_NUM="${PROCESS_NUM:-4}"  # Number of GPUs to use

# ==========================================
# Configuration
# ==========================================

# Model and data paths (use same pattern as sft.sh)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

# General data configuration
GENERAL_DATA_PATH="${GENERAL_DATA_PATH:-../data/general/ultrachat_200k.jsonl}"
GENERAL_DATA_RATIO="${GENERAL_DATA_RATIO:-0.3}"  # 30% general data by default
GENERAL_DATA_SAMPLE="${GENERAL_DATA_SAMPLE:-50000}"  # Limit general data examples

# Training configuration (match sft.sh defaults)
BATCH_SIZE="${BATCH_SIZE:-1024}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"  # Match sft.sh default
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
CUTOFF_LEN="${CUTOFF_LEN:-512}"
SEED="${SEED:-42}"
TRAIN_FROM_SCRATCH="${TRAIN_FROM_SCRATCH:-False}"
FREEZE_LLM="${FREEZE_LLM:-False}"

# Output configuration
MODEL_BASENAME=$(basename "$MODEL_PATH" | tr '/' '_')
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/sft_mixed_${CATEGORY}_${MODEL_BASENAME}_bs${BATCH_SIZE}}"

# WandB configuration
WANDB_PROJECT="${WANDB_PROJECT:-MiniOneRec}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-sft_mixed_${CATEGORY}_${MODEL_BASENAME}_bs${BATCH_SIZE}}"

# ==========================================
# Loop through categories (same as sft.sh)
# ==========================================

# Office_Products, Industrial_and_Scientific
for category in "${CATEGORY}"; do
    # Use same file discovery pattern as sft.sh
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv 2>/dev/null | head -1)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv 2>/dev/null | head -1)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)

    # Use same index file pattern as sft.sh
    sid_index_path="./data/Amazon/index/${category}.index.json"
    item_meta_path="./data/Amazon/index/${category}.item.json"

    echo "========================================="
    echo "Mixed SFT Configuration"
    echo "========================================="
    echo "Model: ${MODEL_PATH}"
    echo "Category: ${category}"
    echo "GPUs: ${PROCESS_NUM}"
    echo ""
    echo "Recommendation Data Files:"
    echo "  Train: ${train_file}"
    echo "  Eval: ${eval_file}"
    echo "  Test: ${test_file}"
    echo "  Info: ${info_file}"
    echo "  SID Index: ${sid_index_path}"
    echo "  Item Meta: ${item_meta_path}"
    echo ""
    echo "General Data:"
    echo "  Path: ${GENERAL_DATA_PATH}"
    echo "  Ratio: ${GENERAL_DATA_RATIO} (recommendation:general = $((100 - ${GENERAL_DATA_RATIO%.*})):${GENERAL_DATA_RATIO%.*})"
    echo "  Max samples: ${GENERAL_DATA_SAMPLE}"
    echo ""
    echo "Training Config:"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "  Micro batch size: ${MICRO_BATCH_SIZE}"
    echo "  Epochs: ${NUM_EPOCHS}"
    echo "  Learning rate: ${LEARNING_RATE}"
    echo "  Max length: ${CUTOFF_LEN}"
    echo "  Seed: ${SEED}"
    echo ""
    echo "Output: ${OUTPUT_DIR}"
    echo "========================================="
    echo ""

    # Check if required recommendation data files exist
    if [[ ! -f "${train_file}" ]]; then
        echo "Error: Training file not found: ${train_file}"
        exit 1
    fi

    if [[ ! -f "${eval_file}" ]]; then
        echo "Error: Validation file not found: ${eval_file}"
        exit 1
    fi

    if [[ ! -f "${sid_index_path}" ]]; then
        echo "Error: SID index file not found: ${sid_index_path}"
        exit 1
    fi

    if [[ ! -f "${item_meta_path}" ]]; then
        echo "Error: Item metadata file not found: ${item_meta_path}"
        exit 1
    fi

    # Check general data
    GENERAL_DATA_ARG=""
    if [[ -f "${GENERAL_DATA_PATH}" ]]; then
        GENERAL_DATA_ARG="--general_data_path ${GENERAL_DATA_PATH} --general_data_ratio ${GENERAL_DATA_RATIO} --general_data_sample ${GENERAL_DATA_SAMPLE}"
        echo "✓ General data file found: ${GENERAL_DATA_PATH}"
    else
        echo "Warning: General data file not found: ${GENERAL_DATA_PATH}"
        echo ""
        echo "To download UltraChat-200k general data, run:"
        echo "  python download_ultrachat.py --output ${GENERAL_DATA_PATH} --limit 50000"
        echo ""
        echo "Continuing with recommendation data only (no mixed training)..."
    fi

    # ==========================================
    # Launch Training (same as sft.sh - using torchrun)
    # ==========================================

    echo ""
    echo "Starting mixed SFT training with ${PROCESS_NUM} GPUs..."
    echo ""

    torchrun --nproc_per_node ${PROCESS_NUM} \
        sft_mixed.py \
        --base_model ${MODEL_PATH} \
        --batch_size ${BATCH_SIZE} \
        --micro_batch_size ${MICRO_BATCH_SIZE} \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --output_dir ${OUTPUT_DIR} \
        --wandb_project ${WANDB_PROJECT} \
        --wandb_run_name ${WANDB_RUN_NAME} \
        --category ${category} \
        --train_from_scratch ${TRAIN_FROM_SCRATCH} \
        --seed ${SEED} \
        --sid_index_path ${sid_index_path} \
        --item_meta_path ${item_meta_path} \
        --freeze_LLM ${FREEZE_LLM} \
        ${GENERAL_DATA_ARG}

    echo ""
    echo "========================================="
    echo "✓ Mixed SFT training completed!"
    echo "========================================="
    echo "Model saved to: ${OUTPUT_DIR}/final_checkpoint"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate general LLM capabilities:"
    echo "     bash eval_llm.sh ${OUTPUT_DIR}/final_checkpoint mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval llm_eval"
    echo ""
    echo "  2. Evaluate recommendation performance:"
    echo "     bash evaluate.sh ${OUTPUT_DIR}/final_checkpoint"
    echo ""
    echo "  3. Run RL training:"
    echo "     bash rl.sh  # Update paths in rl.sh to use ${OUTPUT_DIR}/final_checkpoint"
    echo "========================================="
done
