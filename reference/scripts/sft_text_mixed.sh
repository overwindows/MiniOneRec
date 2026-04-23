#!/usr/bin/env bash
#
# Mixed SFT (Text): Combines Amazon text-based recommendation data with
# general instruction data to preserve LLM capabilities.
#
# USAGE:
#   bash scripts/sft_text_mixed.sh
#
# ENVIRONMENT VARIABLES:
#   GENERAL_JSONL: Path to JSONL file (default: data/general/ultrachat_200k.jsonl)
#   GENERAL_RATIO: Ratio of general data (default: 0.1)
#   GENERAL_SAMPLE: Max general data examples (default: 50000)
#   GENERAL_PROMPT_STYLE: instruction|user_input (default: user_input)
#   MIND_RATIO: Ratio of MIND data (default: 0.0)
#

set -euo pipefail

# ==========================================
# Multi-GPU Configuration
# ==========================================
export NCCL_IB_DISABLE=1        # Disable IB/RoCE
export WANDB_API_KEY="${WANDB_API_KEY:-fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b}"


export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

PROCESS_NUM="${PROCESS_NUM:-1}"

# ==========================================
# Configuration
# ==========================================
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

AMAZON_TRAIN_FILE="${AMAZON_TRAIN_FILE:-./data/Amazon/train/${CATEGORY}*11.csv}"
AMAZON_EVAL_FILE="${AMAZON_EVAL_FILE:-./data/Amazon/valid/${CATEGORY}*11.csv}"
AMAZON_ITEM_META_PATH="${AMAZON_ITEM_META_PATH:-./data/Amazon/index/${CATEGORY}.item.json}"

MIND_BEHAVIORS_PATH="${MIND_BEHAVIORS_PATH:-}"
MIND_NEWS_PATH="${MIND_NEWS_PATH:-}"
MIND_RATIO="${MIND_RATIO:-0.3}"
MIND_MAX_HISTORY="${MIND_MAX_HISTORY:-50}"
MIND_USE_ABSTRACT="${MIND_USE_ABSTRACT:-False}"

GENERAL_JSONL="${GENERAL_JSONL:-../data/general/ultrachat_200k.jsonl}"
GENERAL_RATIO="${GENERAL_RATIO:-0.3}"
GENERAL_SAMPLE="${GENERAL_SAMPLE:-50000}"
GENERAL_PROMPT_STYLE="${GENERAL_PROMPT_STYLE:-user_input}"

BATCH_SIZE="${BATCH_SIZE:-1024}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
CUTOFF_LEN="${CUTOFF_LEN:-512}"
SEED="${SEED:-42}"
TRAIN_FROM_SCRATCH="${TRAIN_FROM_SCRATCH:-False}"

MODEL_BASENAME=$(basename "$MODEL_PATH" | tr '/' '_')
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/sft_text_mixed_${CATEGORY}_${MODEL_BASENAME}_bs${BATCH_SIZE}}"

MAIN_PORT=${MAIN_PORT:-$((20000 + RANDOM % 45000))}

# ==========================================
# File Checks and Logging
# ==========================================
train_file=$(ls -f ${AMAZON_TRAIN_FILE} 2>/dev/null | head -1)
eval_file=$(ls -f ${AMAZON_EVAL_FILE} 2>/dev/null | head -1)

if [[ ! -f "${train_file}" ]]; then
    echo "Error: Training file not found: ${AMAZON_TRAIN_FILE}" >&2
    exit 1
fi

if [[ ! -f "${eval_file}" ]]; then
    echo "Error: Validation file not found: ${AMAZON_EVAL_FILE}" >&2
    exit 1
fi

if [[ ! -f "${AMAZON_ITEM_META_PATH}" ]]; then
    echo "Error: Item metadata file not found: ${AMAZON_ITEM_META_PATH}" >&2
    exit 1
fi

GENERAL_DATA_ARG=""
if [[ -f "${GENERAL_JSONL}" ]]; then
    GENERAL_DATA_ARG="--general_jsonl ${GENERAL_JSONL} --general_ratio ${GENERAL_RATIO} --general_sample ${GENERAL_SAMPLE} --general_prompt_style ${GENERAL_PROMPT_STYLE}"
else
    echo "Warning: General data file not found: ${GENERAL_JSONL}" >&2
    echo "Continuing with recommendation data only (no mixed training)..." >&2
fi

MIND_ARGS=""
if [[ -n "${MIND_BEHAVIORS_PATH}" || -n "${MIND_NEWS_PATH}" ]]; then
    if [[ -z "${MIND_BEHAVIORS_PATH}" || -z "${MIND_NEWS_PATH}" ]]; then
        echo "Error: Both MIND_BEHAVIORS_PATH and MIND_NEWS_PATH must be set together." >&2
        exit 1
    fi
    MIND_ARGS="--mind_behaviors_path ${MIND_BEHAVIORS_PATH} --mind_news_path ${MIND_NEWS_PATH} --mind_ratio ${MIND_RATIO} --mind_max_history ${MIND_MAX_HISTORY} --mind_use_abstract ${MIND_USE_ABSTRACT}"
fi

cat <<EOF_LOG
=========================================
Mixed SFT (Text) Configuration
=========================================
Model: ${MODEL_PATH}
Category: ${CATEGORY}
GPUs: ${PROCESS_NUM}
Train: ${train_file}
Eval: ${eval_file}
Item Meta: ${AMAZON_ITEM_META_PATH}
General JSONL: ${GENERAL_JSONL}
General Ratio: ${GENERAL_RATIO}
General Sample: ${GENERAL_SAMPLE}
General Prompt Style: ${GENERAL_PROMPT_STYLE}
MIND Ratio: ${MIND_RATIO}
Output: ${OUTPUT_DIR}
=========================================
EOF_LOG

# ==========================================
# Launch Training
# ==========================================
set -x

torchrun --nproc_per_node ${PROCESS_NUM} --master_port ${MAIN_PORT} \
    src/sft_text_mixed.py \
    --base_model ${MODEL_PATH} \
    --amazon_train_file ${train_file} \
    --amazon_eval_file ${eval_file} \
    --amazon_category ${CATEGORY} \
    --amazon_item_meta_path ${AMAZON_ITEM_META_PATH} \
    ${MIND_ARGS} \
    ${GENERAL_DATA_ARG} \
    --batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --seed ${SEED} \
    --train_from_scratch ${TRAIN_FROM_SCRATCH} \
    --output_dir ${OUTPUT_DIR}
