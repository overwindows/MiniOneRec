#!/bin/bash
# MIND Dataset SFT Training Script
# Optimized for achieving SOTA on MIND news recommendation

export NCCL_IB_DISABLE=1        # Disable IB/RoCE
export WANDB_API_KEY="${WANDB_API_KEY:-fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b}"

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    PROCESS_NUM=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        PROCESS_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        PROCESS_NUM=1
    fi
fi

# Configuration
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
# For SOTA, consider using larger models:
# MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
# MODEL_PATH=Qwen/Qwen3-8B-Instruct-2507

MIND_SIZE=${MIND_SIZE:-small}  # small or large
if [[ -z "${MIND_ROOT:-}" ]]; then
    if [[ "${MIND_SIZE}" == "large" && -d "../data/MIND_large" ]]; then
        MIND_ROOT="../data/MIND_large"
    elif [[ "${MIND_SIZE}" == "small" && -d "../data/MIND_small" ]]; then
        MIND_ROOT="../data/MIND_small"
    else
        MIND_ROOT="../data/MIND"
    fi
fi

# Paths
TRAIN_BEHAVIORS="${MIND_ROOT}/train/behaviors.tsv"
TRAIN_NEWS="${MIND_ROOT}/train/news.tsv"
EVAL_BEHAVIORS="${MIND_ROOT}/dev/behaviors.tsv"
EVAL_NEWS="${MIND_ROOT}/dev/news.tsv"

# Check if files exist
if [[ ! -f "${TRAIN_BEHAVIORS}" ]] || [[ ! -f "${TRAIN_NEWS}" ]]; then
    echo "Error: MIND training data not found at ${MIND_ROOT}" >&2
    echo "Please set MIND_ROOT environment variable or prepare data first:" >&2
    echo "  python prepare_mind.py --root ${MIND_ROOT} --size ${MIND_SIZE} --splits train,dev" >&2
    exit 1
fi

echo "========================================="
echo "MIND SFT Training"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "MIND data: ${MIND_ROOT} (${MIND_SIZE})"
echo "GPUs: ${PROCESS_NUM}"
echo "========================================="
echo ""

# Training parameters
BATCH_SIZE=${BATCH_SIZE:-1024}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
CUTOFF_LEN=${CUTOFF_LEN:-1024}
USE_ABSTRACT=${USE_ABSTRACT:-0}
MAX_HISTORY=${MAX_HISTORY:-50}

# Output directory
MODEL_BASENAME=$(basename ${MODEL_PATH})
DEFAULT_OUTPUT_DIR="output_dir/sft_mind_${MIND_SIZE}_${MODEL_BASENAME}_bs${BATCH_SIZE}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_abstract"
    ABSTRACT_FLAG="--use_abstract"
else
    ABSTRACT_FLAG=""
fi

echo "Output directory: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE} (micro: ${MICRO_BATCH_SIZE})"
echo "Learning rate: ${LEARNING_RATE}"
echo "Sequence length: ${CUTOFF_LEN}"
echo "Use abstracts: ${USE_ABSTRACT}"
echo "Max history: ${MAX_HISTORY}"
echo ""

# Run training
torchrun --nproc_per_node ${PROCESS_NUM} \
        src/sft_mind.py \
        --base_model ${MODEL_PATH} \
        --batch_size ${BATCH_SIZE} \
        --micro_batch_size ${MICRO_BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --cutoff_len ${CUTOFF_LEN} \
        --max_history ${MAX_HISTORY} \
        --train_behaviors_path ${TRAIN_BEHAVIORS} \
        --train_news_path ${TRAIN_NEWS} \
        --eval_behaviors_path ${EVAL_BEHAVIORS} \
        --eval_news_path ${EVAL_NEWS} \
        --output_dir ${OUTPUT_DIR} \
        --wandb_project MiniOneRec \
        --wandb_run_name sft_mind_${MIND_SIZE}_${MODEL_BASENAME}_bs${BATCH_SIZE} \
        --train_from_scratch False \
        --seed 42 \
        ${ABSTRACT_FLAG}

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
echo "Model saved to: ${OUTPUT_DIR}/final_checkpoint"
echo ""
echo "To evaluate your model:"
echo "  bash scripts/eval_mind.sh ${OUTPUT_DIR}/final_checkpoint dev"
echo ""
echo "For leaderboard submission:"
echo "  bash scripts/eval_mind.sh ${OUTPUT_DIR}/final_checkpoint test"
