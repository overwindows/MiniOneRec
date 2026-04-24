#!/bin/bash

# =========================
# DOCA Pointwise SFT Training Script
# =========================
# Usage:
#   bash scripts/sft_doca_pointwise_ds.sh
#
# Key environment variables (override defaults):
#   MODEL_PATH, DATA_ROOT, BATCH_SIZE, MICRO_BATCH_SIZE,
#   LEARNING_RATE, CUTOFF_LEN, NUM_EPOCHS, NEG_RATIO

# =========================
# NCCL Configuration
# =========================
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=INIT,NET

if [ -n "$NCCL_SOCKET_IFNAME" ]; then
    echo "Using specified NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
else
    if ip link show ib0 &>/dev/null; then
        export NCCL_SOCKET_IFNAME=ib0
        export NCCL_IB_DISABLE=0
        echo "Detected InfiniBand interface: ib0"
    elif ip link show eth0 &>/dev/null; then
        export NCCL_SOCKET_IFNAME=eth0
        export NCCL_IB_DISABLE=1
        echo "Using ethernet interface: eth0"
    else
        echo "No specific interface found, letting NCCL auto-detect"
        export NCCL_IB_DISABLE=1
    fi
fi

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}

# =========================
# PyTorch Distributed
# =========================
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_CUDA_ARCH_LIST="8.0"
export TORCH_DISTRIBUTED_TIMEOUT_SEC=${TORCH_DISTRIBUTED_TIMEOUT_SEC:-3600}
export NCCL_LAUNCH_MODE=PARALLEL

# =========================
# CPU / Threading
# =========================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# =========================
# CUDA
# =========================
export CUDA_DEVICE_MAX_CONNECTIONS=1

# WandB Configuration
export WANDB_API_KEY="${WANDB_API_KEY:-fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b}"
export WANDB_MODE=online
echo "WandB: online mode"

export MASTER_PORT=29503

# Ensure conda is available
export PATH="$HOME/.conda/envs/MiniOneRec/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate MiniOneRec 2>/dev/null || true

# DeepSpeed hostfile
if [ -z "$HOSTFILE" ]; then
    if [ -f "/job/hostfile" ]; then
        HOSTFILE="/job/hostfile"
        echo "Using multi-node hostfile: /job/hostfile"
    elif [ -f "./hostfile" ]; then
        HOSTFILE="./hostfile"
    else
        HOSTFILE="./hostfile"
    fi
fi

# Model and data paths
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
DATA_ROOT=${DATA_ROOT:-data/doca_v8}

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-256}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
CUTOFF_LEN=${CUTOFF_LEN:-4096}
NUM_EPOCHS=${NUM_EPOCHS:-1}
NEG_RATIO=${NEG_RATIO:-2.0}
MAX_INTERESTS=${MAX_INTERESTS:-0}
MAX_CONVERSATION=${MAX_CONVERSATION:-15}
MAX_SHOWN=${MAX_SHOWN:-10}

USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-0}"
DS_CONFIG=${DS_CONFIG:-ds_configs/ds_config_zero2.json}
if [[ ! "$DS_CONFIG" = /* ]]; then
    DS_CONFIG="$(pwd)/$DS_CONFIG"
fi
DS_LAUNCHER=${DS_LAUNCHER:-pdsh}

# Create default hostfile if needed
if [ ! -f "$HOSTFILE" ]; then
    echo "Creating default hostfile for single-node training..."
    echo "localhost slots=8" > $HOSTFILE
fi

echo "Using hostfile: $HOSTFILE"
cat $HOSTFILE

# Data paths
TRAIN_JSONL=${DATA_ROOT}/train.jsonl
EVAL_JSONL=${DATA_ROOT}/dev.jsonl

# Output directory
MODEL_BASENAME=$(basename ${MODEL_PATH})
OUTPUT_NAME="sft_doca_pointwise_${MODEL_BASENAME}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_neg${NEG_RATIO}"
if [[ "${USE_CHAT_TEMPLATE}" -eq 1 ]]; then
    OUTPUT_NAME="${OUTPUT_NAME}_chat"
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/${OUTPUT_NAME}"
else
    OUTPUT_DIR="output_dir/${OUTPUT_NAME}"
fi
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${OUTPUT_NAME}}

echo "DATA_ROOT: ${DATA_ROOT}"
echo "Train JSONL: ${TRAIN_JSONL}"
echo "Eval JSONL: ${EVAL_JSONL}"
echo "Output: ${OUTPUT_DIR}"
echo "Chat template: $(if [[ "${USE_CHAT_TEMPLATE}" -eq 1 ]]; then echo "enabled"; else echo "disabled"; fi)"

export PDSH_RCMD_TYPE=ssh

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

deepspeed --hostfile=$HOSTFILE \
        --master_port=${MASTER_PORT} \
        --launcher=${DS_LAUNCHER} \
        ${DS_LAUNCHER_ARGS:+--launcher_args="$DS_LAUNCHER_ARGS"} \
        src/sft_doca_pointwise_ds.py \
        --base_model ${MODEL_PATH} \
        --train_jsonl ${TRAIN_JSONL} \
        --eval_jsonl ${EVAL_JSONL} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${BATCH_SIZE} \
        --micro_batch_size ${MICRO_BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --cutoff_len ${CUTOFF_LEN} \
        --neg_ratio ${NEG_RATIO} \
        --max_interests ${MAX_INTERESTS} \
        --max_conversation_msgs ${MAX_CONVERSATION} \
        --max_shown ${MAX_SHOWN} \
        $(if [[ "${USE_CHAT_TEMPLATE}" -eq 1 ]]; then echo "--use_chat_template True"; fi) \
        --wandb_project MiniOneRec_DOCA \
        --wandb_run_name ${WANDB_RUN_NAME} \
        --train_from_scratch False \
        --seed 42 \
        --deepspeed_config ${DS_CONFIG} \
        ${RESUME_CHECKPOINT:+--resume_from_checkpoint $RESUME_CHECKPOINT} \
        ${WANDB_RUN_ID:+--wandb_run_id $WANDB_RUN_ID}

TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "Training FAILED with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi
echo "Training completed!"
