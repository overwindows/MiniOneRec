#!/bin/bash

# =========================
# NCCL Configuration
# =========================
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=INIT,NET

# Network interface - auto-detect if not specified
# Common interfaces: eth0, ib0, bond0, enp (AzureML often uses eth0 or ib0)
if [ -n "$NCCL_SOCKET_IFNAME" ]; then
    echo "Using specified NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
else
    # Try to auto-detect the correct interface
    if ip link show ib0 &>/dev/null; then
        export NCCL_SOCKET_IFNAME=ib0
        export NCCL_IB_DISABLE=0
        echo "Detected InfiniBand interface: ib0"
    elif ip link show eth0 &>/dev/null; then
        export NCCL_SOCKET_IFNAME=eth0
        export NCCL_IB_DISABLE=1
        echo "Using ethernet interface: eth0"
    else
        # Let NCCL auto-detect
        echo "No specific interface found, letting NCCL auto-detect"
        export NCCL_IB_DISABLE=1
    fi
fi

# P2P settings (disable if having issues across nodes)
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}

# Prevent silent hangs
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

# Increase timeout for large allreduces (in seconds)
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}

# =========================
# PyTorch Distributed
# =========================
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_CUDA_ARCH_LIST="8.0"
# Increase distributed init timeout (default is 30 min, set to 60 min)
export TORCH_DISTRIBUTED_TIMEOUT_SEC=${TORCH_DISTRIBUTED_TIMEOUT_SEC:-3600}
# Rendezvous timeout for multi-node
export NCCL_LAUNCH_MODE=PARALLEL

# =========================
# CPU / Threading (important!)
# =========================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# =========================
# CUDA (reduce contention)
# =========================
export CUDA_DEVICE_MAX_CONNECTIONS=1

# WandB Configuration
export WANDB_API_KEY="${WANDB_API_KEY:-fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b}"
export WANDB_MODE=online
echo "WandB: online mode"

# Use a different port to avoid conflicts
export MASTER_PORT=29503

# Ensure conda is available and activate environment
export PATH="$HOME/.conda/envs/MiniOneRec/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate MiniOneRec 2>/dev/null || true

# DeepSpeed Configuration
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
# Model and data paths
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B-Instruct}
DATA_ROOT=${DATA_ROOT:-/home/aiscuser/MiniOneRec/data/MIND}

# Training hyperparameters (configurable via environment variables)
# SOTA defaults (8B model, 30 history, 2.0 negs)
BATCH_SIZE=${BATCH_SIZE:-256}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}  # 2 for 8B model
LEARNING_RATE=${LEARNING_RATE:-2e-5}     # 2e-5 for 8B model
CUTOFF_LEN=${CUTOFF_LEN:-2048}
NUM_EPOCHS=${NUM_EPOCHS:-5}
MAX_HISTORY=${MAX_HISTORY:-30}
NEG_RATIO=${NEG_RATIO:-2.0}
USE_ABSTRACT=${USE_ABSTRACT:-False}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-mind_pointwise_$(basename ${MODEL_PATH})_bs${BATCH_SIZE}}
DS_CONFIG=${DS_CONFIG:-ds_configs/ds_config_zero2.json}
# Convert to absolute path for multi-node compatibility
if [[ ! "$DS_CONFIG" = /* ]]; then
    DS_CONFIG="$(pwd)/$DS_CONFIG"
fi
DS_LAUNCHER=${DS_LAUNCHER:-pdsh}  # pdsh or openmpi

# Create default hostfile if it doesn't exist (single node with 8 GPUs)
if [ ! -f "$HOSTFILE" ]; then
    echo "Creating default hostfile for single-node training..."
    echo "localhost slots=8" > $HOSTFILE
fi

echo "Using hostfile: $HOSTFILE"
cat $HOSTFILE

# MIND dataset paths
TRAIN_BEHAVIORS=${DATA_ROOT}/train/behaviors.tsv
TRAIN_NEWS=${DATA_ROOT}/train/news.tsv
EVAL_BEHAVIORS=${DATA_ROOT}/dev/behaviors.tsv
EVAL_NEWS=${DATA_ROOT}/dev/news.tsv

# Output directory (configurable)
OUTPUT_DIR=${OUTPUT_DIR:-output_dir/mind_pointwise_ds}

echo "DATA_ROOT: ${DATA_ROOT}"
echo "Train behaviors: ${TRAIN_BEHAVIORS}"
echo "Train news: ${TRAIN_NEWS}"
echo "Eval behaviors: ${EVAL_BEHAVIORS}"
echo "Eval news: ${EVAL_NEWS}"

export PDSH_RCMD_TYPE=ssh

# Set checkpoint path if resuming (comment out to start fresh)
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # Set via environment variable or edit here
WANDB_RUN_ID="${WANDB_RUN_ID:-}"  # Set WandB run ID to continue same run

deepspeed --hostfile=$HOSTFILE \
        --master_port=${MASTER_PORT} \
        --launcher=${DS_LAUNCHER} \
        ${DS_LAUNCHER_ARGS:+--launcher_args="$DS_LAUNCHER_ARGS"} \
        src/sft_mind_pointwise_ds.py \
        --base_model ${MODEL_PATH} \
        --train_behaviors_path ${TRAIN_BEHAVIORS} \
        --train_news_path ${TRAIN_NEWS} \
        --eval_behaviors_path ${EVAL_BEHAVIORS} \
        --eval_news_path ${EVAL_NEWS} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${BATCH_SIZE} \
        --micro_batch_size ${MICRO_BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --cutoff_len ${CUTOFF_LEN} \
        --max_history ${MAX_HISTORY} \
        --neg_ratio ${NEG_RATIO} \
        --use_abstract ${USE_ABSTRACT} \
        --wandb_project MiniOneRec_MIND \
        --wandb_run_name ${WANDB_RUN_NAME} \
        --train_from_scratch False \
        --seed 42 \
        --deepspeed_config ${DS_CONFIG} \
        ${RESUME_CHECKPOINT:+--resume_from_checkpoint $RESUME_CHECKPOINT} \
        ${WANDB_RUN_ID:+--wandb_run_id $WANDB_RUN_ID}

echo "Training completed!"
