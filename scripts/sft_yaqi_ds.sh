#!/bin/bash

# =========================
# NCCL (STABILITY FIRST)
# =========================
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# Disable RDMA / IB (AML-safe)
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Force TCP and correct NIC
export NCCL_SOCKET_IFNAME=eth0

# Prevent silent hangs
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

# Increase timeout for large allreduces
export NCCL_TIMEOUT=7200

# =========================
# PyTorch Distributed
# =========================
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_CUDA_ARCH_LIST="8.0"

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
export MASTER_PORT=29502

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
MODEL_PATH=Qwen/Qwen3-1.7B

# Data root path (can be overridden by environment variable)
DATA_ROOT=${DATA_ROOT:-/home/aiscuser/MiniOneRec/data/GenRecDatasetV3}

# Create default hostfile if it doesn't exist (single node with 8 GPUs)
if [ ! -f "$HOSTFILE" ]; then
    echo "Creating default hostfile for single-node training..."
    echo "localhost slots=8" > $HOSTFILE
fi

echo "Using hostfile: $HOSTFILE"
cat $HOSTFILE

for category in "GenRecDatasetV2_1"; do
    train_file=${DATA_ROOT}/train/train.csv
    eval_file=${DATA_ROOT}/valid/valid.csv
    test_file=${DATA_ROOT}/test/test.csv
    info_file=${DATA_ROOT}/info/GenRecDatasetV2_1.txt
    echo "DATA_ROOT: ${DATA_ROOT}"
    echo ${train_file} ${eval_file} ${info_file} ${test_file}

    export PDSH_RCMD_TYPE=ssh

    # Set checkpoint path if resuming (comment out to start fresh)
    RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # Set via environment variable or edit here
    WANDB_RUN_ID="${WANDB_RUN_ID:-}"  # Set WandB run ID to continue same run

    deepspeed --hostfile=$HOSTFILE \
            --master_port=${MASTER_PORT} \
            --launcher=pdsh \
            --launcher_args="-S" \
            sft_ds.py \
            --base_model ${MODEL_PATH} \
            --batch_size 1024 \
            --micro_batch_size 2 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/MiniOneRec_v2 \
            --wandb_project MiniOneRec \
            --wandb_run_name sft_GenRecV2_Qwen3-1.7B_bs1024 \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ${DATA_ROOT}/index/GenRecDatasetV2_1.index.json \
            --item_meta_path ${DATA_ROOT}/index/GenRecDatasetV2_1.item.json \
            --freeze_LLM False \
            --deepspeed_config ds_config_zero2.json \
            ${RESUME_CHECKPOINT:+--resume_from_checkpoint $RESUME_CHECKPOINT} \
            ${WANDB_RUN_ID:+--wandb_run_id $WANDB_RUN_ID}

done
