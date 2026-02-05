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
export MASTER_PORT=29504

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
MODEL_PATH=Qwen/Qwen3-1.7B
DATA_ROOT=${DATA_ROOT:-/home/aiscuser/MiniOneRec/data/MIND}

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
OUTPUT_DIR=${OUTPUT_DIR:-output_dir/mind_ranking_ds}

echo "DATA_ROOT: ${DATA_ROOT}"
echo "Train behaviors: ${TRAIN_BEHAVIORS}"
echo "Train news: ${TRAIN_NEWS}"
echo "Eval behaviors: ${EVAL_BEHAVIORS}"
echo "Eval news: ${EVAL_NEWS}"

export PDSH_RCMD_TYPE=ssh

# Set checkpoint path if resuming (comment out to start fresh)
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # Set via environment variable or edit here
WANDB_RUN_ID="${WANDB_RUN_ID:-}"  # Set WandB run ID to continue same run

# Chat template setting (set to True for instruction-tuned models like Qwen3)
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-True}

deepspeed --hostfile=$HOSTFILE \
        --master_port=${MASTER_PORT} \
        --launcher=pdsh \
        --launcher_args="-S" \
        src/sft_mind_ranking_ds.py \
        --base_model ${MODEL_PATH} \
        --train_behaviors_path ${TRAIN_BEHAVIORS} \
        --train_news_path ${TRAIN_NEWS} \
        --eval_behaviors_path ${EVAL_BEHAVIORS} \
        --eval_news_path ${EVAL_NEWS} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 256 \
        --micro_batch_size 2 \
        --num_epochs 3 \
        --learning_rate 1e-4 \
        --cutoff_len 6144 \
        --max_history 30 \
        --neg_ratio 4.0 \
        --use_abstract False \
        --use_chat_template ${USE_CHAT_TEMPLATE} \
        --wandb_project MiniOneRec_MIND \
        --wandb_run_name mind_ranking_Qwen3-1.7B_bs256 \
        --train_from_scratch False \
        --seed 42 \
        --deepspeed_config ds_configs/ds_config_zero2.json \
        ${RESUME_CHECKPOINT:+--resume_from_checkpoint $RESUME_CHECKPOINT} \
        ${WANDB_RUN_ID:+--wandb_run_id $WANDB_RUN_ID}

echo "Training completed!"
echo ""
echo "To evaluate with matching settings:"
echo "  python evaluate_mind_ranking.py \\"
echo "      --model_path ${OUTPUT_DIR}/final_checkpoint \\"
echo "      --behaviors_path ${EVAL_BEHAVIORS} \\"
echo "      --news_path ${EVAL_NEWS} \\"
echo "      --load_training_config"
