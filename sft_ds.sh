#!/bin/bash

# NCCL Configuration for multi-node training
export NCCL_DEBUG=INFO                    # Enable detailed NCCL logging
export NCCL_IB_DISABLE=1                  # Disable InfiniBand
export NCCL_SOCKET_IFNAME=eth0            # Use ethernet interface (adjust if needed)
export NCCL_TIMEOUT=7200                  # Increase timeout to 2 hours (in seconds)
export NCCL_ASYNC_ERROR_HANDLING=1        # Enable async error handling
export NCCL_P2P_DISABLE=1                 # Disable P2P transfers between GPUs on different nodes
export NCCL_BUFFSIZE=2097152              # Reduce buffer size to 2MB (default 4MB)

# PyTorch Distributed Configuration
export TORCH_DISTRIBUTED_DEBUG=DETAIL     # Enable detailed PyTorch distributed logging
export TORCH_NCCL_BLOCKING_WAIT=1         # Enable blocking wait for better error messages
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Match NCCL async error handling
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200  # PyTorch NCCL heartbeat timeout (2 hours)

# WandB Configuration - Disable WandB to avoid authentication issues
export WANDB_MODE=offline                 # Use offline mode

# Use a different port to avoid conflicts
export MASTER_PORT=29501

# Ensure conda is available and activate environment
export PATH="$HOME/.conda/envs/MiniOneRec/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate MiniOneRec 2>/dev/null || true

# DeepSpeed Configuration
# Check common hostfile locations in order of priority
if [ -z "$HOSTFILE" ]; then
    if [ -f "/job/hostfile" ]; then
        HOSTFILE="/job/hostfile"
        echo "Using multi-node hostfile: /job/hostfile"
    elif [ -f "./hostfile" ]; then
        HOSTFILE="./hostfile"
    else
        HOSTFILE="./hostfile"  # Will be created if doesn't exist
    fi
fi
# MODEL_PATH=/nvmedata/hf_checkpoints/Qwen2.5-1.5B-Instruct
# MODEL_PATH=/nvmedata/hf_checkpoints/Qwen3-4B-Instruct-2507
# MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
MODEL_PATH=Qwen/Qwen3-8B

# Create default hostfile if it doesn't exist (single node with 8 GPUs)
if [ ! -f "$HOSTFILE" ]; then
    echo "Creating default hostfile for single-node training..."
    echo "localhost slots=8" > $HOSTFILE
fi

echo "Using hostfile: $HOSTFILE"
cat $HOSTFILE

# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}

    # Export conda environment path to worker nodes
    export PDSH_RCMD_TYPE=ssh

    deepspeed --hostfile=$HOSTFILE \
            --master_port=29501 \
            --launcher=pdsh \
            --launcher_args="-S" \
            sft_ds.py \
            --base_model ${MODEL_PATH} \
            --batch_size 32 \
            --micro_batch_size 2 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/sft_${category}_qwen3-8b \
            --wandb_project MiniOneRec \
            --wandb_run_name sft_${category}_qwen3-8b \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
            --item_meta_path ./data/Amazon/index//Industrial_and_Scientific.item.json \
            --freeze_LLM False \
            --deepspeed_config ds_config_zero2.json
done
