#!/bin/bash

# NCCL Configuration for multi-node training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_TIMEOUT=7200
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_BUFFSIZE=2097152

# PyTorch Distributed Configuration
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

# WandB Configuration (enabled; set WANDB_DISABLED=true to turn off)
export WANDB_MODE=online

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

# Create default hostfile if it doesn't exist (single node with 8 GPUs)
if [ ! -f "$HOSTFILE" ]; then
    echo "Creating default hostfile for single-node training..."
    echo "localhost slots=8" > $HOSTFILE
fi

echo "Using hostfile: $HOSTFILE"
cat $HOSTFILE

for category in "GenRecDatasetV2_1"; do
    train_file=$(ls -f /home/aiscuser/MiniOneRec/data/GenRecDatasetV3/train/train.csv)
    eval_file=$(ls -f /home/aiscuser/MiniOneRec/data/GenRecDatasetV3/valid/valid.csv)
    test_file=$(ls -f /home/aiscuser/MiniOneRec/data/GenRecDatasetV3/test/test.csv)
    info_file=$(ls -f /home/aiscuser/MiniOneRec/data/GenRecDatasetV3/info/GenRecDatasetV2_1.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}

    export PDSH_RCMD_TYPE=ssh

    deepspeed --hostfile=$HOSTFILE \
            --master_port=${MASTER_PORT} \
            --launcher=pdsh \
            --launcher_args="-S" \
            sft_ds.py \
            --base_model ${MODEL_PATH} \
            --batch_size 1024 \
            --micro_batch_size 4 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/MiniOneRec_v2 \
            --wandb_project wandb_proj \
            --wandb_run_name wandb_name \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path /home/aiscuser/MiniOneRec/data/GenRecDatasetV3/index/GenRecDatasetV2_1.index.json \
            --item_meta_path /home/aiscuser/MiniOneRec/data/GenRecDatasetV3/index/GenRecDatasetV2_1.item.json \
            --freeze_LLM False \
            --deepspeed_config ds_config_zero2.json

done
