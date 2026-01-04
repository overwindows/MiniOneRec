#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export WANDB_API_KEY="${WANDB_API_KEY:-}"

PROCESS_NUM=4
MODEL_PATH=Qwen/Qwen3-1.7B
# MODEL_PATH=/nvmedata/hf_checkpoints/Qwen3-4B-Instruct-2507
# MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507

# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    torchrun --nproc_per_node ${PROCESS_NUM} \
            sft.py \
            --base_model ${MODEL_PATH} \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/sft_${category}_qwen2.5-1.5B-Instruct_bs1024 \
            --wandb_project MiniOneRec \
            --wandb_run_name sft_${category}_qwen2.5-1.5B-Instruct_bs1024 \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
            --item_meta_path ./data/Amazon/index//Industrial_and_Scientific.item.json \
            --freeze_LLM False
done
