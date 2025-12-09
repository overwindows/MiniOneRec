#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export WANDB_API_KEY=fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b

PROCESS_NUM=4
# MODEL_PATH=/nvmedata/hf_checkpoints/Qwen2.5-1.5B-Instruct
MODEL_PATH=/nvmedata/hf_checkpoints/Qwen3-4B-Instruct-2507

EXP_NAME=sft_${category}_qwen3-4b-instruct-2507_bs1024

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
            --output_dir output_dir/${EXP_NAME} \
            --wandb_project MiniOneRec \
            --wandb_run_name ${EXP_NAME} \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
            --item_meta_path ./data/Amazon/index//Industrial_and_Scientific.item.json \
            --freeze_LLM False
done
