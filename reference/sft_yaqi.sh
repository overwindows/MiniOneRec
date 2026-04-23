export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
 
# export WANDB_DISABLED=true            # 彻底禁用 (COMMENTED OUT - wandb enabled)
export HF_HUB_DISABLE_TELEMETRY=1     # 可选：顺手关掉 HF tele
 
# Office_Products, Industrial_and_Scientific
for category in "GenRecDatasetV2_1"; do
    train_file=$(ls -f /home/aiscuser//MiniOneRec/data/GenRecDatasetV3/train/train.csv)
    eval_file=$(ls -f /home/aiscuser//MiniOneRec/data/GenRecDatasetV3/valid/valid.csv)
    test_file=$(ls -f /home/aiscuser//MiniOneRec/data/GenRecDatasetV3/test/test.csv)
    info_file=$(ls -f /home/aiscuser//MiniOneRec/data/GenRecDatasetV3/info/GenRecDatasetV2_1.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
   
    torchrun --nproc_per_node 8 \
            sft.py \
            --base_model Qwen/Qwen3-1.7B \
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
            --sid_index_path /home/aiscuser//MiniOneRec/data/GenRecDatasetV3/index/GenRecDatasetV2_1.index.json \
            --item_meta_path /home/aiscuser//MiniOneRec/data/GenRecDatasetV3/index/GenRecDatasetV2_1.item.json \
            --freeze_LLM False
done
