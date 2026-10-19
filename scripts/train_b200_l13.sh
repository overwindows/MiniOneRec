#!/bin/bash
# L13 repro: Qwen3-1.7B (chat), MINDlarge, abstract, pointwise SFT, bs256 ep5 neg2.0 hist30
# Target: 0.7049 dev AUC (team-best single model)
set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/dev/shm/triton
export NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

MODEL=$1
OUT=$2
EXTRA_ARGS=${3:-}

deepspeed --num_gpus=8 --master_port=29513 src/sft_mind_pointwise_ds.py \
  --base_model ${MODEL} \
  --train_behaviors_path data/MIND_large/train/behaviors.tsv \
  --train_news_path data/MIND_large/train/news.tsv \
  --eval_behaviors_path data/MIND_large/dev/behaviors.tsv \
  --eval_news_path data/MIND_large/dev/news.tsv \
  --output_dir ${OUT} \
  --batch_size 256 --micro_batch_size 8 --num_epochs 5 --learning_rate 2e-5 --cutoff_len 2048 \
  --max_history 30 --neg_ratio 2.0 --hard_neg_ratio 0.5 --sample 0 \
  --use_abstract True --use_chat_template True --train_from_scratch False --seed 42 \
  --deepspeed_config ds_configs/ds_config_zero2_gpu.json ${EXTRA_ARGS}
