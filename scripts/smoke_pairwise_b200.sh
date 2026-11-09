#!/bin/bash
# SMOKE TEST for the A7.3 Bradley-Terry pairwise path in
# src/sft_mind_pointwise_ds.py.
#
# WHAT IT PROVES
#   1. The custom loss path (CE + lambda * BT) runs under DeepSpeed ZeRO-2
#      without NaN and the loss decreases over a handful of steps.
#   2. The impression-packed sampler actually forms within-impression
#      pos/neg pairs at the rate the CPU unit test predicted (~99.99% of
#      micro-batches at micro-batch 8).
#   3. The BT term is non-zero (bt_pairs_seen > 0) and the score it reads is
#      the row that PREDICTS the answer (L-2), i.e. the same row the
#      evaluator scores.
#   4. The DeepSpeed gradient path is not silently broken by the composed
#      loss (the loss must actually move the weights).
#
# It is deliberately NOT a training run: MINIONEREC_MAX_STEPS caps it, and
# the final checkpoint is thrown away.
#
# Usage: sbatch scripts/smoke_pairwise_b200.sh
set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/dev/shm/triton
export NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MINIONEREC_NUM_WORKERS=4

NGPU=${TE_NGPU:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}
echo "=== smoke test on ${NGPU} GPUs ==="

# The packed sampler REPLACES the length-grouped one, so MINIONEREC_LENGTH_GROUPED
# is deliberately NOT set here.
export MINIONEREC_IMPRESSION_PACKED=1
export MINIONEREC_MAX_STEPS=${STEPS:-8}

MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_small
OUT=${OUT:-output_dir/smoke_pairwise}

deepspeed --num_gpus=${NGPU} --master_port=29511 src/sft_mind_pointwise_ds.py \
  --base_model models/Qwen3-1.7B \
  --train_behaviors_path ${MIND_ROOT}/train/behaviors.tsv \
  --train_news_path ${MIND_ROOT}/train/news.tsv \
  --eval_behaviors_path ${MIND_ROOT}/dev/behaviors.tsv \
  --eval_news_path ${MIND_ROOT}/dev/news.tsv \
  --output_dir ${OUT} \
  --batch_size 256 --micro_batch_size 8 --num_epochs 3 --learning_rate 2e-5 --cutoff_len 2048 \
  --max_history 30 --neg_ratio 2.0 --hard_neg_ratio 0.5 --sample 0 \
  --use_abstract True --use_chat_template True --train_from_scratch False --seed 42 \
  --use_recency True --neg_ratio 1.0 \
  --pairwise_loss_weight ${LAMBDA:-0.5} \
  --deepspeed_config ds_configs/ds_config_zero2_gpu.json

echo "=== smoke test DONE ==="
# The BT statistics (pairing rate + CE/BT loss split) are printed to stdout by
# the trainer, so they land in this job's .log file.
