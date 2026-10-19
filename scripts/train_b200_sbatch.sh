#!/bin/bash
# Launch one 8-GPU single-node B200 training run via sbatch.
# Usage: sbatch scripts/train_b200_sbatch.sh <MODEL> <OUTDIR> <RUN_TAG> <EXTRA_ARGS...>
#SBATCH --job-name=sota_train
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400000
#SBATCH --time=2-00:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/dev/shm/triton
export NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# Perf knobs (validated by probes 2026-10-19):
#  - length-grouped batches: median prompt ~1250 vs padded 2048 -> ~40% less padding compute
#  - dataloader workers: hide tokenization (chat-template jinja per sample)
export MINIONEREC_LENGTH_GROUPED=1
export MINIONEREC_NUM_WORKERS=8
# Eval every 2000 steps (5000 val samples, bs8 -> ~2min per eval)
export MINIONEREC_EVAL_STEPS=2000
export MINIONEREC_SAVE_STEPS=2000

MODEL=$1
OUT=$2
shift 2

deepspeed --num_gpus=8 --master_port=29500 src/sft_mind_pointwise_ds.py \
  --base_model ${MODEL} \
  --train_behaviors_path data/MIND_large/train/behaviors.tsv \
  --train_news_path data/MIND_large/train/news.tsv \
  --eval_behaviors_path data/MIND_large/dev/behaviors.tsv \
  --eval_news_path data/MIND_large/dev/news.tsv \
  --output_dir ${OUT} \
  --batch_size 256 --micro_batch_size 8 --num_epochs 5 --learning_rate 2e-5 --cutoff_len 2048 \
  --max_history 30 --neg_ratio 2.0 --hard_neg_ratio 0.5 --sample 0 \
  --use_abstract True --use_chat_template True --train_from_scratch False --seed 42 \
  --deepspeed_config ds_configs/ds_config_zero2_gpu.json "$@"