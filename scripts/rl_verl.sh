#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-output_dir/sft_Industrial_and_Scientific_qwen3-4b-instruct-2507_bs1024/final_checkpoint}"
TRAIN_PARQUET="${TRAIN_PARQUET:-data/verl/Industrial_and_Scientific/train.parquet}"
EVAL_PARQUET="${EVAL_PARQUET:-data/verl/Industrial_and_Scientific/eval.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/verl_rl_Industrial_and_Scientific}"
SID_INFO_FILE="${SID_INFO_FILE:-data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt}"

python src/rl_verl.py \
  --model_path "${MODEL_PATH}" \
  --train_parquet "${TRAIN_PARQUET}" \
  --eval_parquet "${EVAL_PARQUET}" \
  --output_dir "${OUTPUT_DIR}" \
  --reward_type rule \
  --num_generations 16 \
  --train_batch_size 1024 \
  --max_prompt_length 512 \
  --max_response_length 128 \
  --learning_rate 1e-6 \
  --total_epochs 1 \
  --temperature 1.0 \
  --rollout_name hf \
  --sid_info_file "${SID_INFO_FILE}"
