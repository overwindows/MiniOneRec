#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-output_dir/sft_text_Industrial_and_Scientific_qwen3-1.7B_bs1024/final_checkpoint}"
TRAIN_PARQUET="${TRAIN_PARQUET:-data/verl/Industrial_and_Scientific/train.parquet}"
EVAL_PARQUET="${EVAL_PARQUET:-data/verl/Industrial_and_Scientific/eval.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/verl_rl_text_Industrial_and_Scientific}"
SID_INFO_FILE="${SID_INFO_FILE:-data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt}"

REWARD_TYPE="${REWARD_TYPE:-sasrec}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
ROLLOUT_NAME="${ROLLOUT_NAME:-hf}"

PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-32}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-5}"

NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

WAND_PROJECT="${WAND_PROJECT:-}"
WAND_RUN_NAME="${WAND_RUN_NAME:-}"

# Optional resources for semantic/sasrec rewards:
# ADA_PATH, SASREC_PATH, SASREC_LEN_SEQ

python src/rl_verl.py \
  --model_path "${MODEL_PATH}" \
  --train_parquet "${TRAIN_PARQUET}" \
  --eval_parquet "${EVAL_PARQUET}" \
  --output_dir "${OUTPUT_DIR}" \
  --reward_type "${REWARD_TYPE}" \
  --num_generations "${NUM_GENERATIONS}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_response_length "${MAX_RESPONSE_LENGTH}" \
  --learning_rate "${LEARNING_RATE}" \
  --total_epochs "${TOTAL_EPOCHS}" \
  --temperature "${TEMPERATURE}" \
  --rollout_name "${ROLLOUT_NAME}" \
  --ppo_mini_batch_size "${PPO_MINI_BATCH_SIZE}" \
  --ppo_micro_batch_size_per_gpu "${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  --kl_loss_coef "${KL_LOSS_COEF}" \
  --kl_loss_type "${KL_LOSS_TYPE}" \
  --save_freq "${SAVE_FREQ}" \
  --test_freq "${TEST_FREQ}" \
  --nnodes "${NNODES}" \
  --n_gpus_per_node "${N_GPUS_PER_NODE}" \
  --sid_info_file "${SID_INFO_FILE}" \
  --wandb_project "${WAND_PROJECT}" \
  --wandb_run_name "${WAND_RUN_NAME}"
