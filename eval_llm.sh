#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${1:-}"
TASKS="${2:-mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval}"
OUTPUT_DIR="${3:-llm_eval}"
LIMIT="${4:-}"  # Optional: limit evaluation (e.g., 0.1 for 10% or 100 for first 100 examples)

if [[ -z "${MODEL_ROOT}" ]]; then
  echo "Usage: $0 <model_or_output_dir> [tasks] [output_dir] [limit]" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $0 Qwen/Qwen3-4B-Instruct-2507                    # Full evaluation (data parallel)" >&2
  echo "  $0 Qwen/Qwen3-4B-Instruct-2507 mmlu llm_eval 0.1  # 10% of samples (data parallel)" >&2
  echo "" >&2
  echo "Environment variables:" >&2
  echo "  NUM_GPUS=4 $0 ...            # Use specific number of GPUs (default: all)" >&2
  exit 1
fi

# Detect number of GPUs
NUM_AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_GPUS="${NUM_GPUS:-${NUM_AVAILABLE_GPUS}}"

echo "========================================="
echo "GPU Configuration:"
echo "  Available GPUs: ${NUM_AVAILABLE_GPUS}"
echo "  Using: ${NUM_GPUS} GPUs (Data Parallel)"
echo "  Speedup: ~${NUM_GPUS}x faster"
echo "========================================="
echo ""

if [[ -d "${MODEL_ROOT}" && -d "${MODEL_ROOT}/checkpoint-0" ]]; then
  for ckpt in "${MODEL_ROOT}"/checkpoint-*; do
    if [[ -d "${ckpt}" ]]; then
      echo "Evaluating checkpoint: ${ckpt}"
      python llm_eval_parallel.py \
        --model_path "${ckpt}" \
        --tasks "${TASKS}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_gpus "${NUM_GPUS}" \
        ${LIMIT:+--limit ${LIMIT}} \
        --verbose
    fi
  done
else
  echo "Evaluating model: ${MODEL_ROOT}"
  python llm_eval_parallel.py \
    --model_path "${MODEL_ROOT}" \
    --tasks "${TASKS}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_gpus "${NUM_GPUS}" \
    ${LIMIT:+--limit ${LIMIT}} \
    --verbose
fi
