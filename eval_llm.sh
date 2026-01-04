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
  echo "  $0 Qwen/Qwen3-4B-Instruct-2507                    # Single GPU evaluation" >&2
  echo "  $0 Qwen/Qwen3-4B-Instruct-2507 mmlu llm_eval 0.1  # 10% of samples" >&2
  echo "" >&2
  echo "Environment variables:" >&2
  echo "  NUM_GPUS=4 $0 ...            # Use multiple GPUs (default: 1)" >&2
  exit 1
fi

# Detect number of GPUs
NUM_AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_GPUS="${NUM_GPUS:-1}"  # Default to single GPU

# Build the Python command based on number of GPUs
build_cmd() {
  local model_path="$1"
  if [[ "${NUM_GPUS}" -eq 1 ]]; then
    # Single GPU mode - use llm_eval.py
    local cmd="python llm_eval.py --model_path \"${model_path}\" --tasks \"${TASKS}\" --output_dir \"${OUTPUT_DIR}\" --verbose"
    if [[ -n "${LIMIT}" ]]; then
      cmd="${cmd} --limit ${LIMIT}"
    fi
  else
    # Multi-GPU mode - use llm_eval_parallel.py
    local cmd="python llm_eval_parallel.py --model_path \"${model_path}\" --tasks \"${TASKS}\" --output_dir \"${OUTPUT_DIR}\" --num_gpus ${NUM_GPUS} --verbose"
    if [[ -n "${LIMIT}" ]]; then
      cmd="${cmd} --limit ${LIMIT}"
    fi
  fi
  echo "${cmd}"
}

# Set HuggingFace cache to local storage (avoid FUSE)
export HF_HOME=/home/aiscuser/.cache/huggingface
export HF_DATASETS_CACHE=/home/aiscuser/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/home/aiscuser/.cache/huggingface/transformers

echo "========================================="
echo "GPU Configuration:"
echo "  Available GPUs: ${NUM_AVAILABLE_GPUS}"
if [[ "${NUM_GPUS}" -eq 1 ]]; then
  echo "  Using: Single GPU mode"
else
  echo "  Using: ${NUM_GPUS} GPUs (Data Parallel)"
  echo "  Speedup: ~${NUM_GPUS}x faster"
fi
echo "Cache locations (avoiding FUSE):"
echo "  HF_HOME: ${HF_HOME}"
echo "  HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "========================================="
echo ""

if [[ -d "${MODEL_ROOT}" && -d "${MODEL_ROOT}/checkpoint-0" ]]; then
  for ckpt in "${MODEL_ROOT}"/checkpoint-*; do
    if [[ -d "${ckpt}" ]]; then
      echo "Evaluating checkpoint: ${ckpt}"
      eval "$(build_cmd "${ckpt}")"
    fi
  done
else
  echo "Evaluating model: ${MODEL_ROOT}"
  eval "$(build_cmd "${MODEL_ROOT}")"
fi
