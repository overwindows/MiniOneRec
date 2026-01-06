#!/usr/bin/env bash
# LLM Evaluation Script with Decoupled Dataset Download
#
# DECOUPLED WORKFLOW:
# ===================
# This script implements a two-stage evaluation workflow to avoid FUSE conflicts:
#
#   Stage 1: Pre-download evaluation datasets (once per task set)
#     - Download all required datasets to a local cache directory
#     - Avoids concurrent FUSE access during evaluation
#     - Dramatically faster than downloading during evaluation
#
#   Stage 2: Run evaluation with offline mode
#     - Uses pre-downloaded datasets from cache
#     - No network I/O during GPU computation
#     - Multiple GPU processes work without FUSE conflicts
#
# USAGE EXAMPLES:
# ===============
#
# 1. Single GPU with pre-download:
#    ./eval_llm.sh Qwen/Qwen3-1.7B
#
# 2. Multi-GPU with pre-download:
#    NUM_GPUS=4 ./eval_llm.sh Qwen/Qwen3-1.7B
#
# 3. With custom tasks and limit:
#    NUM_GPUS=4 ./eval_llm.sh Qwen/Qwen3-1.7B "mmlu,hellaswag" llm_eval 0.1
#
# 4. With custom batch size (higher = faster but more memory):
#    BATCH_SIZE=32 ./eval_llm.sh Qwen/Qwen3-1.7B
#
# 5. Skip pre-download (datasets already cached):
#    SKIP_DOWNLOAD=1 NUM_GPUS=4 ./eval_llm.sh Qwen/Qwen3-1.7B
#
# ENVIRONMENT VARIABLES:
# ======================
#   NUM_GPUS: Number of GPUs to use (default: 1)
#   BATCH_SIZE: Batch size for evaluation (default: 16, or 'auto' for auto-detection)
#   SKIP_DOWNLOAD: Set to 1 to skip pre-download step
#   CACHE_DIR: Custom cache directory (default: ~/.cache/huggingface/datasets)

set -euo pipefail

MODEL_ROOT="${1:-}"
TASKS="${2:-mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval}"
OUTPUT_DIR="${3:-llm_eval_results}"
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
  echo "  BATCH_SIZE=32 $0 ...         # Custom batch size (default: 16)" >&2
  echo "  SKIP_DOWNLOAD=1 $0 ...       # Skip dataset pre-download" >&2
  echo "  CACHE_DIR=/path $0 ...       # Custom cache directory" >&2
  exit 1
fi

# Detect number of GPUs
NUM_AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_GPUS="${NUM_GPUS:-1}"  # Default to single GPU
BATCH_SIZE="${BATCH_SIZE:-16}"  # Default to batch size 16 (much faster than auto=1)
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"  # Default to running pre-download
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/huggingface/datasets}"

# Build the Python command based on number of GPUs
build_cmd() {
  local model_path="$1"
  if [[ "${NUM_GPUS}" -eq 1 ]]; then
    # Single GPU mode - use llm_eval.py
    local cmd="python llm_eval.py --model_path \"${model_path}\" --tasks \"${TASKS}\" --output_dir \"${OUTPUT_DIR}\" --cache_dir \"${CACHE_DIR}\" --batch_size ${BATCH_SIZE} --online_mode --verbose"
    if [[ -n "${LIMIT}" ]]; then
      cmd="${cmd} --limit ${LIMIT}"
    fi
  else
    # Multi-GPU mode - use llm_eval_parallel.py
    local cmd="python llm_eval_parallel.py --model_path \"${model_path}\" --tasks \"${TASKS}\" --output_dir \"${OUTPUT_DIR}\" --num_gpus ${NUM_GPUS} --cache_dir \"${CACHE_DIR}\" --batch_size ${BATCH_SIZE} --online_mode --verbose"
    if [[ -n "${LIMIT}" ]]; then
      cmd="${cmd} --limit ${LIMIT}"
    fi
  fi
  echo "${cmd}"
}

# Set HuggingFace cache locations
export HF_HOME="${HOME}/.cache/huggingface"
export HF_DATASETS_CACHE="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface/transformers"

echo "========================================="
echo "Evaluation Configuration:"
echo "========================================="
echo "Model: ${MODEL_ROOT}"
echo "Tasks: ${TASKS}"
echo "Output directory: ${OUTPUT_DIR}"
if [[ -n "${LIMIT}" ]]; then
  echo "Evaluation limit: ${LIMIT}"
fi
echo ""
echo "GPU Configuration:"
echo "  Available GPUs: ${NUM_AVAILABLE_GPUS}"
if [[ "${NUM_GPUS}" -eq 1 ]]; then
  echo "  Using: Single GPU mode"
else
  echo "  Using: ${NUM_GPUS} GPUs (Data Parallel)"
  echo "  Speedup: ~${NUM_GPUS}x faster (theoretical)"
fi
echo "  Batch size: ${BATCH_SIZE}"
echo ""
echo "Dataset Caching (avoiding FUSE conflicts):"
echo "  Cache directory: ${CACHE_DIR}"
echo "  HF_HOME: ${HF_HOME}"
echo "========================================="
echo ""

# STAGE 1: Pre-download datasets if needed
if [[ "${SKIP_DOWNLOAD}" -eq 0 ]]; then
  echo "STAGE 1: Pre-downloading evaluation datasets..."
  echo "========================================="
  echo "This may take 5-30 minutes depending on tasks."
  echo "Run with SKIP_DOWNLOAD=1 to skip this step."
  echo ""
  
  mkdir -p "${CACHE_DIR}"
  
  # Run pre-download
  python download_eval_datasets.py \
    --tasks "${TASKS}" \
    --cache_dir "${CACHE_DIR}"
  
  echo ""
  echo "✓ Dataset pre-download completed"
  echo "========================================="
  echo ""
fi

# STAGE 2: Run evaluation
echo "STAGE 2: Running evaluation..."
echo "========================================="
echo "Datasets will be loaded from cache."
echo "Model will be loaded from HuggingFace cache or downloaded if needed."
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

echo ""
echo "========================================="
echo "✓ Evaluation completed!"
echo "========================================="
