# LLM Evaluation Script with Decoupled Dataset Download
#
# RECOMMENDED WORKFLOW (Download Once, Evaluate Multiple Times):
# ==============================================================
#
# Step 1: PRE-DOWNLOAD all datasets (run ONCE):
#   python download_eval_datasets.py \
#     --tasks "mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval" \
#     --cache_dir ~/.cache/huggingface/datasets
#
# Step 2: RUN EVALUATION multiple times on PRE-DOWNLOADED datasets (OFFLINE MODE):
#   SKIP_DOWNLOAD=1 ./eval_llm.sh Qwen/Qwen3-1.7B
#   SKIP_DOWNLOAD=1 ./eval_llm.sh Qwen/Qwen3-1.7B "mmlu,hellaswag"
#
# This approach AVOIDS:
#   - FUSE conflicts from concurrent dataset downloads
#   - Slow network I/O during GPU computation
#   - Re-downloading same datasets repeatedly
#
# USAGE EXAMPLES:
# ===============
#
# 1. Full workflow (download + evaluate):
#    ./eval_llm.sh Qwen/Qwen3-1.7B
#
# 2. Evaluate only on pre-downloaded datasets (OFFLINE MODE):
#    SKIP_DOWNLOAD=1 ./eval_llm.sh Qwen/Qwen3-1.7B
#
# 3. With custom batch size (higher = faster but more memory):
#    SKIP_DOWNLOAD=1 BATCH_SIZE=32 ./eval_llm.sh Qwen/Qwen3-1.7B
#
# 4. With custom tasks:
#    SKIP_DOWNLOAD=1 ./eval_llm.sh Qwen/Qwen3-1.7B "mmlu,hellaswag"
#
# 5. Evaluate only subset of examples:
#    SKIP_DOWNLOAD=1 ./eval_llm.sh Qwen/Qwen3-1.7B "mmlu" "" 0.1
#
# ENVIRONMENT VARIABLES:
# ======================
#   SKIP_DOWNLOAD: Set to 1 to skip pre-download (datasets must be already cached)
#   BATCH_SIZE: Batch size for evaluation (default: 16, higher = faster but more memory)
#   CACHE_DIR: Custom cache directory (default: ~/.cache/huggingface/datasets)
#   MASTER_PORT: Port for distributed training (default: 29500)

set -euo pipefail

MODEL_ROOT="${1:-}"
TASKS="${2:-mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval}"
OUTPUT_DIR="${3:-llm_eval_results}"
LIMIT="${4:-}"  # Optional: limit evaluation (e.g., 0.1 for 10% or 100 for first 100 examples)

if [[ -z "${MODEL_ROOT}" ]]; then
  echo "Usage: $0 <model_or_output_dir> [tasks] [output_dir] [limit]" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $0 Qwen/Qwen3-4B-Instruct-2507                    # Evaluate model" >&2
  echo "  $0 Qwen/Qwen3-4B-Instruct-2507 mmlu llm_eval 0.1  # Evaluate with 10% samples" >&2
  echo "" >&2
  echo "Environment variables:" >&2
  echo "  BATCH_SIZE=32 $0 ...         # Custom batch size (default: 16)" >&2
  echo "  SKIP_DOWNLOAD=1 $0 ...       # Skip dataset pre-download (use cached)" >&2
  echo "  CACHE_DIR=/path $0 ...       # Custom cache directory" >&2
  exit 1
fi

# Detect number of GPUs
NUM_AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
BATCH_SIZE="${BATCH_SIZE:-16}"  # Default to batch size 16 (much faster than auto=1)
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"  # Default to running pre-download
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/huggingface/datasets}"

# Build the Python command
build_cmd() {
  local model_path="$1"
  # Always use offline mode - evaluate only on pre-downloaded datasets
  local mode_flag="--offline_mode"
  
  # Use llm_eval.py for all cases (single GPU)
  local cmd="python llm_eval.py --model_path \"${model_path}\" --tasks \"${TASKS}\" --output_dir \"${OUTPUT_DIR}\" --cache_dir \"${CACHE_DIR}\" --batch_size ${BATCH_SIZE} ${mode_flag} --verbose"
  if [[ -n "${LIMIT}" ]]; then
    cmd="${cmd} --limit ${LIMIT}"
  fi
  echo "${cmd}"
}

# Set HuggingFace cache locations
export HF_HOME="${HOME}/.cache/huggingface"
export HF_DATASETS_CACHE="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface/transformers"

# Set distributed training port to avoid conflicts (default 29500, fallback from 9500)
export MASTER_PORT="${MASTER_PORT:-29500}"

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

if [[ -d "${MODEL_ROOT}" && -d "${MODEL_ROOT}/final_checkpoint" ]]; then
  for ckpt in "${MODEL_ROOT}"/final_checkpoint; do
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
