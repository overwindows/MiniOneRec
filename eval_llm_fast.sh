#!/usr/bin/env bash
# Optimized version of eval_llm.sh with pre-download step
set -euo pipefail

MODEL_ROOT="${1:-}"
TASKS="${2:-mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval}"
OUTPUT_DIR="${3:-llm_eval}"
LIMIT="${4:-}"

if [[ -z "${MODEL_ROOT}" ]]; then
  echo "Usage: $0 <model_or_output_dir> [tasks] [output_dir] [limit]" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $0 Qwen/Qwen3-4B-Instruct-2507                    # Full evaluation" >&2
  echo "  $0 Qwen/Qwen3-4B-Instruct-2507 mmlu llm_eval 0.1  # 10% of samples" >&2
  exit 1
fi

# Check if it's a HuggingFace model (not a local path)
if [[ ! -d "${MODEL_ROOT}" ]]; then
  echo "============================================================"
  echo "🔍 Detected HuggingFace model: ${MODEL_ROOT}"
  echo "📥 Pre-downloading to avoid I/O bottlenecks during eval..."
  echo "============================================================"
  echo ""

  # Pre-download the model
  python download_model.py "${MODEL_ROOT}" --tasks "${TASKS}" || {
    echo "⚠️  Pre-download failed, continuing anyway..."
  }

  echo ""
  echo "============================================================"
  echo "🚀 Starting evaluation..."
  echo "============================================================"
  echo ""
fi

# Build the Python command
build_cmd() {
  local model_path="$1"
  local cmd="python llm_eval.py --model_path \"${model_path}\" --tasks \"${TASKS}\" --output_dir \"${OUTPUT_DIR}\" --verbose"
  if [[ -n "${LIMIT}" ]]; then
    cmd="${cmd} --limit ${LIMIT}"
  fi
  echo "${cmd}"
}

# Run evaluation
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
