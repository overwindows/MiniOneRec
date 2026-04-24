#!/bin/bash
# =========================
# DOCA Listwise Eval Script
# =========================
# USAGE EXAMPLES:
# ===============
#
# 1. Quick test (500 feeds, single GPU):
#    CUDA_VISIBLE_DEVICES=0 bash scripts/eval_doca_listwise.sh output_dir/final_checkpoint
#
# 2. Full dev evaluation (single GPU):
#    CUDA_VISIBLE_DEVICES=0 bash scripts/eval_doca_listwise.sh output_dir/final_checkpoint --all
#
# 3. Parallel multi-GPU evaluation:
#    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_doca_listwise.sh output_dir/final_checkpoint
#
# 4. Full evaluation with 8 GPUs:
#    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval_doca_listwise.sh output_dir/final_checkpoint --all
#
# ENVIRONMENT VARIABLES:
# ======================
#   CUDA_VISIBLE_DEVICES: GPU IDs (e.g., "0,1,2,3" for parallel, "0" for single)
#   EVAL_JSONL: Path to evaluation JSONL (default: data/doca/dev.jsonl)
#   MAX_FEEDS: Max feeds to evaluate (default: 500, set 0 for all)

set -euo pipefail

# Parse arguments
MODEL_PATH="${1:-}"
FULL_EVAL="${2:-}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: $0 <model_path> [--all]" >&2
  echo "" >&2
  echo "Arguments:" >&2
  echo "  model_path: Local checkpoint path or HuggingFace model name" >&2
  echo "  --all: Evaluate all feeds (default: quick mode with 500 feeds)" >&2
  exit 1
fi

# Configuration
EVAL_JSONL="${EVAL_JSONL:-data/doca/dev.jsonl}"
MAX_FEEDS_DEFAULT=500

if [[ "${FULL_EVAL}" == "--all" ]]; then
  MAX_FEEDS="${MAX_FEEDS:-0}"
else
  MAX_FEEDS="${MAX_FEEDS:-${MAX_FEEDS_DEFAULT}}"
fi

# Detect parallel mode
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
  PARALLEL_MODE=true
  CUDA_LIST="${CUDA_VISIBLE_DEVICES}"
else
  PARALLEL_MODE=false
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="0"
  fi
fi

echo "========================================="
echo "DOCA List-wise Evaluation"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Eval data: ${EVAL_JSONL}"
if [[ "${MAX_FEEDS}" -eq 0 ]]; then
  echo "Max feeds: ALL"
else
  echo "Max feeds: ${MAX_FEEDS} (use --all for full eval)"
fi
echo ""
echo "GPU Configuration:"
if [[ "${PARALLEL_MODE}" == "true" ]]; then
  echo "  Mode: Parallel multi-GPU"
  echo "  GPUs: ${CUDA_LIST}"
else
  echo "  Mode: Single GPU"
  echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
fi
echo "========================================="
echo ""

# Check eval file exists
if [[ ! -f "${EVAL_JSONL}" ]]; then
  echo "Error: Eval file not found: ${EVAL_JSONL}" >&2
  exit 1
fi

TOTAL_FEEDS=$(wc -l < "${EVAL_JSONL}")
echo "Total feeds in file: ${TOTAL_FEEDS}"
echo ""

if [[ "${PARALLEL_MODE}" == "true" ]]; then
  # ========== PARALLEL MULTI-GPU MODE ==========
  echo "Starting parallel evaluation..."
  echo ""

  TEMP_DIR="./temp_doca/eval_listwise-$$"
  mkdir -p "${TEMP_DIR}"

  # Split JSONL across GPUs
  echo "Splitting data across GPUs..."
  python src/split_doca.py \
    --input_path "${EVAL_JSONL}" \
    --output_path "${TEMP_DIR}" \
    --cuda_list "${CUDA_LIST}"
  echo ""

  # Compute per-GPU max_feeds
  num_gpus=$(echo "$CUDA_LIST" | tr ',' ' ' | wc -w)
  if [[ "${MAX_FEEDS}" -gt 0 ]]; then
    per_gpu_feeds=$((MAX_FEEDS / num_gpus))
  else
    per_gpu_feeds=0
  fi

  # Start parallel evaluation
  cudalist=$(echo "$CUDA_LIST" | tr ',' ' ')
  pids=()
  gpu_pids=()

  for gpu_id in ${cudalist}; do
    shard_file="${TEMP_DIR}/${gpu_id}.jsonl"
    if [[ ! -f "${shard_file}" ]]; then
      echo "WARNING: ${shard_file} not found, skipping GPU $gpu_id"
      continue
    fi

    echo "[GPU $gpu_id] Starting evaluation..."

    cmd="CUDA_VISIBLE_DEVICES=$gpu_id python -u src/evaluate_doca_listwise.py \
      --model_path \"${MODEL_PATH}\" \
      --eval_jsonl \"${shard_file}\" \
      --flash_attn \
      --output_scores_file \"${TEMP_DIR}/${gpu_id}_scores.txt\""

    if [[ "${per_gpu_feeds}" -gt 0 ]]; then
      cmd="${cmd} --max_feeds ${per_gpu_feeds}"
    fi

    eval "$cmd 2>&1 | sed \"s/^/[GPU $gpu_id] /\"" &
    pid=$!
    pids+=($pid)
    gpu_pids+=("$gpu_id:$pid")
    echo "[GPU $gpu_id] PID $pid"
  done

  echo ""
  echo "Waiting for ${#pids[@]} GPU process(es)..."
  echo ""

  failed_gpus=()
  for idx in "${!pids[@]}"; do
    pid=${pids[$idx]}
    gpu_info=${gpu_pids[$idx]}
    gpu=${gpu_info%%:*}

    if wait $pid; then
      echo "✓ GPU $gpu completed"
    else
      echo "✗ GPU $gpu FAILED"
      failed_gpus+=($gpu)
    fi
  done

  echo ""

  if [[ ${#failed_gpus[@]} -gt 0 ]]; then
    echo "WARNING: ${#failed_gpus[@]} GPU(s) failed: ${failed_gpus[@]}"
  fi

  # Merge results
  OUTPUT_FILE="./results_doca/dev_listwise_scores.txt"
  mkdir -p "$(dirname "$OUTPUT_FILE")"

  actual_cuda_list=$(ls "${TEMP_DIR}"/*_scores.txt 2>/dev/null | sed 's/.*\///g' | sed 's/_scores\.txt//g' | tr '\n' ',' | sed 's/,$//')

  echo "Merging results..."
  python src/merge_doca.py \
    --input_path "${TEMP_DIR}" \
    --output_path "${OUTPUT_FILE}" \
    --cuda_list "${actual_cuda_list}"

  echo ""
  echo "========================================="
  echo "✓ Parallel DOCA listwise evaluation completed!"
  echo "========================================="
  echo "Scores saved to: ${OUTPUT_FILE}"

  # Cleanup
  echo ""
  echo "Cleaning up temp files..."
  rm -rf "${TEMP_DIR}"

else
  # ========== SINGLE GPU MODE ==========
  echo "Starting single-GPU evaluation..."
  echo ""

  CMD="python src/evaluate_doca_listwise.py \
    --model_path \"${MODEL_PATH}\" \
    --eval_jsonl \"${EVAL_JSONL}\" \
    --flash_attn \
    --output_scores_file \"./results_doca/dev_listwise_scores.txt\""

  if [[ "${MAX_FEEDS}" -gt 0 ]]; then
    CMD="${CMD} --max_feeds ${MAX_FEEDS}"
  fi

  mkdir -p "./results_doca"
  echo "Running: ${CMD}"
  echo ""
  eval "${CMD}"
fi
