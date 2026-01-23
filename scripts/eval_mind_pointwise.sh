#!/usr/bin/env bash
# MIND Evaluation with Point-wise Format (Multi-GPU Support)
#
# This script evaluates models trained with point-wise SFT
# using Yes/No classification to score each candidate.
#
# USAGE:
#   # Single GPU (quick test)
#   bash scripts/eval_mind_pointwise.sh output_dir/sft_mind_pointwise_*/final_checkpoint dev 100
#
#   # Single GPU (full evaluation)
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_mind_pointwise.sh output_dir/sft_mind_pointwise_*/final_checkpoint dev
#
#   # Multi-GPU parallel evaluation (4-8x faster)
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_mind_pointwise.sh output_dir/sft_mind_pointwise_*/final_checkpoint dev
#
#   # With abstracts
#   USE_ABSTRACT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_mind_pointwise.sh output_dir/sft_mind_pointwise_*/final_checkpoint dev

set -euo pipefail

# Parse arguments
MODEL_PATH="${1:-}"
SPLIT="${2:-dev}"
MAX_IMPRESSIONS="${3:-0}"

# Detect parallel mode based on CUDA_VISIBLE_DEVICES
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
  PARALLEL_MODE=true
  CUDA_LIST="${CUDA_VISIBLE_DEVICES}"
else
  PARALLEL_MODE=false
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="0"
  fi
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: $0 <model_path> [split] [max_impressions]" >&2
  echo "" >&2
  echo "Arguments:" >&2
  echo "  model_path: Path to point-wise SFT checkpoint" >&2
  echo "  split: dev or test (default: dev)" >&2
  echo "  max_impressions: Limit to N impressions (optional, 0=all)" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  # Single GPU" >&2
  echo "  CUDA_VISIBLE_DEVICES=0 $0 model_path dev" >&2
  echo "" >&2
  echo "  # Multi-GPU parallel (4-8x faster)" >&2
  echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 $0 model_path dev" >&2
  exit 1
fi

# Configuration
MIND_ROOT="${MIND_ROOT:-../data/MIND}"
MIND_SIZE="${MIND_SIZE:-small}"
USE_ABSTRACT="${USE_ABSTRACT:-0}"
MAX_HISTORY="${MAX_HISTORY:-0}"  # 0 = no limit (use all history)
OUTPUT_FILE="${OUTPUT_FILE:-}"
FLASH_ATTN="${FLASH_ATTN:-1}"  # Use Flash Attention 2 by default
BATCH_SIZE="${BATCH_SIZE:-8}"  # Batch size for scoring candidates

# Construct paths
DATA_DIR="${MIND_ROOT}/${SPLIT}"
BEHAVIORS_PATH="${DATA_DIR}/behaviors.tsv"
NEWS_PATH="${DATA_DIR}/news.tsv"

echo "========================================="
echo "MIND Point-wise Evaluation"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: MIND${MIND_SIZE} ${SPLIT} split"
echo "Format: Yes/No classification"
echo ""
echo "GPU Configuration:"
if [[ "${PARALLEL_MODE}" == "true" ]]; then
  echo "  Mode: Parallel multi-GPU"
  echo "  GPUs: ${CUDA_LIST}"
else
  echo "  Mode: Single GPU"
  echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
fi
echo ""
echo "Configuration:"
echo "  Use abstract: ${USE_ABSTRACT}"
echo "  Max history: ${MAX_HISTORY:-unlimited}"
echo "  Max impressions: ${MAX_IMPRESSIONS:-all}"
echo "  Flash Attention: ${FLASH_ATTN}"
echo "  Batch size: ${BATCH_SIZE}"
echo "========================================="
echo ""

# Check if data exists
if [[ ! -f "${BEHAVIORS_PATH}" ]] || [[ ! -f "${NEWS_PATH}" ]]; then
  echo "ERROR: MIND data not found!" >&2
  echo "  Behaviors: ${BEHAVIORS_PATH}" >&2
  echo "  News: ${NEWS_PATH}" >&2
  exit 1
fi

echo "✓ Data files found"
echo "  Behaviors: ${BEHAVIORS_PATH} ($(wc -l < "${BEHAVIORS_PATH}") impressions)"
echo "  News: ${NEWS_PATH} ($(wc -l < "${NEWS_PATH}") news articles)"
echo ""

if [[ "${PARALLEL_MODE}" == "true" ]]; then
  # ========== PARALLEL MULTI-GPU MODE ==========
  echo "Starting parallel evaluation..."
  echo ""

  # Create temporary directory
  TEMP_DIR="./temp_mind_pointwise/${SPLIT}-$$"
  mkdir -p "${TEMP_DIR}"

  # Split behaviors across GPUs
  echo "Splitting behaviors across GPUs..."
  python split_mind.py \
    --input_path "${BEHAVIORS_PATH}" \
    --output_path "${TEMP_DIR}" \
    --cuda_list "${CUDA_LIST}"

  echo ""

  # Start parallel evaluation on each GPU
  cudalist=$(echo "$CUDA_LIST" | tr ',' ' ')
  pids=()
  gpu_pids=()

  for gpu_id in ${cudalist}; do
    if [[ ! -f "${TEMP_DIR}/${gpu_id}.tsv" ]]; then
      echo "WARNING: Split file ${TEMP_DIR}/${gpu_id}.tsv not found, skipping GPU $gpu_id"
      continue
    fi

    echo "[GPU $gpu_id] Starting evaluation"

    # Build command
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id python -u evaluate_mind_pointwise.py \
      --model_path \"${MODEL_PATH}\" \
      --behaviors_path \"${TEMP_DIR}/${gpu_id}.tsv\" \
      --news_path \"${NEWS_PATH}\" \
      --max_history ${MAX_HISTORY} \
      --batch_size ${BATCH_SIZE} \
      --output_file \"${TEMP_DIR}/${gpu_id}.txt\""

    if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
      cmd="${cmd} --use_abstract"
    fi

    if [[ "${FLASH_ATTN}" -eq 1 ]]; then
      cmd="${cmd} --flash_attn"
    fi

    if [[ -n "${MAX_IMPRESSIONS}" ]] && [[ "${MAX_IMPRESSIONS}" -gt 0 ]]; then
      num_gpus=$(echo "$CUDA_LIST" | tr ',' ' ' | wc -w)
      per_gpu=$((MAX_IMPRESSIONS / num_gpus))
      cmd="${cmd} --max_impressions ${per_gpu}"
    fi

    # Run in background
    eval "$cmd 2>&1 | sed \"s/^/[GPU $gpu_id] /\"" &
    pid=$!
    pids+=($pid)
    gpu_pids+=("$gpu_id:$pid")
    echo "[GPU $gpu_id] Process started with PID $pid"
  done

  echo ""
  echo "Waiting for ${#pids[@]} GPU process(es) to complete..."
  echo ""

  # Wait for all processes
  failed_gpus=()
  for idx in "${!pids[@]}"; do
    pid=${pids[$idx]}
    gpu_info=${gpu_pids[$idx]}
    gpu=${gpu_info%%:*}

    if wait $pid; then
      echo "✓ GPU $gpu completed successfully"
    else
      echo "✗ ERROR: GPU $gpu failed"
      failed_gpus+=($gpu)
    fi
  done

  echo ""

  # Merge predictions
  if [[ ${#failed_gpus[@]} -gt 0 ]]; then
    echo "WARNING: ${#failed_gpus[@]} GPU(s) failed: ${failed_gpus[@]}"
  fi

  OUTPUT_FILE="${OUTPUT_FILE:-./results_mind/${SPLIT}_pointwise_predictions.txt}"
  mkdir -p "$(dirname "$OUTPUT_FILE")"

  echo "Merging predictions..."
  actual_cuda_list=$(ls "${TEMP_DIR}"/*.txt 2>/dev/null | sed 's/.*\///g' | sed 's/\.txt//g' | tr '\n' ',' | sed 's/,$//')

  python merge_mind.py \
    --input_path "${TEMP_DIR}" \
    --output_path "${OUTPUT_FILE}" \
    --cuda_list "${actual_cuda_list}" \
    --calculate_metrics false

  echo ""
  echo "========================================="
  echo "✓ Parallel evaluation completed!"
  echo "========================================="
  echo "Predictions saved to: ${OUTPUT_FILE}"

  # Calculate metrics for dev split (has ground truth labels)
  if [[ "${SPLIT}" == "dev" ]] && [[ -f "${BEHAVIORS_PATH}" ]]; then
    echo ""
    echo "Calculating metrics from predictions..."
    python calc_mind_metrics.py \
      --predictions "${OUTPUT_FILE}" \
      --behaviors "${BEHAVIORS_PATH}"
  elif [[ "${SPLIT}" == "test" ]]; then
    echo ""
    echo "Note: Test split has no ground truth labels."
    echo "Submit predictions to MIND leaderboard for evaluation."
  fi

  # Cleanup temp files
  rm -rf "${TEMP_DIR}"

else
  # ========== SINGLE GPU MODE ==========
  echo "Running evaluation..."
  echo ""

  # Build command
  CMD="python evaluate_mind_pointwise.py \
    --model_path ${MODEL_PATH} \
    --behaviors_path ${BEHAVIORS_PATH} \
    --news_path ${NEWS_PATH} \
    --max_history ${MAX_HISTORY} \
    --batch_size ${BATCH_SIZE}"

  if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
    CMD="${CMD} --use_abstract"
  fi

  if [[ -n "${MAX_IMPRESSIONS}" ]] && [[ "${MAX_IMPRESSIONS}" -gt 0 ]]; then
    CMD="${CMD} --max_impressions ${MAX_IMPRESSIONS}"
  fi

  if [[ -n "${OUTPUT_FILE}" ]]; then
    CMD="${CMD} --output_file ${OUTPUT_FILE}"
  fi

  if [[ "${FLASH_ATTN}" -eq 1 ]]; then
    CMD="${CMD} --flash_attn"
  fi

  eval "${CMD}"

  echo ""
  echo "========================================="
  echo "✓ Evaluation completed!"
  echo "========================================="
fi
