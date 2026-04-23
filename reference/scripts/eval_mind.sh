#!/usr/bin/env bash
# MIND Dataset Evaluation Script
#
# USAGE EXAMPLES:
# ===============
#
# 1. Quick test on dev split (100 impressions):
#    bash scripts/eval_mind.sh Qwen/Qwen3-1.7B dev 100
#
# 2. Full dev split evaluation (single GPU):
#    CUDA_VISIBLE_DEVICES=7 bash scripts/eval_mind.sh Qwen/Qwen3-1.7B dev
#
# 3. Parallel multi-GPU evaluation (4-8x faster):
#    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_mind.sh Qwen/Qwen3-1.7B dev
#
# 4. With abstracts (better quality, slower):
#    CUDA_VISIBLE_DEVICES=0,1,2,3 USE_ABSTRACT=1 bash scripts/eval_mind.sh Qwen/Qwen3-1.7B dev
#
# 5. Generate predictions for leaderboard submission:
#    CUDA_VISIBLE_DEVICES=0,1,2,3 OUTPUT_FILE=predictions.txt bash scripts/eval_mind.sh Qwen/Qwen3-1.7B test
#
# 6. Custom data root:
#    CUDA_VISIBLE_DEVICES=0,1,2,3 MIND_ROOT=/path/to/data bash scripts/eval_mind.sh Qwen/Qwen3-1.7B dev
#
# 7. Calculate metrics from existing predictions:
#    python calc_mind_metrics.py --predictions ./results_mind/dev_predictions.txt --behaviors ../data/MIND/dev/behaviors.tsv
#
# ENVIRONMENT VARIABLES:
# ======================
#   CUDA_VISIBLE_DEVICES: GPU IDs to use (e.g., "0,1,2,3" for parallel, "1" for single GPU)
#   MIND_ROOT: Root directory containing MIND data (default: ../data/MIND)
#   MIND_SIZE: Dataset size - small or large (default: small)
#   MIND_ZIPS: Directory containing MIND ZIP files (default: ~/wuc/downloaded/zips)
#   USE_ABSTRACT: Set to 1 to use abstracts (default: 0)
#   MAX_HISTORY: Max history items to use (default: 50)
#   BATCH_SIZE: Batch size for scoring candidates (default: 4, reduce if OOM)
#   SKIP_EXTRACT: Set to 1 to skip automatic extraction (default: 0)
#   OUTPUT_FILE: Path to save predictions for MIND leaderboard submission (optional)

set -euo pipefail

# Parse arguments
MODEL_PATH="${1:-}"
SPLIT="${2:-dev}"
MAX_IMPRESSIONS="${3:-}"

# Detect parallel mode based on CUDA_VISIBLE_DEVICES
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
  PARALLEL_MODE=true
  # Extract GPU IDs from CUDA_VISIBLE_DEVICES
  CUDA_LIST="${CUDA_VISIBLE_DEVICES}"
else
  PARALLEL_MODE=false
  # Single GPU mode - use GPU_ID if CUDA_VISIBLE_DEVICES is not set
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
  fi
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: $0 <model_path> [split] [max_impressions]" >&2
  echo "" >&2
  echo "Arguments:" >&2
  echo "  model_path: HuggingFace model name or local checkpoint path" >&2
  echo "  split: train, dev, or test (default: dev)" >&2
  echo "  max_impressions: Limit evaluation to N impressions (optional)" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $0 Qwen/Qwen3-1.7B dev              # Full dev evaluation" >&2
  echo "  $0 Qwen/Qwen3-1.7B dev 100          # Quick test (100 impressions)" >&2
  echo "  USE_ABSTRACT=1 $0 Qwen/Qwen3-1.7B dev  # Use abstracts" >&2
  exit 1
fi

# Configuration
MIND_SIZE="${MIND_SIZE:-small}"
if [[ -z "${MIND_ROOT:-}" ]]; then
  if [[ "${MIND_SIZE}" == "large" && -d "../data/MIND_large" ]]; then
    MIND_ROOT="../data/MIND_large"
  elif [[ "${MIND_SIZE}" == "small" && -d "../data/MIND_small" ]]; then
    MIND_ROOT="../data/MIND_small"
  else
    MIND_ROOT="../data/MIND"
  fi
fi
MIND_ZIPS="${MIND_ZIPS:-${HOME}/wuc/downloaded/zips}"
USE_ABSTRACT="${USE_ABSTRACT:-0}"
MAX_HISTORY="${MAX_HISTORY:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

# Construct paths
DATA_DIR="${MIND_ROOT}/${SPLIT}"
BEHAVIORS_PATH="${DATA_DIR}/behaviors.tsv"
NEWS_PATH="${DATA_DIR}/news.tsv"

echo "========================================="
echo "MIND Evaluation Configuration:"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: MIND${MIND_SIZE} ${SPLIT} split"
echo "Data directory: ${DATA_DIR}"
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
echo "Settings:"
echo "  Use abstracts: ${USE_ABSTRACT}"
echo "  Max history: ${MAX_HISTORY}"
echo "  Batch size: ${BATCH_SIZE}"
if [[ -n "${MAX_IMPRESSIONS}" ]]; then
  echo "  Max impressions: ${MAX_IMPRESSIONS} (quick test mode)"
fi
echo "========================================="
echo ""

# Auto-extract MIND data if not found
if [[ ! -f "${BEHAVIORS_PATH}" ]] || [[ ! -f "${NEWS_PATH}" ]]; then
  if [[ "${SKIP_EXTRACT}" -eq 1 ]]; then
    echo "Error: Data files not found and SKIP_EXTRACT=1" >&2
    echo "  Behaviors: ${BEHAVIORS_PATH}" >&2
    echo "  News: ${NEWS_PATH}" >&2
    exit 1
  fi

  echo "Data files not found. Auto-extracting from ZIP files..."
  echo "  ZIP directory: ${MIND_ZIPS}"
  echo "  Target: MIND${MIND_SIZE} ${SPLIT} split"
  echo ""

  # Check if ZIP directory exists
  if [[ ! -d "${MIND_ZIPS}" ]]; then
    echo "Error: ZIP directory not found: ${MIND_ZIPS}" >&2
    echo "" >&2
    echo "Please set MIND_ZIPS environment variable to the correct path," >&2
    echo "or download MIND dataset first." >&2
    exit 1
  fi

  # Run extraction
  echo "Running: python prepare_mind.py --root ${MIND_ROOT} --size ${MIND_SIZE} --splits ${SPLIT} --local ${MIND_ZIPS}"
  python prepare_mind.py \
    --root "${MIND_ROOT}" \
    --size "${MIND_SIZE}" \
    --splits "${SPLIT}" \
    --local "${MIND_ZIPS}"

  echo ""
  echo "✓ Extraction completed"
  echo ""

  # Verify extraction succeeded
  if [[ ! -f "${BEHAVIORS_PATH}" ]] || [[ ! -f "${NEWS_PATH}" ]]; then
    echo "Error: Extraction failed. Data files still not found." >&2
    exit 1
  fi
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
  TEMP_DIR="./temp_mind/${SPLIT}-$$"
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
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id python -u evaluate_mind.py \
      --model_path \"${MODEL_PATH}\" \
      --behaviors_path \"${TEMP_DIR}/${gpu_id}.tsv\" \
      --news_path \"${NEWS_PATH}\" \
      --max_history ${MAX_HISTORY} \
      --batch_size ${BATCH_SIZE} \
      --output_file \"${TEMP_DIR}/${gpu_id}.txt\""

    if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
      cmd="${cmd} --use_abstract"
    fi

    if [[ -n "${MAX_IMPRESSIONS}" ]]; then
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

  OUTPUT_FILE="${OUTPUT_FILE:-./results_mind/${SPLIT}_predictions.txt}"
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
  echo "✓ Parallel MIND evaluation completed!"
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

else
  # ========== SINGLE GPU MODE ==========
  echo "Starting evaluation..."
  echo ""

  # Build Python command
  CMD="python evaluate_mind.py --model_path \"${MODEL_PATH}\" --behaviors_path \"${BEHAVIORS_PATH}\" --news_path \"${NEWS_PATH}\" --max_history ${MAX_HISTORY} --batch_size ${BATCH_SIZE}"

  if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
    CMD="${CMD} --use_abstract"
  fi

  if [[ -n "${MAX_IMPRESSIONS}" ]]; then
    CMD="${CMD} --max_impressions ${MAX_IMPRESSIONS}"
  fi

  # Add output file for predictions if specified
  if [[ -n "${OUTPUT_FILE}" ]]; then
    CMD="${CMD} --output_file \"${OUTPUT_FILE}\""
  fi

  # Run evaluation
  eval "${CMD}"

  echo ""
  echo "========================================="
  echo "✓ MIND evaluation completed!"
  echo "========================================="
fi
