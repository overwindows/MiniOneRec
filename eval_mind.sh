#!/usr/bin/env bash
# MIND Dataset Evaluation Script
#
# USAGE EXAMPLES:
# ===============
#
# 1. Quick test on dev split (100 impressions):
#    ./eval_mind.sh Qwen/Qwen3-1.7B dev 100
#
# 2. Full dev split evaluation:
#    ./eval_mind.sh Qwen/Qwen3-1.7B dev
#
# 3. With abstracts (better quality, slower):
#    USE_ABSTRACT=1 ./eval_mind.sh Qwen/Qwen3-1.7B dev
#
# 4. Use a specific GPU (e.g., GPU 0 or GPU 2):
#    GPU_ID=0 ./eval_mind.sh Qwen/Qwen3-1.7B dev
#
# 5. Generate predictions for leaderboard submission:
#    OUTPUT_FILE=predictions.txt ./eval_mind.sh Qwen/Qwen3-1.7B test
#
# 6. Custom data root:
#    MIND_ROOT=/path/to/data ./eval_mind.sh Qwen/Qwen3-1.7B dev
#
# ENVIRONMENT VARIABLES:
# ======================
#   GPU_ID: GPU ID to use for evaluation (default: 1)
#   MIND_ROOT: Root directory containing MIND data (default: ../data/MIND)
#   MIND_SIZE: Dataset size - small or large (default: small)
#   MIND_ZIPS: Directory containing MIND ZIP files (default: ~/wuc/downloaded/zips)
#   USE_ABSTRACT: Set to 1 to use abstracts (default: 0)
#   MAX_HISTORY: Max history items to use (default: 50)
#   BATCH_SIZE: Batch size for scoring candidates (default: 4, reduce if OOM)
#   SKIP_EXTRACT: Set to 1 to skip automatic extraction (default: 0)
#   OUTPUT_FILE: Path to save predictions for MIND leaderboard submission (optional)

set -euo pipefail

# GPU Configuration
# Set to specific GPU ID (e.g., 0, 1, 2) or leave empty to use all GPUs
GPU_ID="${GPU_ID:-1}"

MODEL_PATH="${1:-}"
SPLIT="${2:-dev}"
MAX_IMPRESSIONS="${3:-}"

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
MIND_ROOT="${MIND_ROOT:-../data/MIND}"
MIND_SIZE="${MIND_SIZE:-small}"
MIND_ZIPS="${MIND_ZIPS:-${HOME}/wuc/downloaded/zips}"
USE_ABSTRACT="${USE_ABSTRACT:-0}"
MAX_HISTORY="${MAX_HISTORY:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

# Construct paths
DATA_DIR="${MIND_ROOT}/${SPLIT}"
BEHAVIORS_PATH="${DATA_DIR}/behaviors.tsv"
NEWS_PATH="${DATA_DIR}/news.tsv"

# Set GPU for evaluation
if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

echo "========================================="
echo "MIND Evaluation Configuration:"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: MIND${MIND_SIZE} ${SPLIT} split"
echo "Data directory: ${DATA_DIR}"
echo ""
echo "GPU Configuration:"
if [[ -n "${GPU_ID}" ]]; then
  echo "  Using GPU: ${GPU_ID} (CUDA_VISIBLE_DEVICES=${GPU_ID})"
else
  echo "  Using all available GPUs"
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
echo "  Behaviors: ${BEHAVIORS_PATH} ($(wc -l < "${BEHAVIORS_PATH}") lines)"
echo "  News: ${NEWS_PATH} ($(wc -l < "${NEWS_PATH}") lines)"
echo ""
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
