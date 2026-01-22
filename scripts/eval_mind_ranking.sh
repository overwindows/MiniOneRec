#!/usr/bin/env bash
# MIND Evaluation with Ranking-Aware Format
#
# This script evaluates models trained with ranking-aware SFT
# using the SAME multiple-choice format as training.
#
# USAGE:
#   bash scripts/eval_mind_ranking.sh <model_path> [split] [max_impressions]
#
# EXAMPLES:
#   # Quick test (100 impressions)
#   bash scripts/eval_mind_ranking.sh output_dir/sft_mind_ranking_*/final_checkpoint dev 100
#
#   # Full evaluation
#   bash scripts/eval_mind_ranking.sh output_dir/sft_mind_ranking_*/final_checkpoint dev
#
#   # With abstracts
#   USE_ABSTRACT=1 bash scripts/eval_mind_ranking.sh output_dir/sft_mind_ranking_*/final_checkpoint dev

set -euo pipefail

# Parse arguments
MODEL_PATH="${1:-}"
SPLIT="${2:-dev}"
MAX_IMPRESSIONS="${3:-0}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: $0 <model_path> [split] [max_impressions]" >&2
  echo "" >&2
  echo "Arguments:" >&2
  echo "  model_path: Path to ranking-aware SFT checkpoint" >&2
  echo "  split: dev or test (default: dev)" >&2
  echo "  max_impressions: Limit to N impressions (optional, 0=all)" >&2
  exit 1
fi

# Configuration
MIND_ROOT="${MIND_ROOT:-../data/MIND}"
MIND_SIZE="${MIND_SIZE:-small}"
USE_ABSTRACT="${USE_ABSTRACT:-0}"
MAX_HISTORY="${MAX_HISTORY:-50}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

# Construct paths
DATA_DIR="${MIND_ROOT}/${SPLIT}"
BEHAVIORS_PATH="${DATA_DIR}/behaviors.tsv"
NEWS_PATH="${DATA_DIR}/news.tsv"

echo "========================================="
echo "MIND Ranking-Aware Evaluation"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: MIND${MIND_SIZE} ${SPLIT} split"
echo "Format: Multiple-choice (1/2/3/...)"
echo ""
echo "Configuration:"
echo "  Use abstract: ${USE_ABSTRACT}"
echo "  Max history: ${MAX_HISTORY}"
echo "  Max impressions: ${MAX_IMPRESSIONS:-all}"
echo "========================================="
echo ""

# Check if data exists
if [[ ! -f "${BEHAVIORS_PATH}" ]] || [[ ! -f "${NEWS_PATH}" ]]; then
  echo "ERROR: MIND data not found!" >&2
  echo "  Behaviors: ${BEHAVIORS_PATH}" >&2
  echo "  News: ${NEWS_PATH}" >&2
  exit 1
fi

# Build command
CMD="python evaluate_mind_ranking.py \
  --model_path ${MODEL_PATH} \
  --behaviors_path ${BEHAVIORS_PATH} \
  --news_path ${NEWS_PATH} \
  --max_history ${MAX_HISTORY}"

if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
  CMD="${CMD} --use_abstract"
fi

if [[ -n "${MAX_IMPRESSIONS}" ]] && [[ "${MAX_IMPRESSIONS}" -gt 0 ]]; then
  CMD="${CMD} --max_impressions ${MAX_IMPRESSIONS}"
fi

if [[ -n "${OUTPUT_FILE}" ]]; then
  CMD="${CMD} --output_file ${OUTPUT_FILE}"
fi

echo "Running evaluation..."
echo ""

eval "${CMD}"

echo ""
echo "========================================="
echo "✓ Evaluation completed!"
echo "========================================="
