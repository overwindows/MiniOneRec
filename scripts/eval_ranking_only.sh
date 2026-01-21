#!/usr/bin/env bash
# Ranking-Only Evaluation for MIND
#
# This script evaluates models trained with ranking-aware SFT
# using the EXACT SAME multiple-choice format as training.
#
# USAGE:
#   bash scripts/eval_ranking_only.sh <model_path> [split] [max_impressions]
#
# EXAMPLES:
#   # Quick test (100 impressions)
#   bash scripts/eval_ranking_only.sh output_dir/sft_mind_ranking_*/final_checkpoint dev 100
#
#   # Full evaluation
#   bash scripts/eval_ranking_only.sh output_dir/sft_mind_ranking_*/final_checkpoint dev

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
  echo "  max_impressions: Limit to N impressions (default: 0=all)" >&2
  exit 1
fi

# Configuration
MIND_ROOT="${MIND_ROOT:-../data/MIND}"
USE_ABSTRACT="${USE_ABSTRACT:-}"
MAX_HISTORY="${MAX_HISTORY:-50}"

echo "========================================="
echo "MIND Ranking-Only Evaluation"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Split: ${SPLIT}"
echo "Format: Multiple-choice (A/B/C/...)"
echo "Scoring: Letter probabilities only"
echo ""
echo "Configuration:"
echo "  MIND root: ${MIND_ROOT}"
echo "  Max history: ${MAX_HISTORY}"
if [[ -n "${USE_ABSTRACT}" ]]; then
  echo "  Use abstract: yes"
fi
if [[ "${MAX_IMPRESSIONS}" -gt 0 ]]; then
  echo "  Max impressions: ${MAX_IMPRESSIONS}"
else
  echo "  Max impressions: all"
fi
echo "========================================="
echo ""

# Check if model exists
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

# Build command
CMD="python eval_ranking_only.py \
  --model_path ${MODEL_PATH} \
  --split ${SPLIT} \
  --mind_root ${MIND_ROOT} \
  --max_history ${MAX_HISTORY}"

if [[ -n "${USE_ABSTRACT}" ]]; then
  CMD="${CMD} --use_abstract"
fi

if [[ "${MAX_IMPRESSIONS}" -gt 0 ]]; then
  CMD="${CMD} --max_impressions ${MAX_IMPRESSIONS}"
fi

echo "Running evaluation..."
echo ""

eval "${CMD}"

echo ""
echo "========================================="
echo "✓ Evaluation completed!"
echo "========================================="
