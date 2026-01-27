#!/bin/bash
# Evaluate MIND using Cascade/Reranking: Point-wise filter + Ranking rerank
#
# Usage:
#   bash scripts/eval_mind_cascade.sh <pointwise_model> <ranking_model> [top_k] [split]
#
# Example:
#   bash scripts/eval_mind_cascade.sh \
#       output_dir/sft_mind_pointwise_small_Qwen3-1.7B-Base_bs1024_ep3_neg1.0/final_checkpoint \
#       output_dir/sft_mind_ranking_small_Qwen3-1.7B-Base_bs1024_ep3/final_checkpoint \
#       10 dev

set -euo pipefail

POINTWISE_MODEL="${1:-}"
RANKING_MODEL="${2:-}"
TOP_K="${3:-10}"
SPLIT="${4:-dev}"

if [[ -z "${POINTWISE_MODEL}" ]] || [[ -z "${RANKING_MODEL}" ]]; then
    echo "Usage: $0 <pointwise_model> <ranking_model> [top_k] [split]"
    echo ""
    echo "Arguments:"
    echo "  pointwise_model: Path to point-wise SFT model"
    echo "  ranking_model:   Path to ranking SFT model"
    echo "  top_k:           Number of candidates to pass to ranking stage (default: 10)"
    echo "  split:           dev or test (default: dev)"
    exit 1
fi

# Configuration
MIND_ROOT="${MIND_ROOT:-../data/MIND}"
MIND_SIZE="${MIND_SIZE:-small}"
USE_ABSTRACT="${USE_ABSTRACT:-0}"
MAX_HISTORY="${MAX_HISTORY:-0}"
MAX_IMPRESSIONS="${MAX_IMPRESSIONS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FLASH_ATTN="${FLASH_ATTN:-1}"

# Construct paths
BEHAVIORS_PATH="${MIND_ROOT}/${SPLIT}/behaviors.tsv"
NEWS_PATH="${MIND_ROOT}/${SPLIT}/news.tsv"

echo "========================================="
echo "MIND Cascade/Reranking Evaluation"
echo "========================================="
echo "Point-wise: ${POINTWISE_MODEL}"
echo "Ranking:    ${RANKING_MODEL}"
echo "Top-K:      ${TOP_K}"
echo "Split:      ${SPLIT}"
echo "========================================="
echo ""

# Build command
CMD="python evaluate_mind_cascade.py \
    --pointwise_model ${POINTWISE_MODEL} \
    --ranking_model ${RANKING_MODEL} \
    --behaviors_path ${BEHAVIORS_PATH} \
    --news_path ${NEWS_PATH} \
    --top_k ${TOP_K} \
    --max_history ${MAX_HISTORY} \
    --batch_size ${BATCH_SIZE}"

if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
    CMD="${CMD} --use_abstract"
fi

if [[ "${MAX_IMPRESSIONS}" -gt 0 ]]; then
    CMD="${CMD} --max_impressions ${MAX_IMPRESSIONS}"
fi

if [[ "${FLASH_ATTN}" -eq 1 ]]; then
    CMD="${CMD} --flash_attn"
fi

eval "${CMD}"
