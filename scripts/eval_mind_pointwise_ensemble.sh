#!/usr/bin/env bash
# Ensemble evaluation of multiple point-wise models on MIND.
#
# Models are scored sequentially (one at a time) to keep VRAM manageable.
# Final score = weighted average of per-model Yes/No log probs.
#
# USAGE:
#   # 2-model ensemble (equal weights)
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   bash scripts/eval_mind_pointwise_ensemble.sh \
#     output_dir/model_a/final_checkpoint \
#     output_dir/model_b/final_checkpoint
#
#   # 3-model ensemble with weights
#   WEIGHTS="1.0 0.8 0.8" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   bash scripts/eval_mind_pointwise_ensemble.sh \
#     output_dir/model_a/final_checkpoint \
#     output_dir/model_b/final_checkpoint \
#     output_dir/model_c/final_checkpoint
#
#   # With abstracts
#   USE_ABSTRACT=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   bash scripts/eval_mind_pointwise_ensemble.sh model_a model_b

set -euo pipefail

# All positional args are model paths
MODEL_PATHS=("$@")

if [[ ${#MODEL_PATHS[@]} -lt 2 ]]; then
  echo "Usage: $0 <model_path_1> <model_path_2> [model_path_3 ...]" >&2
  echo "" >&2
  echo "Environment variables:" >&2
  echo "  WEIGHTS         Space-separated weights per model (default: equal)" >&2
  echo "  MIND_SIZE       small or large (default: large)" >&2
  echo "  MIND_ROOT       Path to MIND dataset root" >&2
  echo "  SPLIT           dev or test (default: dev)" >&2
  echo "  USE_ABSTRACT    1 to include abstracts (default: 0)" >&2
  echo "  USE_CHAT_TEMPLATE  1 for instruct models (default: 1)" >&2
  echo "  MAX_HISTORY     Max history items, 0=unlimited (default: 0)" >&2
  echo "  BATCH_SIZE      Batch size per model (default: 8)" >&2
  echo "  OUTPUT_FILE     Path to save ranked predictions (optional)" >&2
  exit 1
fi

# Configuration
MIND_SIZE="${MIND_SIZE:-large}"
SPLIT="${SPLIT:-dev}"
USE_ABSTRACT="${USE_ABSTRACT:-0}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"
MAX_HISTORY="${MAX_HISTORY:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WEIGHTS="${WEIGHTS:-}"
MAX_IMPRESSIONS="${MAX_IMPRESSIONS:-0}"

if [[ -z "${MIND_ROOT:-}" ]]; then
  if [[ "${MIND_SIZE}" == "large" && -d "../data/MIND_large" ]]; then
    MIND_ROOT="../data/MIND_large"
  elif [[ "${MIND_SIZE}" == "small" && -d "../data/MIND_small" ]]; then
    MIND_ROOT="../data/MIND_small"
  else
    MIND_ROOT="../data/MIND"
  fi
fi

PYTHON="${PYTHON:-/home/aiscuser/.conda/envs/MiniOneRec/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

BEHAVIORS_PATH="${MIND_ROOT}/${SPLIT}/behaviors.tsv"
NEWS_PATH="${MIND_ROOT}/${SPLIT}/news.tsv"

# Default output file
if [[ -z "${OUTPUT_FILE:-}" ]]; then
  OUTPUT_FILE="./results_mind/${SPLIT}_ensemble_predictions.txt"
fi
mkdir -p "$(dirname "${OUTPUT_FILE}")"

echo "========================================="
echo "MIND Point-wise Ensemble Evaluation"
echo "========================================="
echo "Models (${#MODEL_PATHS[@]}):"
for i in "${!MODEL_PATHS[@]}"; do
  echo "  [$((i+1))] ${MODEL_PATHS[$i]}"
done
echo ""
echo "Dataset: MIND${MIND_SIZE} ${SPLIT}"
echo "Use abstract: ${USE_ABSTRACT}"
echo "Chat template: ${USE_CHAT_TEMPLATE}"
echo "Max history: ${MAX_HISTORY:-unlimited}"
echo "Batch size: ${BATCH_SIZE}"
if [[ -n "${WEIGHTS}" ]]; then
  echo "Weights: ${WEIGHTS}"
fi
echo "Output: ${OUTPUT_FILE}"
echo "========================================="
echo ""

if [[ ! -f "${BEHAVIORS_PATH}" ]]; then
  echo "ERROR: ${BEHAVIORS_PATH} not found" >&2
  exit 1
fi

# Build command
CMD="${PYTHON} src/evaluate_mind_pointwise_ensemble.py"

# Add model paths
CMD="${CMD} --model_paths"
for mp in "${MODEL_PATHS[@]}"; do
  CMD="${CMD} \"${mp}\""
done

CMD="${CMD} \
  --behaviors_path \"${BEHAVIORS_PATH}\" \
  --news_path \"${NEWS_PATH}\" \
  --max_history ${MAX_HISTORY} \
  --batch_size ${BATCH_SIZE} \
  --flash_attn \
  --output_file \"${OUTPUT_FILE}\""

if [[ "${USE_ABSTRACT}" -eq 1 ]]; then
  CMD="${CMD} --use_abstract"
fi

if [[ "${USE_CHAT_TEMPLATE}" -eq 1 ]]; then
  CMD="${CMD} --use_chat_template"
fi

if [[ -n "${WEIGHTS}" ]]; then
  CMD="${CMD} --weights ${WEIGHTS}"
fi

if [[ "${MAX_IMPRESSIONS}" -gt 0 ]]; then
  CMD="${CMD} --max_impressions ${MAX_IMPRESSIONS}"
fi

eval "${CMD}"
