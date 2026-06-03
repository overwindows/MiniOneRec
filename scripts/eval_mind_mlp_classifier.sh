#!/bin/bash
# Evaluate frozen LLM + MLP classifier on MIND dataset.
#
# Usage:
#   bash scripts/eval_mind_mlp_classifier.sh <model_path> <mlp_checkpoint> [dev|test]
#
# Examples:
#   bash scripts/eval_mind_mlp_classifier.sh Qwen/Qwen3-1.7B output_dir/mlp_classifier/best_mlp.pt dev
#   bash scripts/eval_mind_mlp_classifier.sh /path/to/model  /path/to/best_mlp.pt              test

set -e

MODEL_PATH=${1:-${MODEL_PATH:-Qwen/Qwen3-1.7B}}
MLP_CHECKPOINT=${2:-${MLP_CHECKPOINT:-""}}
SPLIT=${3:-${EVAL_SPLIT:-dev}}

if [ -z "$MLP_CHECKPOINT" ]; then
    echo "ERROR: mlp_checkpoint is required (arg 2 or MLP_CHECKPOINT env var)"
    exit 1
fi

MIND_ROOT=${MIND_ROOT:-data/MIND}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_HISTORY=${MAX_HISTORY:-0}
USE_ABSTRACT=${USE_ABSTRACT:-0}

# Offline mode if model path is local
if [[ "$MODEL_PATH" == /* ]] || [[ -d "$MODEL_PATH" ]]; then
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "  MLP Classifier Evaluation"
echo "============================================================"
echo "  Backbone:      ${MODEL_PATH}"
echo "  MLP ckpt:      ${MLP_CHECKPOINT}"
echo "  Split:         ${SPLIT}"
echo "  MIND root:     ${MIND_ROOT}"
echo "  Batch size:    ${BATCH_SIZE}"
echo "============================================================"

USE_ABSTRACT_FLAG=""
if [ "${USE_ABSTRACT}" = "1" ]; then
    USE_ABSTRACT_FLAG="--use_abstract"
fi

python src/evaluate_mind_mlp_classifier.py \
    --model_path       "${MODEL_PATH}" \
    --mlp_checkpoint   "${MLP_CHECKPOINT}" \
    --behaviors_path   "${MIND_ROOT}/${SPLIT}/behaviors.tsv" \
    --news_path        "${MIND_ROOT}/${SPLIT}/news.tsv" \
    --max_history      ${MAX_HISTORY} \
    --batch_size       ${BATCH_SIZE} \
    --flash_attn \
    ${USE_ABSTRACT_FLAG}
