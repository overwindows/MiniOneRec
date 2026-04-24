#!/bin/bash
# =========================
# DOCA Listwise Eval Script
# =========================
# Usage:
#   bash scripts/eval_doca_listwise.sh <checkpoint_path> [--quick]
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_doca_listwise.sh output_dir/sft_doca_listwise_*/final_checkpoint

CHECKPOINT=${1:?Usage: bash scripts/eval_doca_listwise.sh <checkpoint_path> [extra_args...]}
shift

DATA_ROOT=${DATA_ROOT:-data/doca}
EVAL_JSONL=${DATA_ROOT}/dev.jsonl
MAX_FEEDS=${MAX_FEEDS:-0}

echo "Evaluating listwise model: $CHECKPOINT"
echo "Eval data: $EVAL_JSONL"

python src/evaluate_doca_listwise.py \
    --model_path "$CHECKPOINT" \
    --eval_jsonl "$EVAL_JSONL" \
    --max_feeds $MAX_FEEDS \
    --flash_attn \
    "$@"
