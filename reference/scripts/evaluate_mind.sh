#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-"output_dir/sft_text_mixed/final_checkpoint"}
MIND_ROOT=${2:-"/path/to/MIND"}
SPLIT=${3:-"dev"}

BEHAVIORS="$MIND_ROOT/${SPLIT}/behaviors.tsv"
NEWS="$MIND_ROOT/${SPLIT}/news.tsv"

python evaluate_mind.py \
  --model_path "$MODEL_PATH" \
  --behaviors_path "$BEHAVIORS" \
  --news_path "$NEWS"
