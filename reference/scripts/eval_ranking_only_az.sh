#!/usr/bin/env bash
set -euo pipefail

# Evaluate MIND ranking model trained with A-Z options.
#
# Usage:
#   bash scripts/eval_ranking_only_az.sh /path/to/model dev
#   bash scripts/eval_ranking_only_az.sh /path/to/model dev 100

MODEL_PATH=${1:-""}
SPLIT=${2:-"dev"}
MAX_IMPRESSIONS=${3:-0}

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: bash scripts/eval_ranking_only_az.sh /path/to/model [split] [max_impressions]" >&2
  exit 1
fi

python eval_ranking_only_az.py \
  --model_path "${MODEL_PATH}" \
  --split "${SPLIT}" \
  --max_impressions "${MAX_IMPRESSIONS}"
