#!/usr/bin/env bash
set -euo pipefail

# Get the directory of this script and the repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Usage:
#   ./run_cloud_eval.sh pointwise
#   ./run_cloud_eval.sh listwise
#   ./run_cloud_eval.sh selection
#
# Required env vars:
#   SAMBANOVA_API_KEY
#
# Optional env vars (defaults shown):
#   SAMBANOVA_BASE_URL="https://api.sambanova.ai/v1"
#   SAMBANOVA_MODEL="DeepSeek-V3.1"
#   ROOT_PATH="../data/MIND/dev"
#   USE_ABSTRACT="false"   # true/false
#   MAX_HISTORY=0
#   MAX_IMPRESSIONS=0
#   TEMPERATURE=0.1
#   TOP_P=0.1
#   MAX_TOKENS=0            # 0=auto (128 for pointwise, 512 for selection, 4096 for listwise)
#   TOP_K=5                 # Top-K for listwise (5=default, 0=rank all candidates)
#   OUTPUT_FILE=""          # e.g. outputs/preds.tsv

MODE="${1:-}"  # pointwise | listwise | selection
if [[ -z "$MODE" ]]; then
  echo "Usage: $0 pointwise|listwise|selection" >&2
  exit 1
fi

if [[ "$MODE" != "pointwise" && "$MODE" != "listwise" && "$MODE" != "selection" ]]; then
  echo "Mode must be 'pointwise', 'listwise', or 'selection'" >&2
  exit 1
fi

: "${SAMBANOVA_API_KEY:?SAMBANOVA_API_KEY is required}"

SAMBANOVA_BASE_URL="${SAMBANOVA_BASE_URL:-https://api.sambanova.ai/v1}"
SAMBANOVA_MODEL="${SAMBANOVA_MODEL:-DeepSeek-V3.1}"
ROOT_PATH="${ROOT_PATH:-../data/MIND}"
BEHAVIORS_PATH="${ROOT_PATH}/dev/behaviors.tsv"
NEWS_PATH="${ROOT_PATH}/dev/news.tsv"
USE_ABSTRACT="${USE_ABSTRACT:-false}"
MAX_HISTORY="${MAX_HISTORY:-0}"
MAX_IMPRESSIONS="${MAX_IMPRESSIONS:-0}"
TEMPERATURE="${TEMPERATURE:-0.1}"
TOP_P="${TOP_P:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-0}"  # 0=auto (128 for pointwise, 512 for selection, 4096 for listwise)
TOP_K="${TOP_K:-5}"  # Top-K for listwise (5=default, 0=rank all)
OUTPUT_FILE="${OUTPUT_FILE:-}"

ARGS=(
  --mode "$MODE"
  --model "$SAMBANOVA_MODEL"
  --behaviors_path "$BEHAVIORS_PATH"
  --news_path "$NEWS_PATH"
  --api_key "$SAMBANOVA_API_KEY"
  --base_url "$SAMBANOVA_BASE_URL"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --max_tokens "$MAX_TOKENS"
  --top_k "$TOP_K"
  --max_history "$MAX_HISTORY"
  --max_impressions "$MAX_IMPRESSIONS"
)

if [[ "$USE_ABSTRACT" == "true" ]]; then
  ARGS+=(--use_abstract)
fi

if [[ -n "$OUTPUT_FILE" ]]; then
  ARGS+=(--output_file "$OUTPUT_FILE")
fi

python "$REPO_ROOT/src/evaluate_mind_cloud.py" "${ARGS[@]}"
