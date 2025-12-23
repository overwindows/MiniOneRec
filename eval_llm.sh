#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${1:-}"
TASKS="${2:-mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval}"
OUTPUT_DIR="${3:-llm_eval}"

if [[ -z "${MODEL_ROOT}" ]]; then
  echo "Usage: $0 <model_or_output_dir> [tasks] [output_dir]" >&2
  exit 1
fi

if [[ -d "${MODEL_ROOT}" && -d "${MODEL_ROOT}/checkpoint-0" ]]; then
  for ckpt in "${MODEL_ROOT}"/checkpoint-*; do
    if [[ -d "${ckpt}" ]]; then
      python llm_eval.py --model_path "${ckpt}" --tasks "${TASKS}" --output_dir "${OUTPUT_DIR}"
    fi
  done
else
  python llm_eval.py --model_path "${MODEL_ROOT}" --tasks "${TASKS}" --output_dir "${OUTPUT_DIR}"
fi
