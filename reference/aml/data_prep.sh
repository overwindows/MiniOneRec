#!/usr/bin/env bash
set -euo pipefail

DATASET_VERSION="${DATASET_VERSION:-amazon18}"
DATASET_NAME="${DATASET_NAME:-Industrial_and_Scientific}"
USER_K="${USER_K:-5}"
ITEM_K="${ITEM_K:-5}"
ST_YEAR="${ST_YEAR:-1996}"
ST_MONTH="${ST_MONTH:-10}"
ED_YEAR="${ED_YEAR:-2018}"
ED_MONTH="${ED_MONTH:-11}"
METADATA_FILE="${METADATA_FILE:-}"
REVIEWS_FILE="${REVIEWS_FILE:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./data}"

run_amazon18() {
  args=(
    --dataset "${DATASET_NAME}"
    --user_k "${USER_K}"
    --item_k "${ITEM_K}"
    --st_year "${ST_YEAR}"
    --st_month "${ST_MONTH}"
    --ed_year "${ED_YEAR}"
    --ed_month "${ED_MONTH}"
    --output_path "${OUTPUT_PATH}"
  )
  if [[ -n "${METADATA_FILE}" ]]; then
    args+=(--metadata_file "${METADATA_FILE}")
  fi
  if [[ -n "${REVIEWS_FILE}" ]]; then
    args+=(--reviews_file "${REVIEWS_FILE}")
  fi
  python data/amazon18_data_process.py "${args[@]}"
}

run_amazon23() {
  if [[ -z "${METADATA_FILE}" || -z "${REVIEWS_FILE}" ]]; then
    echo "amazon23 requires METADATA_FILE and REVIEWS_FILE." >&2
    exit 1
  fi
  python data/amazon23_data_process.py \
    --dataset "${DATASET_NAME}" \
    --user_k "${USER_K}" \
    --st_year "${ST_YEAR}" \
    --st_month "${ST_MONTH}" \
    --ed_year "${ED_YEAR}" \
    --ed_month "${ED_MONTH}" \
    --metadata_file "${METADATA_FILE}" \
    --reviews_file "${REVIEWS_FILE}" \
    --output_path "${OUTPUT_PATH}"
}

case "${DATASET_VERSION}" in
  amazon18)
    run_amazon18
    ;;
  amazon23)
    run_amazon23
    ;;
  *)
    echo "Unknown DATASET_VERSION: ${DATASET_VERSION} (expected amazon18 or amazon23)" >&2
    exit 1
    ;;
esac
