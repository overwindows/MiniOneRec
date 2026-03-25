#!/usr/bin/env bash
# Ensemble evaluation of multiple point-wise models on MIND.
#
# Architecture (fast, parallel):
#   - Available GPUs are split evenly across models (e.g. 8 GPUs → 4+4 for 2 models)
#   - Each model runs its own parallel eval simultaneously as a background job
#   - ensemble_from_scores.py averages the raw scores → metrics
#
# Total time ≈ single-model eval time (not N × single-model eval time)
#
# Per-model use_chat_template and use_abstract are AUTO-DETECTED from the
# checkpoint directory name (_chat → chat=1, _abstract → abstract=1).
# Override with CHAT_TEMPLATES="1 0 1" and ABSTRACTS="1 0 0" if needed.
#
# USAGE:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   bash scripts/eval_mind_pointwise_ensemble.sh \
#     output_dir/.../abstract_chat/final_checkpoint \
#     output_dir/.../ep7_chat/final_checkpoint
#
#   # 3-model ensemble (GPUs split 3+3+2)
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   bash scripts/eval_mind_pointwise_ensemble.sh model_a model_b model_c

set -euo pipefail

MODEL_PATHS=("$@")
N_MODELS=${#MODEL_PATHS[@]}

if [[ ${N_MODELS} -lt 2 ]]; then
  echo "Usage: $0 <model_path_1> <model_path_2> [model_path_3 ...]" >&2
  echo "" >&2
  echo "Environment variables:" >&2
  echo "  CUDA_VISIBLE_DEVICES  GPUs to use, split evenly across models (e.g. 0,1,2,3,4,5,6,7)" >&2
  echo "  CHAT_TEMPLATES        Space-separated per-model chat_template flags (default: auto-detect)" >&2
  echo "  ABSTRACTS             Space-separated per-model abstract flags (default: auto-detect)" >&2
  echo "  WEIGHTS               Space-separated per-model weights (default: equal)" >&2
  echo "  MIND_SIZE             small or large (default: large)" >&2
  echo "  MIND_ROOT             Path to MIND dataset root" >&2
  echo "  SPLIT                 dev or test (default: dev)" >&2
  echo "  MAX_HISTORY           Max history items, 0=unlimited (default: 0)" >&2
  echo "  BATCH_SIZE            Batch size per GPU process (default: 8)" >&2
  echo "  OUTPUT_FILE           Path to save ranked predictions (optional)" >&2
  exit 1
fi

# Configuration
MIND_SIZE="${MIND_SIZE:-large}"
SPLIT="${SPLIT:-dev}"
MAX_HISTORY="${MAX_HISTORY:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WEIGHTS="${WEIGHTS:-}"
CHAT_TEMPLATES="${CHAT_TEMPLATES:-}"
ABSTRACTS="${ABSTRACTS:-}"
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

if [[ -z "${OUTPUT_FILE:-}" ]]; then
  OUTPUT_FILE="./results_mind/${SPLIT}_ensemble_predictions.txt"
fi

TEMP_DIR="./temp_ensemble_scores/$$"
mkdir -p "${TEMP_DIR}"

# ── Split GPUs evenly across models ─────────────────────────────────────────
ALL_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPU_LIST <<< "${ALL_GPUS}"
N_GPUS=${#GPU_LIST[@]}

# Distribute GPUs: model i gets floor(N_GPUS/N_MODELS) or ceil
declare -a MODEL_GPUS
gpu_idx=0
for (( i=0; i<N_MODELS; i++ )); do
  remaining_models=$(( N_MODELS - i ))
  remaining_gpus=$(( N_GPUS - gpu_idx ))
  count=$(( (remaining_gpus + remaining_models - 1) / remaining_models ))
  gpu_subset=""
  for (( j=0; j<count && gpu_idx<N_GPUS; j++ )); do
    [[ -n "${gpu_subset}" ]] && gpu_subset="${gpu_subset},"
    gpu_subset="${gpu_subset}${GPU_LIST[$gpu_idx]}"
    (( gpu_idx++ ))
  done
  MODEL_GPUS[$i]="${gpu_subset}"
done

# Parse per-model flag overrides
IFS=' ' read -ra CHAT_ARR <<< "${CHAT_TEMPLATES}"
IFS=' ' read -ra ABSTRACT_ARR <<< "${ABSTRACTS}"

echo "========================================="
echo "MIND Point-wise Ensemble Evaluation"
echo "========================================="
echo "Models: ${N_MODELS}  |  Total GPUs: ${N_GPUS}"
echo ""
for i in "${!MODEL_PATHS[@]}"; do
  echo "  [Model $((i+1))]  GPUs: ${MODEL_GPUS[$i]}"
  echo "            ${MODEL_PATHS[$i]}"
done
echo ""
echo "Dataset:  MIND${MIND_SIZE} ${SPLIT}"
echo "Strategy: ${N_MODELS} parallel evals, GPU split $(IFS='/'; echo "${MODEL_GPUS[*]}" | tr ',' 'x' | sed 's/x[^ ]*/&GPU /g')"
echo "========================================="
echo ""

if [[ ! -f "${BEHAVIORS_PATH}" ]]; then
  echo "ERROR: ${BEHAVIORS_PATH} not found" >&2
  exit 1
fi

# ── Step 1: Launch all model evals in parallel ───────────────────────────────
pids=()
LOG_FILES=()

for i in "${!MODEL_PATHS[@]}"; do
  MODEL_PATH="${MODEL_PATHS[$i]}"
  MODEL_NUM=$((i+1))
  SCORE_FILE="${TEMP_DIR}/model_${MODEL_NUM}_scores.txt"
  LOG_FILE="${TEMP_DIR}/model_${MODEL_NUM}.log"
  LOG_FILES+=("${LOG_FILE}")

  # Auto-detect flags from checkpoint dir name
  DIR_NAME=$(basename "$(dirname "${MODEL_PATH}")")
  if [[ "${DIR_NAME}" == "final_checkpoint" ]] || [[ "${DIR_NAME}" == checkpoint-* ]]; then
    DIR_NAME=$(basename "$(dirname "$(dirname "${MODEL_PATH}")")")
  fi

  if [[ ${#CHAT_ARR[@]} -gt $i ]] && [[ -n "${CHAT_ARR[$i]:-}" ]]; then
    USE_CHAT="${CHAT_ARR[$i]}"
  elif [[ "${DIR_NAME}" == *_chat* ]]; then
    USE_CHAT=1
  else
    USE_CHAT=0
  fi

  if [[ ${#ABSTRACT_ARR[@]} -gt $i ]] && [[ -n "${ABSTRACT_ARR[$i]:-}" ]]; then
    USE_ABS="${ABSTRACT_ARR[$i]}"
  elif [[ "${DIR_NAME}" == *_abstract* ]]; then
    USE_ABS=1
  else
    USE_ABS=0
  fi

  echo "[Model ${MODEL_NUM}] Launching on GPUs ${MODEL_GPUS[$i]}"
  echo "  chat=${USE_CHAT}  abstract=${USE_ABS}  → ${SCORE_FILE}"

  CUDA_VISIBLE_DEVICES="${MODEL_GPUS[$i]}" \
  USE_CHAT_TEMPLATE=${USE_CHAT} \
  USE_ABSTRACT=${USE_ABS} \
  MAX_HISTORY=${MAX_HISTORY} \
  BATCH_SIZE=${BATCH_SIZE} \
  OUTPUT_SCORES_FILE=${SCORE_FILE} \
  MIND_SIZE=${MIND_SIZE} \
  MIND_ROOT=${MIND_ROOT} \
  bash scripts/eval_mind_pointwise.sh \
    "${MODEL_PATH}" "${SPLIT}" "${MAX_IMPRESSIONS:-0}" \
    > "${LOG_FILE}" 2>&1 &

  pids+=($!)
done

echo ""
echo "All ${N_MODELS} models running in parallel — waiting for completion..."
echo "(Tailing logs: tail -f ${TEMP_DIR}/model_*.log)"
echo ""

# ── Step 2: Wait for all, stream logs ────────────────────────────────────────
failed=()
for i in "${!pids[@]}"; do
  pid=${pids[$i]}
  model_num=$((i+1))
  if wait "${pid}"; then
    echo "✓ Model ${model_num} completed"
  else
    echo "✗ Model ${model_num} FAILED (see ${LOG_FILES[$i]})"
    failed+=("${model_num}")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "ERROR: Model(s) failed: ${failed[*]}" >&2
  # Print last 30 lines of failed logs
  for i in "${!pids[@]}"; do
    model_num=$((i+1))
    if [[ " ${failed[*]} " == *" ${model_num} "* ]]; then
      echo "--- Log for model ${model_num} ---"
      tail -30 "${LOG_FILES[$i]}"
    fi
  done
  rm -rf "${TEMP_DIR}"
  exit 1
fi

# ── Step 3: Average scores and compute metrics ────────────────────────────────
echo ""
echo "========================================="
echo "Averaging scores across ${N_MODELS} models..."
echo "========================================="

SCORE_FILES=()
for i in "${!MODEL_PATHS[@]}"; do
  SCORE_FILES+=("${TEMP_DIR}/model_$((i+1))_scores.txt")
done

CMD="${PYTHON} src/ensemble_from_scores.py --score_files"
for sf in "${SCORE_FILES[@]}"; do
  CMD="${CMD} \"${sf}\""
done
CMD="${CMD} --behaviors_path \"${BEHAVIORS_PATH}\" --output_file \"${OUTPUT_FILE}\""
if [[ -n "${WEIGHTS}" ]]; then
  CMD="${CMD} --weights ${WEIGHTS}"
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"
eval "${CMD}"

rm -rf "${TEMP_DIR}"

echo ""
echo "========================================="
echo "✓ Ensemble evaluation complete!"
echo "Predictions: ${OUTPUT_FILE}"
echo "========================================="
