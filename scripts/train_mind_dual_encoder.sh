#!/usr/bin/env bash
# Train the from-scratch contrastive dual-encoder on MIND.
#
# Env contract (set by pipeline):
#   TRAIN_BEHAVIORS, TRAIN_NEWS, EVAL_BEHAVIORS, EVAL_NEWS
#   OUTPUT_DIR, D_MODEL, NUM_EPOCHS, BATCH_SIZE, MAX_HISTORY,
#   MAX_NEWS_LEN, NUM_HIST, LR, USE_ABSTRACT, PER_POS_NEG, HISTORY_AS_NEG,
#   INFONCE_WEIGHT, BCE_WEIGHT, TEMPERATURE, DEVICE
set -euo pipefail

PYTHON="${PYTHON:-/home/aiscuser/.conda/envs/MiniOneRec/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

MODEL_BASENAME="${MODEL_BASENAME:-dual_encoder}"
OUT="${OUTPUT_DIR:-output_dir/dual_encoder}"

ARGS=(
  --train_behaviors "${TRAIN_BEHAVIORS}"
  --train_news      "${TRAIN_NEWS}"
  --output_dir      "${OUT}"
  --d_model         "${D_MODEL:-128}"
  --num_epochs      "${NUM_EPOCHS:-8}"
  --batch_size      "${BATCH_SIZE:-256}"
  --max_history     "${MAX_HISTORY:-50}"
  --max_news_len    "${MAX_NEWS_LEN:-64}"
  --num_hist        "${NUM_HIST:-20}"
  --lr              "${LR:-1e-3}"
  --per_pos_neg     "${PER_POS_NEG:-2}"
  --num_workers     "${NUM_WORKERS:-4}"
  --infonce_weight  "${INFONCE_WEIGHT:-1.0}"
  --bce_weight      "${BCE_WEIGHT:-1.0}"
  --temperature     "${TEMPERATURE:-0.07}"
)
if [[ "${USE_ABSTRACT:-0}" == "1" ]]; then
  ARGS+=(--use_abstract)
fi
if [[ "${HISTORY_AS_NEG:-0}" == "1" ]]; then
  ARGS+=(--history_as_neg)
fi

echo "Training dual-encoder → ${OUT}"
echo "Command: ${PYTHON} -u src/train_mind_dual_encoder.py ${ARGS[*]}"

mkdir -p "${OUT}"
"${PYTHON}" -u src/train_mind_dual_encoder.py "${ARGS[@]}"

echo ""
echo "[OK] Dual-encoder training complete. Checkpoints:"
ls -la "${OUT}"
