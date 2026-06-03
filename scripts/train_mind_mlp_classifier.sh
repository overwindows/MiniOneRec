#!/bin/bash
# Train frozen LLM + MLP classifier on MIND dataset.
# No DeepSpeed needed — backbone is frozen, only the MLP trains (single GPU).
#
# Usage:
#   bash scripts/train_mind_mlp_classifier.sh
#   MODEL_PATH=/path/to/model DATA_ROOT=/path/to/MIND bash scripts/train_mind_mlp_classifier.sh

set -e

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
DATA_ROOT=${DATA_ROOT:-data/MIND}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_dir}
OUTPUT_NAME=${OUTPUT_NAME:-mlp_classifier_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_NAME}"

BATCH_SIZE=${BATCH_SIZE:-64}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
MAX_HISTORY=${MAX_HISTORY:-30}
NEG_RATIO=${NEG_RATIO:-2.0}
HARD_NEG_RATIO=${HARD_NEG_RATIO:-0.5}
MLP_HIDDEN_DIM=${MLP_HIDDEN_DIM:-256}
CUTOFF_LEN=${CUTOFF_LEN:-2048}
USE_ABSTRACT=${USE_ABSTRACT:-0}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-3}
SEED=${SEED:-42}

# Offline mode if model path is a local directory
if [[ "$MODEL_PATH" == /* ]] || [[ -d "$MODEL_PATH" ]]; then
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "Local model detected — offline mode enabled"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

# ---------------------------------------------------------------------------
# Echo config
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  MLP Classifier Training"
echo "============================================================"
echo "  Model:        ${MODEL_PATH}"
echo "  Data root:    ${DATA_ROOT}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Batch size:   ${BATCH_SIZE}"
echo "  Epochs:       ${NUM_EPOCHS}"
echo "  LR:           ${LEARNING_RATE}"
echo "  MLP dim:      ${MLP_HIDDEN_DIM}"
echo "  Max history:  ${MAX_HISTORY}"
echo "  Neg ratio:    ${NEG_RATIO}"
echo "  Use abstract: ${USE_ABSTRACT}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
USE_ABSTRACT_FLAG=""
if [ "${USE_ABSTRACT}" = "1" ]; then
    USE_ABSTRACT_FLAG="--use_abstract True"
fi

python src/train_mind_mlp_classifier.py \
    --base_model "${MODEL_PATH}" \
    --train_behaviors_path "${DATA_ROOT}/train/behaviors.tsv" \
    --train_news_path      "${DATA_ROOT}/train/news.tsv" \
    --eval_behaviors_path  "${DATA_ROOT}/dev/behaviors.tsv" \
    --eval_news_path       "${DATA_ROOT}/dev/news.tsv" \
    --output_dir           "${OUTPUT_DIR}" \
    --batch_size           ${BATCH_SIZE} \
    --num_epochs           ${NUM_EPOCHS} \
    --learning_rate        ${LEARNING_RATE} \
    --max_history          ${MAX_HISTORY} \
    --neg_ratio            ${NEG_RATIO} \
    --hard_neg_ratio       ${HARD_NEG_RATIO} \
    --mlp_hidden_dim       ${MLP_HIDDEN_DIM} \
    --cutoff_len           ${CUTOFF_LEN} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --seed                 ${SEED} \
    ${USE_ABSTRACT_FLAG}

echo ""
echo "Training complete. Checkpoint at: ${OUTPUT_DIR}/best_mlp.pt"
echo ""
echo "To evaluate:"
echo "  bash scripts/eval_mind_mlp_classifier.sh ${MODEL_PATH} ${OUTPUT_DIR}/best_mlp.pt dev"
