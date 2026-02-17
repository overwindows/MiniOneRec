#!/usr/bin/env bash
# Checkpoint Sweep Evaluation for MIND RL Models
#
# Converts FSDP checkpoints to HuggingFace format and evaluates each,
# then reports a metrics table sorted by AUC to find the best checkpoint.
#
# USAGE:
#   # Evaluate specific checkpoints
#   bash scripts/eval_mind_sweep.sh output_dir/rl_mind_*/global_step_100 output_dir/rl_mind_*/global_step_500
#
#   # Evaluate all checkpoints in a directory (every Nth)
#   STEP=200 bash scripts/eval_mind_sweep.sh output_dir/rl_mind_pointwise_*/
#
#   # Quick mode (500 impressions per checkpoint)
#   QUICK=1 bash scripts/eval_mind_sweep.sh output_dir/rl_mind_*/
#
# ENVIRONMENT VARIABLES:
#   STEP:       Evaluate every N steps (default: evaluate all global_step_* dirs)
#   QUICK:      Use --quick flag for fast evaluation (default: 0)
#   MIND_SIZE:  Dataset size (default: small)
#   GPU:        GPU to use (default: 0)
#   BASE_MODEL: Base model for config/tokenizer (auto-detected if not set)

set -euo pipefail

# Configuration
MIND_SIZE="${MIND_SIZE:-small}"
GPU="${GPU:-0}"
QUICK="${QUICK:-0}"
STEP="${STEP:-0}"

if [[ -z "${MIND_ROOT:-}" ]]; then
    if [[ "${MIND_SIZE}" == "large" && -d "../data/MIND_large" ]]; then
        MIND_ROOT="../data/MIND_large"
    elif [[ "${MIND_SIZE}" == "small" && -d "../data/MIND_small" ]]; then
        MIND_ROOT="../data/MIND_small"
    else
        MIND_ROOT="../data/MIND"
    fi
fi

BEHAVIORS_PATH="${MIND_ROOT}/dev/behaviors.tsv"
NEWS_PATH="${MIND_ROOT}/dev/news.tsv"

if [[ ! -f "${BEHAVIORS_PATH}" ]] || [[ ! -f "${NEWS_PATH}" ]]; then
    echo "ERROR: MIND dev data not found at ${MIND_ROOT}/dev/" >&2
    exit 1
fi

# Collect checkpoint directories
RL_DIR="${1:-}"
if [[ -z "${RL_DIR}" ]]; then
    echo "Usage: $0 <rl_output_dir> [checkpoint_dir ...]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 output_dir/rl_mind_pointwise_small_*/" >&2
    echo "  STEP=200 $0 output_dir/rl_mind_pointwise_small_*/" >&2
    echo "  QUICK=1 $0 output_dir/rl_mind_pointwise_small_*/" >&2
    exit 1
fi

# Find checkpoint directories
CHECKPOINTS=()
for arg in "$@"; do
    if [[ -d "${arg}" ]]; then
        # If it's a global_step_* dir, use directly
        if [[ "$(basename "${arg}")" == global_step_* ]]; then
            CHECKPOINTS+=("${arg}")
        else
            # It's the RL output dir - find global_step_* subdirs
            for ckpt in "${arg}"/global_step_*; do
                if [[ -d "${ckpt}/actor" ]]; then
                    CHECKPOINTS+=("${ckpt}")
                fi
            done
        fi
    fi
done

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "ERROR: No checkpoints found" >&2
    exit 1
fi

# Sort by step number
IFS=$'\n' CHECKPOINTS=($(for c in "${CHECKPOINTS[@]}"; do echo "$c"; done | sort -t_ -k3 -n)); unset IFS

# Filter by STEP if set
if [[ "${STEP}" -gt 0 ]]; then
    FILTERED=()
    for ckpt in "${CHECKPOINTS[@]}"; do
        step_num=$(basename "${ckpt}" | sed 's/global_step_//')
        if (( step_num % STEP == 0 )); then
            FILTERED+=("${ckpt}")
        fi
    done
    CHECKPOINTS=("${FILTERED[@]}")
fi

echo "========================================="
echo "MIND Checkpoint Sweep Evaluation"
echo "========================================="
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "MIND: ${MIND_SIZE}"
echo "GPU: ${GPU}"
echo "Quick mode: ${QUICK}"
echo "========================================="
echo ""

# Auto-detect base model for config/tokenizer
BASE_MODEL="${BASE_MODEL:-}"
if [[ -z "${BASE_MODEL}" ]]; then
    # Try to find SFT checkpoint
    for candidate in output_dir/sft_mind_pointwise_*/final_checkpoint; do
        if [[ -d "${candidate}" ]] && [[ -f "${candidate}/config.json" ]]; then
            BASE_MODEL="${candidate}"
            break
        fi
    done
fi

if [[ -z "${BASE_MODEL}" ]]; then
    echo "ERROR: Could not auto-detect base model. Set BASE_MODEL env var." >&2
    exit 1
fi

echo "Base model: ${BASE_MODEL}"
echo ""

# Results storage
RESULTS_FILE="/tmp/mind_sweep_results_$$.tsv"
echo -e "Step\tAUC\tMRR\tnDCG@5\tnDCG@10\tCheckpoint" > "${RESULTS_FILE}"

QUICK_FLAG=""
if [[ "${QUICK}" -eq 1 ]]; then
    QUICK_FLAG="--quick"
fi

for ckpt in "${CHECKPOINTS[@]}"; do
    step_name=$(basename "${ckpt}")
    step_num=$(echo "${step_name}" | sed 's/global_step_//')
    hf_dir="${ckpt}_hf"

    echo "--- ${step_name} ---"

    # Convert if needed
    if [[ ! -f "${hf_dir}/model.safetensors" ]] && [[ ! -f "${hf_dir}/pytorch_model.bin" ]]; then
        echo "  Converting FSDP checkpoint..."
        python convert_verl_checkpoint.py \
            --checkpoint_dir "${ckpt}" \
            --output_dir "${hf_dir}" \
            --base_model "${BASE_MODEL}" 2>&1 | tail -2
    else
        echo "  HF checkpoint exists, skipping conversion"
    fi

    # Evaluate
    echo "  Evaluating..."
    metrics=$(CUDA_VISIBLE_DEVICES=${GPU} python -u src/evaluate_mind_pointwise.py \
        --model_path "${hf_dir}" \
        --behaviors_path "${BEHAVIORS_PATH}" \
        --news_path "${NEWS_PATH}" \
        --batch_size 8 \
        ${QUICK_FLAG} 2>&1 | grep -E "^(AUC|MRR|nDCG)" || echo "FAILED")

    if [[ "${metrics}" == "FAILED" ]]; then
        echo "  FAILED"
        echo -e "${step_num}\tFAILED\t-\t-\t-\t${ckpt}" >> "${RESULTS_FILE}"
    else
        auc=$(echo "${metrics}" | grep "^AUC:" | awk '{print $2}')
        mrr=$(echo "${metrics}" | grep "^MRR:" | awk '{print $2}')
        ndcg5=$(echo "${metrics}" | grep "^nDCG@5:" | awk '{print $2}')
        ndcg10=$(echo "${metrics}" | grep "^nDCG@10:" | awk '{print $2}')
        echo "  AUC=${auc} MRR=${mrr} nDCG@5=${ndcg5} nDCG@10=${ndcg10}"
        echo -e "${step_num}\t${auc}\t${mrr}\t${ndcg5}\t${ndcg10}\t${ckpt}" >> "${RESULTS_FILE}"
    fi

    echo ""
done

# Print results table sorted by AUC
echo "========================================="
echo "Results (sorted by AUC, descending)"
echo "========================================="
printf "%-8s %-8s %-8s %-8s %-8s %s\n" "Step" "AUC" "MRR" "nDCG@5" "nDCG@10" "Checkpoint"
echo "---------------------------------------------------------------------------"
tail -n +2 "${RESULTS_FILE}" | sort -t$'\t' -k2 -rn | while IFS=$'\t' read -r step auc mrr ndcg5 ndcg10 ckpt; do
    printf "%-8s %-8s %-8s %-8s %-8s %s\n" "${step}" "${auc}" "${mrr}" "${ndcg5}" "${ndcg10}" "${ckpt}"
done
echo "========================================="

# Find best
best_line=$(tail -n +2 "${RESULTS_FILE}" | sort -t$'\t' -k2 -rn | head -1)
best_step=$(echo "${best_line}" | cut -f1)
best_auc=$(echo "${best_line}" | cut -f2)
echo ""
echo "Best checkpoint: global_step_${best_step} (AUC=${best_auc})"

rm -f "${RESULTS_FILE}"
