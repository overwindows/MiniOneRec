#!/bin/bash

# =========================
# NCCL Configuration
# =========================
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=INIT,NET

if [ -n "$NCCL_SOCKET_IFNAME" ]; then
    echo "Using specified NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
else
    if ip link show ib0 &>/dev/null; then
        export NCCL_SOCKET_IFNAME=ib0
        export NCCL_IB_DISABLE=0
        echo "Detected InfiniBand interface: ib0"
    elif ip link show eth0 &>/dev/null; then
        export NCCL_SOCKET_IFNAME=eth0
        export NCCL_IB_DISABLE=1
        echo "Using ethernet interface: eth0"
    else
        echo "No specific interface found, letting NCCL auto-detect"
        export NCCL_IB_DISABLE=1
    fi
fi

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}

# =========================
# PyTorch Distributed
# =========================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "${MODEL_PATH:-}" == /* ]]; then
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_CUDA_ARCH_LIST="8.0"
export TORCH_DISTRIBUTED_TIMEOUT_SEC=${TORCH_DISTRIBUTED_TIMEOUT_SEC:-3600}
export NCCL_LAUNCH_MODE=PARALLEL

# =========================
# CPU / Threading
# =========================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# =========================
# CUDA
# =========================
export CUDA_DEVICE_MAX_CONNECTIONS=1

# WandB (disabled on AML nodes by default)
if [ "${MINIONEREC_ENABLE_WANDB:-0}" = "1" ]; then
    export WANDB_API_KEY="${WANDB_API_KEY:-fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b}"
    export WANDB_MODE=online
    echo "WandB: online mode (MINIONEREC_ENABLE_WANDB=1)"
else
    export WANDB_MODE=offline
    unset WANDB_API_KEY
    echo "WandB: DISABLED"
fi

export MASTER_PORT=29503

export PATH="$HOME/.conda/envs/MiniOneRec/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate MiniOneRec 2>/dev/null || true

if [ -z "$HOSTFILE" ]; then
    if [ -f "/job/hostfile" ]; then
        HOSTFILE="/job/hostfile"
        echo "Using multi-node hostfile: /job/hostfile"
    else
        HOSTFILE="./hostfile"
    fi
fi

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
DATA_ROOT=${DATA_ROOT:-/home/aiscuser/MiniOneRec/data/MIND}

MIND_SIZE=${MIND_SIZE:-}
if [[ -z "${MIND_SIZE}" ]]; then
    if [[ "${DATA_ROOT}" == *"large"* ]] || [[ "${DATA_ROOT}" == *"MIND_large"* ]]; then
        MIND_SIZE="large"
    else
        MIND_SIZE="small"
    fi
fi

# Hyperparameters (A7.3 default matches L1.3 pointwise settings + pairwise term)
BATCH_SIZE=${BATCH_SIZE:-256}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
CUTOFF_LEN=${CUTOFF_LEN:-8192}
NUM_EPOCHS=${NUM_EPOCHS:-5}
MAX_HISTORY=${MAX_HISTORY:-30}
NEG_RATIO=${NEG_RATIO:-2.0}
HARD_NEG_RATIO=${HARD_NEG_RATIO:-0.5}
TRAIN_SAMPLE=${TRAIN_SAMPLE:-0}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-3}
# A7.3 pairwise (Bradley-Terry) loss weight lambda
PAIRWISE_LOSS_WEIGHT=${PAIRWISE_LOSS_WEIGHT:-0.5}
USE_ABSTRACT=${USE_ABSTRACT:-0}
if [[ "$USE_ABSTRACT" == "1" || "$USE_ABSTRACT" == "True" || "$USE_ABSTRACT" == "true" ]]; then
    USE_ABSTRACT="True"
else
    USE_ABSTRACT="False"
fi
USE_SUBCATEGORY=${USE_SUBCATEGORY:-0}
if [[ "$USE_SUBCATEGORY" == "1" || "$USE_SUBCATEGORY" == "True" || "$USE_SUBCATEGORY" == "true" ]]; then
    USE_SUBCATEGORY="True"
else
    USE_SUBCATEGORY="False"
fi
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-0}"
DS_CONFIG=${DS_CONFIG:-ds_configs/ds_config_zero2.json}
if [[ ! "$DS_CONFIG" = /* ]]; then
    DS_CONFIG="$(pwd)/$DS_CONFIG"
fi
DS_LAUNCHER=${DS_LAUNCHER:-pdsh}

if [ ! -f "$HOSTFILE" ]; then
    echo "Creating default hostfile for single-node training..."
    echo "localhost slots=8" > $HOSTFILE
fi

echo "Using hostfile: $HOSTFILE"
cat $HOSTFILE

TRAIN_BEHAVIORS=${DATA_ROOT}/train/behaviors.tsv
TRAIN_NEWS=${DATA_ROOT}/train/news.tsv
EVAL_BEHAVIORS=${DATA_ROOT}/dev/behaviors.tsv
EVAL_NEWS=${DATA_ROOT}/dev/news.tsv

MODEL_BASENAME=$(basename ${MODEL_PATH})
OUTPUT_NAME="sft_mind_pointwise_pairwise_${MIND_SIZE}_${MODEL_BASENAME}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_neg${NEG_RATIO}_hist${MAX_HISTORY}_pw${PAIRWISE_LOSS_WEIGHT}"
if [[ "${USE_ABSTRACT}" == "True" ]]; then
    OUTPUT_NAME="${OUTPUT_NAME}_abstract"
fi
if [[ "${USE_SUBCATEGORY}" == "True" ]]; then
    OUTPUT_NAME="${OUTPUT_NAME}_subcat"
fi
if [[ "${USE_CHAT_TEMPLATE}" -eq 1 ]]; then
    OUTPUT_NAME="${OUTPUT_NAME}_chat"
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/${OUTPUT_NAME}"
else
    OUTPUT_DIR="output_dir/${OUTPUT_NAME}"
fi
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${OUTPUT_NAME}}

echo "DATA_ROOT: ${DATA_ROOT}"
echo "MIND size: ${MIND_SIZE}"
echo "Pairwise (BT) loss weight: ${PAIRWISE_LOSS_WEIGHT}"
echo "Chat template: $(if [[ "${USE_CHAT_TEMPLATE}" -eq 1 ]]; then echo "enabled"; else echo "disabled"; fi)"

export PDSH_RCMD_TYPE=ssh

RESUME_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${RESUME_CHECKPOINT:-}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

deepspeed --hostfile=$HOSTFILE \
        --master_port=${MASTER_PORT} \
        --launcher=${DS_LAUNCHER} \
        ${DS_LAUNCHER_ARGS:+--launcher_args="$DS_LAUNCHER_ARGS"} \
        src/sft_mind_pointwise_pairwise.py \
        --base_model ${MODEL_PATH} \
        --train_behaviors_path ${TRAIN_BEHAVIORS} \
        --train_news_path ${TRAIN_NEWS} \
        --eval_behaviors_path ${EVAL_BEHAVIORS} \
        --eval_news_path ${EVAL_NEWS} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${BATCH_SIZE} \
        --micro_batch_size ${MICRO_BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --cutoff_len ${CUTOFF_LEN} \
        --max_history ${MAX_HISTORY} \
        --neg_ratio ${NEG_RATIO} \
        --hard_neg_ratio ${HARD_NEG_RATIO} \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
        --sample ${TRAIN_SAMPLE} \
        --use_abstract ${USE_ABSTRACT} \
        --use_subcategory ${USE_SUBCATEGORY} \
        --pairwise_loss_weight ${PAIRWISE_LOSS_WEIGHT} \
        $(if [[ "${USE_CHAT_TEMPLATE}" -eq 1 ]]; then echo "--use_chat_template True"; fi) \
        $(if [ "${MINIONEREC_ENABLE_WANDB:-0}" = "1" ]; then echo "--wandb_project MiniOneRec_MIND --wandb_run_name ${WANDB_RUN_NAME}"; fi) \
        --train_from_scratch False \
        --seed 42 \
        --deepspeed_config ${DS_CONFIG} \
        ${RESUME_CHECKPOINT:+--resume_from_checkpoint $RESUME_CHECKPOINT} \
        ${WANDB_RUN_ID:+--wandb_run_id $WANDB_RUN_ID}

TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "Training FAILED with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi
echo "Training completed!"
