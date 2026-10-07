#!/bin/bash

# =========================
# NCCL / Distributed config (same as sft_mind_pointwise_ds.sh)
# =========================
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=INIT,NET
if [ -n "$NCCL_SOCKET_IFNAME" ]; then
    echo "Using specified NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
else
    if ip link show ib0 &>/dev/null; then
        export NCCL_SOCKET_IFNAME=ib0; export NCCL_IB_DISABLE=0
    elif ip link show eth0 &>/dev/null; then
        export NCCL_SOCKET_IFNAME=eth0; export NCCL_IB_DISABLE=1
    else
        export NCCL_IB_DISABLE=1
    fi
fi
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "${MODEL_PATH:-}" == /* ]]; then
    export TRANSFORMERS_OFFLINE=1; export HF_DATASETS_OFFLINE=1
fi
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_CUDA_ARCH_LIST="8.0"
export TORCH_DISTRIBUTED_TIMEOUT_SEC=${TORCH_DISTRIBUTED_TIMEOUT_SEC:-3600}
export NCCL_LAUNCH_MODE=PARALLEL
export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1

# WandB disabled by default (nodes network-isolated).
if [ "${MINIONEREC_ENABLE_WANDB:-0}" = "1" ]; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline; unset WANDB_API_KEY
fi

export MASTER_PORT=29504
export PATH="$HOME/.conda/envs/MiniOneRec/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate MiniOneRec 2>/dev/null || true

if [ -z "$HOSTFILE" ]; then
    if [ -f "/job/hostfile" ]; then HOSTFILE="/job/hostfile"
    else HOSTFILE="./hostfile"
    fi
fi
[ -f "$HOSTFILE" ] || echo "localhost slots=8" > "$HOSTFILE"
echo "Using hostfile: $HOSTFILE"; cat "$HOSTFILE"

MODEL_PATH=${MODEL_PATH:-APUS-OpenJev-v1-4B}
DATA_ROOT=${DATA_ROOT:-/home/aiscuser/MiniOneRec/data/MIND}
MIND_SIZE=${MIND_SIZE:-}
if [[ -z "${MIND_SIZE}" ]]; then
    if [[ "${DATA_ROOT}" == *"large"* ]] || [[ "${DATA_ROOT}" == *"MIND_large"* ]]; then MIND_SIZE="large"; else MIND_SIZE="small"; fi
fi

BATCH_SIZE=${BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-4}     # JEV published lr
CUTOFF_LEN=${CUTOFF_LEN:-4096}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_HISTORY=${MAX_HISTORY:-30}
NEG_RATIO=${NEG_RATIO:-1.0}
HARD_NEG_RATIO=${HARD_NEG_RATIO:-0.5}
KL_BETA=${KL_BETA:-0.1}
ENABLE_KL=${ENABLE_KL:-1}
TRAIN_SAMPLE=${TRAIN_SAMPLE:--1}
USE_ABSTRACT=${USE_ABSTRACT:-1}
[[ "$USE_ABSTRACT" == "1" || "$USE_ABSTRACT" == "True" || "$USE_ABSTRACT" == "true" ]] && USE_ABSTRACT="True" || USE_ABSTRACT="False"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-0}"
DS_CONFIG=${DS_CONFIG:-ds_configs/ds_config_zero2.json}
[[ ! "$DS_CONFIG" = /* ]] && DS_CONFIG="$(pwd)/$DS_CONFIG"
DS_LAUNCHER=${DS_LAUNCHER:-pdsh}

TRAIN_BEHAVIORS=${DATA_ROOT}/train/behaviors.tsv
TRAIN_NEWS=${DATA_ROOT}/train/news.tsv
EVAL_BEHAVIORS=${DATA_ROOT}/dev/behaviors.tsv
EVAL_NEWS=${DATA_ROOT}/dev/news.tsv

WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
OUTPUT_BASENAME="sft_jev4b_${MIND_SIZE}_bs${BATCH_SIZE}_ep${NUM_EPOCHS}_neg${NEG_RATIO}_hist${MAX_HISTORY}_kl${KL_BETA}"
if [[ "${USE_ABSTRACT}" == "True" ]]; then OUTPUT_BASENAME="${OUTPUT_BASENAME}_abstract"; fi
if [[ "${USE_CHAT_TEMPLATE}" -eq 1 ]]; then OUTPUT_BASENAME="${OUTPUT_BASENAME}_chat"; fi
if [[ -n "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/${OUTPUT_BASENAME}"
else
    OUTPUT_DIR="output_dir/${OUTPUT_BASENAME}"
fi
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${OUTPUT_BASENAME}}"

echo "DATA_ROOT: ${DATA_ROOT}  MIND: ${MIND_SIZE}"
echo "KL_BETA: ${KL_BETA}  ENABLE_KL: ${ENABLE_KL}"
echo "MODEL_PATH: ${MODEL_PATH}"

export PDSH_RCMD_TYPE=ssh

deepspeed --hostfile=$HOSTFILE \
        --master_port=${MASTER_PORT} \
        --launcher=${DS_LAUNCHER} \
        ${DS_LAUNCHER_ARGS:+--launcher_args="$DS_LAUNCHER_ARGS"} \
        src/sft_mind_jev.py \
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
        --kl_beta ${KL_BETA} \
        --enable_kl ${ENABLE_KL} \
        --sample ${TRAIN_SAMPLE} \
        --use_abstract ${USE_ABSTRACT} \
        $(if [[ "${USE_CHAT_TEMPLATE}" -eq 1 ]]; then echo "--use_chat_template True"; fi) \
        --train_from_scratch False \
        --seed 42 \
        --deepspeed_config ${DS_CONFIG} \
        ${WANDB_RUN_NAME:+--wandb_run_name ${WANDB_RUN_NAME}} \
        ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}}

TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "JEV training FAILED with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi
echo "JEV training completed!"
