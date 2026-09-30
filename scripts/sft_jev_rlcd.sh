#!/bin/bash
# RLCD (contrastive self-distill) launcher for JEV-4B on MIND noul pairs.
# Usage from repo root:
#   bash scripts/sft_jev_rlcd.sh --data_root ... --output_dir ...
# Config comes from environment (set by pipeline_executor), defaulting to
# MIND_large abstract hard-neg settings to mirror the C1 SFT run.
set -e

cd "$HOME"
REPO_DIR="$HOME/MiniOneRec"
[ -d "$REPO_DIR" ] || { echo "REPO_DIR $REPO_DIR missing"; exit 1; }
cd "$REPO_DIR"

CONDA_ENV="${CONDA_ENV:-/home/aiscuser/.conda/envs/MiniOneRec}"
source /home/aiscuser/.bashrc
conda activate "$CONDA_ENV"
python - <<'PYEOF'
import sys
try:
    import torch, transformers, deepspeed
    print(f"torch={torch.__version__} transformers={transformers.__version__} deepspeed={deepspeed.__version__} cuda={torch.cuda.is_available()}")
except Exception as e:
    print(f"ENV ERR {e}"); sys.exit(1)
PYEOF

MODEL_PATH="${MODEL_PATH:-shares/users/wuc/models/APUS-OpenJev-v1-4B}"
DATA_ROOT="${DATA_ROOT:-shares/users/wuc/data/MIND_large}"
OUTPUT_DIR="${OUTPUT_DIR:-shares/users/wuc/output_dir/jev_rlcd}"
TRAIN_BEHAVIORS="${TRAIN_BEHAVIORS:-$DATA_ROOT/train/behaviors.tsv}"
TRAIN_NEWS="${TRAIN_NEWS:-$DATA_ROOT/train/news.tsv}"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export MASTER_PORT=${MASTER_PORT:-29505}

# NOTE: on single-node the deepspeed launch uses all 8 GPUs; the "hostfile"
# path is for multi-node. Here we use the SLURM/Azure style launch via
# distributed.launch equivalent through deepspeed.
deepspeed --num_gpus=8 src/sft_mind_jev_rlcd.py \
    --base_model "$MODEL_PATH" \
    --train_behaviors_path "$TRAIN_BEHAVIORS" \
    --train_news_path "$TRAIN_NEWS" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed_config ds_configs/ds_config_zero2.json \
    --use_abstract 1 \
    --max_history "${MAX_HISTORY:-30}" \
    --neg_ratio "${NEG_RATIO:-1.0}" \
    --hard_neg_ratio "${HARD_NEG_RATIO:-0.5}" \
    --micro_batch_size "${MICRO_BATCH_SIZE:-2}" \
    --num_epochs "${NUM_EPOCHS:-3}" \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --cutoff_len "${CUTOFF_LEN:-2048}" \
    --margin "${MARGIN:-0.0}"
