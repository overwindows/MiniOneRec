#!/bin/bash
# Bisect the SIGSEGV that kills the trainer at world_size=1.
#
# WHY
# ---
# scripts/_trainer_val_sbatch.sh segfaults (rc=139) on BOTH variants, right
# after "Length-grouped train dataloader enabled". The production runs
# (scripts/train_eval_small.sh, scripts/train_eval_small_b200.sh) use
# `--num_gpus=8`, i.e. world_size=8, and the running B200 job (16133260) is
# happily logging losses at epoch 2.5. So either
#   (a) the length-grouped dataloader is broken at world_size=1, or
#   (b) DeepSpeed ZeRO-2 is broken at world_size=1, or
#   (c) neither -- the crash is something else entirely.
#
# This runs four configs, ONE PER GPU, CONCURRENTLY, so the whole matrix
# costs one short job:
#
#   A  deepspeed launcher, length-grouped OFF        -> isolates the dataloader
#   B  deepspeed launcher, length-grouped ON, workers=0
#   C  deepspeed launcher, length-grouped ON, workers=2   (the failing config)
#   D  plain torch (no deepspeed), length-grouped ON
#
# A crash means the config segfaults; a clean run means it does not.
#
# Usage: sbatch scripts/_segv_bisect_sbatch.sh
# Env:   SB_MODEL, SB_SAMPLE, SB_MB
#SBATCH --job-name=segvb
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --time=00:40:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -u
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

COMPAT_DIR=/import/ml-sc-scratch1/chenw/cuda-compat-13.0
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if [ -f "${COMPAT_DIR}/libcuda.so.1" ]; then
    echo "=== driver too old for cu130 torch; preloading CUDA forward-compat ==="
    export LD_LIBRARY_PATH=${COMPAT_DIR}:${LD_LIBRARY_PATH:-}
  else
    echo "FATAL: driver too old and no forward-compat lib available -- aborting"
    exit 1
  fi
fi

echo "=== NODE=$(hostname) $(date) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

MODEL=${SB_MODEL:-models/Qwen3.5-4B}
SAMPLE=${SB_SAMPLE:-64}
MB=${SB_MB:-4}
ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_small

GPUS=$(python - <<'PY' 2>/dev/null || echo "0,1,2,3"
import torch
ok = []
for i in range(torch.cuda.device_count()):
    try:
        torch.zeros(1, device=f"cuda:{i}")
        ok.append(i)
    except Exception:
        pass
print(",".join(map(str, ok[:4])) if ok else "0,1,2,3")
PY
)
echo "=== GPUs: ${GPUS} model=${MODEL} sample=${SAMPLE} mb=${MB} ==="

COMMON_ARGS="--base_model ${MODEL} \
  --train_behaviors_path ${ROOT}/train/behaviors.tsv \
  --train_news_path ${ROOT}/train/news.tsv \
  --eval_behaviors_path ${ROOT}/dev/behaviors.tsv \
  --eval_news_path ${ROOT}/dev/news.tsv \
  --batch_size ${MB} --micro_batch_size ${MB} --num_epochs 1 \
  --learning_rate 1e-4 --cutoff_len 2048 --max_history 30 \
  --neg_ratio 2.0 --hard_neg_ratio 0.5 --sample ${SAMPLE} \
  --use_abstract True --use_chat_template True --use_recency True \
  --use_lora True --lora_r 16 --lora_alpha 32 --seed 42"

run_ds () {
  # $1 = label, $2 = gpu, $3 = port, $4 = length-grouped (0/1), $5 = workers
  local name=$1 gpu=$2 port=$3 lg=$4 workers=$5
  echo "----- ${name} gpu=${gpu} length_grouped=${lg} workers=${workers} -----"
  # CUDA_VISIBLE_DEVICES is set so the launcher picks exactly this slot: with
  # no --num_gpus/--include, runner.py turns VISIBLE_DEVICES into
  # --include=localhost:<slot> and then re-derives CUDA_VISIBLE_DEVICES for
  # the child. (With --num_gpus it OVERRIDES the slot and always uses 0.)
  env CUDA_VISIBLE_DEVICES=${gpu} \
    MINIONEREC_LENGTH_GROUPED=${lg} MINIONEREC_NUM_WORKERS=${workers} \
    TRITON_CACHE_DIR=/dev/shm/triton_${name} \
    deepspeed --master_port=${port} src/sft_mind_pointwise_ds.py \
      ${COMMON_ARGS} \
      --output_dir /tmp/segv_${name} \
      --deepspeed_config ds_configs/ds_config_zero2_gpu.json \
      > logs/segv_${name}.log 2>&1
  echo "  ${name} rc=$?"
}

run_torch () {
  # $1 = label, $2 = gpu, $3 = port, $4 = length-grouped (0/1), $5 = workers
  local name=$1 gpu=$2 port=$3 lg=$4 workers=$5
  echo "----- ${name} gpu=${gpu} length_grouped=${lg} workers=${workers} -----"
  env CUDA_VISIBLE_DEVICES=${gpu} \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=${port} \
    MINIONEREC_LENGTH_GROUPED=${lg} MINIONEREC_NUM_WORKERS=${workers} \
    TRITON_CACHE_DIR=/dev/shm/triton_${name} \
    python -u src/sft_mind_pointwise_ds.py \
      ${COMMON_ARGS} \
      --output_dir /tmp/segv_${name} \
      > logs/segv_${name}.log 2>&1
  echo "  ${name} rc=$?"
}

mkdir -p /dev/shm/triton_segv_{a,b,c,d}
IFS=',' read -r G0 G1 G2 G3 <<< "$GPUS"

run_ds  a_nods   "$G0" 29541 0 2 & P0=$!
run_ds  b_lg_w0  "$G1" 29542 1 0 & P1=$!
run_ds  c_lg_w2  "$G2" 29543 1 2 & P2=$!
run_torch d_torch "$G3" 29544 1 2 & P3=$!
wait $P0 $P1 $P2 $P3

echo "----- RESULTS -----"
for n in a_nods b_lg_w0 c_lg_w2 d_torch; do
  echo "--- ${n} ---"
  grep -aE "Length-grouped|Answer-only logits" logs/segv_${n}.log | head -3
  grep -aoE "\{'loss'[^}]*\}" logs/segv_${n}.log | tail -2
  tr '\r' '\n' < logs/segv_${n}.log | grep -aE "Error|Traceback|Segmentation" | head -3
done
echo "=== SEGV BISECT DONE $(date) ==="
