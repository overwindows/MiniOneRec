#!/bin/bash
# Diagnose the DeepSpeed SIGSEGV (rc=-11) in the trainer, and find out whether
# it is specific to world_size=1.
#
# WHY
# ---
# scripts/_segv_bisect_sbatch.sh established:
#   * deepspeed + length-grouped OFF        -> SIGSEGV
#   * deepspeed + length-grouped ON, w=0    -> SIGSEGV
#   * deepspeed + length-grouped ON, w=2    -> SIGSEGV
#   * plain torch (no deepspeed)            -> TRAINS FINE
# So the crash is in the DeepSpeed path, not the custom dataloader. It fires
# at the first training step, before any loss is logged.
#
# The production runs use --num_gpus=8 (world_size=8) and the running B200
# job (16133260) is training normally at epoch 2.5, so the Sunday relaunch
# is probably safe -- but "probably" is not good enough for a 24h queue slot.
#
# This job runs four arms, one per GPU, concurrently:
#   A  deepspeed, world_size=1, LG on   -> crash expected; PYTHONFAULTHANDLER
#                                          dumps the Python stack on SIGSEGV
#   B  deepspeed, world_size=1, LG off  -> crash expected; isolates the
#                                          dataloader again with a stack
#   C  deepspeed, world_size=2, LG on   -> THE KEY ARM: does the crash also
#                                          happen with 2 ranks? If no, it is
#                                          a world_size=1-only quirk and the
#                                          Sunday 8-GPU relaunch is unaffected.
#   D  plain torch, world_size=1, LG on -> control, must train
#
# Usage: sbatch scripts/_segv_fh_sbatch.sh
# Env:   FH_MODEL, FH_SAMPLE, FH_MB
#SBATCH --job-name=segvfh
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --time=00:30:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -u
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# PYTHONFAULTHANDLER makes CPython print the Python-level stack on SIGSEGV,
# which is the only way to see WHERE the native crash happens.
export PYTHONFAULTHANDLER=1

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

MODEL=${FH_MODEL:-models/Qwen3.5-4B}
SAMPLE=${FH_SAMPLE:-64}
MB=${FH_MB:-4}
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
  --num_epochs 1 \
  --learning_rate 1e-4 --cutoff_len 2048 --max_history 30 \
  --neg_ratio 2.0 --hard_neg_ratio 0.5 --sample ${SAMPLE} \
  --use_abstract True --use_chat_template True --use_recency True \
  --use_lora True --lora_r 16 --lora_alpha 32 --seed 42"

# $1 label, $2 visible devices, $3 port, $4 length-grouped, $5 workers,
# $6 world_size, $7 use_deepspeed (1/0)
run () {
  local name=$1 devs=$2 port=$3 lg=$4 workers=$5 ws=$6 ds=$7 noans=${8:-0}
  echo "----- ${name} devs=${devs} ws=${ws} ds=${ds} lg=${lg} no_answer_only=${noans} -----"
  # gradient_accumulation_steps = (batch_size // micro_batch_size) // world_size
  # must stay >= 1, so batch_size has to scale with world_size.
  local bs=$((MB * ws))
  local extra="MINIONEREC_LENGTH_GROUPED=${lg} MINIONEREC_NUM_WORKERS=${workers} TRITON_CACHE_DIR=/dev/shm/triton_${name}"
  [ "${noans}" = "1" ] && extra="${extra} MINIONEREC_NO_ANSWER_ONLY_LOGITS=1"
  if [ "${ds}" = "1" ]; then
    # No --num_gpus: the launcher then derives --include=localhost:<slots>
    # from CUDA_VISIBLE_DEVICES and re-sets it for the child, so the exact
    # slots are honoured. (--num_gpus overrides them and always uses slot 0.)
    env CUDA_VISIBLE_DEVICES=${devs} ${extra} \
      deepspeed --master_port=${port} src/sft_mind_pointwise_ds.py \
        ${COMMON_ARGS} --batch_size ${bs} --micro_batch_size ${MB} \
        --output_dir /tmp/segvfh_${name} \
        --deepspeed_config ds_configs/ds_config_zero2_gpu.json \
        > logs/segvfh_${name}.log 2>&1
  else
    env CUDA_VISIBLE_DEVICES=${devs} ${extra} \
      RANK=0 LOCAL_RANK=0 WORLD_SIZE=${ws} \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=${port} \
      python -u src/sft_mind_pointwise_ds.py \
        ${COMMON_ARGS} --batch_size ${bs} --micro_batch_size ${MB} \
        --output_dir /tmp/segvfh_${name} \
        > logs/segvfh_${name}.log 2>&1
  fi
  echo "  ${name} rc=$?"
}

mkdir -p /dev/shm/triton_segvfh_{a,c,d,e}
IFS=',' read -r G0 G1 G2 G3 <<< "$GPUS"

# Phase 1: the two DeepSpeed arms. Arm A crashes within ~40 s, arm C (2 ranks)
# runs longer. They use G0 and G1+G2, so they do not collide.
run a_ds1_lg "$G0"       29551 1 2 1 1 0 & P0=$!
run c_ds2_lg "$G1,$G2"  29553 1 2 2 1 0 & P2=$!
wait $P0 $P2

# Phase 2: the two plain-torch arms, one on each of G0 and G3 (A has already
# released G0). These are the decisive pair: if only the answer-only arm
# crashes, the logits_to_keep path is the culprit.
run d_torch1     "$G0" 29554 1 2 1 0 0 & P3=$!
run e_torch1_nol "$G3" 29555 1 2 1 0 1 & P4=$!
wait $P3 $P4

echo "----- RESULTS (rc: 0=ok, 245=launcher reports child died, 139=SIGSEGV) -----"
for n in a_ds1_lg c_ds2_lg d_torch1 e_torch1_nol; do
  echo "=== ${n} ==="
  grep -aE "Length-grouped|Answer-only logits" logs/segvfh_${n}.log | head -3
  grep -aoE "\{'loss'[^}]*\}" logs/segvfh_${n}.log | tail -2
  echo "--- faulthandler stack (if any) ---"
  grep -a -A 25 "Fatal Python error\|Current thread" logs/segvfh_${n}.log | head -30
done
echo "=== SEGV FH DONE $(date) ==="
