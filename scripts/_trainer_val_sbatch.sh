#!/bin/bash
# Validate the answer-only-logits path in the REAL trainer on REAL MIND data.
#
# WHY
# ---
# The benchmark (src/bench_train_memory.py) already proved that the sliced
# logits give a bit-identical loss to the full-logits path on a real model.
# What it did NOT exercise is the real trainer wiring: the Dataset's label
# layout, the DataCollator, the PEFT wrapper, DeepSpeed, and the actual
# `Trainer.compute_loss` dispatch.
#
# This runs the real trainer twice on the same data with the same seed --
# once with the answer-only path ON, once forced OFF -- and compares the
# logged losses. If the two runs agree step by step, the optimization is
# safe to enable for the production runs.
#
# The two runs go on two GPUs of the same node, concurrently.
#
# Usage: sbatch scripts/_trainer_val_sbatch.sh
# Env:   TV_MODEL, TV_SAMPLE, TV_MB, TV_NGPU
#SBATCH --job-name=tval
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --time=01:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -u
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/dev/shm/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

COMPAT_DIR=/import/ml-sc-scratch1/chenw/cuda-compat-13.0
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if [ -f "${COMPAT_DIR}/libcuda.so.1" ]; then
    echo "=== driver too old for cu130 torch; preloading CUDA forward-compat ==="
    export LD_LIBRARY_PATH=${COMPAT_DIR}:${LD_LIBRARY_PATH:-}
    if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      echo "FATAL: CUDA still unavailable after forward-compat preload -- aborting"
      exit 1
    fi
  else
    echo "FATAL: driver too old and no forward-compat lib available -- aborting"
    exit 1
  fi
fi

echo "=== NODE=$(hostname) $(date) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

MODEL=${TV_MODEL:-models/Qwen3.5-4B}
SAMPLE=${TV_SAMPLE:-400}
MB=${TV_MB:-4}
NGPU=${TV_NGPU:-2}
ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_small

GPUS=$(python - <<'PY' 2>/dev/null || echo "0,1"
import torch
ok = []
for i in range(torch.cuda.device_count()):
    try:
        torch.zeros(1, device=f"cuda:{i}")
        ok.append(i)
    except Exception:
        pass
print(",".join(map(str, ok[:2])) if ok else "0,1")
PY
)
echo "=== GPUs: ${GPUS} | model=${MODEL} sample=${SAMPLE} mb=${MB} ==="

run () {
  local name=$1 gpu=$2 port=$3 extra=$4
  echo "----- variant=${name} gpu=${gpu} port=${port} ${extra} -----"
  # The distributed env vars are set EXPLICITLY instead of going through the
  # `deepspeed` launcher. Two reasons:
  #
  # 1) The launcher OVERWRITES CUDA_VISIBLE_DEVICES. runner.py deletes it
  #    from the environment and then re-derives it from the resource pool
  #    ("Detected VISIBLE_DEVICES=... but ignoring it because ... --num_gpus
  #    was used"), so `--num_gpus=1` always lands on slot 0 -- both variants
  #    ran on GPU 0 and fought over the same memory. Setting the vars by
  #    hand pins each variant to its own GPU.
  # 2) The launcher is only needed to set RANK/WORLD_SIZE/MASTER_ADDR/
  #    MASTER_PORT/LOCAL_RANK. DeepSpeed's init_distributed() skips MPI
  #    discovery when all of those are present (comm.py: required_env), and
  #    the trainer itself calls dist.init_process_group() at import time
  #    when LOCAL_RANK is set. So a plain `python` call with these vars set
  #    is equivalent to the launcher, minus the device hijack.
  #
  # Each variant also gets its OWN master_port and its OWN Triton cache dir.
  # Sharing a port gives EADDRINUSE; sharing a Triton cache lets two
  # processes compile the same fla kernel concurrently and corrupt it, which
  # shows up as a SIGSEGV (-11) at the first fused-kernel call.
  env ${extra} \
    CUDA_VISIBLE_DEVICES=${gpu} \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=${port} \
    MINIONEREC_LENGTH_GROUPED=1 MINIONEREC_NUM_WORKERS=2 \
    TRITON_CACHE_DIR=/dev/shm/triton_${name} \
    python -u src/sft_mind_pointwise_ds.py \
      --local_rank=0 \
      --base_model "${MODEL}" \
      --train_behaviors_path ${ROOT}/train/behaviors.tsv \
      --train_news_path ${ROOT}/train/news.tsv \
      --eval_behaviors_path ${ROOT}/dev/behaviors.tsv \
      --eval_news_path ${ROOT}/dev/news.tsv \
      --output_dir /tmp/tval_${name} \
      --batch_size ${MB} --micro_batch_size ${MB} --num_epochs 1 \
      --learning_rate 1e-4 --cutoff_len 2048 --max_history 30 \
      --neg_ratio 2.0 --hard_neg_ratio 0.5 --sample ${SAMPLE} \
      --use_abstract True --use_chat_template True --use_recency True \
      --use_lora True --lora_r 16 --lora_alpha 32 --seed 42 \
      --deepspeed_config ds_configs/ds_config_zero2_gpu.json \
      > logs/tval_${name}.log 2>&1
  echo "  variant=${name} rc=$?"
  grep -aE "Answer-only logits" logs/tval_${name}.log | head -2
}

IFS=',' read -r A B <<< "$GPUS"
mkdir -p /dev/shm/triton_answeronly /dev/shm/triton_fulllogits
run answeronly "$A" 29531 "" &
P0=$!
run fulllogits "$B" 29532 "MINIONEREC_NO_ANSWER_ONLY_LOGITS=1" &
P1=$!
wait $P0 $P1

echo "----- COMPARISON -----"
python - <<'PY'
import re, sys

def losses(path):
    """Pull the logged 'loss' values out of a Trainer log."""
    out = []
    try:
        for line in open(path, errors="replace"):
            m = re.search(r"'loss': ([0-9.]+)", line)
            if m:
                out.append(float(m.group(1)))
    except FileNotFoundError:
        pass
    return out

a = losses("logs/tval_answeronly.log")
b = losses("logs/tval_fulllogits.log")
print(f"answer-only logged losses: {len(a)}  full-logits: {len(b)}")
if not a or not b:
    print("MISSING losses -- inspect logs/tval_*.log")
    sys.exit(0)
n = min(len(a), len(b))
print(f"{'step':>4} {'answer-only':>14} {'full-logits':>14} {'abs diff':>12}")
worst = 0.0
for i in range(n):
    d = abs(a[i] - b[i])
    worst = max(worst, d)
    print(f"{i+1:>4} {a[i]:>14.8f} {b[i]:>14.8f} {d:>12.3e}")
print(f"\nworst |diff| over {n} logged steps: {worst:.6e}")
print("VERDICT:", "IDENTICAL" if worst < 1e-5 else "DIFFERS -- investigate")
PY
echo "=== TRAINER VAL DONE $(date) ==="
