#!/bin/bash
# CONTROL for the answer-only A/B: do two IDENTICAL runs also diverge?
#
# WHY
# ---
# src/check_answer_only_grads.py (job 16296985) proved that the answer-only
# path is mathematically equivalent to the stock full-logits path:
#   * forward loss identical to 10 digits (25.8902835846 both paths)
#   * eval-mode gradient difference (5.08e-3 relative) is SMALLER than the
#     same path's own run-to-run repeat noise (6.27e-3) -- i.e. the backward
#     is bf16-nondeterministic on its own.
#
# Yet the real trainer A/B (job 16294944) showed the two configurations'
# loss trajectories diverging from step 2 (24.89 vs 24.87, then 1.161 vs
# 1.255). If two IDENTICAL configurations also diverge the same way, the
# divergence is chaos, not a property of the answer-only path.
#
# This runs THREE arms, one GPU each, concurrently, same seed, same data:
#   ans1  answer-only logits   (logits_to_keep=2)
#   ans2  answer-only logits   (IDENTICAL config, different GPU)
#   full  full logits           (stock path)
#
# Then it prints the three pairwise loss sequences. If |ans1 - ans2| is the
# same size as |ans1 - full|, the answer-only path is not the cause.
#
# Usage: sbatch scripts/_answer_only_chaos_sbatch.sh
# Env:   CH_MODEL, CH_SAMPLE, CH_MB
#SBATCH --job-name=chaos
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
export PYTHONFAULTHANDLER=1

# The A100 nodes' RoCE transport segfaults inside NCCL (job 16294462).
# This script runs a single process per GPU with no process group, but
# DeepSpeed/accelerate probe NCCL anyway, so disable the IB transport.
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

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

MODEL=${CH_MODEL:-models/Qwen3.5-4B}
SAMPLE=${CH_SAMPLE:-400}
MB=${CH_MB:-4}
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
IFS=',' read -r G0 G1 G2 G3 <<< "$GPUS"

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

run_arm () {
  # $1 name, $2 gpu, $3 port, $4 extra env
  local name=$1 gpu=$2 port=$3 extra=$4
  echo "----- arm=${name} gpu=${gpu} -----"
  env ${extra} \
    CUDA_VISIBLE_DEVICES=${gpu} \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=${port} \
    MINIONEREC_LENGTH_GROUPED=1 MINIONEREC_NUM_WORKERS=2 \
    TRITON_CACHE_DIR=/dev/shm/triton_${name} \
    python -u src/sft_mind_pointwise_ds.py \
      ${COMMON_ARGS} --batch_size ${MB} --micro_batch_size ${MB} \
      --output_dir /tmp/chaos_${name} \
      --deepspeed_config ds_configs/ds_config_zero2_gpu.json \
      > logs/chaos_${name}.log 2>&1
  echo "  arm=${name} rc=$?"
  grep -aE "Answer-only logits" logs/chaos_${name}.log | head -2
}

mkdir -p /dev/shm/triton_chaos_{ans1,ans2,full}
run_arm ans1 "$G0" 29841 "" & P0=$!
run_arm ans2 "$G1" 29842 "" & P1=$!
run_arm full "$G2" 29843 "MINIONEREC_NO_ANSWER_ONLY_LOGITS=1" & P2=$!
wait $P0 $P1 $P2

echo "----- LOSS TRAJECTORIES -----"
python - <<'PY'
import re

def losses(path):
    out = []
    try:
        for line in open(path, errors="replace"):
            for m in re.finditer(r"'loss': '([0-9.eE+-]+)'", line):
                out.append(float(m.group(1)))
    except FileNotFoundError:
        pass
    return out

a1 = losses("logs/chaos_ans1.log")
a2 = losses("logs/chaos_ans2.log")
f  = losses("logs/chaos_full.log")
print(f"steps: ans1={len(a1)} ans2={len(a2)} full={len(f)}")
n = min(len(a1), len(a2), len(f))
if n == 0:
    print("MISSING losses -- inspect logs/chaos_*.log")
    raise SystemExit(0)

def spread(x, y):
    if not x or not y:
        return float("nan")
    k = min(len(x), len(y))
    return max(abs(x[i] - y[i]) for i in range(k))

print(f"{'step':>4} {'ans1':>12} {'ans2':>12} {'full':>12} {'|a1-a2|':>11} {'|a1-full|':>11}")
for i in range(n):
    print(f"{i+1:>4} {a1[i]:>12.6g} {a2[i]:>12.6g} {f[i]:>12.6g} "
          f"{abs(a1[i]-a2[i]):>11.3e} {abs(a1[i]-f[i]):>11.3e}")

d_same = spread(a1, a2)
d_path = spread(a1, f)
print(f"\nmax |ans1 - ans2| (IDENTICAL configs) = {d_same:.4e}")
print(f"max |ans1 - full| (different path)    = {d_path:.4e}")
if d_same >= 0.5 * d_path:
    print("VERDICT: CHAOS -- two IDENTICAL runs diverge as much as the "
          "answer-only vs full-logits runs. The answer-only path is NOT the cause.")
else:
    print("VERDICT: PATH MATTERS -- identical runs stay together while the "
          "answer-only path diverges. Investigate further.")
PY

echo "=== CHAOS CONTROL DONE $(date) ==="
