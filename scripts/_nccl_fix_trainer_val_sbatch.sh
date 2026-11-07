#!/bin/bash
# Validate the NCCL fix AND complete the answer-only trainer validation.
#
# WHY
# ---
# scripts/_nccl_transport_sbatch.sh found the SIGSEGV root cause:
#   on the A100 nodes, NCCL selects the RoCE transport
#     "NET/IB : Using [0]bnxt_re0:1/RoCE [1]bnxt_re1:1/RoCE"
#   and `dist.all_reduce` SEGFAULTS inside it -- even at world_size=1.
#   NCCL_IB_DISABLE=1 and NCCL_NET=Socket both fix it.
#   gloo passes, so torch's distributed layer is fine.
#   The B200 nodes pass with the DEFAULT settings, so production on B200 is
#   unaffected.
#
# This script does two things:
#   PART 1  re-runs the collectives at ws=2 and ws=4 with the fix, to confirm
#           it holds for multi-rank (the trainer's real configuration)
#   PART 2  runs the REAL trainer (the answer-only vs full-logits A/B that
#           has been blocked by this crash since it was first attempted)
#           with the fix applied, on two GPUs concurrently
#
# The two parts run concurrently on separate GPU sets.
#
# Usage: sbatch scripts/_nccl_fix_trainer_val_sbatch.sh
# Env:   CF_MODEL, CF_SAMPLE, CF_MB
#SBATCH --job-name=ncclfix
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1

# THE FIX: the A100 nodes' RoCE transport segfaults inside NCCL.
# NCCL_IB_DISABLE=1 makes NCCL fall back to the socket transport.
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
echo "=== NCCL_IB_DISABLE=${NCCL_IB_DISABLE} ==="

MODEL=${CF_MODEL:-models/Qwen3.5-4B}
SAMPLE=${CF_SAMPLE:-400}
MB=${CF_MB:-4}
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

# ---------------------------------------------------------------- PART 1
# Confirm the fix holds at ws=2 and ws=4.
part1 () {
  # Runs on G2,G3 only, because part2 (the trainer A/B) owns G0,G1 at the
  # same time. So this part checks ws=2, and ws=4 is checked afterwards
  # once part2 has released its GPUs.
  echo "--- PART 1: collectives at ws=2 WITH the fix (on G2,G3) ---"
  cat > /tmp/fix_probe.py <<'PY'
import os, sys, datetime, traceback
import torch, torch.distributed as dist
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
tag = sys.argv[1]
try:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=3))
    t = torch.ones(1, device="cuda") * (rank + 1)
    dist.all_reduce(t); torch.cuda.synchronize()
    big = torch.zeros(1024, 256, device="cuda")
    dist.broadcast(big, src=0); torch.cuda.synchronize()
    out = [torch.empty(1, device="cuda") for _ in range(world)]
    dist.all_gather(out, torch.full((1,), float(rank), device="cuda")); torch.cuda.synchronize()
    print(f"[{tag}] rank={rank} ALL PASSED", flush=True)
    dist.destroy_process_group()
except Exception:
    print(f"[{tag}] rank={rank} EXC:", flush=True); traceback.print_exc(); sys.exit(1)
PY
  local WS=$1 devs=$2 port=$3
  local pids=(); local r=0
  for ((r=0; r<WS; r++)); do
    env RANK=${r} LOCAL_RANK=${r} WORLD_SIZE=${WS} MASTER_ADDR=127.0.0.1 \
        MASTER_PORT=${port} CUDA_VISIBLE_DEVICES=${devs} \
      python -u /tmp/fix_probe.py "fix_ws${WS}" \
      > "logs/ncclfix_ws${WS}_r${r}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "  ws=${WS} rc=${rc}"
  grep -ah "ALL PASSED\|EXC:\|Segmentation" logs/ncclfix_ws${WS}_r*.log | head -4
}

# ---------------------------------------------------------------- PART 2
# The real trainer A/B: answer-only logits vs forced full-logits.
# Both variants run on their own GPU with their own port and Triton cache.
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

run_trainer () {
  # $1 name, $2 gpu, $3 port, $4 extra env
  local name=$1 gpu=$2 port=$3 extra=$4
  echo "----- trainer variant=${name} gpu=${gpu} -----"
  # The distributed env vars are set EXPLICITLY instead of going through the
  # `deepspeed` launcher: the launcher OVERWRITES CUDA_VISIBLE_DEVICES when
  # --num_gpus is used, so both variants would land on GPU 0 and fight over
  # memory. DeepSpeed's init_distributed() skips MPI discovery when
  # RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT/LOCAL_RANK are all present, so a
  # plain `python` call with those set is launcher-equivalent.
  env ${extra} \
    CUDA_VISIBLE_DEVICES=${gpu} \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=${port} \
    MINIONEREC_LENGTH_GROUPED=1 MINIONEREC_NUM_WORKERS=2 \
    TRITON_CACHE_DIR=/dev/shm/triton_${name} \
    python -u src/sft_mind_pointwise_ds.py \
      ${COMMON_ARGS} --batch_size ${MB} --micro_batch_size ${MB} \
      --output_dir /tmp/ncclfix_${name} \
      --deepspeed_config ds_configs/ds_config_zero2_gpu.json \
      > logs/ncclfix_${name}.log 2>&1
  echo "  variant=${name} rc=$?"
  grep -aE "Answer-only logits|Length-grouped" logs/ncclfix_${name}.log | head -3
}

part2 () {
  echo "--- PART 2: real trainer, answer-only vs full-logits ---"
  mkdir -p /dev/shm/triton_ncclfix_answeronly /dev/shm/triton_ncclfix_fulllogits
  run_trainer answeronly "$G0" 29831 "" &
  local P0=$!
  run_trainer fulllogits "$G1" 29832 "MINIONEREC_NO_ANSWER_ONLY_LOGITS=1" &
  local P1=$!
  wait $P0 $P1
}

part1 2 "$G2,$G3" 29802 &
P1=$!
part2 &
P2=$!
wait $P1 $P2

# ws=4 now that the trainer A/B has released its GPUs.
echo "--- PART 1b: collectives at ws=4 WITH the fix ---"
part1 4 "$G0,$G1,$G2,$G3" 29804
echo "----- LOSS COMPARISON -----"
python - <<'PY'
import re, sys
def losses(path):
    out = []
    try:
        for line in open(path, errors="replace"):
            # transformers logs the value QUOTED: {'loss': '24.94', ...}
            m = re.search(r"'loss': '?([0-9.eE+-]+)'?", line)
            if m:
                out.append(float(m.group(1)))
    except FileNotFoundError:
        pass
    return out
a = losses("logs/ncclfix_answeronly.log")
b = losses("logs/ncclfix_fulllogits.log")
print(f"answer-only logged losses: {len(a)}   full-logits: {len(b)}")
if not a or not b:
    print("MISSING losses -- inspect logs/ncclfix_*.log")
    sys.exit(0)
n = min(len(a), len(b))
print(f"{'step':>4} {'answer-only':>16} {'full-logits':>16} {'abs diff':>12}")
worst = 0.0
for i in range(n):
    d = abs(a[i] - b[i]); worst = max(worst, d)
    print(f"{i+1:>4} {a[i]:>16.8f} {b[i]:>16.8f} {d:>12.3e}")
print(f"\nworst |diff| over {n} logged steps: {worst:.6e}")
print("VERDICT:", "IDENTICAL" if worst < 1e-5 else "DIFFERS -- investigate")
PY

echo "----- CRASH CHECK -----"
for n in answeronly fulllogits; do
  echo "=== ${n} ==="
  grep -aE "Fatal Python|Segmentation|Traceback|Error" logs/ncclfix_${n}.log | head -4
done
echo "=== NCCL FIX + TRAINER VAL DONE $(date) ==="
