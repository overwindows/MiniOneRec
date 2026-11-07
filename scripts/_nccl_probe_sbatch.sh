#!/bin/bash
# Probe whether NCCL collectives are broken on THIS node.
#
# WHY
# ---
# The trainer SIGSEGVs on A100 inside a NCCL collective:
#   * deepspeed arm   -> dist.broadcast   (deepspeed/comm/torch.py:343,
#                                           engine.py:2016 _broadcast_model)
#   * plain-torch arm -> dist.all_gather  (transformers/trainer_pt_utils.py:209)
# Both are the first collective that carries real data. The production 8-GPU
# B200 job (16133260) runs the same collectives fine, so the question is
# whether NCCL is broken on the A100 nodes specifically.
#
# The A100s are PCIe cards ("NVIDIA A100 80GB PCIe") with no NVLink, while
# the B200s are NVLink. A P2P/NVLink difference is the most likely cause of a
# SIGSEGV inside NCCL, and it would be invisible at ws=8 on B200.
#
# This runs the same three probes at increasing world size:
#   P1  ws=1  : all_reduce + broadcast + all_gather
#   P2  ws=2  : same
#   P3  ws=4  : same
# Each prints the stack of whatever fails. Run it on an A100 node AND on a
# B200 node to get a control.
#
# Usage: sbatch scripts/_nccl_probe_sbatch.sh
# Env:   NP_GPUS (default 4), NP_FORCE_COMPAT=1 (preload forward-compat libcuda)
#SBATCH --job-name=ncclpb
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000
#SBATCH --time=00:15:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -u
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=WARN

COMPAT_DIR=/import/ml-sc-scratch1/chenw/cuda-compat-13.0
if [ "${NP_FORCE_COMPAT:-0}" = "1" ]; then
  echo "=== forcing CUDA forward-compat preload (${COMPAT_DIR}) ==="
  export LD_LIBRARY_PATH=${COMPAT_DIR}:${LD_LIBRARY_PATH:-}
elif ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
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
echo "=== versions ==="
python -c "import torch;print('torch',torch.__version__,'nccl',torch.cuda.nccl.version(),'cuda',torch.version.cuda)"
echo "=== /dev/shm ==="
df -h /dev/shm | tail -1
echo "=== nvidia-smi topo (P2P/NVLink) ==="
nvidia-smi topo -m 2>&1 | head -12
echo "=== NCCL P2P capability ==="
python - <<'PY' 2>&1 | tail -6
import torch
n = torch.cuda.device_count()
print("device_count", n)
for i in range(min(n, 4)):
    for j in range(i + 1, min(n, 4)):
        try:
            ok = torch.cuda.can_device_access_peer(i, j)
            print(f"peer {i}<->{j}: {ok}")
        except Exception as e:
            print(f"peer {i}<->{j}: ERR {e}")
PY

cat > /tmp/nccl_probe.py <<'PY'
import os, sys, datetime, traceback
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
local = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local)

tag = sys.argv[1]
print(f"[{tag}] rank={rank}/{world} device=cuda:{local} "
      f"torch={torch.__version__} nccl={torch.cuda.nccl.version()}", flush=True)

try:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=5))
    print(f"[{tag}] rank={rank} init_process_group OK", flush=True)

    t = torch.ones(1, device="cuda") * (rank + 1)
    dist.all_reduce(t)
    torch.cuda.synchronize()
    print(f"[{tag}] rank={rank} all_reduce OK sum={t.item()}", flush=True)

    big = torch.zeros(1024, 256, device="cuda")
    dist.broadcast(big, src=0)
    torch.cuda.synchronize()
    print(f"[{tag}] rank={rank} broadcast OK", flush=True)

    out = [torch.empty(1, device="cuda") for _ in range(world)]
    dist.all_gather(out, torch.full((1,), float(rank), device="cuda"))
    torch.cuda.synchronize()
    print(f"[{tag}] rank={rank} all_gather OK vals={[o.item() for o in out]}", flush=True)

    dist.barrier()
    print(f"[{tag}] rank={rank} ALL NCCL PROBES PASSED", flush=True)
    dist.destroy_process_group()
except Exception:
    print(f"[{tag}] rank={rank} EXCEPTION:", flush=True)
    traceback.print_exc()
    sys.exit(1)
PY

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
echo "=== usable GPUs: ${GPUS} ==="
IFS=',' read -r G0 G1 G2 G3 <<< "$GPUS"

run_probe () {
  # $1 tag, $2 world_size, $3 devices
  local tag=$1 ws=$2 devs=$3
  echo "----- probe ${tag} ws=${ws} devs=${devs} -----"
  local port=$(( 29600 + ws ))
  local procs=()
  local r=0
  IFS=',' read -r -a DEVARR <<< "$devs"
  for ((r=0; r<ws; r++)); do
    procs+=("RANK=${r}" "LOCAL_RANK=${r}" "WORLD_SIZE=${ws}" "MASTER_ADDR=127.0.0.1" \
            "MASTER_PORT=${port}" "CUDA_VISIBLE_DEVICES=${DEVARR[$r]}")
  done
  env "${procs[@]}" TRITON_CACHE_DIR=/dev/shm/triton_ncclpb \
    python -u /tmp/nccl_probe.py "${tag}" \
    > "logs/ncclpb_${tag}.log" 2>&1 &
  local pid=$!
  wait $pid
  echo "  ${tag} rc=$?"
  tail -8 "logs/ncclpb_${tag}.log"
}

run_probe p1_ws1 1 "$G0"
run_probe p2_ws2 2 "$G0,$G1"
run_probe p3_ws4 4 "$G0,$G1,$G2,$G3"

echo "----- SUMMARY -----"
for t in p1_ws1 p2_ws2 p3_ws4; do
  f="logs/ncclpb_${t}.log"
  [ -f "$f" ] || continue
  echo "=== ${t} ==="
  grep -aE "ALL NCCL PROBES PASSED|EXCEPTION|Segmentation|Fatal Python" "$f" | head -4
done
echo "=== NCCL PROBE DONE $(date) ==="
