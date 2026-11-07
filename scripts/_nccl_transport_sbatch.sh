#!/bin/bash
# Characterise the A100-node NCCL SIGSEGV.
#
# WHY
# ---
# scripts/_nccl_probe_sbatch.sh established:
#   * sc3-c98 (A100 x4, PCIe): dist.all_reduce at world_size=1 -> SIGSEGV
#   * sc3-cp100 (B200 x8):      dist.all_reduce at world_size=1 -> PASSES
# Same driver (595.71.05), same torch (2.11.0+cu130), same NCCL (2.28.9).
# The A100s are PCIe with no NVLink; the B200s have NVLink. That difference
# changes which NCCL transport gets selected, and a bad transport choice on
# A100 is the leading hypothesis for a SIGSEGV inside the collective.
#
# This script varies the transport knobs on ONE node and finds which one
# makes the crash go away:
#   T0  baseline (no NCCL tuning)
#   T1  NCCL_DEBUG=INFO   (does not change behaviour; dumps the transport)
#   T2  NCCL_P2P_DISABLE=1
#   T3  NCCL_SHM_DISABLE=1
#   T4  NCCL_IB_DISABLE=1
#   T5  NCCL_NET=Socket    (force the socket transport)
#   T6  NCCL_DEBUG=INFO + NCCL_SOCKET_IFNAME=<the node's primary interface>
#
# Each transport arm runs at world_size=1 and world_size=2, so we also learn
# whether the crash needs >1 rank.
#
# Usage: sbatch scripts/_nccl_transport_sbatch.sh
# Env:   NT_GPUS (default 4), NT_TAG (label for the log files)
#SBATCH --job-name=nccltr
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000
#SBATCH --time=00:20:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -u
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
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

TAG=${NT_TAG:-x}
echo "=== NODE=$(hostname) $(date) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'nccl',torch.cuda.nccl.version(),'cuda',torch.version.cuda)"

# The interface NCCL should use for the socket transport: the one carrying
# the default route.
IFACE=$(ip route show default 2>/dev/null | awk '{print $5; exit}')
echo "=== default-route interface: ${IFACE:-<none>} ==="

cat > /tmp/nccl_probe2.py <<'PY'
import os, sys, datetime, traceback
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
local = int(os.environ.get("LOCAL_RANK", "0"))
# Every rank sees the SAME device list; LOCAL_RANK selects the device.
torch.cuda.set_device(local)

tag = sys.argv[1]
backend = sys.argv[2] if len(sys.argv) > 2 else "nccl"
print(f"[{tag}] rank={rank}/{world} local={local} backend={backend} "
      f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
      f"torch={torch.__version__} nccl={torch.cuda.nccl.version()}", flush=True)

try:
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=3))
    print(f"[{tag}] rank={rank} init_process_group OK", flush=True)

    # gloo's CUDA support is partial, so the gloo control runs on CPU. The
    # point of the control is "does torch's distributed layer work at all on
    # this node", not "does gloo move GPU tensors".
    dev = "cpu" if backend == "gloo" else "cuda"

    t = torch.ones(1, device=dev) * (rank + 1)
    dist.all_reduce(t)
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"[{tag}] rank={rank} all_reduce OK sum={t.item()}", flush=True)

    big = torch.zeros(1024, 256, device=dev)
    dist.broadcast(big, src=0)
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"[{tag}] rank={rank} broadcast OK", flush=True)

    out = [torch.empty(1, device=dev) for _ in range(world)]
    dist.all_gather(out, torch.full((1,), float(rank), device=dev))
    if dev == "cuda":
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

# Run one transport setting at ws=1 and ws=2.
# $1 label, $2 world_size, $3 backend (nccl|gloo), $4.. extra env assignments
run_arm () {
  # $1 label, $2 world_size, $3 backend (nccl|gloo), $4.. extra env assignments
  local label=$1 ws=$2 backend=$3; shift 3
  # Each rank must see the SAME device list and pick its own with
  # LOCAL_RANK. Give the group exactly ws devices so torch's device_count
  # matches world_size (a mismatch makes set_device raise "invalid device
  # ordinal", which is what the first version of this probe hit).
  local devs
  devs=$(echo "$GPUS" | cut -d, -f1-$ws)
  local port=$(( 29700 + ws ))
  echo "----- arm ${label} ws=${ws} backend=${backend} extra=[$*] -----"
  # Spawn ws processes, one per rank. (An earlier version spawned only one,
  # so the ws=2 gloo control hung waiting for a peer that never came.)
  local pids=()
  local r=0
  for ((r=0; r<ws; r++)); do
    env RANK=${r} LOCAL_RANK=${r} WORLD_SIZE=${ws} \
        MASTER_ADDR=127.0.0.1 MASTER_PORT=${port} \
        CUDA_VISIBLE_DEVICES=${devs} \
        "$@" TRITON_CACHE_DIR=/dev/shm/triton_nccltr \
      python -u /tmp/nccl_probe2.py "${label}" "${backend}" \
      > "logs/nccltr_${TAG}_${label}_r${r}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do
    wait "$p" || rc=1
  done
  echo "  ${label} rc=${rc}"
  for f in logs/nccltr_${TAG}_${label}_r*.log; do
    grep -aE "ALL NCCL PROBES PASSED|EXCEPTION|Segmentation|Fatal Python|all_reduce OK|init_process_group OK" \
      "$f" | head -4
  done
  # Surface the transport NCCL selected.
  grep -ahE "NCCL INFO.*(Bootstrap|NET/Socket|NET/IB|NVLS|P2P|Channel)|Using network" \
    logs/nccltr_${TAG}_${label}_r*.log | head -4
}

run_arm t0_base      1 nccl
run_arm t1_debug     1 nccl NCCL_DEBUG=INFO
run_arm t2_nop2p     1 nccl NCCL_P2P_DISABLE=1
run_arm t3_noshm     1 nccl NCCL_SHM_DISABLE=1
run_arm t4_noib      1 nccl NCCL_IB_DISABLE=1
run_arm t5_socket    1 nccl NCCL_NET=Socket
if [ -n "${IFACE}" ]; then
  run_arm t6_iface    1 nccl NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME="${IFACE}"
fi

# Control: the SAME collectives over the gloo backend. If gloo passes and
# nccl segfaults, the fault is inside NCCL, not in torch's distributed layer
# or in the process group bootstrap.
echo "--- gloo control (same ops, no NCCL) ---"
run_arm t7_gloo_w1 1 gloo
run_arm t7_gloo_w2 2 gloo

run_arm t0_base_w2   2 nccl
run_arm t2_nop2p_w2  2 nccl NCCL_P2P_DISABLE=1
run_arm t5_socket_w2 2 nccl NCCL_NET=Socket

echo "----- SUMMARY (${TAG} on $(hostname)) -----"
for f in logs/nccltr_${TAG}_t*.log; do
  [ -f "$f" ] || continue
  b=$(basename "$f" .log)
  if grep -qa "ALL NCCL PROBES PASSED" "$f"; then
    echo "  ${b}: PASS"
  elif grep -qa "Segmentation" "$f"; then
    echo "  ${b}: SIGSEGV"
  else
    echo "  ${b}: OTHER-FAIL"
  fi
done
echo "=== NCCL TRANSPORT PROBE DONE $(date) ==="
