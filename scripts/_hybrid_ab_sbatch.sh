#!/bin/bash
# sbatch wrapper for the hybrid prefix-cache A/B validation.
#
# Runs on an IDLE A100 node (the B200 nodes are saturated by the ck47000 /
# ck89000 test-chunk jobs, and those two checkpoints already cover all six
# main chunks -- so an A100 cannot contribute non-duplicating chunk work).
#
# What it does:
#   1. A 2-minute SMOKE test of the hybrid path on 8 impressions. If the
#      hybrid forward crashes (mask shape, cache expansion, fp32 state patch),
#      fail here instead of after four 30-minute model loads.
#   2. The full 4-arm A/B (plain / plain_rp / plain_b1 / hybrid).
#
# Usage: sbatch scripts/_hybrid_ab_sbatch.sh
# Env:   AB_CKPT, AB_N, AB_N_B1, AB_TAG, AB_GPUS, AB_NODE
#SBATCH --job-name=hyab
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200000
#SBATCH --time=06:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -u
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/dev/shm/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# ---- CUDA driver compatibility -------------------------------------------
# The cu130 torch build needs driver >= 595.71.05. The A100/H100 nodes carry
# older drivers, where torch.cuda.is_available() is False and the model would
# load on CPU (hours for a 9B). The CUDA forward-compat package ships a newer
# libcuda that works against the old kernel driver. It must ONLY be preloaded
# on old-driver nodes: on a new-driver node it shadows the real libcuda and
# breaks CUDA entirely.
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
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

CKPT=${AB_CKPT:-results/ranker/lg_9b_ckpt89000_snapshot}
TAG=${AB_TAG:-q35}

# ---- resolve the GPUs this job may use ------------------------------------
# Inside a Slurm job the allocated GPUs are the only ones visible (either
# CUDA_VISIBLE_DEVICES masks them or the cgroup device controller hides the
# rest), and the CUDA driver renumbers what it sees to 0..N-1. So the
# CUDA-relative index is the only correct thing to export -- the physical id
# (e.g. "4,5,6,7") would not exist inside the job.
GPUS=$(python - <<'PY' 2>/dev/null || echo "0,1,2,3"
import torch
ok = []
for i in range(torch.cuda.device_count()):
    try:
        torch.zeros(1, device=f"cuda:{i}")
        ok.append(i)
    except Exception:
        pass
print(",".join(map(str, ok)) if ok else "0,1,2,3")
PY
)
echo "=== usable GPUs (cuda-relative): ${GPUS} ==="

# ---- 1) smoke test --------------------------------------------------------
# The hybrid path is brand new: it monkey-patches the cache layer class,
# expands a batch-1 cache to the group size, and feeds a 4-D block mask.
# A crash here is cheap to detect and expensive to discover late.
echo "=== SMOKE: hybrid path, 8 impressions ==="
SMOKE_T0=$(date +%s)
env MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large \
  USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=30 \
  USE_RECENCY=1 HISTORY_SELECTION=recent HISTORY_FORMAT=flat \
  PREFIX_CACHE=1 PREFIX_CACHE_MODE=hybrid PREFIX_GROUP_SIZE=16 \
  BATCH_SIZE=8 CUDA_VISIBLE_DEVICES=${GPUS%%,*} \
  bash scripts/eval_mind_pointwise.sh "$CKPT" dev 8 \
  > logs/smoke_${TAG}.log 2>&1
SMOKE_RC=$?
echo "=== SMOKE rc=${SMOKE_RC} elapsed=$(( $(date +%s) - SMOKE_T0 ))s ==="
tr '\r' '\n' < logs/smoke_${TAG}.log | grep -aE "Prefix KV cache|AUC:|Error|Traceback|error" | head -20
if [ ${SMOKE_RC} -ne 0 ]; then
  echo "FATAL: hybrid smoke test failed; not running the A/B"
  tr '\r' '\n' < logs/smoke_${TAG}.log | tail -40
  exit 1
fi
if ! tr '\r' '\n' < logs/smoke_${TAG}.log | grep -qa "Prefix KV cache: True"; then
  echo "FATAL: hybrid prefix cache was not enabled; not running the A/B"
  tr '\r' '\n' < logs/smoke_${TAG}.log | grep -aE "Prefix KV cache" | head -5
  exit 1
fi

# ---- 2) full A/B ----------------------------------------------------------
echo "=== A/B ==="
AB_CKPT="${CKPT}" AB_TAG="${TAG}" AB_GPUS="${GPUS}" \
AB_N=${AB_N:-1000} AB_N_B1=${AB_N_B1:-200} \
  bash scripts/_hybrid_ab.sh

echo "=== HYBRID AB SBATCH DONE $(date) ==="
