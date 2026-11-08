#!/bin/bash
# Shared GPU preflight for the production scoring scripts.
#
# SOURCE this script, then use ${GPUS} (comma-separated PHYSICAL device
# indices) and ${NGPU} (how many of them).
#
# WHY THIS EXISTS
# ---------------
# 1. OLD DRIVER. The cu130 torch build needs driver >= 595.71.05. Some
#    A100/H100 nodes still carry 565.57.01, where torch.cuda.is_available()
#    is False and the model would load on CPU (days for a 9B). The CUDA
#    forward-compat package ships a newer libcuda that works against the old
#    kernel driver, so preload it when the installed driver is too old.
#
# 2. SHARED NODES. Slurm hands out an arbitrary SUBSET of a node's GPUs
#    (e.g. 2,3,4,5) and exports that list in CUDA_VISIBLE_DEVICES. The
#    physical indices are NOT 0..N-1. Probing 0..N-1 both misses a busy GPU
#    of ours and aborts on a busy GPU that belongs to somebody else.
#    nvidia-smi always lists every physical GPU of the node and ignores
#    CUDA_VISIBLE_DEVICES, so the PHYSICAL indices are what to query.
#
# 3. device_map="auto" silently falls back to CPU when one GPU is
#    unhealthy, which burns hours on a CPU forward pass. Refuse to start
#    instead of discovering that three hours in.
#
# Usage:
#   GPUS=$(mkgpu_preflight 8)     # or: source this file; it sets GPUS/NGPU
#
# Env:
#   GPU_PREFLIGHT_NGPU (default: all of CUDA_VISIBLE_DEVICES, else 1)
#   GPU_PREFLIGHT_MEM  (default 100 MiB -- above this a GPU counts as busy)

GPU_PREFLIGHT_MEM=${GPU_PREFLIGHT_MEM:-100}
GPU_PREFLIGHT_NGPU=${GPU_PREFLIGHT_NGPU:-0}

mkgpu_preflight () {
  local want=${1:-0}

  # ---- 1. CUDA driver compatibility ---------------------------------------
  local compat=/import/ml-sc-scratch1/chenw/cuda-compat-13.0
  local drv
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    if [ -f "${compat}/libcuda.so.1" ]; then
      echo "=== driver ${drv} too old for cu130 torch; preloading CUDA forward-compat ==="
      export LD_LIBRARY_PATH=${compat}:${LD_LIBRARY_PATH:-}
      if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "FATAL: CUDA still unavailable after forward-compat preload -- aborting"
        return 1
      fi
    else
      echo "FATAL: driver ${drv} too old and no forward-compat lib available -- aborting"
      return 1
    fi
  fi

  # ---- 2. which PHYSICAL devices did Slurm give us? -----------------------
  local phys
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    phys="${CUDA_VISIBLE_DEVICES}"
  else
    phys=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
  fi
  local n_phys
  n_phys=$(echo "${phys}" | tr ',' '\n' | grep -c .)

  local ngpu=${want}
  if [ "${ngpu}" -le 0 ]; then ngpu=${GPU_PREFLIGHT_NGPU}; fi
  if [ "${ngpu}" -le 0 ]; then ngpu=${n_phys}; fi
  if [ "${ngpu}" -gt "${n_phys}" ]; then
    echo "FATAL: asked for ${ngpu} GPUs but Slurm allocated only ${n_phys} (${phys})"
    return 1
  fi

  local sel
  sel=$(echo "${phys}" | cut -d, -f1-${ngpu})

  # ---- 3. health check on the PHYSICAL devices we will use ----------------
  local g used
  for g in ${sel//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "${used}" -gt "${GPU_PREFLIGHT_MEM}" ]; then
      echo "FATAL: GPU ${g} has ${used} MiB in use (busy or faulted)"
      return 1
    fi
  done

  echo "=== NODE=$(hostname) $(date) ==="
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv,noheader
  echo "=== using ${ngpu} GPU(s): ${sel} (of ${n_phys} allocated) ==="
  GPUS="${sel}"
  NGPU="${ngpu}"
  export GPUS NGPU
  return 0
}

# When sourced (not executed), run the preflight immediately.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  mkgpu_preflight "$@"
fi
