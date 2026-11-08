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
# 4. A FAULTED GPU THAT STILL LOOKS IDLE. sc3-cp80's GPU 7 reported 0 MiB
#    used (so the memory probe passed) but the model loaded on CPU: the card
#    is broken at a level nvidia-smi does not report. The scorer then aborted
#    the whole unit ("1 GPU(s) failed (7)") and wrote NOTHING, wasting the
#    entire node for hours. A memory probe cannot see this -- only an actual
#    CUDA compute can. So the preflight now runs a real matmul on every
#    candidate GPU and DROPS the ones that fail, and then uses the remaining
#    ones instead of failing the unit.
#
# 5. A GPU THAT IS BUSY IS NOT OURS TO TOUCH. Such a GPU is skipped (not
#    used), rather than aborting the whole unit -- one busy card must not
#    cost us the other seven.
#
# Usage:
#   GPUS=$(mkgpu_preflight 8)     # or: source this file; it sets GPUS/NGPU
#
# Env:
#   GPU_PREFLIGHT_NGPU  (default: all of CUDA_VISIBLE_DEVICES, else 1)
#   GPU_PREFLIGHT_MEM   (default 100 MiB -- above this a GPU counts as busy)
#   GPU_PREFLIGHT_MIN   (default 2 -- refuse to run with fewer usable GPUs;
#                        one GPU is ~8x slower and holds the node for hours)

GPU_PREFLIGHT_MEM=${GPU_PREFLIGHT_MEM:-100}
GPU_PREFLIGHT_NGPU=${GPU_PREFLIGHT_NGPU:-0}
GPU_PREFLIGHT_MIN=${GPU_PREFLIGHT_MIN:-2}

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

  # ---- 3. drop the BUSY GPUs ----------------------------------------------
  # nvidia-smi reports every physical GPU of the node, ignoring
  # CUDA_VISIBLE_DEVICES, so this sees cards that belong to other users too.
  local free busy g used
  free=""
  busy=""
  for g in ${phys//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
    if [ -z "${used}" ] || [ "${used}" -gt "${GPU_PREFLIGHT_MEM}" ]; then
      if [ -z "${busy}" ]; then busy="${g}"; else busy="${busy},${g}"; fi
    else
      if [ -z "${free}" ]; then free="${g}"; else free="${free},${g}"; fi
    fi
  done
  if [ -z "${free}" ]; then
    echo "FATAL: every allocated GPU is busy (${busy}) -- aborting"
    return 1
  fi
  if [ -n "${busy}" ]; then
    echo "=== skipping ${busy} (busy / in use by someone else) ==="
  fi

  # ---- 4. real CUDA compute test on the FREE GPUs ------------------------
  #
  # A single python process tests all of them (one torch import, then a loop),
  # so the extra cost is one import, not one per GPU. A GPU is usable only if
  # it can allocate and run a matmul. This catches the faulted-but-idle card
  # that the memory probe cannot see.
  #
  # The python process sees ONLY the free GPUs (CUDA_VISIBLE_DEVICES is set
  # to the free list), so its device ordinal i corresponds to the PHYSICAL
  # index free[i]. The mapping is applied here, in the shell.
  local usable_ord
  usable_ord=$(CUDA_VISIBLE_DEVICES="${free}" python - <<'PY' 2>/dev/null
import sys
try:
    import torch
except Exception:
    sys.exit(1)
if not torch.cuda.is_available():
    sys.exit(1)
ok = []
for i in range(torch.cuda.device_count()):
    try:
        a = torch.randn(512, 512, device=f"cuda:{i}")
        float((a @ a).sum().item())
        ok.append(i)
    except Exception:
        pass
print(",".join(str(i) for i in ok))
PY
)
  if [ -z "${usable_ord}" ]; then
    echo "FATAL: no free GPU can actually run a CUDA op (${free}) -- aborting"
    return 1
  fi

  # torch ordinal -> physical index
  local usable faulted i p
  usable=""
  faulted=""
  local n=0
  for i in ${usable_ord//,/ }; do
    p=$(echo "${free}" | cut -d, -f$((i+1)))
    if [ -z "${usable}" ]; then usable="${p}"; else usable="${usable},${p}"; fi
    n=$((n+1))
  done
  for p in ${free//,/ }; do
    case ",${usable}," in
      *",${p},"*) ;;
      *) if [ -z "${faulted}" ]; then faulted="${p}"; else faulted="${faulted},${p}"; fi ;;
    esac
  done

  local n_usable=${n}
  if [ "${n_usable}" -lt "${n_phys}" ]; then
    echo "WARNING: ${n_phys} GPU(s) allocated but only ${n_usable} can run CUDA; dropping ${faulted}"
  fi

  # ---- 5. pick the GPUs we will actually use ------------------------------
  if [ "${n_usable}" -lt "${ngpu}" ]; then
    # Running with fewer GPUs is strictly better than failing the unit: the
    # scorer shards impressions across whatever GPUs it is given, and the
    # completeness check is against the CHUNK's line count, not the GPU
    # count. But one GPU is ~8x slower and would hold the node for hours.
    if [ "${n_usable}" -lt "${GPU_PREFLIGHT_MIN}" ]; then
      echo "FATAL: only ${n_usable} usable GPU(s) (need >= ${GPU_PREFLIGHT_MIN}) -- aborting"
      return 1
    fi
    echo "WARNING: running with ${n_usable} GPU(s) instead of ${ngpu}"
    ngpu=${n_usable}
  fi

  local sel
  sel=$(echo "${usable}" | cut -d, -f1-${ngpu})

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
