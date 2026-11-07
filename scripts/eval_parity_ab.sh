#!/bin/bash
# PAIRED prompt-parity A/B: score ONE over-length dev subset with BOTH arms,
# back to back, ON THE SAME NODE.
#
# WHY THIS IS PAIRED
# ------------------
# The parity path (src/evaluate_mind_pointwise.py `_build_parity_prompt_ids`)
# reproduces training's history-shrink loop. For an impression whose prompt
# already FITS in 2048 tokens the parity path is a no-op: the rendered prompt
# is identical, so the two arms produce identical scores. The arms can
# therefore only differ on impressions whose prompt OVERFLOWS.
#
# The subset (data/MIND_large/dev/chunks/parity/p0*.tsv) contains exactly
# those impressions (built by scripts/build_overlong_subset.py, deliberately
# over-inclusive so the fitting ones inside it act as a control group).
#
# Both arms run here, in the SAME job, on the SAME GPUs, minutes apart. That
# makes the comparison free of node-to-node hardware variation -- the whole
# point: a cross-node A/B on different A100s/H100s would confound the parity
# effect with bf16 batching noise.
#
#   arm OFF : MINIONEREC_PROMPT_PARITY=0  (current production eval path)
#   arm ON  : MINIONEREC_PROMPT_PARITY=1  (train/eval parity path)
#
# The subset is the SLOWEST population (longest prompts), so the wall clock
# here is an upper bound for a full-dev parity re-score.
#
# Usage:
#   PA_PART=00 PA_CKPT=<path> sbatch scripts/eval_parity_ab.sh
# Env: PA_PART 00..05 | PA_CKPT | PA_NGPU (default: all allocated)
#      PA_MODE hybrid|packed | PA_OUTDIR | PA_MAX_IMPRESSIONS (default 0 = all)
#SBATCH --job-name=parab
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400000
#SBATCH --time=06:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

PART=$(printf "%02d" ${PA_PART:?set PA_PART=00..05})
CKPT=${PA_CKPT:-results/ranker/lg_9b_ckpt45500_snapshot}
MODE=${PA_MODE:-hybrid}
OUTDIR=${PA_OUTDIR:-results/ranker/parity_ab}
MAXIMP=${PA_MAX_IMPRESSIONS:-0}

# ---- CUDA driver compatibility (A100/H100 nodes carry an old driver) -------
# The cu130 torch build needs driver >= 595.71.05. On the older A100/H100
# drivers torch.cuda.is_available() is False and the 9B model would load on
# CPU (hours). The CUDA forward-compat package ships a newer libcuda that
# works against the old kernel driver. It must ONLY be preloaded when torch
# reports no CUDA: on a new-driver node it shadows the real libcuda.
COMPAT_DIR=/import/ml-sc-scratch1/chenw/cuda-compat-13.0
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if [ -f "${COMPAT_DIR}/libcuda.so.1" ]; then
    echo "=== driver ${DRV} too old for cu130 torch; preloading CUDA forward-compat ==="
    export LD_LIBRARY_PATH=${COMPAT_DIR}:${LD_LIBRARY_PATH:-}
    if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      echo "FATAL: CUDA still unavailable after forward-compat preload -- aborting"
      exit 1
    fi
  else
    echo "FATAL: driver ${DRV} too old and no forward-compat lib -- aborting"
    exit 1
  fi
fi

# ---- GPU selection --------------------------------------------------------
# Slurm exports CUDA_VISIBLE_DEVICES with the ALLOCATED device indices, which
# on a shared node are NOT 0..N-1. Honour them: overriding them would land
# two processes on a GPU another job owns. torch.cuda.device_count() already
# respects CUDA_VISIBLE_DEVICES, so it is the number of GPUs we may use.
NTORCH=$(python -c "import torch; print(torch.cuda.device_count())")
NGPU=${PA_NGPU:-$NTORCH}
if [ "$NGPU" -gt "$NTORCH" ]; then
  echo "FATAL: asked for ${NGPU} GPUs but torch sees only ${NTORCH}"
  exit 1
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  # Take the first NGPU entries of what Slurm gave us.
  GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | cut -d, -f1-${NGPU})
else
  GPUS=$(seq -s, 0 $((NGPU - 1)))
fi

# ---- GPU health pre-check -------------------------------------------------
# device_map="auto" silently falls back to CPU on an unhealthy GPU, which
# burns hours on a CPU forward pass. Refuse to start instead. The indices are
# the PHYSICAL ones Slurm handed us (CUDA_VISIBLE_DEVICES entries).
for g in $(echo "${GPUS}" | tr ',' ' '); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 100 ]; then
    echo "FATAL: GPU $g has ${used} MiB in use (busy or faulted)"
    exit 1
  fi
done
echo "=== NODE=$(hostname) $(date) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
echo "=== part=${PART} ckpt=${CKPT} mode=${MODE} gpus=${GPUS} ==="

# Per-job Triton cache: /dev/shm is small and shared on some nodes.
export TRITON_CACHE_DIR=/dev/shm/triton_parab_${PART}_${SLURM_JOB_ID:-$$}
mkdir -p "${TRITON_CACHE_DIR}"

# The mini split dir (behaviors.tsv + news.tsv symlinks) for the parity subset.
SUBROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/dev/chunks/parity
SPLITNAME=split_p${PART}
MINI=${SUBROOT}/${SPLITNAME}
mkdir -p ${MINI}
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/dev/news.tsv ${MINI}/news.tsv
ln -sf ${SUBROOT}/p${PART}.tsv ${MINI}/behaviors.tsv

mkdir -p ${OUTDIR}

run_arm () {
  local PARITY=$1
  local TAG=$2
  local OUT=${OUTDIR}/p${PART}_${TAG}.tsv
  local LOG=logs/parab_${MODE}_${TAG}_${PART}_${SLURM_JOB_ID:-$$}.log
  rm -f "$OUT"
  local T0=$(date +%s)
  echo "=== ARM ${TAG} (MINIONEREC_PROMPT_PARITY=${PARITY}) start $(date) ==="
  export MINIONEREC_PROMPT_PARITY=${PARITY}
  MIND_ROOT=${SUBROOT} \
  USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=30 \
  USE_RECENCY=1 \
  HISTORY_SELECTION=recent \
  HISTORY_FORMAT=flat \
  PREFIX_CACHE_MODE=${MODE} \
  OUTPUT_SCORES_FILE=${OUT} \
  CUDA_VISIBLE_DEVICES=${GPUS} \
  bash scripts/eval_mind_pointwise.sh ${CKPT} ${SPLITNAME} ${MAXIMP} > "$LOG" 2>&1
  local T1=$(date +%s)
  echo "=== ARM ${TAG} done $(date)  elapsed=$((T1-T0))s ==="
  tr '\r' '\n' < "$LOG" | grep -aE "Prefix KV cache:|AUC:|Error|Traceback" | head -8
  wc -l "$OUT"
}

# Arm OFF first, then arm ON: the same node, the same GPUs, minutes apart.
run_arm 0 off
run_arm 1 on

# ---- verify: both arms scored the same number of impressions --------------
python3 - "$PART" "$OUTDIR" "$MAXIMP" <<'PY'
import sys, os
part, outdir, maximp = sys.argv[1], sys.argv[2], int(sys.argv[3])
sub = f"data/MIND_large/dev/chunks/parity/p{part}.tsv"
total = sum(1 for _ in open(sub))
want = min(maximp, total) if maximp > 0 else total
for tag in ("off", "on"):
    p = f"{outdir}/p{part}_{tag}.tsv"
    n = sum(1 for _ in open(p)) if os.path.exists(p) else -1
    print(f"VERIFY {p}: lines={n} (want {want})")
    assert n == want, f"{tag} arm: line count {n} != {want}"
print("VERIFY OK: both arms scored the same impressions")
PY

echo "=== PARITY A/B part ${PART} DONE $(date) ==="
