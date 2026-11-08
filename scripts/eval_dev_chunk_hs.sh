#!/bin/bash
# Score ONE sub-chunk of the MIND_large DEV split with a checkpoint.
#
# WHY A NEW SCRIPT (and not eval_dev_subchunk_large.sh)
# ------------------------------------------------------
# Two reasons, both about not breaking running jobs or the eval protocol:
#
#  1) PREFIX CACHE MODE. scripts/eval_mind_pointwise.sh hardcodes
#     PREFIX_CACHE_MODE="${PREFIX_CACHE_MODE:-packed}". The "packed" mode
#     needs plain full-attention KV layers, so on a Qwen3.5 backbone
#     (GatedDeltaNet linear-attention layers) it silently DISABLES the
#     prefix cache -- the eval prints "Prefix KV cache: False" and falls
#     back to the slow plain path. The measured cost on 1 GPU / 1000
#     impressions: hybrid 11m15s vs plain 1h19m58s = 7.1x slower.
#     The hybrid path is validated (job 16278212): its deviation from the
#     plain path is the SAME SIZE as the plain path's own bf16 batching
#     noise. So DE_MODE=hybrid is the production path for Qwen3.5.
#
#  2) PROMPT PARITY. scripts/eval_dev_subchunk_large.sh hardcodes
#     HISTORY_SELECTION=recent, so a checkpoint trained with a different
#     history selection is scored off-distribution (this is the bug that
#     made the dedupbf105k ensemble member's dev and test scores
#     disagree). Here DE_HISTORY_SELECTION is honoured.
#
# This script is safe on A100/H100 nodes: the scoring path never creates a
# process group (src/evaluate_mind_pointwise.py does not import
# torch.distributed), so the A100 RoCE/NCCL SIGSEGV cannot fire here.
#
# Usage:
#   DE_PART=00 DE_CKPT=<path> DE_MODE=hybrid sbatch scripts/eval_dev_chunk_hs.sh
# Env: DE_PART 00..05 | DE_CKPT | DE_MODE hybrid|packed | DE_NGPU (default: all)
#      DE_OUTDIR | DE_HISTORY_SELECTION (default recent) | DE_USE_RECENCY (default 1)
#      DE_MAX_IMPRESSIONS (default 0 = all)
#      DE_PARITY 0|1  (1 = reproduce training's prompt-shrink loop per candidate)
#      DE_TAG    label appended to the output filename (keeps A/B arms apart)
#SBATCH --job-name=devchunk
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400000
#SBATCH --time=08:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

PART=$(printf "%02d" ${DE_PART:?set DE_PART=00..05})
CKPT=${DE_CKPT:?set DE_CKPT}
MODE=${DE_MODE:-hybrid}
OUTDIR=${DE_OUTDIR:-results/ranker/dev_scores/hs}
TAG=${DE_TAG:-}
PARITY=${DE_PARITY:-0}
if [ -n "${TAG}" ]; then
  OUT=${OUTDIR}/hs_${MODE}${TAG}_${PART}.tsv
else
  OUT=${OUTDIR}/hs_${MODE}_${PART}.tsv
fi
# ---- GPU selection --------------------------------------------------------
# Slurm exports CUDA_VISIBLE_DEVICES with the ALLOCATED device indices, which
# on a shared node are NOT 0..N-1. Honour them: overriding them would land
# two processes on a GPU another job owns. torch.cuda.device_count() already
# respects CUDA_VISIBLE_DEVICES, so it is the number of GPUs we may use.
NTORCH=$(python -c "import torch; print(torch.cuda.device_count())")
NGPU=${DE_NGPU:-$NTORCH}
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
HS=${DE_HISTORY_SELECTION:-recent}
REC=${DE_USE_RECENCY:-1}
MAXIMP=${DE_MAX_IMPRESSIONS:-0}

# ---- CUDA driver compatibility (A100/H100 nodes may carry an old driver) ---
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

# ---- GPU health pre-check -------------------------------------------------
# device_map="auto" silently falls back to CPU when a GPU is unhealthy, which
# burns hours on a CPU forward pass. Refuse to start instead.
#
# The check must look at the PHYSICAL devices we were allocated, not at
# physical 0..NGPU-1: on a shared node Slurm hands out an arbitrary subset of
# the node's GPUs (e.g. 2,3,4,5), so probing 0..NGPU-1 both misses a busy GPU
# of ours and aborts on a busy GPU that belongs to somebody else.
# nvidia-smi always lists every physical GPU of the node and ignores
# CUDA_VISIBLE_DEVICES, so ${GPUS} (the physical indices) is what to query.
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 100 ]; then
    echo "FATAL: GPU $g has ${used} MiB in use (busy or faulted)"
    exit 1
  fi
done
echo "=== NODE=$(hostname) $(date) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv,noheader
echo "=== part=${PART} ckpt=${CKPT} mode=${MODE} ngpu=${NGPU} hs=${HS} recency=${REC} ==="

# Per-job Triton cache: /dev/shm is small and shared on some nodes.
export TRITON_CACHE_DIR=/dev/shm/triton_hs_${MODE}_${PART}_${SLURM_JOB_ID:-$$}
mkdir -p "${TRITON_CACHE_DIR}"

# The mini split dir (behaviors.tsv + news.tsv symlinks) already exists under
# data/MIND_large/dev/chunks/sub/d00/split_<PART>; create it if missing so
# the job is self-contained.
MINI=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/dev/chunks/sub/d00/split_${PART}
mkdir -p ${MINI}
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/dev/news.tsv ${MINI}/news.tsv
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/dev/chunks/sub/d00/chunk_${PART}.tsv ${MINI}/behaviors.tsv

mkdir -p ${OUTDIR}
# The parity switch is read by src/evaluate_mind_pointwise.py straight from the
# environment (MINIONEREC_PROMPT_PARITY), so no running script needs editing.
export MINIONEREC_PROMPT_PARITY=${PARITY}
MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/dev/chunks/sub \
USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=30 \
USE_RECENCY=${REC} \
HISTORY_SELECTION=${HS} \
HISTORY_FORMAT=flat \
PREFIX_CACHE_MODE=${MODE} \
OUTPUT_SCORES_FILE=${OUT} \
CUDA_VISIBLE_DEVICES=${GPUS} \
bash scripts/eval_mind_pointwise.sh ${CKPT} d00/split_${PART} ${MAXIMP} 2>&1 | tee logs/devchunk_${MODE}${TAG}_${PART}_${SLURM_JOB_ID:-$$}.log

# ---- verify: line count and per-impression candidate count ----
# With DE_MAX_IMPRESSIONS the scorer only processes the first N impressions of
# the chunk, so the expected line count is min(N, chunk size).
#
# The candidate-count check must NOT restrict `want` to the first N entries:
# in parallel mode the scorer splits the chunk into contiguous per-GPU blocks
# and caps EACH block at N/num_gpus, so the scored impressions are NOT the
# first N of the chunk (they are N/num_gpus from the start of each block).
# Building `want` from the whole chunk checks each scored line against its
# own true candidate count regardless of which impressions were picked.
python3 - "$PART" "$OUT" "$MAXIMP" <<'PY'
import sys
part, out, maximp = sys.argv[1], sys.argv[2], int(sys.argv[3])
want = {}
with open(f"data/MIND_large/dev/chunks/sub/d00/chunk_{part}.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        want[p[0]] = len(p[4].split())
n_total = len(want)
n = bad = unknown = 0
with open(out) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        exp = want.get(p[0])
        if exp is None:
            unknown += 1
        elif len(p) - 1 != exp:
            bad += 1
        n += 1
expect_lines = min(maximp, n_total) if maximp > 0 else n_total
print(f"VERIFY {out}: lines={n} (want {expect_lines})  "
      f"mismatches={bad}  unknown_ids={unknown}")
assert n == expect_lines, f"line count {n} != {expect_lines}"
assert bad == 0, f"{bad} candidate-count mismatches"
assert unknown == 0, f"{unknown} scored impression ids are not in the chunk"
PY

echo "=== dev chunk ${PART} (${MODE}) DONE $(date) ==="
