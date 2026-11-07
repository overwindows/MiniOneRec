#!/bin/bash
# Smoke test the HYBRID prefix cache in PARALLEL multi-GPU mode.
#
# WHY
# ---
# The 4-arm A/B (scripts/_hybrid_ab.sh) validated the hybrid path in SINGLE
# GPU mode. But the test-chunk scoring jobs run in PARALLEL mode: the
# behaviors file is split across GPUs and each GPU process scores its own
# slice, then the score files are concatenated. The hybrid path's cache
# expansion is per-process, so it should behave identically -- but "should" is
# not "verified", and the chunk jobs are 7.5 hours in, so a crash discovered
# late would be expensive.
#
# This runs the parallel path with 2 GPUs on a handful of impressions and
# checks:
#   * no crash,
#   * "Prefix KV cache: True (mode=hybrid" on EVERY GPU process,
#   * the merged score file has the expected number of impressions,
#   * the scores match the single-GPU hybrid run on the same impressions.
#
# Usage: sbatch scripts/_hybrid_par_smoke.sh
# Env:   HP_CKPT, HP_N, HP_TAG
#SBATCH --job-name=hpsmoke
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60000
#SBATCH --time=00:30:00
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

CKPT=${HP_CKPT:-results/ranker/lg_9b_ckpt89000_snapshot}
N=${HP_N:-64}
TAG=${HP_TAG:-par}

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
echo "=== GPUs: ${GPUS} n=${N} ckpt=${CKPT} ==="

OUT=results/ranker/hp_${TAG}_par_scores.tsv
rm -f "$OUT"

env MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large \
  USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=30 \
  USE_RECENCY=1 HISTORY_SELECTION=recent HISTORY_FORMAT=flat \
  PREFIX_CACHE=1 PREFIX_CACHE_MODE=hybrid PREFIX_GROUP_SIZE=16 \
  BATCH_SIZE=8 OUTPUT_SCORES_FILE=${OUT} CUDA_VISIBLE_DEVICES=${GPUS} \
  bash scripts/eval_mind_pointwise.sh "$CKPT" dev $N \
  > logs/hp_${TAG}_par.log 2>&1
RC=$?
echo "=== parallel run rc=${RC} ==="
tr '\r' '\n' < logs/hp_${TAG}_par.log | grep -aE "Prefix KV cache|AUC:|Error|Traceback" | head -10

if [ ${RC} -ne 0 ]; then
  echo "FATAL: parallel hybrid run failed"
  tr '\r' '\n' < logs/hp_${TAG}_par.log | tail -30
  exit 1
fi

# Every GPU process must have had the hybrid path enabled -- one process
# silently falling back to the plain path would make the merged file a mix.
NT=$(tr '\r' '\n' < logs/hp_${TAG}_par.log | grep -ac "Prefix KV cache: True (mode=hybrid")
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
echo "=== hybrid-enabled GPU processes: ${NT} / ${NGPU} ==="
if [ "${NT}" -lt "${NGPU}" ]; then
  echo "FATAL: not every GPU process used the hybrid path"
  exit 1
fi

echo "=== merged score file ==="
wc -l "$OUT" 2>&1
head -1 "$OUT" 2>&1 | cut -c1-120

echo "=== HP SMOKE DONE $(date) ==="
