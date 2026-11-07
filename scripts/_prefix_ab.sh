#!/bin/bash
# A/B the shared-prefix KV cache against the plain full-prompt path.
#
# Both paths must produce the SAME scores for the same (history, candidate):
# the prefix path reuses the shared prompt prefix's KV cache and packs the
# per-candidate suffixes, so any divergence is a correctness bug (mask, block
# boundary, or bf16 accumulation order).
#
# The two arms run CONCURRENTLY on two GPUs of the same node (each arm gets
# its own CUDA_VISIBLE_DEVICES), so the wall time is the slow arm, not the sum.
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/dev/shm/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

COMPAT_DIR=/import/ml-sc-scratch1/chenw/cuda-compat-13.0
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if [ -f "${COMPAT_DIR}/libcuda.so.1" ]; then
    echo "=== old driver; preloading CUDA forward-compat ==="
    export LD_LIBRARY_PATH=${COMPAT_DIR}:${LD_LIBRARY_PATH:-}
  else
    echo "FATAL: no compat lib"; exit 1
  fi
fi

echo "=== NODE=$(hostname) $(date) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

CKPT=${AB_CKPT:-output_dir/small_runs/lg_q8b/checkpoint-14500}
N=${AB_N:-1000}
TAG=${AB_TAG:-q8b}

arm () {
  local pc=$1 gpu=$2
  local OUT=results/ranker/ab_${TAG}_pc${pc}_scores.tsv
  echo "----- prefix_cache=$pc on GPU $gpu n=$N -> $OUT -----"
  MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large \
  USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=30 \
  USE_RECENCY=1 HISTORY_SELECTION=recent HISTORY_FORMAT=flat \
  PREFIX_CACHE=$pc BATCH_SIZE=8 OUTPUT_SCORES_FILE=${OUT} \
  CUDA_VISIBLE_DEVICES=${gpu} \
  bash scripts/eval_mind_pointwise.sh "$CKPT" dev $N > logs/ab_${TAG}_pc${pc}.log 2>&1
  tr '\r' '\n' < logs/ab_${TAG}_pc${pc}.log | grep -aE "Prefix KV cache|AUC:|MRR:|nDCG" | head -6
}

arm 1 0 &
P1=$!
arm 0 1 &
P0=$!
wait $P1 $P0

echo "----- DIFF -----"
python - "$TAG" <<'PY'
import sys, numpy as np
tag = sys.argv[1]
def load(p):
    d = {}
    for line in open(p):
        parts = line.rstrip("\n").split("\t")
        d[parts[0]] = np.array([float(x) for x in parts[1:]])
    return d
a = load(f"results/ranker/ab_{tag}_pc1_scores.tsv")
b = load(f"results/ranker/ab_{tag}_pc0_scores.tsv")
common = sorted(set(a) & set(b))
print(f"impressions: prefix={len(a)} plain={len(b)} common={len(common)}")
diffs = []
rank_flips = 0
tot_c = 0
for k in common:
    x, y = a[k], b[k]
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    diffs.append(np.abs(x - y).max())
    tot_c += n
    if not np.array_equal(np.argsort(np.argsort(-x)), np.argsort(np.argsort(-y))):
        rank_flips += 1
diffs = np.array(diffs)
print(f"max |dscore| per impression: mean={diffs.mean():.6f} p50={np.median(diffs):.6f} "
      f"p95={np.percentile(diffs,95):.6f} max={diffs.max():.6f}")
print(f"impressions with any rank flip: {rank_flips}/{len(common)}")
print(f"total candidates compared: {tot_c}")
PY
echo "=== AB DONE ==="
