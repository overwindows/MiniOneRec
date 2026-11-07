#!/bin/bash
# Score ONE sub-chunk of the MIND_large test split with a checkpoint (8 GPUs).
# Usage: TE_SUB=<c01|c02> TE_PART=NN [TE_CKPT=path] sbatch scripts/eval_test_subchunk_large.sh
# Writes raw scores to results/ranker/test_scores/sub_<cXX>_<NN>.tsv
#SBATCH --job-name=lg_tsub
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400000
#SBATCH --time=12:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

SUB=${TE_SUB:?set TE_SUB=c01|c02}
PART=$(printf "%02d" ${TE_PART})
CKPT=${TE_CKPT:-results/ranker/lg_9b_ckpt32500_snapshot}
OUTDIR=${TE_OUTDIR:-results/ranker/test_scores}
OUT=${OUTDIR}/sub_${SUB}_${PART}.tsv

echo "=== test sub-chunk ${SUB}/${PART} ckpt=${CKPT} out=${OUT} ==="
mkdir -p ${OUTDIR}

# ---- GPU health pre-check ----
# Job 14951791 lost a whole sub-chunk because device_map="auto" silently
# fell back to CPU on one unhealthy GPU and the merge step waits for ALL
# GPUs. Refuse to start rather than burn hours on a CPU forward pass.
if ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: torch.cuda.is_available() is False on this node -- the driver"
  echo "       is older than the cu130 torch build, so the model would load"
  echo "       on CPU. Use a B200 node."
  exit 1
fi
for g in 0 1 2 3 4 5 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 100 ]; then
    echo "FATAL: GPU $g has ${used} MiB in use (busy or faulted)"
    exit 1
  fi
done
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

# Build a mini split dir: news.tsv + sub-chunk as behaviors.tsv
MINI=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/chunks/sub/${SUB}/split_${PART}
mkdir -p ${MINI}
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/news.tsv ${MINI}/news.tsv
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/chunks/sub/${SUB}/chunk_${PART}.tsv ${MINI}/behaviors.tsv

MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/chunks/sub \
USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=30 \
USE_RECENCY=${EVAL_USE_RECENCY:-1} \
HISTORY_SELECTION=${EVAL_HISTORY_SELECTION:-recent} \
HISTORY_FORMAT=flat \
PREFIX_CACHE_MODE=${EVAL_PREFIX_CACHE_MODE:-auto} \
OUTPUT_SCORES_FILE=${OUT} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/eval_mind_pointwise.sh ${CKPT} ${SUB}/split_${PART} 2>&1 | tee logs/test_sub_${SUB}_${PART}.log

# ---- verify: line count and per-impression candidate count ----
python3 - "$SUB" "$PART" "$OUT" <<'PY'
import sys
sub, part, out = sys.argv[1], sys.argv[2], sys.argv[3]
want = {}
with open(f"data/MIND_large/test/chunks/sub/{sub}/chunk_{part}.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        want[p[0]] = len(p[4].split())
n = bad = 0
with open(out) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) - 1 != want.get(p[0], -1):
            bad += 1
        n += 1
print(f"VERIFY {out}: lines={n} (want {len(want)})  mismatches={bad}")
assert n == len(want), f"line count {n} != {len(want)}"
assert bad == 0, f"{bad} candidate-count mismatches"
PY

echo "=== sub-chunk ${SUB}/${PART} DONE ==="
