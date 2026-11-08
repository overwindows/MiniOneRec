#!/bin/bash
# Score ONE sub-chunk of the MIND_large DEV split with a checkpoint (8 GPUs).
#
# WHY SUB-CHUNKS FOR DEV
# ----------------------
# A full-dev eval takes 23.5 h on one node (job 14704401). The stack ranker
# must be retrained on the ckpt-45500 dev scores, so that 23.5 h is on the
# critical path. Splitting dev into 6 sequential slices of 62,745 impressions
# turns it into 6 parallel jobs of ~3 h each.
#
# The 6 slices are exact sequential line ranges of data/MIND_large/dev/
# behaviors.tsv (verified: concatenating chunk_00..chunk_05 reproduces the
# original impression order 1..376471), so the merged score file is
# byte-identical in ordering to a single full-dev run.
#
# Usage: DE_PART=NN [DE_CKPT=path] [DE_OUTDIR=dir] sbatch scripts/eval_dev_subchunk_large.sh
# Writes raw scores to ${DE_OUTDIR}/sub_d00_${DE_PART}.tsv
#SBATCH --job-name=lg_dsub
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
CKPT=${DE_CKPT:-results/ranker/lg_9b_ckpt45500_snapshot}
OUTDIR=${DE_OUTDIR:-results/ranker/dev_scores/ck45500}
OUT=${OUTDIR}/sub_d00_${PART}.tsv

echo "=== dev sub-chunk d00/${PART} ckpt=${CKPT} out=${OUT} ==="
mkdir -p ${OUTDIR}

# ---- GPU health pre-check ----
# device_map="auto" silently falls back to CPU when one GPU is unhealthy,
# which burns hours on a CPU forward pass. Refuse to start instead.
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

# The mini split dir (behaviors.tsv symlink + news.tsv symlink) is created
# once by scripts/prepare_dev_subchunks.sh; create it here too so the job
# is self-contained.
MINI=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/dev/chunks/sub/d00/split_${PART}
mkdir -p ${MINI}
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/dev/news.tsv ${MINI}/news.tsv
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/dev/chunks/sub/d00/chunk_${PART}.tsv ${MINI}/behaviors.tsv

# USE_RECENCY=1 must match the training config AND the test-side scoring
# (scripts/eval_test_subchunk_large.sh hardcodes 1). A mismatch shifts the
# llm_score distribution and breaks the ranker's llm_rank / llm_z calibration.
#
# HISTORY_SELECTION / USE_RECENCY used to be hardcoded to "recent"/1 here,
# which silently mis-scored any checkpoint trained with a different history
# policy (e.g. the dedup_backfill 1.7B model). They are now overridable via
# the SAME env var names used by the test-side scripts, so a dev/test pair
# can never drift apart:
#   EVAL_HISTORY_SELECTION (default recent) | EVAL_USE_RECENCY (default 1)
# EVAL_MAX_HISTORY: how many history items the prompt may carry (0 = unlimited).
#
# THIS USED TO BE HARDCODED TO 30 -- the same budget TRAINING used -- so eval
# silently cut the real user history of ~35% of dev impressions down to their
# last 30 items. That cap is an INFERENCE-TIME choice, not a model property:
# raising it to unlimited at eval time, with the SAME weights, was measured
# paired on 40,000 dev impressions and gained
#   +0.00392 AUC (SE 0.00036, t=+10.77)
# with the delta bit-identically 0.00000 on the 65% of impressions that carry
# <=30 items, and growing monotonically with history length
# (31-50: +0.00551, 51-100: +0.01317, >100: +0.02091).
# Cost: 1.19x wall-clock. Keep the dev and test caps identical.
EVAL_MAX_HISTORY=${EVAL_MAX_HISTORY:-0}
MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/dev/chunks/sub \
USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=${EVAL_MAX_HISTORY} \
USE_RECENCY=${EVAL_USE_RECENCY:-1} \
HISTORY_SELECTION=${EVAL_HISTORY_SELECTION:-recent} \
HISTORY_FORMAT=flat \
PREFIX_CACHE_MODE=${EVAL_PREFIX_CACHE_MODE:-auto} \
OUTPUT_SCORES_FILE=${OUT} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/eval_mind_pointwise.sh ${CKPT} d00/split_${PART} 2>&1 | tee logs/dev_sub_${PART}.log

# ---- verify: line count and per-impression candidate count ----
python3 - "$PART" "$OUT" <<'PY'
import sys
part, out = sys.argv[1], sys.argv[2]
want = {}
with open(f"data/MIND_large/dev/chunks/sub/d00/chunk_{part}.tsv") as f:
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

echo "=== dev sub-chunk d00/${PART} DONE ==="
