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
# Delegates to the shared preflight: it resolves the PHYSICAL device indices
# Slurm actually allocated (a shared node hands out e.g. 2,3,4,5, not 0..3),
# refuses to start if one of OUR GPUs is busy, and preloads the CUDA
# forward-compat lib on nodes whose driver is too old for the cu130 torch
# build (A100/H100 carry 565.57.01 => torch.cuda.is_available() is False and
# the 9B would silently load on CPU).
# shellcheck source=scripts/_gpu_preflight.sh
source scripts/_gpu_preflight.sh
mkgpu_preflight "${TE_NGPU:-0}" || exit 1

# Build a mini split dir: news.tsv + sub-chunk as behaviors.tsv
MINI=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/chunks/sub/${SUB}/split_${PART}
mkdir -p ${MINI}
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/news.tsv ${MINI}/news.tsv
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/chunks/sub/${SUB}/chunk_${PART}.tsv ${MINI}/behaviors.tsv

# EVAL_MAX_HISTORY: how many history items the prompt may carry (0 = unlimited).
#
# THIS USED TO BE HARDCODED TO 30 -- the same budget TRAINING used -- so eval
# silently cut the real user history of ~43% of test impressions down to their
# last 30 items. That cap is an INFERENCE-TIME choice, not a model property:
# raising it to unlimited at eval time, with the SAME weights, was measured
# paired on 40,000 dev impressions and gained
#   +0.00392 AUC (SE 0.00036, t=+10.77)
# with the delta bit-identically 0.00000 on the 65% of impressions that carry
# <=30 items, and growing monotonically with history length
# (31-50: +0.00551, 51-100: +0.01317, >100: +0.02091).
# Cost: 1.19x wall-clock. Keep the dev and test caps identical.
EVAL_MAX_HISTORY=${EVAL_MAX_HISTORY:-0}
MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/chunks/sub \
USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=${EVAL_MAX_HISTORY} \
USE_RECENCY=${EVAL_USE_RECENCY:-1} \
HISTORY_SELECTION=${EVAL_HISTORY_SELECTION:-recent} \
HISTORY_FORMAT=flat \
PREFIX_CACHE_MODE=${EVAL_PREFIX_CACHE_MODE:-auto} \
OUTPUT_SCORES_FILE=${OUT} \
CUDA_VISIBLE_DEVICES=${GPUS} \
bash scripts/eval_mind_pointwise.sh ${CKPT} ${SUB}/split_${PART} 2>&1 | tee logs/test_sub_${SUB}_${PART}${EVAL_LOG_TAG:-}.log

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
