#!/bin/bash
# Score ONE sub-chunk of the MIND_large TRAIN split with a checkpoint (8 GPUs).
#
# WHY
# ---
# The stack ranker must be trained on TRAIN labels (dev is evaluation-only).
# The train feature table (large_train_features.tsv) carries llm_score from
# the l13_large model; the production stack uses lg_9b, so the train split
# must be re-scored with the SAME checkpoint that scores dev/test.
#
# 34 sequential sub-chunks of 65,853 impressions (last 59,599) = 2,232,748
# train impressions. Concatenating chunk_00..chunk_33 reproduces the original
# train impression order, so the merged score file aligns row-for-row with
# large_train_features.tsv.
#
# Usage: TR_PART=NN [TR_CKPT=path] sbatch scripts/eval_train_subchunk_large.sh
# Writes raw scores to results/ranker/train_scores/ck45500/sub_tr00_${TR_PART}.tsv
#SBATCH --job-name=lg_trsub
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

PART=$(printf "%02d" $((10#${TR_PART:?set TR_PART=00..33})))
CKPT=${TR_CKPT:-results/ranker/lg_9b_ckpt45500_snapshot}
OUTDIR=${TR_OUTDIR:-results/ranker/train_scores/ck45500}
OUT=${OUTDIR}/sub_tr00_${PART}.tsv

echo "=== train sub-chunk tr00/${PART} ckpt=${CKPT} out=${OUT} ==="
mkdir -p ${OUTDIR}

# ---- GPU health pre-check (same as test/dev sub-chunk jobs) ----
# Delegates to the shared preflight: it resolves the PHYSICAL device indices
# Slurm actually allocated (a shared node hands out e.g. 2,3,4,5, not 0..3),
# refuses to start if one of OUR GPUs is busy, and preloads the CUDA
# forward-compat lib on nodes whose driver is too old for the cu130 torch
# build (A100/H100 carry 565.57.01 => torch.cuda.is_available() is False and
# the 9B would silently load on CPU).
# shellcheck source=scripts/_gpu_preflight.sh
source scripts/_gpu_preflight.sh
mkgpu_preflight "${TR_NGPU:-0}" || exit 1

# Build a mini split dir: news.tsv + sub-chunk as behaviors.tsv
MINI=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/train/chunks/sub/tr00/split_${PART}
mkdir -p ${MINI}
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/train/news.tsv ${MINI}/news.tsv
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/train/chunks/sub/tr00/chunk_${PART}.tsv ${MINI}/behaviors.tsv

# USE_RECENCY=1 etc. must match the dev/test scoring config exactly.
# HISTORY_SELECTION / USE_RECENCY use the SAME env var names as the dev/test
# scripts so the three splits can never drift apart.
# EVAL_MAX_HISTORY: how many history items the prompt may carry (0 = unlimited).
#
# THIS USED TO BE HARDCODED TO 30 -- the same budget TRAINING used -- so eval
# silently cut the real user history of ~35% of train impressions down to their
# last 30 items. That cap is an INFERENCE-TIME choice, not a model property:
# raising it to unlimited at eval time, with the SAME weights, was measured
# paired on 40,000 dev impressions and gained
#   +0.00392 AUC (SE 0.00036, t=+10.77)
# with the delta bit-identically 0.00000 on the 65% of impressions that carry
# <=30 items, and growing monotonically with history length
# (31-50: +0.00551, 51-100: +0.01317, >100: +0.02091).
# Cost: 1.19x wall-clock. Keep the train, dev and test caps identical --
# the stack ranker is FITTED on these train features, so a cap mismatch
# between train and test would shift llm_score / llm_rank / llm_z.
EVAL_MAX_HISTORY=${EVAL_MAX_HISTORY:-0}
MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/train/chunks/sub \
USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=${EVAL_MAX_HISTORY} \
USE_RECENCY=${EVAL_USE_RECENCY:-1} \
HISTORY_SELECTION=${EVAL_HISTORY_SELECTION:-recent} \
HISTORY_FORMAT=flat \
OUTPUT_SCORES_FILE=${OUT} \
CUDA_VISIBLE_DEVICES=${GPUS} \
bash scripts/eval_mind_pointwise.sh ${CKPT} tr00/split_${PART} 2>&1 | tee logs/train_sub_${PART}${EVAL_LOG_TAG:-}.log

# ---- verify: line count and per-impression candidate count ----
python3 - "$PART" "$OUT" <<'PY'
import sys
part, out = sys.argv[1], sys.argv[2]
want = {}
with open(f"data/MIND_large/train/chunks/sub/tr00/chunk_{part}.tsv") as f:
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

echo "=== train sub-chunk tr00/${PART} DONE ==="