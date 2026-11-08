#!/bin/bash
# Score ONE preference checkpoint on the 40k uniform dev subset (chunk_90).
#
# WHY THIS SCRIPT
# ---------------
# The DPO/RLCD-from-ck45500 checkpoints had NEVER been scored on dev. The
# screen compares them against the SFT base (ck45500) PAIRED on the SAME
# impressions, which is the only way to resolve a ~0.001 AUC effect:
#
#   same model, two code paths : per-impression delta sd = 0.03118
#   two different models       : per-impression delta sd = 0.11535
#   => at N=40,000: paired SE 0.00016 (noise) / 0.00058 (real difference)
#      2-sigma detectable delta = 0.00115
#
# The subset's own AUC LEVEL is biased by sampling noise (ck45500 on chunk_90
# = 0.72143 vs 0.71806 on full dev), so ONLY the paired delta is meaningful.
# The winner is afterwards re-scored on the FULL dev split for the final
# number.
#
# USAGE
#   DE_CKPT=<path> DE_TAG=<label> DE_NODE=sc3-c128 sbatch scripts/eval_screen90.sh
#
# ENV
#   DE_CKPT   (required) checkpoint dir: a PEFT adapter dir, or a merged model dir
#   DE_TAG    (required) short label used for the output filename
#   DE_OUTDIR (default results/ranker/dev_scores/screen90)
#   DE_NODE   (default sc3-c128) -- just for the log line; Slurm picks the node
#   DE_NGPU   (default 8)  number of GPUs to use
#   DE_PART   (default 90) chunk index
#
# IMPORTANT: a PEFT adapter dir has NO config.json -- the scorer detects
# adapter_config.json and loads base_model_name_or_path from it. Verify with
# "PEFT adapter checkpoint; base model: models/Qwen3.5-9B" in the log.
#SBATCH --job-name=scr90
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400000
#SBATCH --time=08:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/scr90_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/scr90_%j.err

set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec

CKPT=${DE_CKPT:?set DE_CKPT}
TAG=${DE_TAG:?set DE_TAG}
PART=${DE_PART:-90}
NGPU=${DE_NGPU:-8}
OUTDIR=${DE_OUTDIR:-results/ranker/dev_scores/screen90}

# The chunk must exist before the job starts: the scorer reads it from the
# shared filesystem, and a missing chunk would silently score 0 impressions.
CHUNK=data/MIND_large/dev/chunks/sub/d00/chunk_${PART}.tsv
if [ ! -f "${CHUNK}" ]; then
  echo "FATAL: ${CHUNK} missing -- run scripts/build_dev_subset.py first"
  exit 1
fi
# The checkpoint must exist AND be non-empty. An adapter dir without
# adapter_model.safetensors would be scored as the BASE model and produce a
# fake "no change" result -- that is exactly the failure that killed the
# rlcd3000 arm (checkpoint-3000 did not exist; save_total_limit=2 had
# already pruned it).
if [ ! -d "${CKPT}" ]; then
  echo "FATAL: checkpoint dir ${CKPT} does not exist"
  exit 1
fi
if [ -f "${CKPT}/adapter_config.json" ] && [ ! -f "${CKPT}/adapter_model.safetensors" ]; then
  echo "FATAL: ${CKPT} has adapter_config.json but no adapter_model.safetensors"
  exit 1
fi
if [ ! -f "${CKPT}/adapter_model.safetensors" ] && [ ! -f "${CKPT}/model.safetensors" ]; then
  echo "FATAL: ${CKPT} has neither adapter_model.safetensors nor model.safetensors"
  exit 1
fi

echo "=== ckpt=${CKPT} tag=${TAG} part=${PART} ngpu=${NGPU} $(date) ==="
DE_PART=${PART} DE_CKPT=${CKPT} DE_MODE=hybrid DE_NGPU=${NGPU} \
  DE_OUTDIR=${OUTDIR}/${TAG} DE_TAG=_${TAG} \
  bash scripts/eval_dev_chunk_hs.sh

echo "=== screen arm ${TAG} DONE $(date) ==="
