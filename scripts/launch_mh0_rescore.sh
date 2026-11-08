#!/bin/bash
# Launch the MAX_HISTORY=0 (unlimited history) production re-score.
#
# WHY
# ---
# The MAX_HISTORY lever was measured paired on 40,000 dev impressions:
# lifting the inference-side history cap from 30 to unlimited, with the SAME
# ckpt-45500 weights, gained +0.00392 AUC (SE 0.00036, t=+10.77). The delta
# is bit-identically 0.00000 on the 65% of impressions that carry <=30 items
# and grows monotonically with history length (31-50: +0.00551, 51-100:
# +0.01317, >100: +0.02091). The cap was an inference-side choice hardcoded
# in the production scorers to match the TRAINING budget; the model actually
# handles more context than training gave it.
#
# This re-scores dev / test / train with EVAL_MAX_HISTORY=0 so the whole
# pipeline (LLM scores -> ranker features -> ranker -> submission) is
# consistent under the new cap.
#
# USAGE
#   bash scripts/launch_mh0_rescore.sh dev            # 6 dev sub-chunks
#   bash scripts/launch_mh0_rescore.sh test           # 36 test sub-chunks
#   bash scripts/launch_mh0_rescore.sh train          # 34 train sub-chunks
#   bash scripts/launch_mh0_rescore.sh all
#
# Env:
#   MH0_CKPT   (default results/ranker/lg_9b_ckpt45500_snapshot)
#   MH0_NODES  (default "sc3-c126 sc3-c127 sc3-c128 sc3-c129 sc-c96 sc3-c98")
#   MH0_NODE_GPUS  (default "8 8 8 8 8 4")  GPUs to request per node
#   MH0_RESERVATION (default "sn40-hdi-1 sn40-hdi-0 - - - -")  per node
#
# Jobs are PINNED to a node with --nodelist, so Slurm serialises the queue
# on that node: N pinned jobs on one node still occupy exactly ONE node and
# one GPU set at a time. The footprint is (number of nodes in MH0_NODES),
# not (number of jobs).
set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"

WHICH=${1:?usage: launch_mh0_rescore.sh dev|test|train|all}
CKPT=${MH0_CKPT:-results/ranker/lg_9b_ckpt45500_snapshot}
NODES=(${MH0_NODES:-sc3-c126 sc3-c127 sc3-c128 sc3-c129 sc-c96 sc3-c98})
NODE_GPUS=(${MH0_NODE_GPUS:-8 8 8 8 8 4})
RESERVATIONS=(${MH0_RESERVATION:-sn40-hdi-1 sn40-hdi-0 - - - -})
MAXJOBS=${MH0_MAXJOBS:-0}          # 0 = no limit

if [ ! -d "${CKPT}" ]; then
  echo "FATAL: checkpoint dir ${CKPT} does not exist"
  exit 1
fi
if [ -f "${CKPT}/adapter_config.json" ] && [ ! -f "${CKPT}/adapter_model.safetensors" ]; then
  echo "FATAL: ${CKPT} has adapter_config.json but no adapter_model.safetensors"
  exit 1
fi

# ---- build the unit list --------------------------------------------------
UNITS=()
add_units () {
  local kind=$1
  case $kind in
    dev)
      local p
      for p in 00 01 02 03 04 05; do
        UNITS+=("dev|scripts/eval_dev_subchunk_large.sh|DE_PART=${p}|results/ranker/dev_scores/mh0")
      done
      ;;
    test)
      local s p
      for s in c00 c01 c02 c03 c04 c05; do
        for p in 00 01 02 03 04 05; do
          UNITS+=("test|scripts/eval_test_subchunk_large.sh|TE_SUB=${s},TE_PART=${p}|results/ranker/test_scores/mh0")
        done
      done
      ;;
    train)
      local p
      for p in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 \
               16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33; do
        UNITS+=("train|scripts/eval_train_subchunk_large.sh|TR_PART=${p}|results/ranker/train_scores/mh0")
      done
      ;;
  esac
}

case $WHICH in
  dev)   add_units dev ;;
  test)  add_units test ;;
  train) add_units train ;;
  all)   add_units dev; add_units test; add_units train ;;
  *) echo "FATAL: unknown target '${WHICH}'"; exit 1 ;;
esac

echo "=== mh0 re-score: ${WHICH} -- ${#UNITS[@]} units ==="
echo "=== ckpt=${CKPT}  EVAL_MAX_HISTORY=0 (unlimited history) ==="

i=0
for u in "${UNITS[@]}"; do
  if [ "$MAXJOBS" -gt 0 ] && [ "$i" -ge "$MAXJOBS" ]; then break; fi
  IFS='|' read -r kind script envs outdir <<< "$u"
  n=${NODES[$((i % ${#NODES[@]}))]}
  g=${NODE_GPUS[$((i % ${#NODE_GPUS[@]}))]}
  r=${RESERVATIONS[$((i % ${#RESERVATIONS[@]}))]}

  mkdir -p "${outdir}"
  RES=""
  if [ "$r" != "-" ]; then RES="--reservation=${r}"; fi

  # 8 GPUs on an H200 finish a 62k-impression sub-chunk in ~1 h with the
  # hybrid prefix cache; a 4-GPU A100 needs ~4-5 h. Keep the limits generous
  # -- the jobs are serialised per node, so a timeout would stall the queue.
  if [ "$g" -ge 8 ]; then TL=08:00:00; else TL=14:00:00; fi

  # --export takes COMMA-separated VAR=VAL pairs (no spaces).
  EXPORT="ALL,EVAL_MAX_HISTORY=0,EVAL_LOG_TAG=_mh0"
  EXPORT="${EXPORT},DE_CKPT=${CKPT},TE_CKPT=${CKPT},TR_CKPT=${CKPT}"
  EXPORT="${EXPORT},DE_OUTDIR=${outdir},TE_OUTDIR=${outdir},TR_OUTDIR=${outdir}"
  EXPORT="${EXPORT},${envs}"

  # shellcheck disable=SC2086
  JOB=$(sbatch \
    --job-name=mh0_${kind} \
    --partition=gpuonly \
    --nodes=1 \
    --gres=gpu:${g} \
    --cpus-per-task=32 \
    --mem=400000 \
    --time=${TL} \
    --nodelist="${n}" \
    --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/mh0_${kind}_%j.log \
    --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/mh0_${kind}_%j.err \
    --export="${EXPORT}" \
    "${script}")

  echo "  ${kind} ${envs} -> node=${n} gres=${g} res=${r}  ${JOB}"
  i=$((i+1))
done

echo "=== submitted ${i} job(s) ==="
