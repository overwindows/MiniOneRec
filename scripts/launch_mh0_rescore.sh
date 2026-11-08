#!/bin/bash
# Launch the MAX_HISTORY=0 (unlimited history) production re-score.
#
# WHY
# ---
# The MAX_HISTORY lever was measured paired on 40,000 dev impressions:
# lifting the inference-side history cap from 30 to unlimited, with the SAME
# ckpt-45500 weights, gained +0.00392 AUC (SE 0.00036, t=+10.77) at the LLM
# level, and +0.00333 AUC (SE 0.00035, t=+9.47) after the v6 stack ranker
# (scripts/verify_mh0_stack.py). Zero retraining: the cap was an inference-side
# choice hardcoded in the production scorers to match the TRAINING budget.
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
#   bash scripts/launch_mh0_rescore.sh test --dry-run # print, submit nothing
#
# Env:
#   MH0_CKPT   (default results/ranker/lg_9b_ckpt45500_snapshot)
#   MH0_POOL   node pool, "node:gpus:reservation" entries separated by spaces.
#               Reservation "-" means no --reservation flag. Units are assigned
#               to pool slots round-robin.
#   MH0_MAXJOBS (default 0 = no limit)
#   MH0_ONLY   comma-separated unit ids to restrict to (e.g. test_c00_03)
#
#   Default pool: 3x B200 (8 GPU, needs the sn50-hdi-1 reservation), 1x A100
#   (8 GPU), 1x A100 (4 GPU), 1x H100 (2 GPU).
#
#   The B200 nodes REQUIRE --reservation=sn50-hdi-1: without it sbatch leaves
#   the job PENDING with ReqNodeNotAvail (verified by probe 16393545/16393547).
#
# JOBS ARE PINNED TO A NODE with --nodelist, so Slurm serialises the queue on
# that node: N pinned jobs on one node still occupy exactly ONE node and one
# GPU set at a time. The footprint is (number of nodes in the pool), not
# (number of jobs).
#
# IDEMPOTENT: a unit is skipped when its output file already exists OR when a
# job for it is already queued/running, so re-running the launcher after a
# partial failure only submits what is missing.
set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"

WHICH=${1:?usage: launch_mh0_rescore.sh dev|test|train|all [--dry-run]}
shift || true
DRYRUN=0
for a in "$@"; do
  [ "$a" = "--dry-run" ] && DRYRUN=1
done

CKPT=${MH0_CKPT:-results/ranker/lg_9b_ckpt45500_snapshot}
# node:gpus:reservation
POOL=(${MH0_POOL:-sc3-cp105:8:sn50-hdi-1 sc3-cp107:8:sn50-hdi-1 sc3-cp108:8:sn50-hdi-1 \
                 sc-c96:8:- sc3-c98:4:- sc-c120:2:-})
MAXJOBS=${MH0_MAXJOBS:-0}          # 0 = no limit
ONLY=${MH0_ONLY:-}

if [ ! -d "${CKPT}" ]; then
  echo "FATAL: checkpoint dir ${CKPT} does not exist"
  exit 1
fi
if [ -f "${CKPT}/adapter_config.json" ] && [ ! -f "${CKPT}/adapter_model.safetensors" ]; then
  echo "FATAL: ${CKPT} has adapter_config.json but no adapter_model.safetensors"
  exit 1
fi

# ---- build the unit list --------------------------------------------------
# id|kind|script|envs|outdir|filename|chunk
#
# The chunk path is the behaviors file the unit scores. Its line count is the
# EXPECTED line count of the score file, which the launcher uses to tell a
# finished unit from a truncated one (a GPU crash mid-run leaves a partial
# score file behind -- see the failed_gpus guard in eval_mind_pointwise.sh).
UNITS=()
add_units () {
  local kind=$1
  case $kind in
    dev)
      local p
      for p in 00 01 02 03 04 05; do
        UNITS+=("dev_${p}|dev|scripts/eval_dev_subchunk_large.sh|DE_PART=${p}|results/ranker/dev_scores/mh0|sub_d00_${p}.tsv|data/MIND_large/dev/chunks/sub/d00/chunk_${p}.tsv")
      done
      ;;
    test)
      local s p
      for s in c00 c01 c02 c03 c04 c05; do
        for p in 00 01 02 03 04 05; do
          UNITS+=("test_${s}_${p}|test|scripts/eval_test_subchunk_large.sh|TE_SUB=${s},TE_PART=${p}|results/ranker/test_scores/mh0|sub_${s}_${p}.tsv|data/MIND_large/test/chunks/sub/${s}/chunk_${p}.tsv")
        done
      done
      ;;
    train)
      local p
      for p in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 \
               16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33; do
        UNITS+=("train_${p}|train|scripts/eval_train_subchunk_large.sh|TR_PART=${p}|results/ranker/train_scores/mh0|sub_tr00_${p}.tsv|data/MIND_large/train/chunks/sub/tr00/chunk_${p}.tsv")
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
echo "=== pool: ${POOL[*]} ==="
[ -n "${ONLY}" ] && echo "=== restricted to: ${ONLY} ==="
[ "$DRYRUN" = "1" ] && echo "=== DRY RUN: nothing will be submitted ==="

i=0
skipped=0

# Which unit is a queued/running job working on?
#
# New jobs are named mh0_<uid>, so the name identifies the unit exactly. But
# the OLDER launcher submitted one job per KIND (all six dev jobs were named
# "mh0_dev"), so the name alone cannot say which part each one is scoring.
# Those jobs do record it in their log header ("=== dev sub-chunk d00/03 ..."),
# so read it back instead of guessing.
unit_of_job () {
  local jid=$1 kind=$2
  local log="logs/mh0_${kind}_${jid}.log"
  [ -f "$log" ] || return 1
  # "=== dev sub-chunk d00/03 ckpt=..."  /  "=== test sub-chunk c00/03 ..."
  #                             "=== train sub-chunk tr00/03 ..."
  local sub part
  sub=$(sed -n 's/^=== [a-z]* sub-chunk \([a-z0-9]*\)\/\([0-9]*\) .*/\1/p' "$log" | head -1)
  part=$(sed -n 's/^=== [a-z]* sub-chunk \([a-z0-9]*\)\/\([0-9]*\) .*/\2/p' "$log" | head -1)
  [ -z "$part" ] && return 1
  case "$kind" in
    dev)   echo "dev_${part}" ;;
    test)  echo "test_${sub}_${part}" ;;
    train) echo "train_${part}" ;;
    *)     return 1 ;;
  esac
}

# Is any queued/running job already covering this unit?
unit_is_busy () {
  local uid=$1 kind=$2
  local jid
  # exact per-unit name (new-style jobs)
  for jid in $(squeue -h -n "mh0_${uid}" -o "%i" 2>/dev/null); do
    echo "    (job ${jid} mh0_${uid})"
    return 0
  done
  # legacy per-kind name: match on the unit the job's log says it is scoring
  for jid in $(squeue -h -n "mh0_${kind}" -o "%i" 2>/dev/null); do
    local u
    u=$(unit_of_job "$jid" "$kind" 2>/dev/null || true)
    if [ "$u" = "$uid" ]; then
      echo "    (job ${jid} mh0_${kind} is scoring ${uid})"
      return 0
    fi
  done
  return 1
}

for u in "${UNITS[@]}"; do
  IFS='|' read -r uid kind script envs outdir fname chunk <<< "$u"

  # MH0_ONLY filter.
  if [ -n "${ONLY}" ] && [[ ",${ONLY}," != *",${uid},"* ]]; then
    continue
  fi
  if [ "$MAXJOBS" -gt 0 ] && [ "$i" -ge "$MAXJOBS" ]; then break; fi

  # Idempotency: skip a unit whose output is COMPLETE, or whose job is
  # already queued/running.
  #
  # "Complete" means the score file has exactly as many lines as the chunk it
  # scores -- NOT merely that the file exists. A GPU crash mid-run leaves a
  # truncated score file behind (jobs 16381792/16381793 wrote 31,372 and
  # 47,059 of 62,745 lines), and treating "exists" as "done" would skip those
  # units forever and silently ship a short score table.
  if [ -s "${outdir}/${fname}" ]; then
    want=$(wc -l < "${chunk}")
    got=$(wc -l < "${outdir}/${fname}")
    if [ "$got" -eq "$want" ]; then
      echo "  SKIP ${uid} (${outdir}/${fname} complete: ${got} lines)"
      skipped=$((skipped+1)); i=$((i+1)); continue
    fi
    echo "  RE-RUN ${uid} (${outdir}/${fname} is TRUNCATED: ${got}/${want} lines)"
    # The scoring script truncates its output file before writing, so the
    # stale partial file needs no explicit removal. (Deleting it here would
    # also make --dry-run destructive, which it must not be.)
  fi
  if busy=$(unit_is_busy "$uid" "$kind"); then
    echo "  SKIP ${uid} (already queued: ${busy})"
    skipped=$((skipped+1)); i=$((i+1)); continue
  fi

  # Round-robin over the pool.
  slot=$((i % ${#POOL[@]}))
  IFS=':' read -r n g r <<< "${POOL[$slot]}"

  mkdir -p "${outdir}"
  RES=""
  if [ "$r" != "-" ]; then RES="--reservation=${r}"; fi

  # 8 GPUs on an H200 finish a 62k-impression sub-chunk in ~30 min with the
  # hybrid prefix cache; an A100 is ~2x slower per GPU and a 2-GPU H100 ~8x.
  # Keep the limits generous -- the jobs are serialised per node, so a timeout
  # would stall the queue.
  if [ "$g" -ge 8 ]; then TL=10:00:00; else TL=16:00:00; fi

  # --export takes COMMA-separated VAR=VAL pairs (no spaces).
  EXPORT="ALL,EVAL_MAX_HISTORY=0,EVAL_LOG_TAG=_mh0"
  EXPORT="${EXPORT},DE_CKPT=${CKPT},TE_CKPT=${CKPT},TR_CKPT=${CKPT}"
  EXPORT="${EXPORT},DE_OUTDIR=${outdir},TE_OUTDIR=${outdir},TR_OUTDIR=${outdir}"
  EXPORT="${EXPORT},${envs}"

  if [ "$DRYRUN" = "1" ]; then
    echo "  ${uid} -> node=${n} gres=${g} res=${r}  [dry-run]"
    i=$((i+1)); continue
  fi

  # shellcheck disable=SC2086
  JOB=$(sbatch \
    --job-name=mh0_${uid} \
    --partition=gpuonly \
    --nodes=1 \
    --gres=gpu:${g} \
    --cpus-per-task=32 \
    --mem=400000 \
    --time=${TL} \
    --nodelist="${n}" \
    ${RES} \
    --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/mh0_${uid}_%j.log \
    --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/mh0_${uid}_%j.err \
    --export="${EXPORT}" \
    "${script}")

  echo "  ${uid} -> node=${n} gres=${g} res=${r}  ${JOB}"
  i=$((i+1))
done

echo "=== submitted ${i} job(s), skipped ${skipped} already-done/queued unit(s) ==="
