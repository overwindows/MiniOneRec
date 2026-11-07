#!/bin/bash
# Generic auto-resume chain for MIND_large training runs that hit the 24h
# wall before finishing 3 epochs.
#
# Usage: bash scripts/resume_run.sh <RUN_TAG> <MODEL> [EXTRA_TRAIN_ARGS...]
#   RUN_TAG  e.g. lg_4b / lg_gemma26 / lg_q8b / lg_dedupbf
#   MODEL    e.g. models/Qwen3.5-4B
#
# Mirrors resume_lg9b.sh: waits for the current job, picks the highest
# checkpoint-<step>, resubmits with --resume_from_checkpoint, repeats until
# final_checkpoint is written after this watcher started.
#
# MIND_ROOT and TE_USE_RECENCY are REQUIRED on resume: without them the job
# silently trains on MIND_small without the recency prompt.
# --reservation / --constraint are ALSO required: without them the job
# lands on a node that has no gpu:b200m180 gres and dies at launch.
set -u
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"

RUN_TAG=$1
MODEL=$2
shift 2

OUT=output_dir/small_runs/${RUN_TAG}
LOG=logs/resume_${RUN_TAG}.log
MAX_RESUMES=12

exec >> $LOG 2>&1
echo "=== resume watcher start $(date) ==="

# The output dir may be shared with an earlier MIND_small run of the same
# tag, so only a final_checkpoint written AFTER this watcher started counts.
START_TS=$(date +%s)

# The job that is running right now (if any).
CUR_JOB=$(squeue -u chenw -h -o "%i %j" | awk -v t="${RUN_TAG}" '$2==t{print $1}' | head -1)
echo "=== watching current job: ${CUR_JOB:-none} ==="

for attempt in $(seq 1 $MAX_RESUMES); do
  if [ -n "${CUR_JOB:-}" ]; then
    while squeue -h -j "${CUR_JOB}" 2>/dev/null | grep -q .; do
      sleep 120
    done
    echo "=== job ${CUR_JOB} finished $(date) ==="
  fi

  if [ -f "${OUT}/final_checkpoint/config.json" ]; then
    FC_TS=$(stat -c %Y "${OUT}/final_checkpoint/config.json")
    if [ "$FC_TS" -gt "$START_TS" ]; then
      echo "=== training complete (final_checkpoint written by this run) $(date) ==="
      break
    fi
    echo "=== final_checkpoint is stale (mtime $FC_TS <= start $START_TS), ignoring ==="
  fi

  CKPT=$(ls -d ${OUT}/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  if [ -z "${CKPT}" ]; then
    echo "=== no checkpoint found, aborting $(date) ==="
    break
  fi
  CKPT=${OUT}/checkpoint-${CKPT}
  echo "=== attempt ${attempt}: resuming from ${CKPT} $(date) ==="

  OUT_JOB=$(MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large \
  TE_USE_RECENCY=1 \
  sbatch --time=1-00:00:00 --job-name=${RUN_TAG} \
    --reservation=sn50-hdi-1 --constraint="host_os_release=10.2" -x sc3-cp80 \
    scripts/train_eval_small_b200.sh ${RUN_TAG} ${MODEL} \
    "$@" \
    --resume_from_checkpoint ${CKPT} | awk '{print $4}')
  echo "=== submitted job ${OUT_JOB} ==="
  CUR_JOB=${OUT_JOB}

  sleep 90
done
echo "=== resume watcher done $(date) ==="