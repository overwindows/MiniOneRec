#!/bin/bash
# Auto-resume chain for the lg_9b MIND_large training run.
#
# The job hits the 24h wall before finishing 3 epochs, so it must be
# resumed from the latest checkpoint. This watcher:
#   1. waits for the current job to leave the queue
#   2. picks the highest-numbered checkpoint-<step>
#   3. resubmits with --resume_from_checkpoint <ckpt>
#   4. repeats until training completes (final_checkpoint exists)
#
# MIND_ROOT and TE_USE_RECENCY are REQUIRED on resume: without them the
# job silently trains on MIND_small without the recency prompt.
#
# --reservation / --constraint are ALSO required: without them the job
# lands on a node that has no gpu:b200m180 gres and dies at launch.
# (That is what killed the previous watcher chain.)
set -u
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"

RUN_TAG=lg_9b
OUT=output_dir/small_runs/${RUN_TAG}
LOG=logs/resume_${RUN_TAG}.log
MAX_RESUMES=12

exec >> $LOG 2>&1
echo "=== resume watcher start $(date) ==="

# The output dir is shared with an earlier MIND_small run of the same tag, so
# final_checkpoint may already exist from THAT run. Only a final_checkpoint
# written AFTER this watcher started belongs to the MIND_large run.
START_TS=$(date +%s)

# The job that is running right now (submitted by hand before this watcher).
CUR_JOB=$(squeue -u chenw -h -o "%i %j" | awk '$2=="small_te"{print $1}' | head -1)
echo "=== watching current job: ${CUR_JOB:-none} ==="

for attempt in $(seq 1 $MAX_RESUMES); do
  # wait until the job we are tracking leaves the queue
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
  sbatch --time=1-00:00:00 --job-name=small_te \
    --reservation=sn50-hdi-1 --constraint="host_os_release=10.2" -x sc3-cp80 \
    scripts/train_eval_small_b200.sh ${RUN_TAG} models/Qwen3.5-9B \
    --use_lora True --lora_r 16 --lora_alpha 32 --learning_rate 1e-4 \
    --micro_batch_size 4 --use_recency True \
    --resume_from_checkpoint ${CKPT} | awk '{print $4}')
  echo "=== submitted job ${OUT_JOB} ==="
  CUR_JOB=${OUT_JOB}

  sleep 90
done
echo "=== resume watcher done $(date) ==="
