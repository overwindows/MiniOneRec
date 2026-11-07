#!/bin/bash
# Sunday restart dispatcher — relaunches everything that was paused/cancelled
# on Fri 2026-11-06.
#
# Scheduled with `at` for Sun 2026-11-08 09:05 PST.
#
# WHY 09:05 AND NOT 08:00
# -----------------------
# The reservation `sn50-hdi-1` is DAILY 09:00 -> 00:00 (scontrol shows
# StartTime=09:00 EndTime=00:00 Flags=DAILY). A job submitted at 08:00 waits
# for the window to open anyway, but submitting at 09:05 lands the jobs
# inside the window with margin instead of racing its edge.
#
# What gets relaunched:
#   A) 6x te_ck89000  — score the lg_9b ckpt-89000 snapshot on the MIND_large
#                       test split (6 chunks x 395k impressions). A chunk whose
#                       score file already exists is SKIPPED: the original 6
#                       jobs are still running, so re-submitting would
#                       duplicate 48 B200 GPUs of work.
#   B) 4x training    — resume lg_4b / lg_gemma26 / lg_q8b / lg_dedupbf from
#                       their last deepspeed checkpoint.
#   C) 4x watchers    — resume_run.sh chains so a 24h wall-clock expiry
#                       auto-resubmits the training job instead of dying.
#
# NOT relaunched: te_ck47000 (still running, finishes on its own) and
# lg_9b/small_te (still running, flagship run).
set -u
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"

LOG=logs/sunday_restart.log
exec >> "$LOG" 2>&1
echo "=== SUNDAY RESTART START $(date) ==="

# Idempotency guard: `at` and a manual run could both fire. Refuse to run
# twice so we never double-submit 10 jobs.
LOCK=logs/.sunday_restart.done
if [ -f "$LOCK" ]; then
  echo "=== already ran (lock $LOCK present), exiting ==="
  exit 0
fi
touch "$LOCK"

# Nodes known-bad from the cluster runbook do-not-use list. cp79 / cp84 /
# cp88 are NOT excluded: they each ran a full 8-GPU job successfully today.
EXCL="sc3-cp78,sc3-cp80,sc3-cp83,sc3-cp98,sc3-cp104,sc3-cp106"
SBATCH_OPTS=(--reservation=sn50-hdi-1 --constraint="host_os_release=10.2" -x "$EXCL")
MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large

# ---------------------------------------------------------------- A) ck89000
# Written to results/ranker/test_scores/ck89000_<chunk>.tsv, one line per
# impression, scores tab-separated in candidate order.
#
# The original 6 jobs (16180106 / 16187236-40) are STILL RUNNING as of
# 2026-11-07 and each takes ~18 h. Re-submitting them would duplicate 48
# B200 GPUs of work and slow down the training jobs we actually need. So:
#   - if ANY te_ck89000 job is in the queue, skip the whole section;
#   - otherwise submit the chunks whose score file is missing.
echo "--- A) ck89000 test scoring ---"
CK89000_JOBS=$(squeue -u chenw -h -n te_ck89000 2>/dev/null | wc -l)
if [ "${CK89000_JOBS}" -gt 0 ]; then
  echo "  SKIPPED: ${CK89000_JOBS} te_ck89000 job(s) already in the queue"
  squeue -u chenw -o "%.10i %.12j %.8T %.10M %R" -n te_ck89000
else
  for CHUNK in 0 1 2 3 4 5; do
    CH=$(printf "%02d" ${CHUNK})
    OUT=results/ranker/test_scores/ck89000_${CH}.tsv
    if [ -s "${OUT}" ]; then
      echo "  chunk ${CH}: ${OUT} exists, skipping"
      continue
    fi
    TE_CHUNK=${CHUNK} \
    TE_CKPT=results/ranker/lg_9b_ckpt89000_snapshot \
    TE_TAG=ck89000 \
    sbatch "${SBATCH_OPTS[@]}" --job-name=te_ck89000 \
      scripts/eval_test_chunk_large.sh | awk '{print "  te_ck89000 chunk '"${CHUNK}"' -> "$4}'
  done
fi

# ------------------------------------------------------------- B) training
# Each run resumes from its newest checkpoint. MIND_ROOT and TE_USE_RECENCY
# are mandatory: without them the job silently trains on MIND_small and drops
# the recency prompt.
#
# WHAT CHANGED (2026-11-07) -- THE ORIGINAL 4x RESUME PLAN WAS DEAD ON ARRIVAL
# ----------------------------------------------------------------------
# Measured before relaunching anything:
#   * lg_4b      ep 0.1798 of 3.0,  647 steps/h  -> 170 h = 7.1 days left
#   * lg_q8b     ep 0.3724 of 3.0,  538 steps/h  -> 190 h = 7.9 days left
#   * lg_dedupbf ep 0.3595 of 3.0,  521 steps/h  -> 197 h = 8.2 days left
#   * lg_gemma26 ep 0.0642 of 3.0,  289 steps/h  -> 396 h = 16.5 days left
# None of them can finish. Their dev AUC at these epochs would be ~0.69-0.70,
# versus lg_9b's 0.7181 (ck45500) -- so the resumes would burn 4 nodes for
# 1-2.5 weeks to produce numbers we already know are worse.
#
# lg_9b itself is DONE as a model: dev AUC by checkpoint is
#   ck45500 (ep 1.17) 0.7181  <- best
#   ck47000 (ep 1.21) 0.7163
#   ck89000 (ep 2.29) 0.7146
# i.e. it peaked early and then declined. More epochs of the same recipe do
# not help.
#
# So: NO training resumes. The GPUs go to the levers that can actually
# produce a better SUBMITTED model in the time available.
#
# Section B used to call submit_train() for 4 runs; that helper and its calls
# are gone. What is running instead (submitted 2026-11-07, all already in
# the queue when this script fires):
#   * pref45500_dpo / pref45500_rlcd  -- DPO and RLCD from ck45500, our best
#     checkpoint (dev AUC 0.7181), on the B200 reservation.
#   * parab (6 parts)                -- the paired prompt-parity A/B on the
#     idle A100/H100 nodes.
#   * te_ck89000 chunks              -- the last missing test score files.
echo "--- B) no training resumes here (see comment); the real jobs are already submitted ---"

# ------------------------------------------------------------- C) watchers
# REMOVED 2026-11-07. The original Section C launched 4x resume_run.sh
# watchers for lg_4b / lg_gemma26 / lg_q8b / lg_dedupbf, but Section B no
# longer starts those runs, so the watchers had nothing to watch and would
# have RESUBMITTED them from scratch -- including the lg_dedupbf watcher,
# which carried `--use_dedup_backfill True`, a flag that is NOT one of
# train()'s parameters (fire silently drops it, so the run would have trained
# `history_selection=recent` under a "dedup_backfill" name -- exactly the
# false label that made the 9B dedupbf runs duplicates of lg_9b).
#
# On top of that, all four runs provably cannot finish in the time available
# (measured in the Section B comment: 170 h / 190 h / 197 h / 396 h
# remaining), so a watcher chain would hold 4 B200 nodes for weeks to
# produce models we already know score below lg_9b.
#
# The GPUs go to the levers that can produce a better SUBMITTED model:
# DPO/RLCD from ck45500, the prompt-parity A/B on the idle A100/H100 nodes,
# and the hybrid-mode dev/test re-scores.
echo "--- C) no training resume watchers (see comment) ---"

sleep 20
echo "--- queue after launch ---"
squeue -u chenw -o "%.10i %.12j %.8T %.10M %R"
echo "=== SUNDAY RESTART DONE $(date) ==="
