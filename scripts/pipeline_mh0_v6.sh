#!/bin/bash
# Promote the MAX_HISTORY=0 (mh0) re-score using the EXISTING v6 ranker.
#
# WHY THIS EXISTS SEPARATELY FROM pipeline_mh0_v7.sh
# ---------------------------------------------------
# The mh0 win was MEASURED with the EXISTING v6 ranker:
# scripts/verify_mh0_stack.py swapped only the llm_score column of the
# canonical dev feature table between the cap-30 and unlimited dumps and
# re-applied the v6 ranker to both arms. The FULL-DEV run of that comparison
# is the artifact of record:
#
#   results/ranker/mh0v6_paired_dev_summary.json
#   STACK v6 on the full dev split (376,471 impressions)
#     0.72337 -> 0.72642, paired delta +0.00305 (SE 0.00013, t=+22.78)
#
# (An earlier 40,000-impression subset run of the same script reported
# +0.00333 / t=+9.47, but its log crashed before writing a summary and the
# number was never reproduced -- the full-dev figure above supersedes it.)
#
# So the v6 ranker applied to mh0 scores is a MEASURED configuration, and it
# needs NO train re-score at all -- only dev + test scores. The train
# re-score exists solely to REFIT the ranker (v7), which is an unmeasured
# extra, so it must not block the submission.
#
# v6 also happens to be the model whose dev AUC (0.7234) and Codabench test
# AUC (0.7232, sub 1036308) we have already published, so the delta from v6
# to v6+mh0 is exactly the delta we claim.
#
# CHAIN
# -----
#   1. wait for the 6 dev + 36 test mh0 sub-chunk scores
#   2. merge each -> lg9b_ckpt45500_{dev,test}_scores_mh0.tsv
#   3. swap llm_score into the DEV feature table, apply v6 -> the ONE honest
#      dev number (per-impression AUC over all 376,471 dev impressions)
#   4. swap llm_score into the TEST feature table, apply v6 -> leaderboard
#      format -> zip
#
# EVALUATION PROTOCOL
# -------------------
# The ranker is FITTED on TRAIN labels (already done: the v6 model file).
# The ranker is NOT refitted here, so dev plays no part in fitting and the
# number produced at step 3 is a measurement, not a selection.
#
# ONE QUALIFICATION, stated honestly: the history cap itself (30 vs 0 vs
# 50 vs 100) was chosen on a 40,000-impression uniform random SUBSET of dev
# (chunk_90, seed 42) -- that subset is ~10.6% of dev, so it is a mild
# double-dip, not a clean held-out selection. The subset screen ranked
# unlimited > 100 > 50 > 30, but the 0-vs-100 difference was NOT significant
# (paired delta -0.00036, t=-1.75), so the choice among the top arms rests on
# the monotone dose-response and on the mechanism (see
# memory/project_max_history_lever.md), not on a significant pairwise win.
# The figure this pipeline reports (+0.00305, t=+22.78) is measured on the
# FULL 376,471-impression dev split and is not the number that was selected
# on.
set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"

LOG=logs/pipeline_mh0_v6.log
exec >> $LOG 2>&1
echo "=== mh0 v6 pipeline start $(date) ==="

DEV_DIR=results/ranker/dev_scores/mh0
TEST_DIR=results/ranker/test_scores/mh0

DEV_MERGED=results/ranker/lg9b_ckpt45500_dev_scores_mh0.tsv
TEST_MERGED=results/ranker/lg9b_ckpt45500_test_scores_mh0.tsv

DEV_TOTAL=$(wc -l < data/MIND_large/dev/behaviors.tsv)
TEST_TOTAL=$(wc -l < data/MIND_large/test/behaviors.tsv)

MODEL=results/ranker/ranker_v6_train_popwit.txt

# ---- 1. wait for every sub-chunk to be COMPLETE ----
# "Complete" means the score file has exactly as many lines as the chunk it
# scores -- not merely that the file exists. A GPU crash mid-run leaves a
# truncated score file behind, and a truncated file still parses and still has
# a plausible line count, so the guard has to compare counts.
wait_sub () {
  local dir=$1 fname=$2 chunk=$3
  # `local want=$(...)` masks a failure: if wc cannot read the chunk, `want`
  # becomes the empty string and the comparison below is never true, so the
  # loop spins forever. Bind first, then check the exit status.
  local want
  want=$(wc -l < "${chunk}") || { echo "FATAL: cannot read ${chunk}"; exit 1; }
  while true; do
    if [ -f "${dir}/${fname}" ]; then
      n=$(wc -l < "${dir}/${fname}") || { echo "FATAL: cannot read ${dir}/${fname}"; exit 1; }
      if [ "$n" -eq "$want" ]; then
        echo "${fname} OK: n=$n (want $want)"
        return 0
      fi
      echo "${fname} partial: n=$n (want $want)"
    else
      echo "${fname} missing"
    fi
    sleep 300
  done
}

echo "--- waiting for dev sub-chunks ---"
for p in 00 01 02 03 04 05; do
  wait_sub "${DEV_DIR}" "sub_d00_${p}.tsv" "data/MIND_large/dev/chunks/sub/d00/chunk_${p}.tsv"
done

# ---- 2. merge, in impression order, and assert the counts ----
echo "--- merging dev scores ---"
cat ${DEV_DIR}/sub_d00_0{0,1,2,3,4,5}.tsv > ${DEV_MERGED}
N=$(wc -l < ${DEV_MERGED})
echo "dev merged lines: $N (want $DEV_TOTAL)"
[ "$N" -eq "$DEV_TOTAL" ] || { echo "FATAL: dev merged count mismatch"; exit 1; }

# ---- 3. the ONE honest dev evaluation ----
# Swap only the llm_score column of the canonical leak-free dev table, then
# apply v6. llm_rank / llm_z are derived at apply time from that column, so
# they follow automatically.
#
# DEV IS EVALUATED BEFORE THE TEST SCORES FINISH, on purpose: dev is the
# held-out split and the test split has no labels, so the dev number is the
# one that carries information. Waiting for the test scores first would delay
# the reportable result by an hour for no gain.
echo "--- swapping llm_score into the dev feature table ---"
python src/swap_llm_score.py \
  --features results/ranker/ck45500_dev_features_causalpop.tsv \
  --scores ${DEV_MERGED} \
  --behaviors data/MIND_large/dev/behaviors.tsv \
  --output results/ranker/mh0v6_dev_features_causalpop.tsv

echo "--- applying ranker v6 to dev (the single honest eval) ---"
python src/apply_ranker_devpop.py \
  --features results/ranker/mh0v6_dev_features_causalpop.tsv \
  --model ${MODEL} \
  --output results/ranker/mh0v6_dev_predictions.txt

python src/calc_mind_metrics.py \
  --predictions results/ranker/mh0v6_dev_predictions.txt \
  --behaviors data/MIND_large/dev/behaviors.tsv

echo "=== DEV EVALUATION COMPLETE $(date) ==="

# ---- 4. wait for the test scores, then merge ----
echo "--- waiting for test sub-chunks ---"
for s in c00 c01 c02 c03 c04 c05; do
  for p in 00 01 02 03 04 05; do
    wait_sub "${TEST_DIR}" "sub_${s}_${p}.tsv" "data/MIND_large/test/chunks/sub/${s}/chunk_${p}.tsv"
  done
done

echo "--- merging test scores ---"
cat ${TEST_DIR}/sub_c0{0,1,2,3,4,5}_0{0,1,2,3,4,5}.tsv > ${TEST_MERGED}
N=$(wc -l < ${TEST_MERGED})
echo "test merged lines: $N (want $TEST_TOTAL)"
[ "$N" -eq "$TEST_TOTAL" ] || { echo "FATAL: test merged count mismatch"; exit 1; }

# ---- 5. the test submission ----
echo "--- swapping llm_score into the test feature table ---"
python src/swap_llm_score.py \
  --features results/ranker/ck45500_test_features_devpopcat.tsv \
  --scores ${TEST_MERGED} \
  --behaviors data/MIND_large/test/behaviors.tsv \
  --output results/ranker/mh0v6_test_features_devpopcat.tsv

echo "--- applying ranker v6 to test ---"
python src/apply_ranker_devpop.py \
  --features results/ranker/mh0v6_test_features_devpopcat.tsv \
  --model ${MODEL} \
  --output results/ranker/mh0v6_test_predictions.txt

echo "--- converting to leaderboard format ---"
python src/convert_to_leaderboard_format.py \
  --predictions results/ranker/mh0v6_test_predictions.txt \
  --behaviors data/MIND_large/test/behaviors.tsv \
  --output results/ranker/submission_mh0_v6.txt

SUB_LINES=$(wc -l < results/ranker/submission_mh0_v6.txt)
echo "submission lines: $SUB_LINES (want $TEST_TOTAL)"
if [ "$SUB_LINES" -ne "$TEST_TOTAL" ]; then
  echo "FATAL: submission line count != test impression count"
  exit 1
fi

mkdir -p results/submissions
# The zip MUST contain a single member named `prediction.txt` -- the official
# msnews/MIND evaluate.py hardcodes that name.
bash scripts/build_submission_zip.sh \
  results/ranker/submission_mh0_v6.txt \
  results/submissions/MINDlarge_test_mh0_v6.zip
ls -la results/submissions/MINDlarge_test_mh0_v6.zip

echo "=== mh0 v6 pipeline DONE $(date) ==="
