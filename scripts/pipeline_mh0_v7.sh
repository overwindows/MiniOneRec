#!/bin/bash
# Promote the MAX_HISTORY=0 (mh0) re-score to a new stack ranker + submission.
#
# WHY
# ---
# Raising the inference-side history cap from 30 to unlimited is the biggest
# lever measured in this project, and it needs NO retraining of the LLM:
#
#   LLM level : +0.00392 AUC (SE 0.00036, t=+10.77), paired on 40,000 dev
#               impressions with the SAME ckpt-45500 weights.
#   STACK v6  : +0.00305 AUC (SE 0.00013, t=+22.78) on the FULL dev split
#               (376,471 impressions), the EXISTING v6 ranker, artifact
#               results/ranker/mh0v6_paired_dev_summary.json.
#
# The mechanism is verified, not assumed: on the full dev split the paired
# delta is statistically zero on the 245,360 impressions with <=30 history
# items (whose prompt is byte-identical under both caps: -0.000031, SE
# 0.000099, t=-0.31) and grows monotonically with history length
# (31-50: +0.003850, 51-100: +0.011101, >100: +0.015727).
#
# WHAT THIS PIPELINE DOES
# -----------------------
# The cap change touches ONLY the llm_score column of the ranker feature
# table. Every other column (cf_score, the popularity families, the
# history-derived features, the position features) is a function of the DATA,
# not of the LLM. The feature tables are huge -- the test table is 93,115,002
# rows x 25 columns (17 GB) -- so rebuilding them from scratch just to swap
# one column would waste an hour of CPU and 17 GB of scratch disk.
#
# So: src/swap_llm_score.py streams the feature table and rewrites ONLY the
# llm_score column, with a hard alignment guard (it verifies the impression_id
# of every row group against the score file and aborts on any mismatch).
#
# Chain:
#   1. wait for the 6 dev + 34 train + 36 test mh0 sub-chunk scores
#   2. merge each -> lg9b_ckpt45500_{dev,train,test}_scores_mh0.tsv
#   3. refit the ranker on the mh0 TRAIN scores -> ranker_v7_mh0_train_popwit.txt
#      (the trainer overrides llm_score from --llm-train BEFORE deriving
#      llm_rank/llm_z, so the train feature table needs NO rebuild)
#   4. swap llm_score into the DEV feature table, apply v7 -> the ONE honest
#      dev number (per-impression AUC over all 376,471 dev impressions)
#   5. swap llm_score into the TEST feature table, apply v7 -> leaderboard
#      format -> zip
#
# WHY THE FEATURE-SIDE HISTORY CAP STAYS AT 30
# --------------------------------------------
# src/build_ranker_features.py has --max_history default 30, and the existing
# train/dev/test tables all have hist_len max = 30. Lifting it would require
# rebuilding all three tables (10 GB + 2.5 GB + 17 GB) for a change that
# touches only ~6.8% of the v6 model's gain (the history-derived features),
# versus 37.6% for the LLM columns -- and it would make the v7 configuration
# DIFFERENT from the one that was actually measured (+0.00305 on the full dev
# split).
# Keeping the cap at 30 keeps train/dev/test consistent AND keeps v7 equal to
# the measured configuration.
#
# EVALUATION PROTOCOL
# -------------------
# The ranker is fitted on TRAIN labels only. Dev is touched exactly once, at
# step 4, to produce the single reportable number. There is no validation
# split and no model selection on dev.
#
# ONE QUALIFICATION, stated honestly: the history cap itself (30 vs 0 vs
# 50 vs 100) was chosen on a 40,000-impression uniform random SUBSET of dev
# (chunk_90, seed 42) -- that subset is ~10.6% of dev, so it is a mild
# double-dip, not a clean held-out selection. The subset screen ranked
# unlimited > 100 > 50 > 30, but the 0-vs-100 difference was NOT significant
# (paired delta -0.00036, t=-1.75), so the choice among the top arms rests on
# the monotone dose-response and on the mechanism, not on a significant
# pairwise win.
set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"

LOG=logs/pipeline_mh0_v7.log
exec >> $LOG 2>&1
echo "=== mh0 v7 pipeline start $(date) ==="

DEV_DIR=results/ranker/dev_scores/mh0
TRAIN_DIR=results/ranker/train_scores/mh0
TEST_DIR=results/ranker/test_scores/mh0

DEV_MERGED=results/ranker/lg9b_ckpt45500_dev_scores_mh0.tsv
TRAIN_MERGED=results/ranker/lg9b_ckpt45500_train_scores_mh0.tsv
TEST_MERGED=results/ranker/lg9b_ckpt45500_test_scores_mh0.tsv

DEV_TOTAL=$(wc -l < data/MIND_large/dev/behaviors.tsv)
TRAIN_TOTAL=$(wc -l < data/MIND_large/train/behaviors.tsv)
TEST_TOTAL=$(wc -l < data/MIND_large/test/behaviors.tsv)

MODEL=results/ranker/ranker_v7_mh0_train_popwit.txt

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

echo "--- waiting for train sub-chunks ---"
for p in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 \
         16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33; do
  wait_sub "${TRAIN_DIR}" "sub_tr00_${p}.tsv" "data/MIND_large/train/chunks/sub/tr00/chunk_${p}.tsv"
done

echo "--- waiting for test sub-chunks ---"
for s in c00 c01 c02 c03 c04 c05; do
  for p in 00 01 02 03 04 05; do
    wait_sub "${TEST_DIR}" "sub_${s}_${p}.tsv" "data/MIND_large/test/chunks/sub/${s}/chunk_${p}.tsv"
  done
done

# ---- 2. merge, in impression order, and assert the counts ----
echo "--- merging dev scores ---"
cat ${DEV_DIR}/sub_d00_0{0,1,2,3,4,5}.tsv > ${DEV_MERGED}
N=$(wc -l < ${DEV_MERGED})
echo "dev merged lines: $N (want $DEV_TOTAL)"
[ "$N" -eq "$DEV_TOTAL" ] || { echo "FATAL: dev merged count mismatch"; exit 1; }

echo "--- merging train scores ---"
cat ${TRAIN_DIR}/sub_tr00_{00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33}.tsv > ${TRAIN_MERGED}
N=$(wc -l < ${TRAIN_MERGED})
echo "train merged lines: $N (want $TRAIN_TOTAL)"
[ "$N" -eq "$TRAIN_TOTAL" ] || { echo "FATAL: train merged count mismatch"; exit 1; }

echo "--- merging test scores ---"
cat ${TEST_DIR}/sub_c0{0,1,2,3,4,5}_0{0,1,2,3,4,5}.tsv > ${TEST_MERGED}
N=$(wc -l < ${TEST_MERGED})
echo "test merged lines: $N (want $TEST_TOTAL)"
[ "$N" -eq "$TEST_TOTAL" ] || { echo "FATAL: test merged count mismatch"; exit 1; }

# ---- 3. refit the ranker on the mh0 TRAIN scores ----
# The trainer overrides llm_score from --llm-train (X[off:off+n, 0] = s) and
# derives llm_rank/llm_z AFTER that override (within_imp_transforms(Xm, g)),
# so no train feature rebuild is needed.
echo "--- training ranker v7 on the mh0 train scores ---"
python src/train_ranker_train_split.py \
  --train-feats results/ranker/large_train_features.tsv \
  --llm-train ${TRAIN_MERGED} \
  --out ${MODEL} \
  --joint-mask --wit --subcat --pop-wit \
  --rounds 2000 --threads 64

if [ ! -f "${MODEL}.feats.json" ]; then
  echo "FATAL: model ${MODEL}.feats.json never appeared"
  exit 1
fi
ls -la ${MODEL} ${MODEL}.feats.json

# ---- 4. the ONE honest dev evaluation ----
# Swap only the llm_score column of the canonical leak-free dev table, then
# apply v7. llm_rank / llm_z are derived at apply time from that column, so
# they follow automatically.
echo "--- swapping llm_score into the dev feature table ---"
python src/swap_llm_score.py \
  --features results/ranker/ck45500_dev_features_causalpop.tsv \
  --scores ${DEV_MERGED} \
  --output results/ranker/mh0_dev_features_causalpop.tsv

echo "--- applying ranker v7 to dev (the single honest eval) ---"
python src/apply_ranker_devpop.py \
  --features results/ranker/mh0_dev_features_causalpop.tsv \
  --model ${MODEL} \
  --output results/ranker/v7_mh0_dev_predictions.txt

python src/calc_mind_metrics.py \
  --predictions results/ranker/v7_mh0_dev_predictions.txt \
  --behaviors data/MIND_large/dev/behaviors.tsv

# ---- 5. the test submission ----
echo "--- swapping llm_score into the test feature table ---"
python src/swap_llm_score.py \
  --features results/ranker/ck45500_test_features_devpopcat.tsv \
  --scores ${TEST_MERGED} \
  --output results/ranker/mh0_test_features_devpopcat.tsv

echo "--- applying ranker v7 to test ---"
python src/apply_ranker_devpop.py \
  --features results/ranker/mh0_test_features_devpopcat.tsv \
  --model ${MODEL} \
  --output results/ranker/v7_mh0_test_predictions.txt

echo "--- converting to leaderboard format ---"
python src/convert_to_leaderboard_format.py \
  --predictions results/ranker/v7_mh0_test_predictions.txt \
  --behaviors data/MIND_large/test/behaviors.tsv \
  --output results/ranker/submission_mh0_v7.txt

SUB_LINES=$(wc -l < results/ranker/submission_mh0_v7.txt)
echo "submission lines: $SUB_LINES (want $TEST_TOTAL)"
if [ "$SUB_LINES" -ne "$TEST_TOTAL" ]; then
  echo "FATAL: submission line count != test impression count"
  exit 1
fi

mkdir -p results/submissions
# The zip MUST contain a single member named `prediction.txt` -- the official
# msnews/MIND evaluate.py hardcodes that name.
bash scripts/build_submission_zip.sh \
  results/ranker/submission_mh0_v7.txt \
  results/submissions/MINDlarge_test_mh0_v7.zip
ls -la results/submissions/MINDlarge_test_mh0_v7.zip

echo "=== mh0 v7 pipeline DONE $(date) ==="
