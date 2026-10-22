#!/bin/bash
# Stage-1 combined train+eval job for MIND_small (submit via sngpu/sbatch).
# Runs: train -> dev AUC eval -> append result line to results/small_results.tsv
# Usage: sbatch scripts/train_eval_small.sh <RUN_TAG> <MODEL> <EXTRA_TRAIN_ARGS...>
#SBATCH --job-name=small_te
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:b200m180:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400000
#SBATCH --time=08:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/dev/shm/triton
export NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MINIONEREC_LENGTH_GROUPED=1
export MINIONEREC_NUM_WORKERS=8
export MINIONEREC_EVAL_STEPS=500
export MINIONEREC_SAVE_STEPS=500

RUN_TAG=$1
MODEL=$2
shift 2

# Eval-stage prompt flags must match training. They are derived from the
# training args ("$@", fire style: --flag V, --flag=V, or bare --flag for True)
# so the two stages cannot disagree. A TE_* env var (TE_MAX_HISTORY,
# TE_USE_RECENCY, TE_USE_PROFILE_SUMMARY, TE_USE_IMPRESSION_TIMESTAMP,
# TE_USE_SUBCATEGORY, TE_USE_ENTITY, TE_HISTORY_SELECTION, TE_USE_CHAT_TEMPLATE,
# TE_USE_ABSTRACT) still overrides, with a warning if it contradicts training.
_bool () { case "${1,,}" in true|1|"") echo 1 ;; *) echo 0 ;; esac; }
TR_MAX_HISTORY=30 TR_USE_RECENCY=0 TR_USE_PROFILE_SUMMARY=0 TR_USE_IMPRESSION_TIMESTAMP=0
TR_USE_SUBCATEGORY=0 TR_USE_ENTITY=0 TR_HISTORY_SELECTION=recent TR_USE_CHAT_TEMPLATE=1 TR_USE_ABSTRACT=1
_args=("$@")
for ((i = 0; i < ${#_args[@]}; i++)); do
  a=${_args[i]}
  if [[ $a == --*=* ]]; then v=${a#*=}; a=${a%%=*}
  else v=${_args[i+1]:-}; [[ $v == --* ]] && v=""; fi
  case $a in
    --max_history) TR_MAX_HISTORY=$v ;;
    --history_selection) TR_HISTORY_SELECTION=$v ;;
    --use_recency) TR_USE_RECENCY=$(_bool "$v") ;;
    --use_profile_summary) TR_USE_PROFILE_SUMMARY=$(_bool "$v") ;;
    --use_impression_timestamp) TR_USE_IMPRESSION_TIMESTAMP=$(_bool "$v") ;;
    --use_subcategory) TR_USE_SUBCATEGORY=$(_bool "$v") ;;
    --use_entity) TR_USE_ENTITY=$(_bool "$v") ;;
    --use_chat_template) TR_USE_CHAT_TEMPLATE=$(_bool "$v") ;;
    --use_abstract) TR_USE_ABSTRACT=$(_bool "$v") ;;
  esac
done
for k in MAX_HISTORY USE_RECENCY USE_PROFILE_SUMMARY USE_IMPRESSION_TIMESTAMP \
         USE_SUBCATEGORY USE_ENTITY HISTORY_SELECTION USE_CHAT_TEMPLATE USE_ABSTRACT; do
  te=TE_$k tr=TR_$k
  if [[ -n ${!te:-} && ${!te} != "${!tr}" ]]; then
    echo "WARNING: ${te}=${!te} overrides training value ${!tr} (eval prompt != training prompt)"
  fi
  printf -v "$te" '%s' "${!te:-${!tr}}"
done
echo "Eval prompt flags: max_history=${TE_MAX_HISTORY} selection=${TE_HISTORY_SELECTION} chat=${TE_USE_CHAT_TEMPLATE} abstract=${TE_USE_ABSTRACT} recency=${TE_USE_RECENCY} profile=${TE_USE_PROFILE_SUMMARY} ts=${TE_USE_IMPRESSION_TIMESTAMP} subcat=${TE_USE_SUBCATEGORY} entity=${TE_USE_ENTITY}"

OUT=output_dir/small_runs/${RUN_TAG}
mkdir -p results

echo "=== [${RUN_TAG}] TRAIN ==="
deepspeed --num_gpus=8 --master_port=29500 src/sft_mind_pointwise_ds.py \
  --base_model ${MODEL} \
  --train_behaviors_path data/MIND_small/train/behaviors.tsv \
  --train_news_path data/MIND_small/train/news.tsv \
  --eval_behaviors_path data/MIND_small/dev/behaviors.tsv \
  --eval_news_path data/MIND_small/dev/news.tsv \
  --output_dir ${OUT} \
  --batch_size 256 --micro_batch_size 8 --num_epochs 3 --learning_rate 2e-5 --cutoff_len 2048 \
  --max_history 30 --neg_ratio 2.0 --hard_neg_ratio 0.5 --sample 0 \
  --use_abstract True --use_chat_template True --train_from_scratch False --seed 42 \
  --deepspeed_config ds_configs/ds_config_zero2_gpu.json "$@"

echo "=== [${RUN_TAG}] EVAL dev AUC ==="
MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_small \
USE_ABSTRACT=${TE_USE_ABSTRACT} USE_CHAT_TEMPLATE=${TE_USE_CHAT_TEMPLATE} FLASH_ATTN=0 \
MAX_HISTORY=${TE_MAX_HISTORY} \
USE_RECENCY=${TE_USE_RECENCY} \
USE_PROFILE_SUMMARY=${TE_USE_PROFILE_SUMMARY} \
USE_IMPRESSION_TIMESTAMP=${TE_USE_IMPRESSION_TIMESTAMP} \
USE_SUBCATEGORY=${TE_USE_SUBCATEGORY} \
USE_ENTITY=${TE_USE_ENTITY} \
HISTORY_SELECTION=${TE_HISTORY_SELECTION} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/eval_mind_pointwise.sh ${OUT}/final_checkpoint dev 2>&1 | tee logs/eval_${RUN_TAG}.log

AUC=$(grep -aoE "AUC: +[0-9.]+" logs/eval_${RUN_TAG}.log | tail -1 | grep -oE "[0-9.]+$" || true)
echo -e "${RUN_TAG}\t$(date +%F\ %T)\t${AUC:-FAILED}" >> results/small_results.tsv
echo "=== [${RUN_TAG}] DONE AUC=${AUC:-FAILED} ==="