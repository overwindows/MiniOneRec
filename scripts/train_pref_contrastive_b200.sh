#!/bin/bash
# Contrastive fine-tuning (DPO or RLCD) on top of a pointwise SFT checkpoint,
# then full dev AUC eval. Submit via sngpu/sbatch.
#
# Usage:
#   sbatch scripts/train_pref_contrastive_b200.sh <RUN_TAG> <OBJECTIVE> <SFT_CKPT> [EXTRA...]
#     OBJECTIVE: dpo | rlcd
#
# Env overrides (must match the SFT run's training config):
#   PC_USE_RECENCY (default 1), PC_USE_IMPRESSION_TIMESTAMP (default 0),
#   PC_USE_PROFILE_SUMMARY (default 0), PC_USE_ENTITY (default 0),
#   PC_USE_SUBCATEGORY (default 0), PC_HISTORY_SELECTION (default recent),
#   PC_HISTORY_FORMAT (default flat),
#   PC_BETA (default 0.1), PC_LR (default 5e-6), PC_EPOCHS (default 1.0)
#   PC_MIND_ROOT (default data/MIND_small) -- set to data/MIND_large for
#     MIND_large runs; the train/eval paths are derived from it.
#   PC_ADAPTER (default empty) -- path to an existing PEFT adapter to
#     CONTINUE (the policy is that adapter, the DPO reference is a frozen
#     copy of it). Required when the SFT checkpoint is an adapter dir.
#   PC_RESULTS (default results/small_results.tsv) -- where to append the AUC.
#SBATCH --job-name=pref_ct
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
export MINIONEREC_NUM_WORKERS=8
export MINIONEREC_SAVE_STEPS=2000

RUN_TAG=$1
OBJECTIVE=$2
SFT_CKPT=$3
shift 3

PC_USE_RECENCY=${PC_USE_RECENCY:-1}
PC_USE_IMPRESSION_TIMESTAMP=${PC_USE_IMPRESSION_TIMESTAMP:-0}
PC_USE_PROFILE_SUMMARY=${PC_USE_PROFILE_SUMMARY:-0}
PC_USE_ENTITY=${PC_USE_ENTITY:-0}
PC_USE_SUBCATEGORY=${PC_USE_SUBCATEGORY:-0}
PC_HISTORY_SELECTION=${PC_HISTORY_SELECTION:-recent}
PC_HISTORY_FORMAT=${PC_HISTORY_FORMAT:-flat}
PC_BETA=${PC_BETA:-0.1}
PC_LR=${PC_LR:-5e-6}
PC_EPOCHS=${PC_EPOCHS:-1.0}
PC_MIND_ROOT=${PC_MIND_ROOT:-/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_small}
PC_RESULTS=${PC_RESULTS:-results/small_results.tsv}
PC_ADAPTER=${PC_ADAPTER:-}

# If SFT_CKPT is a PEFT adapter dir (adapter_config.json, no config.json),
# the base weights live in adapter_config.json's base_model_name_or_path and
# the adapter itself is what we continue training. The trainer resolves the
# base automatically, but the POLICY must be the adapter, so pass it via
# --adapter_path.
BASE_MODEL=${SFT_CKPT}
if [ -f "${SFT_CKPT}/adapter_config.json" ] && [ ! -f "${SFT_CKPT}/config.json" ]; then
  BASE_MODEL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['base_model_name_or_path'])" "${SFT_CKPT}/adapter_config.json")
  PC_ADAPTER=${PC_ADAPTER:-${SFT_CKPT}}
  echo "SFT_CKPT is a PEFT adapter: base=${BASE_MODEL} adapter=${PC_ADAPTER}"
fi

OUT=output_dir/small_runs/${RUN_TAG}
mkdir -p results

echo "=== [${RUN_TAG}] CONTRASTIVE (${OBJECTIVE}) on ${SFT_CKPT} ==="
deepspeed --num_gpus=8 --master_port=29500 src/sft_mind_pref_contrastive.py \
  --base_model ${BASE_MODEL} \
  ${PC_ADAPTER:+--adapter_path ${PC_ADAPTER}} \
  --objective ${OBJECTIVE} \
  --beta ${PC_BETA} \
  --learning_rate ${PC_LR} \
  --num_epochs ${PC_EPOCHS} \
  --train_behaviors_path ${PC_MIND_ROOT}/train/behaviors.tsv \
  --train_news_path ${PC_MIND_ROOT}/train/news.tsv \
  --output_dir ${OUT} \
  --batch_size 128 --micro_batch_size 4 --max_len 2048 \
  --max_history 30 --neg_ratio 1.0 --hard_neg_ratio 0.5 --sample 0 \
  --use_abstract True --use_chat_template True --seed 42 \
  --use_recency ${PC_USE_RECENCY} \
  --use_impression_timestamp ${PC_USE_IMPRESSION_TIMESTAMP} \
  --use_profile_summary ${PC_USE_PROFILE_SUMMARY} \
  --use_entity ${PC_USE_ENTITY} \
  --use_subcategory ${PC_USE_SUBCATEGORY} \
  --history_selection ${PC_HISTORY_SELECTION} \
  --history_format ${PC_HISTORY_FORMAT} \
  --deepspeed_config ds_configs/ds_config_zero2_gpu.json "$@"

echo "=== [${RUN_TAG}] EVAL dev AUC ==="
MIND_ROOT=${PC_MIND_ROOT} \
USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 \
MAX_HISTORY=30 \
USE_RECENCY=${PC_USE_RECENCY} \
USE_IMPRESSION_TIMESTAMP=${PC_USE_IMPRESSION_TIMESTAMP} \
USE_PROFILE_SUMMARY=${PC_USE_PROFILE_SUMMARY} \
USE_ENTITY=${PC_USE_ENTITY} \
USE_SUBCATEGORY=${PC_USE_SUBCATEGORY} \
HISTORY_SELECTION=${PC_HISTORY_SELECTION} \
HISTORY_FORMAT=${PC_HISTORY_FORMAT} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/eval_mind_pointwise.sh ${OUT}/final_checkpoint dev 2>&1 | tee logs/eval_${RUN_TAG}.log

AUC=$(grep -aoE "AUC: +[0-9.]+" logs/eval_${RUN_TAG}.log | tail -1 | grep -oE "[0-9.]+$" || true)
echo -e "${RUN_TAG}\t$(date +%F\ %T)\t${AUC:-FAILED}" >> ${PC_RESULTS}
echo "=== [${RUN_TAG}] DONE AUC=${AUC:-FAILED} ==="