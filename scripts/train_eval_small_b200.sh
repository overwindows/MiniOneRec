#!/bin/bash
# Stage-1 combined train+eval job for MIND_small (submit via sngpu/sbatch).
# Runs: train -> dev AUC eval -> append result line to results/small_results.tsv
# Usage: sbatch scripts/train_eval_small.sh <RUN_TAG> <MODEL> <EXTRA_TRAIN_ARGS...>
#SBATCH --job-name=small_te
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
# Node-agnostic by default: take any single GPU-less-than-8 node. Override at
# submit time with e.g. --gres=gpu:h200m141:8 --nodelist=sc3-c127 when a
# specific node type is wanted. (sbatch --gres on the command line replaces
# this line.)
#SBATCH --gres=gpu:8
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
# The length-grouped sampler is the production default. It can be turned off
# with MINIONEREC_LENGTH_GROUPED=0 -- required by the impression-packed
# sampler, which replaces it (the packed buckets are already
# length-homogeneous, so running both would be redundant).
export MINIONEREC_LENGTH_GROUPED=${MINIONEREC_LENGTH_GROUPED:-1}
# The A100 nodes' Broadcom RoCE transport SIGSEGVs inside NCCL all_reduce.
# Socket-only works everywhere, so force it for any non-B200 node.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
# H200 (SM90) SDPA backward produces inf/NaN gradients on this model/shape
# (measured 2026-11-08: 36/40 optimizer steps skipped under sdpa, 0 under
# eager). B200/A100 are unaffected. Auto-select eager on H200.
_gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if [[ "${_gpu_name}" == *"H200"* ]]; then
  export MINIONEREC_ATTN_IMPL=${MINIONEREC_ATTN_IMPL:-eager}
  echo "=== H200 detected: using MINIONEREC_ATTN_IMPL=${MINIONEREC_ATTN_IMPL} ==="
fi
export MINIONEREC_NUM_WORKERS=8
export MINIONEREC_EVAL_STEPS=500
export MINIONEREC_SAVE_STEPS=500

# ---- CUDA driver compatibility -------------------------------------------
# The cu130 torch build needs driver >= 595.71.05. Some A100/H100 nodes still
# carry 565.57.01, where torch.cuda.is_available() is False and the model would
# load on CPU. The CUDA forward-compat package ships a newer libcuda that works
# against the old kernel driver. It must ONLY be preloaded on old-driver nodes:
# on a new-driver node it shadows the real libcuda and breaks CUDA entirely.
COMPAT_DIR=/import/ml-sc-scratch1/chenw/cuda-compat-13.0
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if [ -f "${COMPAT_DIR}/libcuda.so.1" ]; then
    echo "=== driver too old for cu130 torch; preloading CUDA forward-compat ==="
    export LD_LIBRARY_PATH=${COMPAT_DIR}:${LD_LIBRARY_PATH:-}
    if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      echo "FATAL: CUDA still unavailable after forward-compat preload -- aborting"
      exit 1
    fi
  else
    echo "FATAL: driver too old and no forward-compat lib available -- aborting"
    exit 1
  fi
fi

# Number of GPUs in this allocation. B200 nodes have 8, A100 nodes have 4.
# The launcher and the gradient-accumulation math both depend on it.
NGPU=${TE_NGPU:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}
echo "=== training on ${NGPU} GPUs ==="

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
TR_USE_SUBCATEGORY=0 TR_USE_ENTITY=0 TR_HISTORY_SELECTION=recent TR_HISTORY_FORMAT=flat TR_USE_CHAT_TEMPLATE=1 TR_USE_ABSTRACT=1
_args=("$@")
for ((i = 0; i < ${#_args[@]}; i++)); do
  a=${_args[i]}
  if [[ $a == --*=* ]]; then v=${a#*=}; a=${a%%=*}
  else v=${_args[i+1]:-}; [[ $v == --* ]] && v=""; fi
  case $a in
    --max_history) TR_MAX_HISTORY=$v ;;
    --history_selection) TR_HISTORY_SELECTION=$v ;;
    --history_format) TR_HISTORY_FORMAT=$v ;;
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
         USE_SUBCATEGORY USE_ENTITY HISTORY_SELECTION HISTORY_FORMAT USE_CHAT_TEMPLATE USE_ABSTRACT; do
  te=TE_$k tr=TR_$k
  if [[ -n ${!te:-} && ${!te} != "${!tr}" ]]; then
    echo "WARNING: ${te}=${!te} overrides training value ${!tr} (eval prompt != training prompt)"
  fi
  printf -v "$te" '%s' "${!te:-${!tr}}"
done
echo "Eval prompt flags: max_history=${TE_MAX_HISTORY} selection=${TE_HISTORY_SELECTION} format=${TE_HISTORY_FORMAT} chat=${TE_USE_CHAT_TEMPLATE} abstract=${TE_USE_ABSTRACT} recency=${TE_USE_RECENCY} profile=${TE_USE_PROFILE_SUMMARY} ts=${TE_USE_IMPRESSION_TIMESTAMP} subcat=${TE_USE_SUBCATEGORY} entity=${TE_USE_ENTITY}"

OUT=output_dir/small_runs/${RUN_TAG}
mkdir -p results

# The DeepSpeed rendezvous port. It only has to be unique among jobs that can
# reach each other; jobs on different nodes have different host addresses, but
# pinning it per run keeps concurrent same-node launches from colliding.
MASTER_PORT=${MASTER_PORT:-29500}

# MIND_ROOT selects the dataset (default small; set to data/MIND_large for
# large runs — paths are derived, so both splits work unchanged).
MIND_ROOT="${MIND_ROOT:-/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_small}"
MIND_TAG=$(basename "${MIND_ROOT}")

echo "=== [${RUN_TAG}] TRAIN (${MIND_TAG}) ==="
deepspeed --num_gpus=${NGPU} --master_port=${MASTER_PORT} src/sft_mind_pointwise_ds.py \
  --base_model ${MODEL} \
  --train_behaviors_path ${MIND_ROOT}/train/behaviors.tsv \
  --train_news_path ${MIND_ROOT}/train/news.tsv \
  --eval_behaviors_path ${MIND_ROOT}/dev/behaviors.tsv \
  --eval_news_path ${MIND_ROOT}/dev/news.tsv \
  --output_dir ${OUT} \
  --batch_size 256 --micro_batch_size 8 --num_epochs 3 --learning_rate 2e-5 --cutoff_len 2048 \
  --max_history 30 --neg_ratio 2.0 --hard_neg_ratio 0.5 --sample 0 \
  --use_abstract True --use_chat_template True --train_from_scratch False --seed 42 \
  --deepspeed_config ds_configs/ds_config_zero2_gpu.json "$@"

echo "=== [${RUN_TAG}] EVAL dev AUC ==="
MIND_ROOT=${MIND_ROOT} \
USE_ABSTRACT=${TE_USE_ABSTRACT} USE_CHAT_TEMPLATE=${TE_USE_CHAT_TEMPLATE} FLASH_ATTN=0 \
MAX_HISTORY=${TE_MAX_HISTORY} \
USE_RECENCY=${TE_USE_RECENCY} \
USE_PROFILE_SUMMARY=${TE_USE_PROFILE_SUMMARY} \
USE_IMPRESSION_TIMESTAMP=${TE_USE_IMPRESSION_TIMESTAMP} \
USE_SUBCATEGORY=${TE_USE_SUBCATEGORY} \
USE_ENTITY=${TE_USE_ENTITY} \
HISTORY_SELECTION=${TE_HISTORY_SELECTION} \
HISTORY_FORMAT=${TE_HISTORY_FORMAT} \
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU - 1))) \
bash scripts/eval_mind_pointwise.sh ${OUT}/final_checkpoint dev 2>&1 | tee logs/eval_${RUN_TAG}.log

AUC=$(grep -aoE "AUC: +[0-9.]+" logs/eval_${RUN_TAG}.log | tail -1 | grep -oE "[0-9.]+$" || true)
echo -e "${RUN_TAG}\t$(date +%F\ %T)\t${AUC:-FAILED}" >> results/small_results.tsv
echo "=== [${RUN_TAG}] DONE AUC=${AUC:-FAILED} ==="