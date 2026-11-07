#!/bin/bash
# DPO / RLCD on top of the BEST lg_9b checkpoint (ck45500), MIND_large.
#
# WHY THIS EXISTS
# ---------------
# The only DPO run we have (job 14991262, output_dir/small_runs/lg_9b_dpo)
# started from ck32500 and scored dev 0.7136 -- BELOW its own base ck32500
# (0.7134) and far below ck45500 (0.7181). So "DPO from ck32500" failed.
# But "DPO from ck45500" was never tried, and ck45500 is our best single
# model. DPO optimizes the answer-boundary margin
#     r = logP(" Yes" | pos) - logP(" Yes" | neg)
# which is exactly the quantity the per-impression AUC metric ranks on, so
# it is the most on-metric preference lever we have.
#
# It also trains on the SAME prompt format as SFT (the pairwise dataset
# delegates to data.py encode_example), so the only changed variable is the
# objective.
#
# TWO ARMS, ONE NODE EACH (so a failure in one does not take the other out):
#   dpo    -- Bradley-Terry DPO, beta 0.1 (the recipe that ran before)
#   rlcd   -- self-distill hinge, no reference model, margin 0.75
#
# NO EVAL STAGE IN THIS JOB. The eval is a separate job (scripts/
# eval_dev_chunk_hs.sh) because a full MIND_large dev eval takes hours and
# would blow the time limit; splitting it also lets us pick the checkpoint
# by eval_loss before spending dev-split budget.
#
# Usage:
#   sbatch scripts/train_pref_ck45500.sh <RUN_TAG> <OBJECTIVE> [extra args]
# Env: PC_BETA, PC_LR, PC_EPOCHS, PC_SAMPLE, PC_MIND_ROOT
#SBATCH --job-name=pref45500
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
# The A100/H100 nodes select the Broadcom RoCE transport and NCCL SIGSEGVs
# inside it (`dist.all_reduce` crashes even at world_size=1). Disabling the
# IB transport makes NCCL fall back to the socket transport; validated at
# ws=2 and ws=4 (job 16294462 / the _nccl_fix_trainer_val probe). The B200
# nodes are unaffected either way, so this is safe on both.
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MINIONEREC_NUM_WORKERS=8
export MINIONEREC_SAVE_STEPS=1000

RUN_TAG=$1
OBJECTIVE=$2
shift 2

PC_BETA=${PC_BETA:-0.1}
PC_LR=${PC_LR:-5e-6}
PC_EPOCHS=${PC_EPOCHS:-1.0}
PC_SAMPLE=${PC_SAMPLE:-300000}
PC_MIND_ROOT=${PC_MIND_ROOT:-/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large}

# ck45500 is a LoRA adapter dir (adapter_config.json, no config.json); its
# base_model_name_or_path is models/Qwen3.5-9B. The DPO trainer resolves the
# base from adapter_config.json, but the POLICY must be the adapter, so we
# pass it via --adapter_path. The reference model is a frozen copy of the
# SAME adapter, i.e. exactly the SFT model the policy started from.
SFT_CKPT=results/ranker/lg_9b_ckpt45500_snapshot
BASE_MODEL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['base_model_name_or_path'])" "${SFT_CKPT}/adapter_config.json")
echo "=== [${RUN_TAG}] ${OBJECTIVE} from ${SFT_CKPT} (base=${BASE_MODEL}) ==="

OUT=output_dir/small_runs/${RUN_TAG}
mkdir -p results

deepspeed --num_gpus=8 --master_port=29500 src/sft_mind_pref_contrastive.py \
  --base_model ${BASE_MODEL} \
  --adapter_path ${SFT_CKPT} \
  --objective ${OBJECTIVE} \
  --beta ${PC_BETA} \
  --learning_rate ${PC_LR} \
  --num_epochs ${PC_EPOCHS} \
  --train_behaviors_path ${PC_MIND_ROOT}/train/behaviors.tsv \
  --train_news_path ${PC_MIND_ROOT}/train/news.tsv \
  --output_dir ${OUT} \
  --batch_size 128 --micro_batch_size 4 --max_len 2048 \
  --max_history 30 --neg_ratio 1.0 --hard_neg_ratio 0.5 --sample ${PC_SAMPLE} \
  --use_abstract True --use_chat_template True --seed 42 \
  --use_recency 1 \
  --use_impression_timestamp 0 \
  --use_profile_summary 0 \
  --use_entity 0 \
  --use_subcategory 0 \
  --history_selection recent \
  --history_format flat \
  --deepspeed_config ds_configs/ds_config_zero2_gpu.json "$@"

echo "=== [${RUN_TAG}] TRAIN DONE $(date) ==="
echo "=== next: score dev with PREFIX_CACHE_MODE=hybrid via scripts/eval_dev_chunk_hs.sh ==="
ls -d ${OUT}/checkpoint-* 2>/dev/null || true
