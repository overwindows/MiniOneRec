#!/bin/bash
# Decide whether the answer-only (`logits_to_keep`) loss path produces the
# SAME GRADIENTS as the stock full-logits path.
#
# WHY
# ---
# The forward loss is already proven bit-identical
# (src/bench_train_memory.py: 2.248723268508911 for both paths). But the
# real trainer A/B (job 16294944) showed the two configurations' loss
# trajectories diverging from step 2 onward (24.89 vs 24.87, then 1.161 vs
# 1.255). Either the BACKWARD differs (a real bug) or bf16 nondeterminism on
# a chaotic landscape (first loss 24.94 == p(answer) ~ 1e-11) amplifies a
# rounding-level difference.
#
# src/check_answer_only_grads.py answers this with two phases on the SAME
# real MIND batch, in the SAME process:
#   PHASE 1  model.eval()  -> dropout off, so the ONLY difference is the path
#   PHASE 2  model.train() -> dropout on, so the noise floor is measurable
# It then compares the path difference against the noise floor.
#
# This job runs that script on ONE GPU. The A100 nodes are used because they
# are idle; the script does NOT use torch.distributed, so the A100 RoCE/NCCL
# bug does not apply.
#
# Usage: sbatch scripts/_answer_only_grad_sbatch.sh
# Env:   AG_MODEL, AG_BATCH, AG_NBATCH
#SBATCH --job-name=agcheck
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --time=00:30:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -u
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONFAULTHANDLER=1
export TRITON_CACHE_DIR=/dev/shm/triton_agcheck

# The A100 nodes' RoCE transport segfaults inside NCCL. This script does not
# use torch.distributed, but DeepSpeed/accelerate may still probe it, so
# disable the IB transport defensively.
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

COMPAT_DIR=/import/ml-sc-scratch1/chenw/cuda-compat-13.0
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if [ -f "${COMPAT_DIR}/libcuda.so.1" ]; then
    echo "=== driver too old for cu130 torch; preloading CUDA forward-compat ==="
    export LD_LIBRARY_PATH=${COMPAT_DIR}:${LD_LIBRARY_PATH:-}
  else
    echo "FATAL: driver too old and no forward-compat lib available -- aborting"
    exit 1
  fi
fi

echo "=== NODE=$(hostname) $(date) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

MODEL=${AG_MODEL:-models/Qwen3.5-4B}
BATCH=${AG_BATCH:-4}
NBATCH=${AG_NBATCH:-2}

mkdir -p /dev/shm/triton_agcheck
python -u src/check_answer_only_grads.py \
  --model "${MODEL}" \
  --batch_size "${BATCH}" \
  --n_batches "${NBATCH}" \
  --out results/ranker/answer_only_grad_check.json \
  2>&1 | tee logs/answer_only_grad_check.log

echo "=== GRAD CHECK DONE $(date) ==="
