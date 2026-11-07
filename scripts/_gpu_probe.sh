#!/bin/bash
# One-shot probe: how does Slurm expose the allocated GPUs to the job?
#
# The hybrid A/B runs four arms concurrently, one GPU each, and each arm
# re-exports CUDA_VISIBLE_DEVICES. If Slurm hands the job physical devices
# that are NOT 0..N-1 (e.g. "4,5,6,7" on an 8-GPU node), then hardcoding
# 0,1,2,3 would either grab another user's GPUs or fail to find a device.
# This prints the facts so the A/B can map arms to devices correctly.
#SBATCH --job-name=gpuprobe
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20000
#SBATCH --time=00:05:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -u
cd /import/ml-sc-scratch5/chenw/MiniOneRec
echo "=== NODE=$(hostname) $(date) ==="
echo "CUDA_VISIBLE_DEVICES=[${CUDA_VISIBLE_DEVICES:-<unset>}]"
echo "SLURM_JOB_GPUS=[${SLURM_JOB_GPUS:-<unset>}]"
echo "SLURM_STEP_GPUS=[${SLURM_STEP_GPUS:-<unset>}]"
echo "SLURM_LOCALID=[${SLURM_LOCALID:-<unset>}]"
echo "--- scontrol show job \$SLURM_JOB_ID ---"
scontrol show job "${SLURM_JOB_ID}" 2>&1 | tr ' ' '\n' | grep -aE "TresPerNode|TRES=|NodeList|AllocTRES" | head -10
echo "--- nvidia-smi -L ---"
nvidia-smi -L 2>&1
echo "--- nvidia-smi --query-gpu=index,uuid,memory.used --format=csv ---"
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader 2>&1
echo "=== GPUPROBE DONE ==="
