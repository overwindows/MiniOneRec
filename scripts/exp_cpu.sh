#!/bin/bash
# Run a CPU-only ranker experiment on an idle cpuonly node.
#
# WHY A SLURM CPU NODE
# -------------------
# The login node has 36 cores and a load average of ~85 (three LightGBM
# experiments already running there). The cpuonly partition has idle nodes
# with 288-672 cores each, so the experiment gets a clean machine instead of
# competing for cores.
#
# Usage: sbatch scripts/exp_cpu.sh <script.py> <logname> [extra args...]
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200000
#SBATCH --time=12:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SCRIPT=$1
NAME=$2
shift 2

echo "=== ${NAME} start $(date) on $(hostname) (${SLURM_CPUS_PER_TASK} cpus) ==="
python -u "${SCRIPT}" "$@" 2>&1 | tee "logs/${NAME}.log"
echo "=== ${NAME} DONE $(date) ==="
