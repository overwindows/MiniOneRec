#!/bin/bash
#SBATCH --job-name=pfxab
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200000
#SBATCH --time=04:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err
set -u
export AB_N=1000 AB_TAG=q8b
export AB_CKPT=output_dir/small_runs/lg_q8b/checkpoint-14500
bash scripts/_prefix_ab.sh
