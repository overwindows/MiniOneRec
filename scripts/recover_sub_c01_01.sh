#!/bin/bash
# Recover the impressions that job 14951791 lost to a CPU fallback.
#
# WHY
# ---
# Job 14951791 (sub-chunk c01/01) ran 8 scoring processes on 8 GPUs. GPU 7's
# process logged "Model loaded on device: cpu" -- device_map="auto" silently
# fell back to CPU because that GPU was unhealthy -- and then spun on CPU for
# 3.5 hours without producing output. The job's merge step waits for ALL 8
# GPUs, so the whole sub-chunk was lost even though 7/8 GPUs finished.
#
# This script re-scores ONLY the missing impression range (518617..526827,
# 8211 impressions) on a fresh node, and writes the result to
# results/ranker/test_scores/sub_c01_01.tsv by concatenating the 7 good
# per-GPU files from the failed job with the newly scored range.
#
# It also pre-checks every GPU on the node and refuses to start if any of
# them is unhealthy, so a bad device can never silently cost hours again.
#
#SBATCH --job-name=lg_trecover
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400000
#SBATCH --time=04:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

CKPT=results/ranker/lg_9b_ckpt32500_snapshot
PARTIAL=results/ranker/test_scores/partial_c01_01
OUT=results/ranker/test_scores/sub_c01_01.tsv
# The recovery range is only 8211 impressions, so 4 GPUs is plenty and lets
# the job start on the idle A100 nodes instead of waiting for a B200.
GPU_LIST=${TE_GPUS:-0,1,2,3}

echo "=== GPU health pre-check (GPUs: ${GPU_LIST}) ==="
# The A100 nodes in this cluster run driver 12070 (CUDA 12.7) while the
# installed torch is cu130 (CUDA 13.0), so torch.cuda.is_available() is
# False there and device_map="auto" silently loads the model on CPU. Refuse
# to start on such a node instead of burning hours on a CPU forward pass.
if ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: torch.cuda.is_available() is False on this node -- the driver"
  echo "       is older than the cu130 torch build, so the model would load"
  echo "       on CPU. Use a B200 node."
  exit 1
fi
echo "  torch CUDA: OK"
for g in ${GPU_LIST//,/ }; do
  # The CUDA driver itself holds ~1 MiB per GPU, so only a real allocation
  # (>100 MiB) means the device is busy or faulted.
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 100 ]; then
    echo "FATAL: GPU $g has ${used} MiB in use (busy or faulted)"
    exit 1
  fi
done
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

echo "=== building recovery behaviors (8211 impressions) ==="
mkdir -p data/MIND_large/test/chunks/sub/c01/split_01_recover
python3 - <<'PY'
src = "data/MIND_large/test/chunks/sub/c01/chunk_01.tsv"
out = "data/MIND_large/test/chunks/sub/c01/split_01_recover/behaviors.tsv"
LO, HI = 518617, 526827
n = 0
with open(src) as f, open(out, "w") as g:
    for line in f:
        i = int(line.split("\t", 1)[0])
        if LO <= i <= HI:
            g.write(line)
            n += 1
        elif i > HI:
            break
print(f"wrote {n} impressions to {out}")
assert n == 8211, n
PY
ln -sf /import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/news.tsv \
       data/MIND_large/test/chunks/sub/c01/split_01_recover/news.tsv

echo "=== scoring recovery range ==="
MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large/test/chunks/sub \
USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=30 \
USE_RECENCY=1 \
HISTORY_SELECTION=recent \
HISTORY_FORMAT=flat \
OUTPUT_SCORES_FILE=results/ranker/test_scores/recover_c01_01_scores.tsv \
CUDA_VISIBLE_DEVICES=${GPU_LIST} \
bash scripts/eval_mind_pointwise.sh ${CKPT} c01/split_01_recover 2>&1 | tail -30

echo "=== assembling sub_c01_01.tsv ==="
python3 - <<'PY'
import os
partial = "results/ranker/test_scores/partial_c01_01"
recov = "results/ranker/test_scores/recover_c01_01_scores.tsv"
out = "results/ranker/test_scores/sub_c01_01.tsv"

rows = {}
for g in range(8):
    p = f"{partial}/{g}_scores.txt"
    if not os.path.exists(p):
        continue
    with open(p) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rows[parts[0]] = parts[1:]

with open(recov) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        rows[parts[0]] = parts[1:]

want = []
with open("data/MIND_large/test/chunks/sub/c01/chunk_01.tsv") as f:
    for line in f:
        want.append(line.split("\t", 1)[0])

missing = [i for i in want if i not in rows]
if missing:
    raise SystemExit(f"FATAL: {len(missing)} impressions still missing, "
                     f"first={missing[0]} last={missing[-1]}")

with open(out, "w") as g:
    for i in want:
        g.write(i + "\t" + "\t".join(rows[i]) + "\n")
print(f"wrote {len(want)} impressions to {out}")
PY

echo "=== verifying ==="
python3 - <<'PY'
cnt = {}
with open("data/MIND_large/test/chunks/sub/c01/chunk_01.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        cnt[p[0]] = len(p[4].split())
n = bad = 0
with open("results/ranker/test_scores/sub_c01_01.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) - 1 != cnt[p[0]]:
            bad += 1
        n += 1
print(f"lines={n} (want {len(cnt)})  candidate-count mismatches={bad}")
assert n == len(cnt) and bad == 0
PY

echo "=== recovery DONE $(date) ==="
