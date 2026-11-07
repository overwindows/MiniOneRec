#!/bin/bash
# Quantify the training-memory levers on an idle A100 node.
#
# WHY
# ---
# The B200 training runs sit at ~50 GB / 183 GB (27%) of VRAM, but the GPUs
# report 93-100% utilization and 65-90% of their 1000 W power limit. So the
# memory is not "wasted" -- it is headroom that is NOT being converted into
# throughput, because the micro-batch is capped by activation + logits memory
# rather than by the GPU.
#
# The question this answers: for a given micro-batch, how much VRAM does each
# lever save, and how much throughput does the resulting headroom buy?
#
# Levers measured (per micro-batch size):
#   A  full logits   + gradient checkpointing  (current production config)
#   B  answer-only   + gradient checkpointing  (logits_to_keep)
#   C  full logits   - gradient checkpointing
#   D  answer-only   - gradient checkpointing
#
# The four micro-batch sizes run CONCURRENTLY, one GPU each, so one node
# produces the whole table in the time of the largest single run.
#
# Usage: sbatch scripts/_bench_mem_sbatch.sh
# Env:   BM_MODEL, BM_SEQ_LEN, BM_GPUS
#SBATCH --job-name=bmem
#SBATCH --partition=gpuonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --time=02:00:00
#SBATCH --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.log
#SBATCH --error=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/%x_%j.err

set -u
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/dev/shm/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=/import/ml-sc-scratch5/chenw/MiniOneRec/src

# ---- CUDA driver compatibility -------------------------------------------
# The cu130 torch build needs driver >= 595.71.05. Some A100/H100 nodes carry
# an older driver, where torch.cuda.is_available() is False and the model
# would load on CPU. The forward-compat package ships a newer libcuda that
# works against the old kernel driver. It must ONLY be preloaded on
# old-driver nodes: on a new-driver node it shadows the real libcuda.
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

echo "=== NODE=$(hostname) $(date) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

MODEL=${BM_MODEL:-models/Qwen3.5-4B}
SEQ=${BM_SEQ_LEN:-2048}
# Micro-batch sizes to sweep, one per GPU. Default spans "fits easily" to
# "does not fit" for a 4B model at seq_len 2048 on an 80 GB A100.
MBS=${BM_MBS:-"4 8 16 32"}

# Resolve the CUDA-relative device indices (inside a Slurm job the visible
# devices are renumbered to 0..N-1).
GPUS=$(python - <<'PY' 2>/dev/null || echo "0,1,2,3"
import torch
ok = []
for i in range(torch.cuda.device_count()):
    try:
        torch.zeros(1, device=f"cuda:{i}")
        ok.append(i)
    except Exception:
        pass
print(",".join(map(str, ok)) if ok else "0,1,2,3")
PY
)
echo "=== usable GPUs (cuda-relative): ${GPUS} ==="

mkdir -p logs results/ranker
# Clear stale benchmark outputs so the RESULTS table cannot mix in a previous
# run's numbers.
rm -f results/ranker/bench_mem_*.json logs/bench_mem_mb*.log
IFS=',' read -r G0 G1 G2 G3 <<< "$GPUS"

# One micro-batch size per GPU, running concurrently.
GPU_LIST=($(echo "$GPUS" | tr ',' ' '))
i=0
for MB in $MBS; do
  if [[ $i -ge ${#GPU_LIST[@]} ]]; then
    echo "WARNING: more micro-batch sizes (${MBS}) than GPUs (${GPUS}); skipping mb=${MB}"
    break
  fi
  G=${GPU_LIST[$i]}
  echo "----- micro_batch=${MB} on GPU ${G} -----"
  CUDA_VISIBLE_DEVICES=${G} python -u src/bench_train_memory.py \
    --model "${MODEL}" --micro_batch_size ${MB} --seq_len ${SEQ} \
    --steps 5 --lora 1 \
    --out results/ranker/bench_mem_${MODEL##*/}_mb${MB}.json \
    > logs/bench_mem_mb${MB}.log 2>&1 &
  i=$((i+1))
done
wait

echo "----- RESULTS -----"
python - <<'PY'
import glob, json, os
rows = []
for p in sorted(glob.glob("results/ranker/bench_mem_*.json")):
    try:
        data = json.load(open(p))
    except Exception as e:
        print(f"{p}: unreadable ({e})"); continue
    mb = os.path.basename(p).split("_mb")[-1].split(".")[0]
    for r in data:
        if "peak_alloc_gb" not in r and "match" not in r:
            continue
        rows.append((mb, r))
if not rows:
    print("no benchmark results found")
for mb, r in rows:
    if "match" in r:
        print(f"mb={mb}  loss agreement: {r['match']}  ({r['a']:.8f} vs {r['b']:.8f})")
        continue
    if r.get("oom"):
        print(f"mb={mb:>3}  {r['name']:<24} OOM")
    else:
        tps = r.get("tok_per_s")
        print(f"mb={mb:>3}  {r['name']:<24} peak_alloc={r['peak_alloc_gb']:7.2f} GB  "
              f"peak_reserved={r['peak_reserved_gb']:7.2f} GB  "
              f"tok/s={tps:,.0f}" if tps else
              f"mb={mb:>3}  {r['name']:<24} peak_alloc={r['peak_alloc_gb']:7.2f} GB")
PY
echo "=== BENCH MEM DONE $(date) ==="
