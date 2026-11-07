#!/bin/bash
# Probe: (1) is the fla/Triton linear-attention fast path active on this GPU?
#        (2) how much faster is the prefix-cache path vs the plain path?
#        (3) does the Qwen3-8B LoRA adapter load + score through the prefix cache?
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/dev/shm/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# compat lib only if CUDA is unavailable (never preload on a new-driver node)
COMPAT_DIR=/import/ml-sc-scratch1/chenw/cuda-compat-13.0
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if [ -f "${COMPAT_DIR}/libcuda.so.1" ]; then
    echo "=== old driver; preloading CUDA forward-compat ==="
    export LD_LIBRARY_PATH=${COMPAT_DIR}:${LD_LIBRARY_PATH:-}
  else
    echo "FATAL: no compat lib"; exit 1
  fi
fi

echo "=== NODE=$(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
try:
    import triton
    print("triton", triton.__version__)
except Exception as e:
    print("triton MISSING", e)
import fla
print("fla", fla.__version__)
# Does the fla chunk kernel actually dispatch to Triton on THIS gpu?
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import inspect
    print("fla chunk kernel module:", chunk_gated_delta_rule.__module__)
except Exception as e:
    print("fla import err", e)
PY

run () {
  local name=$1 ckpt=$2 prefix=$3 n=$4
  echo "----- $name prefix_cache=$prefix n=$n -----"
  local t0=$(date +%s)
  MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large \
  USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=30 \
  USE_RECENCY=1 HISTORY_SELECTION=recent HISTORY_FORMAT=flat \
  PREFIX_CACHE=$prefix BATCH_SIZE=8 CUDA_VISIBLE_DEVICES=0 \
  timeout 1200 bash scripts/eval_mind_pointwise.sh "$ckpt" dev $n 2>&1 \
    | tr '\r' '\n' | grep -aE "Prefix KV cache|AUC:|PEFT|base model|Model loaded|Error|Traceback" | head -8
  echo "$name prefix=$prefix elapsed=$(( $(date +%s) - t0 ))s"
}

run "q8b_adapter" output_dir/small_runs/lg_q8b/checkpoint-14500 1 128
run "q8b_adapter" output_dir/small_runs/lg_q8b/checkpoint-14500 0 128
run "q35_9b"      results/ranker/lg_9b_ckpt89000_snapshot   1 128
run "q35_9b"      results/ranker/lg_9b_ckpt89000_snapshot   0 128
echo "=== PROBE DONE ==="
