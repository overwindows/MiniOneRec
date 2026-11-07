#!/bin/bash
# Validate the HYBRID prefix cache against the plain path, on a hybrid
# (GatedDeltaNet) backbone.
#
# WHY THIS EXISTS
# ---------------
# The packed prefix cache (pack the candidate suffixes into one sequence behind
# a shared prefix KV) cannot be used on Qwen3.5: the linear-attention layers
# carry a causal conv state and a recurrent state, both sequential over the
# token axis, so a packed sequence would let one candidate's suffix bleed into
# the next. The hybrid mode instead runs the prefix once at batch 1, replicates
# its states across the batch, and gives every candidate its own batch row.
#
# The split is EXACT, not an approximation:
#   * the gated delta rule is partition-invariant, so handing the prefix's
#     recurrent state S_P to the suffix reproduces the full-sequence outputs;
#   * the causal conv's left-context handoff reproduces the full-sequence tail.
# So any divergence from the plain path is a bug (mask, padding side, or bf16
# accumulation order), not an expected approximation.
#
# THE FOUR ARMS
# -------------
#   plain    -- the production path: prefix cache OFF, batch 8, LEFT padding.
#   plain_rp -- prefix cache OFF, batch 8, RIGHT padding.
#   plain_b1 -- prefix cache OFF, batch 1, i.e. NO padding at all. This is the
#               ground truth: every prompt is scored alone, so nothing can be
#               contaminated by another row.
#   hybrid   -- prefix cache ON, hybrid mode (the candidate implementation).
#
# Reading the result:
#   * plain vs plain_b1  -> does LEFT padding perturb the recurrence? It
#     should NOT: apply_mask_to_padding_states zeroes the pad hidden states
#     (so q=k=v=0 there), and the pads LEAD the sequence, so the recurrent
#     state is still zero when the decay is applied. A prior single-layer
#     simulation measured 0.0-1.28e-8 relative error for 1..150 pads.
#     This arm therefore also measures the plain path's own bf16 batching
#     noise, which is the real baseline for judging the hybrid arm.
#   * plain_rp vs plain_b1 -> RIGHT padding: the trimmed recurrent mask now
#     covers the real tokens, and the pads sit AFTER the answer position.
#     This is the padding layout the hybrid path uses internally.
#   * hybrid vs plain_b1 -> does the hybrid prefix cache reproduce the truth?
#
# All arms read the same behaviors file and take the first N impressions, so
# the per-impression score sets are directly comparable.
#
# Usage: sbatch scripts/_hybrid_ab.sh
# Env:   AB_CKPT, AB_N, AB_TAG, AB_GPUS
cd /import/ml-sc-scratch5/chenw/MiniOneRec
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/dev/shm/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# ---- CUDA driver compatibility -------------------------------------------
# The cu130 torch build needs driver >= 595.71.05. Some A100/H100 nodes still
# carry an older driver, where torch.cuda.is_available() is False and the
# model would load on CPU. The CUDA forward-compat package ships a newer
# libcuda that works against the old kernel driver. It must ONLY be preloaded
# on old-driver nodes: on a new-driver node it shadows the real libcuda and
# breaks CUDA entirely.
COMPAT_DIR=/import/ml-sc-scratch1/chenw/cuda-compat-13.0
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if [ -f "${COMPAT_DIR}/libcuda.so.1" ]; then
    echo "=== old driver; preloading CUDA forward-compat ==="
    export LD_LIBRARY_PATH=${COMPAT_DIR}:${LD_LIBRARY_PATH:-}
  else
    echo "FATAL: no compat lib"; exit 1
  fi
fi

echo "=== NODE=$(hostname) $(date) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

CKPT=${AB_CKPT:-results/ranker/lg_9b_ckpt89000_snapshot}
N=${AB_N:-1000}
TAG=${AB_TAG:-q35}

# ---- resolve which GPUs this job may use ----------------------------------
# The four arms each take ONE GPU and each re-exports CUDA_VISIBLE_DEVICES.
#
# The device id to export is the CUDA-RELATIVE index, NOT the physical one.
# Inside a Slurm job the allocated GPUs are already the only ones visible --
# either CUDA_VISIBLE_DEVICES masks them, or the cgroup device controller
# hides the rest -- and in BOTH cases the CUDA driver renumbers what it sees
# to 0..N-1. So exporting 0,1,2,3 picks exactly the four allocated GPUs.
#
# (Exporting the physical ids instead -- e.g. "4,5,6,7" -- would be wrong:
# inside the job those renumbered to 0..3, so index 4 would not exist.)
#
# If Slurm exposed nothing and every GPU on the node is visible, probe which
# ones actually allocate, so we never grab a device that is not ours.
resolve_gpus () {
  python - <<'PY' 2>/dev/null || echo ""
import os, torch
free = []
for i in range(torch.cuda.device_count()):
    try:
        torch.zeros(1, device=f"cuda:{i}")
        free.append(i)
    except Exception:
        pass
print(",".join(map(str, free)))
PY
}

DEVS=($(resolve_gpus | tr ',' ' '))
echo "=== allocated GPUs (cuda-relative): ${DEVS[*]} | SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} ==="
if [[ ${#DEVS[@]} -lt 4 ]]; then
  echo "FATAL: need 4 usable GPUs, found ${#DEVS[@]} (${DEVS[*]})"
  exit 1
fi
GPUS="${DEVS[0]},${DEVS[1]},${DEVS[2]},${DEVS[3]}"

mkdir -p results/ranker logs

# ---- probe: which delta-rule kernel is actually running? ------------------
# The transformers warning "The fast path is not available..." fires when ANY
# of causal_conv1d_fn / causal_conv1d_update / chunk_gated_delta_rule /
# fused_recurrent_gated_delta_rule is missing. causal_conv1d IS missing here,
# so the warning fires even when fla's Triton chunk kernel is active. Print
# the resolved kernel identity so this is settled by evidence, not by
# inference from the warning.
echo "--- kernel probe ---"
python - <<'PY'
import torch, transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
print("torch", torch.__version__, "transformers", transformers.__version__)
print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
for name, sym in [("causal_conv1d_fn", m.causal_conv1d_fn),
                  ("causal_conv1d_update", m.causal_conv1d_update),
                  ("chunk_gated_delta_rule", m.chunk_gated_delta_rule),
                  ("fused_recurrent_gated_delta_rule", m.fused_recurrent_gated_delta_rule)]:
    print(f"  {name:32s} -> {getattr(sym, '__module__', None)}")
print("  is_fast_path_available          ->", m.is_fast_path_available)
try:
    import triton
    print("  triton", triton.__version__)
except Exception as e:
    print("  triton MISSING", e)
PY

# arm <name> <prefix_cache> <gpu> <n> <extra env assignments...>
arm () {
  local name=$1 pc=$2 gpu=$3 n=$4; shift 4
  local OUT=results/ranker/ab_${TAG}_${name}_scores.tsv
  echo "----- arm=${name} prefix_cache=${pc} extra='$*' on GPU ${gpu} n=${n} -----"
  local t0=$(date +%s)
  env MIND_ROOT=/import/ml-sc-scratch5/chenw/MiniOneRec/data/MIND_large \
    USE_ABSTRACT=1 USE_CHAT_TEMPLATE=1 FLASH_ATTN=0 MAX_HISTORY=30 \
    USE_RECENCY=1 HISTORY_SELECTION=recent HISTORY_FORMAT=flat \
    PREFIX_CACHE=${pc} PREFIX_CACHE_MODE=hybrid PREFIX_GROUP_SIZE=16 \
    BATCH_SIZE=8 OUTPUT_SCORES_FILE=${OUT} CUDA_VISIBLE_DEVICES=${gpu} \
    "$@" \
    bash scripts/eval_mind_pointwise.sh "$CKPT" dev $n > logs/ab_${TAG}_${name}.log 2>&1
  echo "  arm=${name} elapsed=$(( $(date +%s) - t0 ))s"
  tr '\r' '\n' < logs/ab_${TAG}_${name}.log \
    | grep -aE "Prefix KV cache|AUC:|MRR:|nDCG|Error|Traceback" | head -8
}

# The batch-1 ground-truth arm is ~3x slower than the batched arms (3x more
# forwards, same total tokens), so it runs on fewer impressions. The diff
# only compares the impressions both arms produced.
N_B1=${AB_N_B1:-200}

# Launch the four arms concurrently, one GPU each.
IFS=',' read -r G0 G1 G2 G3 <<< "$GPUS"
arm plain    0 "$G0" "$N"                       &
P0=$!
arm plain_rp 0 "$G1" "$N" RIGHT_PAD=1           &
P1=$!
arm plain_b1 0 "$G2" "$N_B1" BATCH_SIZE=1       &
P2=$!
arm hybrid   1 "$G3" "$N"                       &
P3=$!
wait $P0 $P1 $P2 $P3

echo "----- DIFF -----"
python - "$TAG" <<'PY'
import sys, numpy as np
tag = sys.argv[1]

def load(p):
    d = {}
    for line in open(p):
        parts = line.rstrip("\n").split("\t")
        d[parts[0]] = np.array([float(x) for x in parts[1:]])
    return d

ref = load(f"results/ranker/ab_{tag}_plain_b1_scores.tsv")   # ground truth
names = ["plain", "plain_rp", "hybrid"]

def compare(name, a, b):
    common = sorted(set(a) & set(b))
    if not common:
        print(f"{name}: NO COMMON IMPRESSIONS (a={len(a)} b={len(b)})")
        return
    diffs, rhos, flips, tot, spreads = [], [], 0, 0, []
    for k in common:
        x, y = a[k], b[k]
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        if n < 2:
            continue
        diffs.append(np.abs(x - y).max())
        spreads.append(x.std())
        tot += n
        if not np.array_equal(np.argsort(np.argsort(-x)), np.argsort(np.argsort(-y))):
            flips += 1
        rx = np.argsort(np.argsort(-x)).astype(float)
        ry = np.argsort(np.argsort(-y)).astype(float)
        if rx.std() > 0 and ry.std() > 0:
            rhos.append(np.corrcoef(rx, ry)[0, 1])
    diffs = np.array(diffs); spreads = np.array(spreads)
    print(f"{name}: impressions common={len(common)} candidates={tot}")
    print(f"  max |dscore|: mean={diffs.mean():.6f} p50={np.median(diffs):.6f} "
          f"p95={np.percentile(diffs,95):.6f} max={diffs.max():.6f}")
    print(f"  delta / within-impression std: mean={np.mean(diffs/spreads):.4f}")
    print(f"  impressions with any rank flip: {flips}/{len(common)}")
    if rhos:
        r = np.array(rhos)
        print(f"  per-impression Spearman rho: mean={r.mean():.6f} "
              f"p05={np.percentile(r,5):.6f} min={r.min():.6f}")

print("ground truth = plain_b1 (batch 1, no padding)")
for nm in names:
    try:
        compare(f"{nm:9s} vs plain_b1", load(f"results/ranker/ab_{tag}_{nm}_scores.tsv"), ref)
    except FileNotFoundError as e:
        print(f"{nm}: MISSING ({e})")
PY
echo "=== HYBRID AB DONE ==="
