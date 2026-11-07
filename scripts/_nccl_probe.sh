#!/bin/bash
export PATH="/import/snvm-sc-scratch2/chenw/miniconda3/envs/streaming/bin:$PATH"
export LD_LIBRARY_PATH=/import/ml-sc-scratch1/chenw/cuda-compat-13.0:${LD_LIBRARY_PATH:-}
export NCCL_DEBUG=WARN
echo "NODE=$(hostname)"
python - <<'PY' 2>&1 | tail -12
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = torch.distributed.get_rank()
t = torch.ones(1024, device="cuda") * (r+1)
dist.all_reduce(t)
torch.cuda.synchronize()
print("NCCL all_reduce OK rank", r, "sum", t[0].item())
dist.destroy_process_group()
PY
