"""Pin down the H200 SDPA backward overflow to a SPECIFIC kernel.

WHAT WE KNOW (measured, 2026-11-08):
  * The same command (Qwen3-1.7B, bf16, sdpa, answer-only logits, packed
    sampler, lambda=0.5) trains fine on A100 (SM80) and B200 (SM100) and
    FAILS on H200 (SM90): DeepSpeed skips 36/40 optimizer steps.
  * The FORWARD loss is finite on the H200 (27.76), so the forward is fine --
    it is the BACKWARD that produces inf/NaN.
  * Switching the attention backend to `eager` FIXES the H200 (0 skips,
    loss 26.8 at step 10 and falling).

SO THE SUSPECT IS A torch SDPA BACKEND ON SM90.

This probe runs the exact Qwen3-1.7B attention shapes through each SDPA
backend separately and reports whether the BACKWARD gradient is finite:

  Qwen3-1.7B: 16 query heads, 8 kv heads (GQA ratio 2), head_dim 128,
  28 layers, bf16, cutoff_len 2048, micro-batch 8, LEFT padding.

Each backend is selected with `torch.nn.attention.sdpa_kernel(SDPBackend.X)`:
  FLASH_ATTENTION, EFFICIENT_ATTENTION, CUDNN_ATTENTION, MATH.

It also tests the two mask regimes the trainer actually uses:
  * `causal_no_mask`  -- is_causal=True, no explicit mask (what SDPA uses
                         when there is no padding)
  * `additive_mask`   -- an explicit bf16 additive mask with -inf on the
                         padded positions (what SDPA uses when the batch has
                         left padding, which the trainer ALWAYS has)

Usage:  python src/probe_sdpa_backend_overflow.py
"""

import os
import sys
import json

import torch
import torch.nn.functional as F

# Qwen3-1.7B attention geometry.
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128
N_LAYERS = 28
MICRO_BATCH = 8
SEQ_LEN = int(os.environ.get("PROBE_SEQ", "2048"))
DTYPE = torch.bfloat16

# A bf16 "additive" mask with -inf on the padded positions, the regime the
# trainer hits (left padding -> the mask is never None).
FINF = torch.finfo(DTYPE).min


def _qkv(seq, device):
    g = torch.Generator(device="cpu").manual_seed(1234)
    q = torch.randn(MICRO_BATCH, N_HEADS, seq, HEAD_DIM, generator=g).to(device=device, dtype=DTYPE)
    k = torch.randn(MICRO_BATCH, N_KV_HEADS, seq, HEAD_DIM, generator=g).to(device=device, dtype=DTYPE)
    v = torch.randn(MICRO_BATCH, N_KV_HEADS, seq, HEAD_DIM, generator=g).to(device=device, dtype=DTYPE)
    return q, k, v


def _mask(seq, device, mode):
    if mode == "causal_no_mask":
        return None
    # Left padding: the first `pad` positions are padding.
    pad = max(1, seq // 8)
    m = torch.zeros(seq, seq, device=device, dtype=DTYPE)
    m[:, :pad] = FINF
    # Causal: position i may attend to 0..i.
    causal = torch.full((seq, seq), FINF, device=device, dtype=DTYPE)
    causal = torch.triu(causal, diagonal=1)
    m = m + causal
    # A fully-masked row would produce NaN in ANY backend; make sure the
    # first real row keeps at least one unmasked key.
    m[pad, :pad + 1] = 0
    return m.view(1, 1, seq, seq)


def run_one(backend_name, mask_mode, device):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {
        "FLASH": SDPBackend.FLASH_ATTENTION,
        "EFFICIENT": SDPBackend.EFFICIENT_ATTENTION,
        "CUDNN": SDPBackend.CUDNN_ATTENTION,
        "MATH": SDPBackend.MATH,
    }
    backend = backends[backend_name]
    seq = SEQ_LEN
    q, k, v = _qkv(seq, device)
    mask = _mask(seq, device, mask_mode)

    q = q.clone().requires_grad_(True)
    k = k.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)

    out = None
    err = None
    try:
        with sdpa_kernel(backend):
            # GQA: expand kv heads, the same thing transformers' sdpa path does
            # when the mask prevents enable_gqa.
            kk = k.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
            vv = v.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
            if mask_mode == "causal_no_mask":
                out = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
            else:
                out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
            loss = out.float().pow(2).mean()
            loss.backward()
    except Exception as e:  # a backend that refuses the shape is itself a result
        return {
            "backend": backend_name, "mask": mask_mode, "seq": seq,
            "status": "EXCEPTION", "error": f"{type(e).__name__}: {e}"[:200],
        }

    grads = {"q": q.grad, "k": k.grad, "v": v.grad}
    res = {"backend": backend_name, "mask": mask_mode, "seq": seq, "status": "ok"}
    for name, g in grads.items():
        if g is None:
            res[f"grad_{name}"] = "None"
            continue
        finite = bool(torch.isfinite(g).all())
        res[f"grad_{name}_finite"] = finite
        if finite:
            res[f"grad_{name}_maxabs"] = float(g.detach().abs().max().item())
        else:
            bad = int((~torch.isfinite(g)).sum().item())
            res[f"grad_{name}_nonfinite"] = f"{bad}/{g.numel()}"
    res["out_finite"] = bool(torch.isfinite(out).all())
    if res["out_finite"]:
        res["out_maxabs"] = float(out.detach().abs().max().item())
    return res


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 1
    device = "cuda"
    props = torch.cuda.get_device_properties(0)
    print(f"=== SDPA BACKEND OVERFLOW PROBE ===")
    print(f"device: {props.name}  SM {props.major}.{props.minor}  torch {torch.__version__}")
    print(f"shapes: B={MICRO_BATCH} H={N_HEADS} KV={N_KV_HEADS} D={HEAD_DIM} L={SEQ_LEN} dtype={DTYPE}")
    print(f"cudnn_sdp={torch.backends.cuda.cudnn_sdp_enabled()} "
          f"flash_sdp={torch.backends.cuda.flash_sdp_enabled()} "
          f"mem_efficient_sdp={torch.backends.cuda.mem_efficient_sdp_enabled()}")

    results = []
    for mask_mode in ("causal_no_mask", "additive_mask"):
        for backend_name in ("FLASH", "EFFICIENT", "CUDNN", "MATH"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                r = run_one(backend_name, mask_mode, device)
            except Exception as e:
                r = {"backend": backend_name, "mask": mask_mode, "seq": SEQ_LEN,
                     "status": "OUTER_EXCEPTION", "error": f"{type(e).__name__}: {e}"[:200]}
            results.append(r)
            flags = []
            for key in ("grad_q_finite", "grad_k_finite", "grad_v_finite"):
                if key in r:
                    flags.append(f"{key[5:-7]}={'OK' if r[key] else 'NONFINITE'}")
            nonfinite = {k: v for k, v in r.items() if "nonfinite" in k}
            print(f"  {backend_name:>10} / {mask_mode:<16} -> {r['status']:<8} "
                  f"out_finite={r.get('out_finite')} {' '.join(flags)} {nonfinite if nonfinite else ''}"
                  + (f"  ERR={r['error']}" if r.get("error") else ""))

    print("\n=== SUMMARY ===")
    bad = [r for r in results
           if r.get("status") != "ok"
           or any(r.get(k) is False for k in ("grad_q_finite", "grad_k_finite", "grad_v_finite"))]
    if bad:
        print("BACKENDS PRODUCING A NON-FINITE BACKWARD GRADIENT:")
        for r in bad:
            print(f"  - {r['backend']} / {r['mask']}: "
                  + json.dumps({k: v for k, v in r.items() if k not in ("backend", "mask")}))
    else:
        print("  none -- every backend produced a finite backward gradient "
              "(so the overflow is NOT in the SDPA kernel itself)")

    out_path = os.environ.get("PROBE_OUT", "")
    if out_path:
        with open(out_path, "w") as f:
            json.dump({"device": props.name, "sm": f"{props.major}.{props.minor}",
                       "results": results}, f, indent=2)
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
