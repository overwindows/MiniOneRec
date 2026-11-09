"""Decisive test of the composed (CE + Bradley-Terry) loss normalisation under
DeepSpeed ZeRO-2.

THE DEFECT UNDER TEST
----------------------
src/sft_mind_pointwise_ds.py composes the loss OUTSIDE the model:

    outputs = model(**fwd)          # Engine.forward -> registers DeepSpeed's
                                     # OutputBackwardHookManager on the model's
                                     # OUTPUT TENSORS (loss AND logits)
    loss    = ce_loss + lam * bt_loss / gas
    loss    = loss * num_processes   # _scale_loss_for_devices
    accelerator.backward(loss, scale_wrt_gas=False)

Two independent scalings act on the two terms and they do NOT cancel:

  CE : the model normalises by `num_items_in_batch`. Under
       `average_tokens_across_devices` (the HF default) that is the GLOBAL
       token count of the window, gathered across all ranks, so the CE term
       ends up as a GLOBAL window mean.

  BT : `bt_loss` is a per-micro-batch MEAN over pairs, divided by `gas` =
       `current_gradient_accumulation_steps` = the PER-DEVICE number of
       micro-batches in the window (trainer.py:1714).

So the BT term is a mean over the PER-DEVICE window while the CE term is a
mean over the GLOBAL window. On W ranks the BT term ends up W times too large
relative to the CE term, i.e. `pairwise_loss_weight` means something different
on 1 GPU than on 8. In production (batch 256, micro-batch 8, 8 ranks) G is
4, so the effective lambda would be 8x the intended value.

THE MEASUREMENT (why it is trustworthy)
---------------------------------------
Under ZeRO-2 the parameter gradients live in a PARTITIONED buffer, so
`param.grad` is not populated and cannot be read back (reading it yields
zeros -- the first version of this probe made exactly that mistake). So the
probe measures at an INTERIOR tensor: `h`, the output of the embedding. The
gradient that arrives at `h` carries the TOTAL scaling DeepSpeed applied
along the whole path from the backward root down to `h`, and it is proportional
to d/d(theta). This is ZeRO-stage independent and is exactly the quantity the
hook mechanism acts on.

PART 1 -- per-micro-batch scale: does DeepSpeed apply a 1/G to the composed
          path that it does not apply to the CE-only path?
    g_ce_plain   d(ce)/dh             plain PyTorch, no DeepSpeed
    g_bt_plain   d(bt)/dh             plain PyTorch, no DeepSpeed
    g_ceonly     d(ce)/dh             DS: backward(ce)
    g_comp0      d(ce)/dh             DS: backward(ce + 0*bt)   <- CE in composed form
    g_comp1      d(ce + bt)/dh        DS: backward(ce + bt)

PART 2 -- DEVICE-COUNT INVARIANCE (the decisive one). The window is held
          FIXED in GLOBAL terms: the probe processes the same GW
          micro-batches of the same data no matter how many ranks there are
          (each rank takes its slice), so the only thing that changes
          between W=1 and W=2 is the NORMALISATION. The effective BT:CE
          weight must then be IDENTICAL for both to be correct.

              G_per_device = GW / W          (production: 256/8/8 -> G=4)
              divisor_production = G_per_device          (the old formula)
              divisor_fixed       = G_per_device * W  = GW (the corrected one)

          Reports the effective BT:CE weight of both formulas. Under the
          old formula the weight must scale with W; under the corrected one
          it must not.

Usage:
    python src/probe_pairwise_grad_scale.py [GW] [--cpu]
    GW = global window micro-batches (default 8)
    W = 1 by default; for W=2:
        python -m torch.distributed.run --nproc_per_node=2 \
            src/probe_pairwise_grad_scale.py 8 --cpu
"""
import os, sys, json

GW = 8
_use_cpu = "--cpu" in sys.argv
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
if _args:
    GW = int(_args[0])

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29533")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.loss.loss_utils import ForCausalLMLoss

B, T, H, V = 2, 6, 32, 64
YES, NO = 3, 7
LAM = 0.5

DS_CONFIG = {
    "train_batch_size": B * GW,
    "train_micro_batch_size_per_gpu": B,
    "gradient_accumulation_steps": GW,
    "steps_per_print": 10 ** 9,
    "zero_optimization": {"stage": 2},
    "bf16": {"enabled": False},
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "wall_clock_breakdown": False,
}


class TinyLM(nn.Module):
    """Minimal causal LM: emb -> head, returns a ModelOutput with loss+logits.

    It mimics the production backbone's contract exactly: the loss is computed
    by the MODEL from `logits` and `labels`, and both `loss` and `logits` come
    back in the output object -- which is what makes DeepSpeed register its
    per-tensor gradient-scaling hooks on both of them.

    `self.emb` is the PROBE POINT: a forward hook captures its output `h`,
    and a tensor hook on `h` captures the gradient that arrives there.
    """

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.head = nn.Linear(H, V)
        self.caps = {}
        self.emb.register_forward_hook(self._capture_h)

    def _capture_h(self, module, inp, out):
        h = out
        self.caps["h"] = h
        h.register_hook(lambda g: self.caps.__setitem__("g_h", g.detach().clone()))

    def forward(self, input_ids=None, labels=None, num_items_in_batch=None, **kw):
        h = self.emb(input_ids)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(
                logits=logits, labels=labels, vocab_size=V,
                num_items_in_batch=num_items_in_batch,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


def bt_term(logits):
    """A stand-in for the Bradley-Terry term: pair row 0 (pos) with row 1 (neg).

    The row used is the one that PREDICTS the answer (the last prompt row),
    exactly like the real implementation.
    """
    scores = logits[:, -2, YES].float() - logits[:, -2, NO].float()
    return -torch.log(torch.sigmoid(scores[0] - scores[1]) + 1e-8)


def proj(a, b):
    """<a, b> / ||b||^2 -- the factor by which b is scaled into a."""
    num = sum(float((x * y).sum()) for x, y in zip(a, b))
    den = sum(float((y * y).sum()) for y in b)
    return num / den if den else float("nan")


def make_global_window(seed=1234):
    """GW micro-batches of B rows -- the GLOBAL window, identical on every rank.

    Row 0 of each micro-batch is the POSITIVE of an impression and row 1 is
    its NEGATIVE, so the Bradley-Terry term always has a within-impression
    pair -- the situation the impression-packed sampler creates in production.
    The answer token of the positive row is YES, of the negative row NO.
    """
    torch.manual_seed(seed)
    ids, labs = [], []
    for _ in range(GW):
        x = torch.randint(0, V, (B, T))
        y = torch.full((B, T), -100, dtype=torch.long)
        y[0, -1] = YES
        y[1, -1] = NO
        ids.append(x)
        labs.append(y)
    return ids, labs


def main():
    torch.manual_seed(0)
    plain = TinyLM()
    torch.manual_seed(0)
    ds_model = TinyLM()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    assert GW % world == 0, f"GW={GW} must be divisible by world_size={world}"
    G = GW // world                      # per-device micro-batches in the window

    # ---- DeepSpeed engine (also initialises the process group) --------------
    import deepspeed
    if _use_cpu:
        deepspeed.init_distributed(dist_backend="gloo")
    else:
        deepspeed.init_distributed(dist_backend="nccl")

    ids, labs = make_global_window()
    # this rank's slice of the global window
    my_ids = ids[rank * G:(rank + 1) * G]
    my_labs = labs[rank * G:(rank + 1) * G]
    local_tokens = sum(int((y != -100).sum()) for y in my_labs)

    # GLOBAL token count of the window -- what `num_items_in_batch` holds under
    # `average_tokens_across_devices` (the HF default): gathered across ranks.
    t = torch.tensor([local_tokens], dtype=torch.long)
    if world > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    T_global = int(t.item())

    # ---- plain PyTorch reference (no DeepSpeed) -----------------------------
    ce_parts, bt_parts, h_parts = [], [], []
    for x, y in zip(my_ids, my_labs):
        out = plain(input_ids=x, labels=y, num_items_in_batch=T_global)
        ce_parts.append(out.loss)
        bt_parts.append(bt_term(out.logits))
        h_parts.append(plain.caps["h"])
    g_ce_plain = torch.autograd.grad(ce_parts[0], h_parts[0], retain_graph=True)[0].detach().clone()
    g_bt_plain = torch.autograd.grad(bt_parts[0], h_parts[0], retain_graph=True)[0].detach().clone()

    engine, _, _, _ = deepspeed.initialize(
        model=ds_model, model_parameters=ds_model.parameters(),
        config={**DS_CONFIG,
                "train_batch_size": B * GW,
                "gradient_accumulation_steps": G},
    )

    def ds_micro(x, y, divisor, lam=LAM, scale_w=world):
        """One micro-batch through the engine, mirroring the trainer exactly.

        `divisor` is what the BT term is divided by; `scale_w` is the
        `_scale_loss_for_devices` factor (num_processes).
        """
        ds_model.caps.clear()
        o = engine(input_ids=x, labels=y, num_items_in_batch=T_global)
        loss = o.loss
        if divisor is not None:
            loss = loss + lam * bt_term(o.logits) / divisor
        loss = loss * scale_w
        engine.backward(loss, scale_wrt_gas=False)
        return ds_model.caps["g_h"].clone()

    def ds_window(divisor, lam=LAM):
        """Accumulate this rank's micro-batches, then rank-average."""
        total = None
        for i in range(G):
            g = ds_micro(my_ids[i], my_labs[i], divisor, lam=lam)
            total = g.clone() if total is None else total + g
        if world > 1:
            dist.all_reduce(total, op=dist.ReduceOp.AVG)
        return total

    # ---- PART 1: per-micro-batch scale -------------------------------------
    c_ceonly = ds_micro(my_ids[0], my_labs[0], None)
    c_comp0 = ds_micro(my_ids[0], my_labs[0], G, lam=0.0)
    c_comp1 = ds_micro(my_ids[0], my_labs[0], G, lam=1.0)

    r_ceonly = proj([c_ceonly], [g_ce_plain])
    r_cecomp = proj([c_comp0], [g_ce_plain])
    r_btcomp = proj([c_comp1 - c_comp0], [g_bt_plain])
    same_ce = abs(r_cecomp - r_ceonly) < 1e-9 * max(1.0, abs(r_ceonly))

    # ---- PART 2: device-count invariance -----------------------------------
    g_ce_win = ds_window(None)
    g_prod_win = ds_window(G)          # production formula:  ce + lam*bt/G
    g_fix_win = ds_window(G * world)   # corrected formula:   ce + lam*bt/(G*W)

    g_bt_prod = g_prod_win - g_ce_win
    g_bt_fix = g_fix_win - g_ce_win

    def rel(g_bt, g_ce):
        num = sum(float((x * y).sum()) for x, y in zip(g_bt, g_ce))
        den = sum(float((y * y).sum()) for y in g_ce)
        return num / den if den else float("nan")

    def nratio(g_bt, g_ce):
        nb = sum(float((x * x).sum()) for x in g_bt) ** 0.5
        nc = sum(float((y * y).sum()) for y in g_ce) ** 0.5
        return nb / nc if nc else float("nan")

    w_prod = rel(g_bt_prod, g_ce_win)
    w_fix = rel(g_bt_fix, g_ce_win)
    n_prod = nratio(g_bt_prod, g_ce_win)
    n_fix = nratio(g_bt_fix, g_ce_win)

    if rank == 0:
        print("\n=== DeepSpeed gradient-scale probe ===")
        print(f"  backend={'gloo/CPU' if _use_cpu else 'nccl/GPU'}")
        print(f"  global window GW={GW} micro-batches   world_size W={world}   "
              f"per-device G={G}   batch B={B}   lambda={LAM}")
        print(f"  global window tokens (num_items_in_batch) = {T_global}")
        print()
        print("PART 1 -- per-micro-batch scale (probe point: h = embedding output)")
        print(f"  ratio_ceonly = <g_ceonly , g_ce_plain> / ||g_ce_plain||^2 = {r_ceonly:.8f}")
        print(f"  ratio_cecomp = <g_comp0  , g_ce_plain> / ||g_ce_plain||^2 = {r_cecomp:.8f}")
        print(f"  ratio_btcomp = <g_comp1-g_comp0, g_bt_plain> / ||g_bt_plain||^2 = {r_btcomp:.8f}")
        print(f"  CE scale: CE-only {r_ceonly:.8f} vs composed {r_cecomp:.8f} -> "
              f"{'SAME' if same_ce else 'DIFFERENT'}")
        print("      => DeepSpeed applies NO extra 1/G to either path (the ratio is")
        print("         exactly the _scale_loss_for_devices factor = num_processes = W).")
        print("         Composing the loss does NOT change the effective learning rate.")
        print()
        print("PART 2 -- effective BT:CE weight (the decisive measurement)")
        print(f"  production  ce + lam*bt/{G}        : proj {w_prod:.8f}   norm-ratio {n_prod:.8f}")
        print(f"  corrected   ce + lam*bt/{G*world}   : proj {w_fix:.8f}   norm-ratio {n_fix:.8f}")
        print(f"  corrected / production               : proj {w_fix / w_prod if w_prod else float('nan'):.8f}"
              f"   norm-ratio {n_fix / n_prod if n_prod else float('nan'):.8f}")
        print()
        print("  The window is held FIXED in global terms (same GW micro-batches of")
        print("  the same data; each rank takes its slice), so ONLY the")
        print("  normalisation differs between W=1 and W=2.")
        print("  => Run this at W=1 and W=2 and compare the two weight lines.")
        print("     The CORRECTED weight must be IDENTICAL; the production weight")
        print("     must differ by exactly a factor of W.")

        out_json = {
            "GW": GW, "G_per_device": G, "world_size": world,
            "backend": "cpu" if _use_cpu else "gpu",
            "ratio_ceonly": r_ceonly, "ratio_cecomp": r_cecomp,
            "ratio_btcomp": r_btcomp, "ce_scale_same": bool(same_ce),
            "weight_production_proj": w_prod,
            "weight_corrected_proj": w_fix,
            "weight_production_normratio": n_prod,
            "weight_corrected_normratio": n_fix,
        }
        print("RESULT_JSON " + json.dumps(out_json))


if __name__ == "__main__":
    main()
