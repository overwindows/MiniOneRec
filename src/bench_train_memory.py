"""Measure the GPU memory and throughput of each training-memory lever.

WHY THIS EXISTS
---------------
The B200 training runs sit at ~50 GB / 183 GB (27%) of VRAM while the GPUs
report 93-100% utilization and 65-90% of their power limit. So the memory is
not "wasted" -- it is headroom that is not being converted into throughput,
because the micro-batch is capped by whatever the activation + logits memory
costs, not by the GPU.

This script measures, for one backbone and one micro-batch size:

  A. full logits + gradient checkpointing   (the current production config)
  B. answer-only logits + gradient checkpointing
  C. full logits, no gradient checkpointing
  D. answer-only logits, no gradient checkpointing

and reports peak allocated/reserved VRAM and tokens/s for each. The point is
to find the config that fits the largest micro-batch on a given GPU, because
a larger micro-batch is what actually buys throughput.

It also re-checks that the answer-only loss is numerically identical to the
full-logits loss, so the memory saving is not bought with a wrong gradient.

Usage (single GPU):
    python src/bench_train_memory.py --model models/Qwen3.5-4B \
        --micro_batch_size 4 --seq_len 2048
"""

import argparse
import json
import os
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_mind_pointwise_ds import PointwiseTrainer, answer_token_len

# Module-level switch so the benchmark's Trainer subclass can force the
# answer-only path ON or OFF. It is read inside __init__ (via
# _check_logits_to_keep_support), so it cannot be an instance attribute.
_FORCE_KEEP = [False]


def _log(*a):
    print(*a, flush=True)


def _resolve_model_dir(model: str) -> str:
    """The repo stores models as `models/<name>`; accept a bare name too."""
    if os.path.isdir(model):
        return model
    cand = os.path.join("models", model)
    if os.path.isdir(cand):
        return cand
    return model


def build_batch(tokenizer, batch_size, seq_len, device):
    """A batch shaped like the real training data.

    The real examples are `prompt + " Yes"|" No"` with labels = [-100]*P + target.
    We only need the SHAPE to be right for the memory measurement: a long
    prompt of ~seq_len tokens plus a 1-token answer, left-padded.
    """
    filler = "the quick brown fox jumps over the lazy dog " * (seq_len // 10)
    prompts, answers = [], []
    for i in range(batch_size):
        p = f"User History: {filler}\nCandidate: {filler[:200]}\nIs this relevant? Answer:"
        prompts.append(p)
        answers.append(" Yes" if i % 2 == 0 else " No")

    tok = tokenizer(prompts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=seq_len - 1)
    ans_ids = tokenizer(answers, return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = torch.cat([tok["input_ids"], ans_ids], dim=1)
    attention_mask = torch.cat(
        [tok["attention_mask"], torch.ones_like(ans_ids)], dim=1
    )
    labels = torch.full_like(input_ids, -100)
    labels[:, -ans_ids.shape[1]:] = ans_ids
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def measure(model, batch, answer_len, use_logits_to_keep, grad_ckpt, steps, device):
    """Run a few fwd+bwd steps and report peak memory + throughput."""
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        # With frozen base weights + LoRA, the checkpointed blocks would get no
        # gradient signal (the embedding output does not require grad), so the
        # checkpoint would not be recomputed. This is the standard PEFT fix.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_disable()
        model.config.use_cache = False

    # A minimal Trainer subclass so we exercise the REAL compute_loss path
    # (the one that decides whether logits_to_keep is used).
    # `_FORCE_KEEP` is read during __init__ (the base class calls
    # _check_logits_to_keep_support there), so it must be a module-level
    # switch rather than an instance attribute set after super().__init__.
    class _Bench(PointwiseTrainer):
        def _check_logits_to_keep_support(self) -> bool:
            if not _FORCE_KEEP[0]:
                return False
            return super()._check_logits_to_keep_support()

    _FORCE_KEEP[0] = use_logits_to_keep
    trainer = _Bench(
        model=model,
        train_dataset=None,
        args=transformers.TrainingArguments(
            output_dir="/tmp/bench_mem",
            per_device_train_batch_size=batch["input_ids"].shape[0],
            report_to=[],
            logging_steps=10**9,
            save_strategy="no",
            use_cpu=False,
        ),
        answer_len=answer_len,
    )
    _FORCE_KEEP[0] = False

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()

    losses = []
    t0 = None
    for step in range(steps):
        if step == 2:  # warmup: let cuBLAS/cuDNN pick kernels
            torch.cuda.synchronize()
            t0 = time.time()
        loss = trainer.compute_loss(model, dict(batch))
        loss.backward()
        model.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))

    torch.cuda.synchronize()
    elapsed = (time.time() - t0) if t0 else float("nan")
    ntok = batch["input_ids"].numel() * max(steps - 2, 1)

    return {
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        "tok_per_s": ntok / elapsed if elapsed == elapsed and elapsed > 0 else None,
        "logits_to_keep": bool(use_logits_to_keep),
        "grad_checkpointing": bool(grad_ckpt),
        "oom": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-4B")
    ap.add_argument("--micro_batch_size", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lora", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda"
    mdir = _resolve_model_dir(args.model)
    _log(f"=== bench: model={mdir} micro_batch={args.micro_batch_size} "
         f"seq_len={args.seq_len} lora={args.lora} ===")
    _log(f"gpu: {torch.cuda.get_device_name(0)}  "
         f"total={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(mdir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        mdir, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ))
    model.train()

    a_len = answer_token_len(tokenizer)
    batch = build_batch(tokenizer, args.micro_batch_size, args.seq_len, device)
    _log(f"batch: input_ids={tuple(batch['input_ids'].shape)} answer_len={a_len}")

    configs = [
        ("A full-logits + ckpt", False, True),
        ("B answer-only + ckpt", True, True),
        ("C full-logits no-ckpt", False, False),
        ("D answer-only no-ckpt", True, False),
    ]

    results = []
    for name, keep, ckpt in configs:
        _log(f"--- {name} ---")
        try:
            r = measure(model, batch, a_len, keep, ckpt, args.steps, device)
            r["name"] = name
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(
                    e, torch.OutOfMemoryError):
                raise
            torch.cuda.empty_cache()
            # The failed backward may have left partial gradients behind; drop
            # them so the next config starts clean.
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            r = {"name": name, "oom": True, "logits_to_keep": keep,
                 "grad_checkpointing": ckpt}
            _log(f"    OOM")
        results.append(r)
        _log(f"    {json.dumps(r)}")

    # The full-logits and answer-only losses must agree: the memory saving
    # must not come with a different gradient.
    a = next((r for r in results if r["name"].startswith("A")), None)
    b = next((r for r in results if r["name"].startswith("B")), None)
    if a and b and not a.get("oom") and not b.get("oom"):
        same = abs(a["loss_first"] - b["loss_first"]) < 1e-4
        _log(f"loss agreement A vs B: {a['loss_first']:.8f} vs {b['loss_first']:.8f} "
             f"-> {'MATCH' if same else 'MISMATCH'}")
        results.append({"name": "loss_agreement", "match": same,
                        "a": a["loss_first"], "b": b["loss_first"]})

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        _log(f"wrote {args.out}")
    _log("=== BENCH DONE ===")


if __name__ == "__main__":
    main()
