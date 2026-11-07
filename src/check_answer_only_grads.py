"""Decisive equivalence test for the answer-only (`logits_to_keep`) loss path.

WHY
---
The forward loss is already proven bit-identical (src/bench_train_memory.py:
2.248723268508911 for both paths). But the REAL trainer A/B showed the two
configurations' loss trajectories diverging from step 2 onward (24.89 vs
24.87, then 1.161 vs 1.255). Two explanations:

  (a) the answer-only BACKWARD produces different gradients (a real bug), or
  (b) bf16 nondeterminism amplified by a chaotic loss landscape.

    The first logged loss is 24.94, i.e. the model is at p(answer) ~ 1e-11:
    the landscape is extremely steep, and a bf16-level difference in the
    gradient (the sliced lm_head GEMM has a different M dimension, so cuBLAS
    tiles it differently) gets amplified by AdamW's normalized updates.

This script separates them with two phases on the SAME real batch, in the
SAME process, on the SAME model:

  PHASE 1 -- model.eval() (LoRA dropout OFF, so the only difference between
             the two paths is the path itself):
               A  answer-only path
               B  full-logits path
             If |grad(A) - grad(B)| is at bf16 epsilon, the paths are
             mathematically equivalent and the divergence is chaos.

  PHASE 2 -- model.train() (LoRA dropout ON, so the run-to-run noise floor
             is measurable):
               A1, A2  answer-only path, twice   -> noise floor
               B1, B2  full-logits path, twice   -> noise floor
             Compare |A1 - A2| (noise) against |A1 - B1| (path).

If the path difference is the same order as the noise floor, the answer-only
path is safe to enable for the production runs.

Usage:
    python src/check_answer_only_grads.py --model models/Qwen3.5-4B
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import MINDPointwiseSFTDataset  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-4B")
    ap.add_argument("--root", default="data/MIND_small")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_batches", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/ranker/answer_only_grad_check.json")
    return ap.parse_args()


def grads_of(model, trainer, batch, use_answer_only, device):
    """One forward+backward; returns (loss, {param_name: grad_tensor})."""
    model.zero_grad(set_to_none=True)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if use_answer_only:
        labels = inputs["labels"]
        keep = trainer.answer_len
        inputs["labels"] = torch.cat(
            [labels.new_full((labels.shape[0], 1), -100), labels[:, -keep:]], dim=1
        )
        inputs["logits_to_keep"] = trainer._logits_to_keep
    loss = trainer.compute_loss(model, inputs)
    loss.backward()
    g = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            g[name] = p.grad.detach().float().clone()
    return float(loss.detach()), g


def compare(g1, g2):
    """Relative difference between two gradient dicts."""
    names = sorted(set(g1) & set(g2))
    if not names:
        return {"n_tensors": 0, "global_rel_diff": float("nan"),
                "worst_tensor_rel_diff": float("nan"), "worst_tensor": "",
                "norm_A": 0.0, "norm_B": 0.0}
    worst, worst_name = 0.0, ""
    sq1 = sq2 = sqd = 0.0
    for n in names:
        a, b = g1[n], g2[n]
        if a.shape != b.shape:
            continue
        na = float(a.norm())
        nb = float(b.norm())
        nd = float((a - b).norm())
        if na > 0:
            rel = nd / na
            if rel > worst:
                worst, worst_name = rel, n
        sq1 += na * na
        sq2 += nb * nb
        sqd += nd * nd
    g1n, g2n, gd = float(np.sqrt(sq1)), float(np.sqrt(sq2)), float(np.sqrt(sqd))
    return {
        "n_tensors": len(names),
        "global_rel_diff": gd / g1n if g1n > 0 else float("nan"),
        "worst_tensor_rel_diff": worst,
        "worst_tensor": worst_name,
        "norm_A": g1n,
        "norm_B": g2n,
    }


def main():
    args = parse_args()
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.sft_mind_pointwise_ds import PointwiseTrainer, answer_token_len

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=== answer-only gradient equivalence check ===")
    print(f"model={args.model} batch_size={args.batch_size} n_batches={args.n_batches}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    ds = MINDPointwiseSFTDataset(
        behaviors_path=f"{args.root}/train/behaviors.tsv",
        news_path=f"{args.root}/train/news.tsv",
        tokenizer=tok,
        max_len=args.max_len,
        sample=400,
        seed=args.seed,
        max_history=30,
        neg_ratio=2.0,
        hard_neg_ratio=0.5,
        use_abstract=True,
        use_chat_template=True,
        use_recency=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tok, label_pad_token_id=-100, padding=True, return_tensors="pt"
    )

    answer_len = answer_token_len(tok)
    trainer = PointwiseTrainer(
        model=model,
        train_dataset=None,
        args=transformers.TrainingArguments(
            output_dir="/tmp/grad_check",
            per_device_train_batch_size=args.batch_size,
            report_to=[],
            logging_steps=10 ** 9,
            save_strategy="no",
            use_cpu=False,
        ),
        answer_len=answer_len,
    )
    print(f"answer_len={answer_len} logits_to_keep={trainer._logits_to_keep} "
          f"supported={trainer._supports_logits_to_keep}")

    results = []
    for bi in range(args.n_batches):
        rows = [ds[i] for i in range(bi * args.batch_size, (bi + 1) * args.batch_size)]
        batch = collator(rows)
        print(f"\n--- batch {bi}: input_ids {tuple(batch['input_ids'].shape)} "
              f"labels {tuple(batch['labels'].shape)} "
              f"n_label_tokens={(batch['labels'] != -100).sum().item()}")

        # ---- PHASE 1: eval mode (dropout off) -> pure path difference ----
        model.eval()
        la, ga = grads_of(model, trainer, batch, True, device)
        lb, gb = grads_of(model, trainer, batch, False, device)
        la2, ga2 = grads_of(model, trainer, batch, True, device)
        p1_path = compare(ga, gb)
        p1_repeat = compare(ga, ga2)
        print(f"  [eval ] loss A={la:.10f} B={lb:.10f} A2={la2:.10f}")
        print(f"  [eval ] PATH  (A vs B): global_rel={p1_path['global_rel_diff']:.3e} "
              f"worst_tensor={p1_path['worst_tensor_rel_diff']:.3e} "
              f"({p1_path['worst_tensor']})")
        print(f"  [eval ] REPEAT(A vs A): global_rel={p1_repeat['global_rel_diff']:.3e}")

        # ---- PHASE 2: train mode (dropout on) -> noise floor ----
        model.train()
        la1, ga1 = grads_of(model, trainer, batch, True, device)
        la2, ga2 = grads_of(model, trainer, batch, True, device)
        lb1, gb1 = grads_of(model, trainer, batch, False, device)
        lb2, gb2 = grads_of(model, trainer, batch, False, device)
        noise_on = compare(ga1, ga2)
        noise_off = compare(gb1, gb2)
        path_diff = compare(ga1, gb1)
        print(f"  [train] loss A1={la1:.10f} A2={la2:.10f} B1={lb1:.10f} B2={lb2:.10f}")
        print(f"  [train] NOISE (A1 vs A2): global_rel={noise_on['global_rel_diff']:.3e}")
        print(f"  [train] NOISE (B1 vs B2): global_rel={noise_off['global_rel_diff']:.3e}")
        print(f"  [train] PATH  (A1 vs B1): global_rel={path_diff['global_rel_diff']:.3e}")

        results.append({
            "batch": bi,
            "eval_mode": {
                "loss_answer_only": la, "loss_full_logits": lb, "loss_answer_only_2": la2,
                "loss_abs_diff": abs(la - lb),
                "path": p1_path, "repeat": p1_repeat,
            },
            "train_mode": {
                "loss_answer_only_1": la1, "loss_answer_only_2": la2,
                "loss_full_logits_1": lb1, "loss_full_logits_2": lb2,
                "noise_answer_only": noise_on, "noise_full_logits": noise_off,
                "path": path_diff,
            },
        })

    # PHASE 1 decides: with dropout off, the two paths must agree to bf16
    # epsilon. PHASE 2 tells us whether the trainer's divergence is noise.
    p1_path_med = float(np.median([r["eval_mode"]["path"]["global_rel_diff"] for r in results]))
    p1_loss_med = float(np.median([r["eval_mode"]["loss_abs_diff"] for r in results]))
    p2_noise_med = float(np.median([
        max(r["train_mode"]["noise_answer_only"]["global_rel_diff"],
            r["train_mode"]["noise_full_logits"]["global_rel_diff"])
        for r in results]))
    p2_path_med = float(np.median([r["train_mode"]["path"]["global_rel_diff"] for r in results]))

    if p1_path_med < 1e-5:
        verdict = ("EQUIVALENT: with dropout off the two paths agree to bf16 epsilon; "
                   "the trainer's trajectory divergence is bf16 chaos, not a bug")
    elif p2_path_med <= max(3.0 * p2_noise_med, 1e-6):
        verdict = ("EQUIVALENT within the dropout noise floor "
                   "(path difference <= noise); divergence is chaos, not a bug")
    else:
        verdict = "DIFFERS: the path difference exceeds the noise floor -- real bug"

    summary = {
        "model": args.model,
        "batch_size": args.batch_size,
        "n_batches": args.n_batches,
        "median_eval_path_global_rel": p1_path_med,
        "median_eval_loss_abs_diff": p1_loss_med,
        "median_train_noise_global_rel": p2_noise_med,
        "median_train_path_global_rel": p2_path_med,
        "verdict": verdict,
        "batches": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SUMMARY =====")
    print(f"[eval ] path grad rel-diff (median): {p1_path_med:.3e}")
    print(f"[eval ] path loss abs-diff (median): {p1_loss_med:.3e}")
    print(f"[train] noise grad rel-diff (median): {p2_noise_med:.3e}")
    print(f"[train] path  grad rel-diff (median): {p2_path_med:.3e}")
    print(f"VERDICT: {verdict}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
