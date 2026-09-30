"""
Train APUS-OpenJev (JEV-4B) on MIND with a within-model distillation term.

This is Campaign-1 of the JEV training campaign. We fine-tune JEV-4B as a
point-wise decision model on MIND using the OpenJev **noul** A/B contract that
src/evaluate_mind_jev.py scores with, so the trained model's P(A)=P(yes) is
directly comparable to our dev AUC evals.

Distillation term (published OpenJev recipe shape, user-selected single-forward
teacher KL): the loss is

    L = 0.5 * CE(forward_1) + 0.5 * CE(forward_2) + beta * KL(P_2 || P_1)

over the A/B answer token, where forward_1/forward_2 are two stochastic passes
of the same JEV backbone under dropout (R-Drop). Because JEV is a stacked
linear-attention model with no dual-exit wiring in this Single-forward setting,
this is the correct non-degenerate within-model distillation; beta defaults to
0.1 to match the published KL coefficient. Set --kl_beta 0 (or --enable_kl False)
for a plain point-wise SFT baseline without distillation.

Usage (multi-node, from repo root):
    MODEL_PATH=... DATA_ROOT=... OUTPUT_DIR=... bash scripts/sft_jev.sh
or directly:
    deepspeed --hostfile=hostfile src/sft_mind_jev.py \\
        --base_model ... --train_behaviors_path ... --output_dir ... \\
        --deepspeed_config ds_configs/ds_config_zero2.json --kl_beta 0.1
"""

import os
import sys
import random
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
)
import fire

from jev_noul_data import JEVNoulPointwiseDataset

# Pre-initialize process group with a long timeout for slow datastore writes.
if "LOCAL_RANK" in os.environ and not dist.is_initialized():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))

# Add parent dir to path (repo root, where data.py lives).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class JEVWithinModelDistillTrainer(Trainer):
    """Trainer whose compute_loss does two stochastic forwards and adds a KL
    distillation term on the A/B answer logits: CE + beta * KL."""

    def __init__(self, kl_beta=0.1, enable_kl=True, **kwargs):
        super().__init__(**kwargs)
        self.kl_beta = kl_beta
        self.enable_kl = enable_kl

    def _answer_logits(self, model, inputs):
        """Single forward; return (answer_logits, answer_target) for last num_labels = 1 answer token."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # we compute CE ourselves on the single answer token
            return_dict=True,
        )
        logits = out.logits  # (B, L, V)

        nz = labels != -100
        last = nz.sum(dim=1) - 1  # index of the answer token (label) in each row
        # logits[i] predicts the token at position i+1 -> the answer at `last` is
        # predicted by logits at `last - 1`.
        predict_pos = last - 1
        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        ans_logits = logits[batch_idx, predict_pos]  # (B, V)
        ans_target = labels.gather(1, last.unsqueeze(1)).squeeze(1)  # (B,)
        return ans_logits, ans_target

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # `num_items_in_batch` is passed by transformers 5.x Trainer; accept and ignore.
        logits1, target = self._answer_logits(model, inputs)
        ce1 = F.cross_entropy(logits1, target)

        if self.enable_kl and self.kl_beta > 0:
            logits2, _ = self._answer_logits(model, inputs)
            ce2 = F.cross_entropy(logits2, target)
            p2 = torch.softmax(logits2, dim=-1)
            lp1 = F.log_softmax(logits1, dim=-1)
            lp2 = F.log_softmax(logits2, dim=-1)
            # Symmetric KL between the two stochastic passes (R-Drop).
            kl12 = (p2 * (lp2 - lp1)).sum(dim=-1).mean()
            p1 = torch.softmax(logits1, dim=-1)
            kl21 = (p1 * (lp1 - lp2)).sum(dim=-1).mean()
            kl = 0.5 * (kl12 + kl21)
            loss = 0.5 * ce1 + 0.5 * ce2 + self.kl_beta * kl
        else:
            loss = ce1

        return (loss, {"loss": loss}) if return_outputs else loss


def resolve_report_to(wandb_project):
    """WandB is unreachable from AzureML nodes; disable unless explicitly enabled."""
    if not wandb_project:
        return "none"
    if os.environ.get("MINIONEREC_ENABLE_WANDB", "") == "1":
        return "wandb"
    return "none"


def train(
    local_rank: int = -1,
    base_model: str = "",
    train_behaviors_path: str = "",
    train_news_path: str = "",
    eval_behaviors_path: str = "",
    eval_news_path: str = "",
    output_dir: str = "",
    use_abstract: bool = False,
    use_subcategory: bool = False,
    max_history: int = 0,
    neg_ratio: float = 1.0,
    hard_neg_ratio: float = 0.5,
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    cutoff_len: int = 2048,
    resume_from_checkpoint: str = None,
    train_from_scratch: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    deepspeed_config: str = "",
    early_stopping_patience: int = 3,
    use_chat_template: bool = None,
    kl_beta: float = 0.1,
    enable_kl: bool = True,
):
    """Train JEV-4B on MIND noul pointwise data with within-model distillation."""
    set_seed(seed)

    if use_chat_template is None:
        use_chat_template = "instruct" in base_model.lower() or "chat" in base_model.lower()

    if not base_model:
        raise ValueError("Please specify --base_model")
    if not resume_from_checkpoint:
        resume_from_checkpoint = None

    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    from pathlib import Path as _Path
    _local = os.path.isabs(base_model) or os.path.isdir(base_model)
    _base_model_arg = _Path(base_model) if _local else base_model
    _local_kwargs = {"local_files_only": True} if _local else {}

    if train_from_scratch:
        config = AutoConfig.from_pretrained(_base_model_arg, **_local_kwargs)
        model = AutoModelForCausalLM.from_config(config)
    else:
        _config = AutoConfig.from_pretrained(_base_model_arg, **_local_kwargs)
        _arch = " ".join(_config.architectures or [])
        model = AutoModelForCausalLM.from_pretrained(
            _base_model_arg,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            **_local_kwargs,
        )
        if "Qwen3_5" in _arch or "ForConditionalGeneration" in _arch:
            print(f"  JEV hybrid backbone detected ({_arch}): sdpa attention.")

    tokenizer = AutoTokenizer.from_pretrained(_base_model_arg, trust_remote_code=True, **_local_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    train_data = JEVNoulPointwiseDataset(
        behaviors_path=train_behaviors_path,
        news_path=train_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        max_history=max_history,
        neg_ratio=neg_ratio,
        hard_neg_ratio=hard_neg_ratio,
        use_abstract=use_abstract,
        use_subcategory=use_subcategory,
        use_chat_template=use_chat_template,
    )

    val_data = None
    if eval_behaviors_path and eval_news_path:
        val_data = JEVNoulPointwiseDataset(
            behaviors_path=eval_behaviors_path,
            news_path=eval_news_path,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=min(5000, len(train_data) // 10) if sample <= 0 else min(1000, sample // 10),
            seed=seed,
            max_history=max_history,
            neg_ratio=neg_ratio,
            hard_neg_ratio=hard_neg_ratio,
            use_abstract=use_abstract,
            use_subcategory=use_subcategory,
            use_chat_template=use_chat_template,
        )

    print(f"\nJEV-4B noul training:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data) if val_data else 0}")
    print(f"  Max history: {'unlimited' if max_history == 0 else max_history}")
    print(f"  Cutoff: {cutoff_len}, micro_batch: {micro_batch_size}")
    print(f"  Distill: kl_beta={kl_beta}, enable_kl={enable_kl}")
    print(f"  Format: OpenJev noul A/B; P(A)=P(yes)")

    training_args_dict = {
        "per_device_train_batch_size": micro_batch_size,
        "per_device_eval_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "warmup_steps": 100,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "bf16": True,
        "logging_steps": 10,
        "logging_first_step": True,
        "eval_strategy": "steps" if val_data else "no",
        "save_strategy": "steps",
        "eval_steps": 256 if val_data else None,
        "save_steps": 2048,
        "output_dir": output_dir,
        "save_total_limit": 2,
        "load_best_model_at_end": True if val_data else False,
        "ddp_find_unused_parameters": False if ddp else None,
        "report_to": resolve_report_to(wandb_project),
        "run_name": wandb_run_name if wandb_run_name else None,
        "metric_for_best_model": "eval_loss" if val_data else None,
        "greater_is_better": False,
        "disable_tqdm": False,
    }
    if deepspeed_config and os.path.exists(deepspeed_config):
        training_args_dict["deepspeed"] = deepspeed_config
        print(f"Using DeepSpeed config: {deepspeed_config}")

    training_args = transformers.TrainingArguments(**training_args_dict)

    trainer = JEVWithinModelDistillTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if val_data else None,
        kl_beta=kl_beta,
        enable_kl=enable_kl,
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_checkpoint_path = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)

    # JEV's tokenizer carries its chat template in chat_template.jinja rather
    # than inline (see evaluate_mind_jev.py fallback), so save_pretrained won't
    # copy it. Without it the eval scorer cannot apply the chat template on the
    # noul prompt. Copy it over (rank 0 only, like trainer.save_model).
    if not ddp or dist.get_rank() == 0:
        if _local:
            src_jinja = os.path.join(str(_base_model_arg), "chat_template.jinja")
            dst_jinja = os.path.join(final_checkpoint_path, "chat_template.jinja")
            if os.path.exists(src_jinja) and not os.path.exists(dst_jinja):
                import shutil
                shutil.copy(src_jinja, dst_jinja)
                print(f"  Copied chat_template.jinja to final_checkpoint")

    print(f"\n✓ JEV-4B noul training completed!")
    print(f"  Model saved to: {final_checkpoint_path}")
    print(f"  Evaluate with: python src/evaluate_mind_jev.py --model_path {final_checkpoint_path} "
          f"--behaviors_path .../dev/behaviors.tsv --news_path .../dev/news.tsv")


if __name__ == "__main__":
    fire.Fire(train)
