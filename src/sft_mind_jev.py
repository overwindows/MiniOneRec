"""
Train APUS-OpenJev on MIND with an explicit full-depth objective and prompt mode.

Historical defaults retain full-vocabulary answer-token CE and the legacy prompt.
The decision ablation uses BCEWithLogits(z_A-z_B, click_label), reading only the
last prompt position. native_click uses canonical Noul true/false criteria.
This is not native two-exit training: published JEV has trained low/high exits,
but cross-depth distillation is not implemented here. The earlier R-Drop term
must not be confused with native deep-to-shallow distillation.

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
import warnings
import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
)
import fire

from jev_noul_data import JEVNoulPointwiseDataset
from jev_decision import LOSS_TYPES, PROMPT_CONTRACTS, candidate_decision_loss

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
    """Full-depth legacy CE or candidate-only decision loss; no KL term."""

    def __init__(self, kl_beta=0.1, enable_kl=True, loss_type="full_vocab",
                 decision_token_ids=None, **kwargs):
        if loss_type not in LOSS_TYPES:
            raise ValueError(f"Unknown JEV loss type: {loss_type}")
        if loss_type == "decision" and (enable_kl or decision_token_ids is None):
            raise ValueError("Decision loss requires A/B IDs and enable_kl=False")
        super().__init__(**kwargs)
        self.kl_beta = kl_beta
        self.enable_kl = enable_kl
        self.loss_type = loss_type
        self.decision_token_ids = decision_token_ids
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.loss_type == "decision":
            loss, outputs = candidate_decision_loss(model, inputs, self.decision_token_ids)
        else:
            outputs = model(**inputs, return_dict=True)
            loss = outputs.loss

        # NaN guard: halt with a clear message instead of silently feeding a NaN
        # loss through optimization (where it poisons the whole run and surfaces
        # later as a spurious EarlyStopping trigger via eval_loss=NaN).
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "JEV SFT loss is NaN/Inf. This indicates unstable training "
                "(bf16 overflow or a bad answer-token index), NOT a normal "
                "high-CE value. Check answer token ids / mixed precision BEFORE "
                "spending GPU on a doomed run."
            )

        return (loss, outputs) if return_outputs else loss


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
    loss_type: str = "full_vocab",
    prompt_contract: str = "legacy",
):
    """Train a full-depth JEV decision model; historical defaults are preserved."""
    if loss_type not in LOSS_TYPES or prompt_contract not in PROMPT_CONTRACTS:
        raise ValueError("Unsupported JEV loss or prompt contract")
    if loss_type == "decision" and enable_kl:
        raise ValueError("Use enable_kl=False for the candidate-only decision ablation")
    if enable_kl:
        warnings.warn("Legacy enable_kl is ignored: this trainer applies no KL term.")
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
        # JEV is a hybrid ForConditionalGeneration backbone, but AutoModelForCausalLM
        # resolves it to Qwen3_5ForCausalLM and ignores the vision tower — the same
        # proven path sft_mind_pointwise_ds.py uses to train these Qwen3.5 hybrid
        # backbones to 0.70 AUC in this exact env.
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
    model.config.jev_loss_type = loss_type
    model.config.jev_prompt_contract = prompt_contract

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
        prompt_contract=prompt_contract,
    )

    val_data = None
    if eval_behaviors_path and eval_news_path:
        # In-band eval dataset is intentionally not wired in: dev AUC is produced
        # by the external noul scorer (evaluate_mind_jev.py) after training, which
        # is the only signal that matters. keep the block so the CLI args stay valid.
        pass

    print(f"\nJEV-4B noul training:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Max history: {'unlimited' if max_history == 0 else max_history}")
    print(f"  Cutoff: {cutoff_len}, micro_batch: {micro_batch_size}")
    print(f"  Loss: {loss_type}; prompt contract: {prompt_contract}; KL: disabled")

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
        # NO in-band HF eval. Dev AUC is produced by the EXTERNAL noul scorer
        # (src/evaluate_mind_jev.py via run_jev_eval.py) after training, which is
        # the only signal that matters. The in-band HF eval on this stacked
        # linear-attention backbone emitted eval_loss=NaN from the very first
        # eval, and metric_for_best_model=eval_loss + EarlyStoppingCallback then
        # killed every C1 run at ~0.14ep before it could learn (see EXPERIMENT_PLAN
        # C1fix row). Full-vocab next-token CE over history is legitimately O(50-150);
        # reading that as divergence was a red herring — the NaN eval was the killer.
        "eval_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 2048,
        "output_dir": output_dir,
        "save_total_limit": 2,
        "load_best_model_at_end": False,
        "ddp_find_unused_parameters": False if ddp else None,
        "report_to": resolve_report_to(wandb_project),
        "run_name": wandb_run_name if wandb_run_name else None,
        "disable_tqdm": False,
    }
    if deepspeed_config and os.path.exists(deepspeed_config):
        training_args_dict["deepspeed"] = deepspeed_config
        print(f"Using DeepSpeed config: {deepspeed_config}")

    training_args = transformers.TrainingArguments(**training_args_dict)

    trainer = JEVWithinModelDistillTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        kl_beta=kl_beta,
        enable_kl=enable_kl,
        loss_type=loss_type,
        decision_token_ids=train_data.decision_token_ids,
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
