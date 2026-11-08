"""
Contrastive fine-tuning (DPO or RLCD) on top of pointwise Yes/No SFT for MIND.

Two objectives on the SAME data shape (within-impression clicked>unclicked
pairs, pointwise Yes/No prompts):

  --objective dpo   Direct Preference Optimization. The implicit reward is the
                    answer-boundary logit margin
                        r = logP(" Yes" | pos) - logP(" Yes" | neg)
                    (both members share the history; only the candidate
                    differs, so length bias is minimal). Reference model =
                    frozen SFT checkpoint. Loss:
                        -log sigmoid(beta * (r_pi - r_ref))

  --objective rlcd  RL from Contrastive Distillation (self-distill hinge, no
                    reference model):
                        D = logP(" Yes"_pos) - logP(" Yes"_neg)
                        L = max(0, margin - D)

Both read the Yes/No logits at the answer boundary (last real token position),
identical to evaluate_mind_pointwise.py's scoring, so training directly
optimizes the quantity the eval measures.

Usage (8xB200, repo root):
  deepspeed --num_gpus=8 src/sft_mind_pref_contrastive.py \
    --base_model output_dir/small_runs/recency/final_checkpoint \
    --objective dpo --beta 0.1 \
    --train_behaviors_path data/MIND_small/train/behaviors.tsv \
    --train_news_path data/MIND_small/train/news.tsv \
    --output_dir output_dir/small_runs/dpo_recency \
    --deepspeed_config ds_configs/ds_config_zero2_gpu.json
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
    PreTrainedModel,
    Trainer,
)
import fire

# Pre-init DDP with long timeout (mirrors sft_mind_pointwise_ds.py).
if "LOCAL_RANK" in os.environ and not dist.is_initialized():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))

# Secondary process groups (DeepSpeed ZeRO) are created via dist.new_group()
# WITHOUT a timeout, inheriting torch's 10-min NCCL default. A >10-min rank
# stall then trips the watchdog and kills the job (exit 122; see histfull
# post-mortem). Wrap new_group so every group gets the same 2h default.
if "LOCAL_RANK" in os.environ:
    _orig_new_group = dist.new_group

    def _new_group_default_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", datetime.timedelta(hours=2))
        return _orig_new_group(*args, **kwargs)

    dist.new_group = _new_group_default_timeout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind_pairwise_pref_data import MINDPairwisePrefDataset, PairwisePrefCollator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class ContrastivePrefTrainer(Trainer):
    """DPO / RLCD on the Yes/No answer-boundary logit margin.

    Batch layout (from PairwisePrefCollator): first B rows preferred, last B
    rows dispreferred. compute_loss runs the policy on all 2B rows; for DPO a
    frozen reference model (passed via ref_model) scores the same batch.
    """

    def __init__(self, objective="dpo", beta=0.1, margin=0.0, yes_token_id=None,
                 no_token_id=None, ref_model=None, **kwargs):
        super().__init__(**kwargs)
        # compute_loss returns a per-micro-batch mean, so Trainer must divide
        # it by the grad-accum steps. Qwen3's forward takes **kwargs, which
        # makes Trainer assume the loss is already token-normalised across
        # the accumulation window and skip that division (GA-x gradients).
        self.model_accepts_loss_kwargs = False
        self.objective = objective
        self.beta = beta
        self.margin = margin
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.ref_model = ref_model

    def _answer_yesno_logits(self, model, input_ids, attention_mask):
        """(B,L,V) -> (B,V) logits at the answer boundary.

        Rows are LEFT-padded and every row ends with the answer token
        (" Yes"/" No"), so the answer is always the last column L-1 and the
        logits predicting it sit at L-2 (standard causal shift). Indexing by
        real_len would be wrong here: with left padding the last real token
        is at L-1, not real_len-1.
        """
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            return_dict=True,
            logits_to_keep=2,  # only columns L-2, L-1; skips the (B,L,V) lm_head
        ).logits
        ans_logits = logits[:, -2, :]  # (B, V)
        return ans_logits

    def _margin(self, model, input_ids, attention_mask):
        """logP(" Yes") margin of preferred over dispreferred (shape (B,))."""
        ans_logits = self._answer_yesno_logits(model, input_ids, attention_mask)
        log_probs = F.log_softmax(ans_logits.float(), dim=-1)
        yes_lp = log_probs[:, self.yes_token_id]  # (B,)
        b = input_ids.shape[0] // 2
        s_pos = yes_lp[:b]
        s_neg = yes_lp[b:]
        return s_pos - s_neg

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        policy_margin = self._margin(model, input_ids, attention_mask)

        if self.objective == "dpo":
            with torch.no_grad():
                ref_margin = self._margin(self.ref_model, input_ids, attention_mask)
            logits = self.beta * (policy_margin - ref_margin)
            loss = -F.logsigmoid(logits).mean()
        elif self.objective == "rlcd":
            # Hinge on the BOUNDED probability margin (sigmoid of the logit
            # gap), not the raw log-prob gap: logP(" Yes") is unbounded
            # below, so a raw hinge lets the model trivially satisfy it by
            # pushing logP(" No") to -inf on the dispreferred side (observed:
            # margin collapsed to ~1e-7 within 100 steps while pair_acc
            # stayed at chance). sigmoid keeps the objective in (0,1) and
            # matches the eval's Yes-vs-No probability comparison.
            p_pos = torch.sigmoid(policy_margin)
            hinge = F.relu(self.margin - p_pos)
            loss = hinge.mean()
        else:
            raise ValueError(f"unknown objective {self.objective}")

        # Logging: fraction of pairs correctly ordered by the policy.
        acc = (policy_margin > 0).float().mean().detach()

        if return_outputs:
            return (loss, {'loss': loss, 'pair_acc': acc, 'margin': policy_margin.mean().detach()})
        self._log_extra = {'pair_acc': acc.item(), 'margin': policy_margin.mean().item()}
        return loss

    def log(self, logs, *args, **kwargs):
        extra = getattr(self, '_log_extra', None)
        if extra:
            logs.update(extra)
        return super().log(logs, *args, **kwargs)


def train(
    local_rank: int = -1,
    base_model: str = "",
    train_behaviors_path: str = "",
    train_news_path: str = "",
    output_dir: str = "",
    objective: str = "dpo",          # "dpo" | "rlcd"
    beta: float = 0.1,               # DPO beta (inverse temperature on margin)
    margin: float = 0.75,            # RLCD hinge margin (on sigmoid prob gap in (0,1))
    max_len: int = 2048,
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: float = 1.0,
    learning_rate: float = 5e-6,     # DPO-typical: much lower than SFT
    max_history: int = 30,
    neg_ratio: float = 1.0,
    hard_neg_ratio: float = 0.5,
    use_abstract: bool = True,
    use_subcategory: bool = False,
    use_chat_template: bool = True,
    use_recency: bool = False,
    use_profile_summary: bool = False,
    use_impression_timestamp: bool = False,
    use_entity: bool = False,
    history_selection: str = "recent",  # must match the SFT checkpoint's training
    history_format: str = "flat",       # must match the SFT checkpoint's training
    pairs_per_positive: int = 1,
    warmup_steps: int = 50,
    deepspeed_config: str = "",
    wandb_project: str = "",
    adapter_path: str = "",             # existing PEFT adapter to continue from
    use_lora: bool = False,             # add a NEW adapter on top of a full model
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
):
    """Contrastive fine-tune (DPO or RLCD) on top of a pointwise SFT checkpoint."""

    set_seed(seed)

    if not base_model:
        raise ValueError("Please specify --base_model (the SFT checkpoint)")

    gradient_accumulation_steps = batch_size // (micro_batch_size * 2)  # 2 fwd per pair
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != 1:
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

    # Load policy model (the SFT checkpoint)
    from pathlib import Path as _Path
    _local = os.path.isabs(base_model) or os.path.isdir(base_model)
    _base_model_arg = _Path(base_model) if _local else base_model
    _local_kwargs = {"local_files_only": True} if _local else {}

    # A PEFT adapter dir has no config.json -- only adapter_config.json with a
    # base_model_name_or_path field. AutoModelForCausalLM cannot load it, so
    # resolve the real base weights and load the adapter on top.
    _adapter_cfg = _base_model_arg / "adapter_config.json" if _local else None
    if _adapter_cfg is not None and _adapter_cfg.is_file() and \
            not (_base_model_arg / "config.json").is_file():
        import json as _json
        _base_name = _json.load(open(_adapter_cfg)).get("base_model_name_or_path")
        if not _base_name:
            raise ValueError(f"{base_model} is an adapter dir but its "
                             "adapter_config.json has no base_model_name_or_path")
        _base_model_arg = _Path(_base_name) if os.path.isabs(_base_name) else _Path(_base_name)
        print(f"{base_model} is a PEFT adapter; base weights = {_base_model_arg}")

    model = AutoModelForCausalLM.from_pretrained(
        _base_model_arg,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        **_local_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(_base_model_arg, trust_remote_code=True, **_local_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # PEFT handling. Two mutually exclusive modes:
    #
    #   adapter_path set -> CONTINUE an existing adapter. The policy is that
    #     adapter (trainable) and the DPO reference is a frozen copy of the
    #     SAME adapter, so the reference is exactly the SFT model the policy
    #     started from. This is the standard "DPO on top of a LoRA SFT" setup
    #     and is what the lg_9b run needs (its checkpoint is an adapter dir,
    #     not a full model).
    #
    #   use_lora set -> add a NEW adapter on top of a full base model.
    #
    # In both cases the adapter is MERGED into the base weights at save time,
    # so final_checkpoint is a plain CausalLM dir the eval pipeline consumes
    # unchanged.
    if adapter_path and use_lora:
        raise ValueError("use --adapter_path OR --use_lora, not both")

    if adapter_path:
        from peft import PeftModel
        _adapter_arg = _Path(adapter_path) if (os.path.isabs(adapter_path)
                                               or os.path.isdir(adapter_path)) else adapter_path
        model = PeftModel.from_pretrained(model, _adapter_arg, is_trainable=True)
        # Gradient checkpointing + frozen base: inputs to checkpointed blocks
        # no longer require grad, which silently zeroes the adapter gradients.
        model.enable_input_require_grads()
        model.print_trainable_parameters()
        print(f"Continuing adapter from: {adapter_path}")
    elif use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[m.strip() for m in
                            (lora_target_modules.split(",") if isinstance(lora_target_modules, str)
                             else lora_target_modules) if m.strip()],
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    # Reference model for DPO: frozen copy of the SFT checkpoint. Placed on
    # the same device as the policy (Trainer only moves `model`; a bare
    # ref_model stays on CPU and the first ref forward fails with a
    # device-mismatch error).
    ref_model = None
    if objective == "dpo":
        if adapter_path:
            from peft import PeftModel
            _adapter_arg = _Path(adapter_path) if (os.path.isabs(adapter_path)
                                                   or os.path.isdir(adapter_path)) else adapter_path
            ref_model = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(
                    _base_model_arg,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    **_local_kwargs,
                ),
                _adapter_arg,
                is_trainable=False,
            )
        else:
            ref_model = AutoModelForCausalLM.from_pretrained(
                _base_model_arg,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                **_local_kwargs,
            )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        if torch.cuda.is_available():
            ref_model.to(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

    # Yes/No token ids (same resolution as evaluate_mind_pointwise.py)
    yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode(" No", add_special_tokens=False)
    yes_token_id = yes_tokens[0] if len(yes_tokens) == 1 else yes_tokens[-1]
    no_token_id = no_tokens[0] if len(no_tokens) == 1 else no_tokens[-1]
    print(f"Yes token: {yes_token_id} ('{tokenizer.decode([yes_token_id])}'), "
          f"No token: {no_token_id} ('{tokenizer.decode([no_token_id])}')")

    train_data = MINDPairwisePrefDataset(
        behaviors_path=train_behaviors_path,
        news_path=train_news_path,
        tokenizer=tokenizer,
        max_len=max_len,
        sample=sample,
        seed=seed,
        max_history=max_history,
        neg_ratio=neg_ratio,
        hard_neg_ratio=hard_neg_ratio,
        use_abstract=use_abstract,
        use_subcategory=use_subcategory,
        use_chat_template=use_chat_template,
        use_recency=use_recency,
        use_profile_summary=use_profile_summary,
        use_impression_timestamp=use_impression_timestamp,
        use_entity=use_entity,
        history_selection=history_selection,
        history_format=history_format,
        pairs_per_positive=pairs_per_positive,
    )

    print(f"\nContrastive fine-tuning ({objective}):")
    print(f"  Base (SFT ckpt): {base_model}")
    print(f"  Pairs: {len(train_data)}")
    print(f"  beta={beta} margin={margin} lr={learning_rate} epochs={num_epochs}")
    print(f"  Effective batch (pairs): {micro_batch_size} x {gradient_accumulation_steps} x {world_size}")

    training_args_dict = {
        "per_device_train_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": os.environ.get("MINIONEREC_NO_GRADIENT_CHECKPOINTING", "") != "1",
        "dataloader_num_workers": int(os.environ.get("MINIONEREC_NUM_WORKERS", "0")),
        "warmup_steps": warmup_steps,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "bf16": True,
        "logging_steps": 10,
        "logging_first_step": True,
        "eval_strategy": "no",
        "save_strategy": "steps",
        "save_steps": int(os.environ.get("MINIONEREC_SAVE_STEPS", "2048")),
        "output_dir": output_dir,
        "save_total_limit": 2,
        "ddp_find_unused_parameters": False,
        "report_to": "none",
        "disable_tqdm": False,
        "remove_unused_columns": False,
    }

    if deepspeed_config and os.path.exists(deepspeed_config):
        training_args_dict["deepspeed"] = deepspeed_config
        print(f"Using DeepSpeed config: {deepspeed_config}")

    training_args = transformers.TrainingArguments(**training_args_dict)

    trainer = ContrastivePrefTrainer(
        objective=objective,
        beta=beta,
        margin=margin,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        ref_model=ref_model,
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=PairwisePrefCollator(tokenizer, pad_to_multiple_of=8),
    )

    model.config.use_cache = False

    trainer.train()

    final_checkpoint_path = os.path.join(output_dir, "final_checkpoint")
    if adapter_path or use_lora:
        # Merge adapters into the base weights so final_checkpoint is a plain
        # CausalLM directory loadable by the standard eval pipeline. Only rank 0
        # writes (all ranks writing the same dir on NFS corrupts shards). ZeRO-3
        # partitions params, so gather them before merging on that path.
        zero3 = trainer.is_deepspeed_enabled and \
            trainer.accelerator.state.deepspeed_plugin.zero_stage == 3
        if zero3:
            import deepspeed
            gather = deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=0)
        else:
            import contextlib
            gather = contextlib.nullcontext()
        with gather:
            if trainer.is_world_process_zero():
                merged = model.merge_and_unload()
                merged.save_pretrained(final_checkpoint_path)
                tokenizer.save_pretrained(final_checkpoint_path)
        trainer.accelerator.wait_for_everyone()
    else:
        trainer.save_model(final_checkpoint_path)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(final_checkpoint_path)
    print(f"  Model saved to: {final_checkpoint_path}")
    print(f"Next: TE_USE_RECENCY={1 if use_recency else 0} bash scripts/eval_mind_pointwise.sh {final_checkpoint_path} dev")


if __name__ == "__main__":
    # Reject unknown flags BEFORE training starts -- see src/cli_guard.py.
    from cli_guard import check_unknown_flags
    check_unknown_flags(train)
    fire.Fire(train)