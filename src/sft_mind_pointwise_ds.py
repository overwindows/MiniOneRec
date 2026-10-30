"""
Train a model on MIND dataset using Point-wise SFT with DeepSpeed support.

This script uses a point-wise approach where each (history, candidate) pair
is scored independently with a Yes/No classification.

Key differences from list-wise (sft_mind_ranking.py):
- Each candidate is evaluated independently
- Binary classification: "Is this article relevant? Yes/No"
- More training signal (every candidate gets a label)
- Shorter context per sample
- No position bias

Training format:
    Prompt: "User History: ... Candidate: [article] Is this relevant? Answer:"
    Target: " Yes" or " No"

Usage:
    deepspeed --hostfile=hostfile src/sft_mind_pointwise_ds.py \
        --base_model Qwen/Qwen3-1.7B \
        --train_behaviors_path ../data/MIND/train/behaviors.tsv \
        --train_news_path ../data/MIND/train/news.tsv \
        --eval_behaviors_path ../data/MIND/dev/behaviors.tsv \
        --eval_news_path ../data/MIND/dev/news.tsv \
        --output_dir output_dir/sft_mind_pointwise \
        --deepspeed_config ds_config_zero2.json
"""

import os
import sys
import random
import datetime
import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
import fire

# Pre-initialize process group with a 2-hour timeout before DeepSpeed does it.
# Default PyTorch pg_timeout is 30 min — too short for large checkpoint writes
# to slow network-mounted storage (e.g. AML datastore).
if "LOCAL_RANK" in os.environ and not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(hours=2),
    )

# Secondary process groups (e.g. DeepSpeed ZeRO's parameter-parallel groups)
# are created via dist.new_group() WITHOUT a timeout argument, so they inherit
# torch's 10-minute NCCL default — NOT the 2h timeout set above. A transient
# rank stall (NFS hiccup, slow dataloader worker) longer than 10 min then trips
# the watchdog on that group and kills the whole job (exit 122; seen on histfull
# @ step 3958: PG 1 ALLREDUCE Timeout(ms)=600000 while PG 0 was healthy).
# Wrap new_group so every group gets the same 2h default.
if "LOCAL_RANK" in os.environ:
    _orig_new_group = dist.new_group

    def _new_group_default_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", datetime.timedelta(hours=2))
        return _orig_new_group(*args, **kwargs)

    dist.new_group = _new_group_default_timeout

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import MINDPointwiseSFTDataset

def resolve_report_to(wandb_project):
    """Decide the HF `report_to` target for this run.

    WandB CANNOT be reached from AzureML/Singularity compute nodes: the node is
    network-isolated (`setup_multi_node.sh` logs "WandB login failed") and even
    with a valid key HF's WandbCallback -> wandb.init() crashes rank 0 with
    "TypeError: the JSON object must be str, bytes or bytearray, not NoneType"
    from wandb's query_with_timeout (server.py:39) reading a None flags field.
    Under DeepSpeed that kills rank 0, hangs the barrier, and the other ranks
    dump a misleading NCCL DistBackendError as a secondary symptom.

    So remote wandb is disabled by default (report_to="none"); checkpoints and
    eval still write to disk, so training results are unaffected. Set
    MINIONEREC_ENABLE_WANDB=1 to force real logging (e.g. on a node with
    outbound access).
    """
    if not wandb_project:
        return "none"
    if os.environ.get("MINIONEREC_ENABLE_WANDB", "") == "1":
        return "wandb"
    return "none"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    local_rank: int = -1,  # injected by DeepSpeed launcher; ignored here
    base_model: str = "",
    train_behaviors_path: str = "",
    train_news_path: str = "",
    eval_behaviors_path: str = "",
    eval_news_path: str = "",
    output_dir: str = "",
    use_abstract: bool = False,
    use_subcategory: bool = False,
    max_history: int = 0,  # 0 = no limit
    neg_ratio: float = 1.0,  # Negatives per positive
    hard_neg_ratio: float = 0.5,  # Fraction of negatives from same category
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,  # Shorter than list-wise since single candidate
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    train_from_scratch: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_run_id: str = "",
    deepspeed_config: str = "",
    early_stopping_patience: int = 3,
    use_chat_template: bool = None,  # Auto-detect if None
    use_recency: bool = False,  # Mark 5 most recent history items with "(recent)"
    use_profile_summary: bool = False,  # Prepend top-3 category interest summary
    use_impression_timestamp: bool = False,  # Add day/time-of-day context from impression timestamp
    use_entity: bool = False,  # Append Wikidata entity labels ("Topics: ...") to article text
    history_selection: str = "recent",  # "recent" | "relevant" | "dedup_backfill" | "dedup_only" | "catcap"
    history_format: str = "flat",  # "flat" | "grouped_profile" | "two_block" | "title_only"
    use_lora: bool = False,  # LoRA fine-tuning (for backbones too large for full FT)
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",  # comma-separated; fire may pass a tuple
    optimizer: str = "adamw",  # "adamw" | "muon" (Muon on hidden 2D matrices, AdamW on the rest)
    muon_lr: float = 0.02,    # Muon lr is in spectral-norm units, NOT comparable to AdamW lr
):
    """Train with point-wise SFT format (Yes/No classification) using DeepSpeed"""

    set_seed(seed)

    # Auto-detect if model is instruct variant (if use_chat_template not explicitly set)
    if use_chat_template is None:
        use_chat_template = "instruct" in base_model.lower() or "chat" in base_model.lower()
        if use_chat_template:
            print(f"Auto-detected instruct model: will use chat template")
        else:
            print(f"Using raw text format (no chat template)")

    # Resume existing WandB run if run_id is provided
    if wandb_run_id:
        os.environ['WANDB_RUN_ID'] = wandb_run_id
        os.environ['WANDB_RESUME'] = 'allow'
        print(f"Resuming WandB run: {wandb_run_id}")

    if not base_model:
        raise ValueError("Please specify --base_model")

    # Normalize empty string to None so HF Trainer doesn't try to load a nonexistent path
    if not resume_from_checkpoint:
        resume_from_checkpoint = None

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Load model — treat absolute paths and existing directories as local
    from pathlib import Path as _Path
    _local = os.path.isabs(base_model) or os.path.isdir(base_model)
    _base_model_arg = _Path(base_model) if _local else base_model
    _local_kwargs = {"local_files_only": True} if _local else {}

    # Attention implementation: use sdpa (PyTorch-native) across the board.
    # flash_attention_2 is nominally faster for dense causal LMs, but it is an
    # external wheel whose ABI can mismatch the node's torch (undefined symbol
    # `_ZN3c105ErrorC2...` at import) — seen node-variant on AzureML where
    # setup_multi_node.sh may install torch==2.6.0 vs a flash_attn built for a
    # different torch. sdpa has no such external dependency, dispatches both
    # dense and the hybrid Qwen3.5 GatedDeltaNet backbone (which FA2 cannot
    # handle anyway), so it is the robust default. Qwen3.5 checkpoints declare
    # multimodal architectures but AutoModelForCausalLM loads the pure-text
    # Qwen3_5ForCausalLM, which automatically ignores the vision tower weights.
    if train_from_scratch:
        config = AutoConfig.from_pretrained(_base_model_arg, **_local_kwargs)
        model = AutoModelForCausalLM.from_config(config)
    else:
        _config = AutoConfig.from_pretrained(_base_model_arg, **_local_kwargs)
        _arch = " ".join(_config.architectures or [])
        _hybrid_backbone = "ForConditionalGeneration" in _arch or "Qwen3_5" in _arch
        model = AutoModelForCausalLM.from_pretrained(
            _base_model_arg,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            **_local_kwargs,
        )
        if _hybrid_backbone:
            print(f"  Detected hybrid backbones ({_arch}): using sdpa attention (vision tower not loaded)")

    tokenizer = AutoTokenizer.from_pretrained(_base_model_arg, trust_remote_code=True, **_local_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # LoRA: wrap the loaded model with PEFT adapters. The base weights stay
    # frozen; only adapters train. trainer.save_model() on a PeftModel writes
    # adapter weights only (small); for eval the adapter must be merged or the
    # eval script must load base + adapter — we MERGE at save time so the
    # final_checkpoint is a plain CausalLM dir the existing eval pipeline can
    # consume unchanged.
    if use_lora:
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
        # Gradient checkpointing + frozen base: inputs to checkpointed blocks
        # no longer require grad, which silently zeroes the adapter gradients.
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    # Load point-wise datasets
    train_data = MINDPointwiseSFTDataset(
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
        use_recency=use_recency,
        use_profile_summary=use_profile_summary,
        use_impression_timestamp=use_impression_timestamp,
        use_entity=use_entity,
        history_selection=history_selection,
        history_format=history_format,
    )

    val_data = MINDPointwiseSFTDataset(
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
        use_recency=use_recency,
        use_profile_summary=use_profile_summary,
        use_impression_timestamp=use_impression_timestamp,
        use_entity=use_entity,
        history_selection=history_selection,
        history_format=history_format,
    )

    print(f"\nTraining with Point-wise SFT:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Max history: {'unlimited' if max_history == 0 else max_history}")
    print(f"  Neg ratio: {neg_ratio}")
    print(f"  Cutoff length: {cutoff_len}")
    print(f"  Format: Yes/No classification")
    print(f"  Chat template: {'enabled' if use_chat_template else 'disabled (raw text)'}")

    # Prepare training arguments with optional DeepSpeed
    training_args_dict = {
        "per_device_train_batch_size": micro_batch_size,
        "per_device_eval_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        # Long sequences (cutoff_len up to 8192) with micro_batch 4 produce very
        # large activation tensors; without checkpointing the forward pass OOMs
        # an 80GB A100 during the loss logits materialization (seen as CUDA OOM
        # in fixed_cross_entropy on the first step). Checkpointing trades a
        # little compute for a large drop in activation memory.
        # On large-VRAM GPUs (B200 180GB) checkpointing is a pure ~2x slowdown:
        # disable via MINIONEREC_NO_GRADIENT_CHECKPOINTING=1.
        "gradient_checkpointing": os.environ.get("MINIONEREC_NO_GRADIENT_CHECKPOINTING", "") != "1",
        "dataloader_num_workers": int(os.environ.get("MINIONEREC_NUM_WORKERS", "0")),
        "warmup_steps": 100,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "bf16": True,
        "logging_steps": 10,
        "logging_first_step": True,
        "eval_strategy": "steps" if val_data else "no",
        "save_strategy": "steps",
        "eval_steps": int(os.environ.get("MINIONEREC_EVAL_STEPS", "256")) if val_data else None,
        "save_steps": int(os.environ.get("MINIONEREC_SAVE_STEPS", "2048")),
        "output_dir": output_dir,
        "save_total_limit": 2,
        "load_best_model_at_end": True if val_data else False,
        "ddp_find_unused_parameters": False if ddp else None,
        # NOTE: `group_by_length` was removed from TrainingArguments in
        # transformers 5.x (it was deprecated in 4.5x). The repo default is
        # False (no length grouping), which is transformers 5.x default too,
        # so dropping it is behaviour-neutral here.
        "report_to": resolve_report_to(wandb_project),
        "run_name": wandb_run_name if wandb_run_name else None,
        "metric_for_best_model": "eval_loss" if val_data else None,
        "greater_is_better": False,
        "disable_tqdm": False,
    }

    # Add DeepSpeed config if provided
    if deepspeed_config and os.path.exists(deepspeed_config):
        training_args_dict["deepspeed"] = deepspeed_config
        print(f"Using DeepSpeed config: {deepspeed_config}")

    training_args = transformers.TrainingArguments(**training_args_dict)

    # Muon optimizer (opt-in): orthogonalized momentum updates on hidden 2D
    # weight matrices, AdamW on embeddings / lm_head / scalars (Muon's own
    # guidance — it is not designed for those). SingleDeviceMuonWithAuxAdam
    # keeps one optimizer object so the HF Trainer scheduler and DeepSpeed
    # integration work unchanged. Note Muon lr is in spectral-norm units;
    # 0.02-0.05 is the usual range regardless of model size.
    optimizer_obj = None
    if optimizer == "muon":
        if training_args_dict.get("deepspeed"):
            raise ValueError(
                "optimizer=muon is incompatible with DeepSpeed ZeRO: ZeRO hands the "
                "optimizer flat 1-D fp32 partitions, so Muon's Newton-Schulz step never "
                "sees the 2-D weight matrices (zeropower_via_newtonschulz5 asserts on the "
                "first step). Launch with torchrun and no --deepspeed_config to use Muon."
            )
        from muon import SingleDeviceMuonWithAuxAdam
        # NOTE: membership must be id-based — `p not in list` invokes tensor
        # __eq__ (elementwise), which crashes on shape mismatch.
        _hidden_ids = {
            id(p) for n, p in model.named_parameters()
            if p.ndim >= 2 and p.requires_grad
            and "embed" not in n and "lm_head" not in n
        }
        hidden_matrix_params = [
            p for _, p in model.named_parameters() if id(p) in _hidden_ids
        ]
        other_params = [
            p for _, p in model.named_parameters()
            if p.requires_grad and id(p) not in _hidden_ids
        ]
        param_groups = [
            dict(params=other_params, lr=learning_rate, betas=(0.9, 0.95),
                 eps=1e-10, weight_decay=0.0, use_muon=False),
            dict(params=hidden_matrix_params, lr=muon_lr, momentum=0.95,
                 weight_decay=0.0, use_muon=True),
        ]
        optimizer_obj = SingleDeviceMuonWithAuxAdam(param_groups)
        print(f"Muon optimizer: {len(hidden_matrix_params)} hidden matrix params, "
              f"{len(other_params)} aux params (AdamW lr={learning_rate}, Muon lr={muon_lr})")

    # Initialize trainer
    # Early stopping is gated on completing at least one full epoch: eval-loss
    # often dips and recovers within epoch 1 (LR warmup + data-order effects),
    # and stopping at epoch 0.3 wastes the run. The wrapper below defers the
    # HF EarlyStoppingCallback until trainer.state.epoch >= 1.0.
    class _EpochGatedEarlyStopping(transformers.TrainerCallback):
        def __init__(self, patience, min_epoch=1.0):
            self.inner = EarlyStoppingCallback(early_stopping_patience=patience)
            self.min_epoch = min_epoch
        def on_step_end(self, args, state, control, **kwargs):
            if state.epoch is not None and state.epoch >= self.min_epoch:
                return self.inner.on_step_end(args, state, control, **kwargs)
            return control
        def on_train_begin(self, args, state, control, **kwargs):
            return self.inner.on_train_begin(args, state, control, **kwargs)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None,
        args=training_args,
        **({"optimizers": (optimizer_obj, None)} if optimizer_obj is not None else {}),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[_EpochGatedEarlyStopping(early_stopping_patience)] if val_data else None,
    )

    # Length-grouped training batches (opt-in via MINIONEREC_LENGTH_GROUPED=1).
    # transformers 5.x removed `group_by_length`; without it every micro-batch pads
    # to its longest member (~2048) while the median prompt is ~1250 tokens — ~40%
    # wasted compute. This sampler restores the old LengthGroupedSampler behavior:
    # random megabatches sorted by length, so batches are homogeneous.
    if os.environ.get("MINIONEREC_LENGTH_GROUPED", "") == "1":
        from torch.utils.data import DataLoader

        def _length_grouped_dataloader(ds, batch_size, collate_fn, num_workers, seed, world_size, rank):
            # Token lengths are not precomputed (samples store raw history/candidate
            # dicts; tokenization happens in __getitem__), so estimate length from
            # the raw text: chars/4 is a robust token proxy for English news text.
            def _est_len(s):
                cand = s.get("candidate", {}) or {}
                hist = s.get("history", []) or []
                txt = cand.get("text", "") + " ".join(h.get("text", "") for h in hist)
                return len(txt) // 4
            try:
                lengths = [_est_len(s) for s in ds.samples]
            except Exception:
                return None
            import random as _random
            rng = _random.Random(seed)
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            # megabatches of 50*batch_size sorted internally by length
            mega = max(50 * batch_size, batch_size * world_size)
            grouped = []
            for i in range(0, len(indices), mega):
                chunk = sorted(indices[i:i + mega], key=lambda j: lengths[j])
                grouped.extend(chunk)
            # Contiguous micro-batches of the length-sorted order
            batches = [grouped[i:i + batch_size] for i in range(0, len(grouped), batch_size)]

            # CRITICAL: shard at the BATCH level across DDP ranks. The Trainer's
            # default dataloader installs a DistributedSampler; this custom one
            # must do its own sharding, otherwise every rank trains on the FULL
            # dataset (8x duplicated compute, 8x too many optimizer steps).
            class _GroupedBatchSampler(torch.utils.data.Sampler):
                def __init__(self, batches, world_size, rank, seed):
                    self.batches = batches
                    self.world_size = world_size
                    self.rank = rank
                    self.seed = seed
                    self.epoch = 0
                def set_epoch(self, epoch):
                    self.epoch = epoch
                def __iter__(self):
                    # reshuffle batch order each epoch (different batch composition)
                    rng2 = _random.Random(self.seed + 1000 * self.epoch)
                    self.epoch += 1
                    order = list(range(len(self.batches)))
                    rng2.shuffle(order)
                    # trim to an equal shard per rank so all ranks iterate the
                    # same number of batches (avoids DDP/DeepSpeed barrier hang)
                    n = (len(order) // self.world_size) * self.world_size
                    for b in order[:n][self.rank::self.world_size]:
                        yield self.batches[b]
                def __len__(self):
                    return len(self.batches) // self.world_size

            class _GroupedDS(torch.utils.data.Dataset):
                def __init__(self, ds):
                    self.ds = ds
                def __len__(self):
                    return len(self.ds)
                def __getitem__(self, i):
                    return self.ds[i]

            return DataLoader(
                _GroupedDS(ds),
                batch_sampler=_GroupedBatchSampler(batches, world_size, rank, seed),
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
            )

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        _dl = _length_grouped_dataloader(
            train_data,
            training_args.per_device_train_batch_size,
            trainer.data_collator,
            training_args.dataloader_num_workers,
            seed,
            world_size,
            rank,
        )
        if _dl is not None:
            trainer.get_train_dataloader = lambda: _dl
            print(f"Length-grouped train dataloader enabled (world_size={world_size}, rank={rank})")

    model.config.use_cache = False

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model (use trainer.save_model for proper DeepSpeed ZeRO handling)
    final_checkpoint_path = os.path.join(output_dir, "final_checkpoint")
    if use_lora:
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

    print(f"\n✓ Point-wise SFT training completed!")
    print(f"  Model saved to: {final_checkpoint_path}")
    print(f"\nTo evaluate:")
    print(f"  bash scripts/eval_mind_pointwise.sh {output_dir}/final_checkpoint dev")


if __name__ == "__main__":
    fire.Fire(train)
