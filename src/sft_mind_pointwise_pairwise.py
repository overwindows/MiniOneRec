"""
Train a model on MIND dataset using a HYBRID point-wise + pairwise objective
(Bradley-Terry) with DeepSpeed support.

This is a fork of `sft_mind_pointwise_ds.py`. It keeps the standard point-wise
CE over the Yes/No answer token, and ADDS a pairwise Bradley-Terry term so the
model also learns to score clicked candidates above non-clicked ones within the
same impression (A7.3, EXPERIMENT_PLAN.md):

    L_BT = -log(sigmoid(score_pos - score_neg))
    score(candidate) = logit(" Yes") - logit(" No") at the final PROMPT position

Total loss = L_CE + pairwise_loss_weight * L_BT

Rationale (arXiv:2312.16018, GLIMPSE 2409.17711): purely point-wise CE ignores
the ranking structure within an impression. Pairing positives against negatives
sharpens the relative score gap, which is what the AUC metric actually measures.
This objective implementation does not establish a benchmark gain.

Implementation notes:
- Uses `MINDPointwiseSFTDataset(return_meta=True)` so each sample carries its
  `impression_id` and `label`, letting the trainer pair pos/neg candidates from
  the same impression.
- The one-token target is removed before the ranking forward pass. Full-vocab
  CE and the yes-minus-no margin use the same final prompt logits.
- All positive/negative pairs PRESENT IN A BATCH are averaged per impression.
  No cross-impression fallback is used. The legacy shuffled/sampled dataset
  is NOT a qualified full-impression/all-candidate training path.
- Opt-in `full_impression_training=True` retains complete source impressions,
  computes all pairs across checkpointed candidate chunks, and verifies every
  fixed epoch before final saving. It requires micro_batch_size=1, no dev
  paths/sampling, and single-process native training; DDP/resume are not qualified.

Usage:
    deepspeed --hostfile=hostfile src/sft_mind_pointwise_pairwise.py \
        --base_model Qwen/Qwen3-1.7B \
        --train_behaviors_path ../data/MIND/train/behaviors.tsv \
        --train_news_path ../data/MIND/train/news.tsv \
        --eval_behaviors_path ../data/MIND/dev/behaviors.tsv \
        --eval_news_path ../data/MIND/dev/news.tsv \
        --output_dir output_dir/sft_mind_pointwise_pairwise \
        --deepspeed_config ds_config_zero2.json \
        --pairwise_loss_weight 0.5
"""

import os
import sys
import random
import datetime
import math
import warnings
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import MINDPointwiseSFTDataset
from jev_decision import candidate_decision_loss
from mind_pairwise_loss import complete_impression_loss, validate_pairwise_metadata, within_impression_bt
from mind_impression_data import MINDImpressionSFTDataset
from mind_impression_training import CompleteEpochCoverage, CompleteImpressionCollator


PAIRWISE_OBJECTIVE = "causal_margin_bt_v2"
COMPLETE_OBJECTIVE = "complete_impression_causal_margin_bt_v1"


def resolve_report_to(wandb_project):
    """Same policy as sft_mind_pointwise_ds.py; wandb is unreachable on AML nodes."""
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


class _Collator:
    """DataCollator for the pairwise trainer.

    Pads a batch like DataCollatorForSeq2Seq, but ALSO keeps the per-sample
    meta fields (impression_id, label) so `compute_loss` can build pos/neg
    pairs from the same impression.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch_size = len(features)
        # Left-pad input_ids/attention_mask/labels to the batch max length.
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[i, max_len - n:] = f["input_ids"]
            attention_mask[i, max_len - n:] = f["attention_mask"]
            labels[i, max_len - n:] = f["labels"]

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "impression_id" in features[0]:
            out["impression_id"] = [f["impression_id"] for f in features]
            out["label"] = torch.tensor(
                [f["label"] for f in features], dtype=torch.long)
        return out


class _PairwiseTrainer(transformers.Trainer):
    """Full-vocab CE + within-impression BT on target-free decision margins."""

    def __init__(self, yes_token_id, no_token_id, pairwise_loss_weight=0.5, **kwargs):
        if not math.isfinite(pairwise_loss_weight) or pairwise_loss_weight < 0:
            raise ValueError("Pairwise loss weight must be finite and nonnegative")
        self._yes_id = yes_token_id
        self._no_id = no_token_id
        self._pw = pairwise_loss_weight
        self._warned_missing_pairs = False
        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = {key: value for key, value in inputs.items()
                        if key not in ("impression_id", "label")}
        if self._pw == 0:
            outputs = model(**model_inputs)
            loss = outputs.loss
        else:
            if "impression_id" not in inputs or "label" not in inputs:
                raise ValueError("Pairwise loss requires impression_id and binary label metadata")
            labels = model_inputs["labels"]
            if not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.shape[1] < 2:
                raise ValueError("Pairwise labels must include a nonempty prompt and final answer")
            cls = inputs["label"]
            impressions = inputs["impression_id"]
            self._validate_metadata(cls, impressions, labels.shape[0])
            if not torch.all(labels[:, -1] == torch.where(
                    cls.to(labels.device) == 1, self._yes_id, self._no_id)):
                raise ValueError("Pairwise metadata must match the supervised Yes/No answer")
            loss, outputs = candidate_decision_loss(
                model, model_inputs, (self._yes_id, self._no_id), loss_type="full_vocab")
            scores = outputs["logits"][:, 0] - outputs["logits"][:, 1]
            bt = self._pairwise_term(scores, cls, impressions)
            if bt is None:
                if not self._warned_missing_pairs:
                    warnings.warn(
                        "No within-impression positive/negative pairs in this batch; CE only. "
                        "Shuffled candidate batches are not full-impression ranking training.",
                        RuntimeWarning, stacklevel=2)
                    self._warned_missing_pairs = True
            else:
                loss = loss + self._pw * bt
            outputs["loss"] = loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Pairwise training loss contains NaN/Inf")
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _validate_metadata(cls, impressions, batch_size):
        validate_pairwise_metadata(cls, impressions, batch_size)

    def _pairwise_term(self, scores, cls, impressions):
        """Average all present positive/negative pairs equally per impression."""
        return within_impression_bt(scores, cls, impressions)


class CompleteImpressionTrainer(_PairwiseTrainer):
    def __init__(self, *, expected_source, candidate_chunk_size, **kwargs):
        args = kwargs["args"]
        if (args.world_size != 1 or args.per_device_train_batch_size != 1
                or args.dataloader_drop_last or args.remove_unused_columns or args.max_steps > 0
                or args.num_train_epochs < 1 or not float(args.num_train_epochs).is_integer()
                or kwargs.get("eval_dataset") is not None or args.deepspeed):
            raise ValueError(
                "Complete-source mode requires single-process fixed epochs, one impression per "
                "batch, no dropping, no step cutoff/dev evaluation or unqualified DeepSpeed")
        if (not isinstance(candidate_chunk_size, int) or isinstance(candidate_chunk_size, bool)
                or candidate_chunk_size < 1):
            raise ValueError("Candidate chunk size must be a positive integer")
        self.expected_source = dict(expected_source)
        self.candidate_chunk_size = candidate_chunk_size
        self.epoch_coverage = []
        self._active_epoch = False
        super().__init__(**kwargs)
        self.add_callback(CompleteEpochCoverage(self))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self._active_epoch:
            validate_pairwise_metadata(
                inputs["label"], inputs["impression_id"], inputs["input_ids"].shape[0])
            if not inputs["impression_id"]:
                raise ValueError("A complete-source training impression cannot be empty")
            identifier = inputs["impression_id"][0]
            if identifier in self._consumed_ids:
                raise ValueError(f"Impression consumed more than once in this epoch: {identifier}")
        loss, outputs = complete_impression_loss(
            model, inputs, (self._yes_id, self._no_id),
            chunk_size=self.candidate_chunk_size, pairwise_weight=self._pw)
        if self._active_epoch:
            self._consumed_ids.add(identifier)
            self._exposures += inputs["expected_candidates"]
            self._positives += int(inputs["label"].sum().item())
        return (loss, outputs) if return_outputs else loss


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
    cutoff_len: int = 2048,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    train_from_scratch: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_run_id: str = "",
    deepspeed_config: str = "",
    early_stopping_patience: int = 3,
    use_chat_template: bool = None,  # Auto-detect if None
    use_recency: bool = False,
    use_profile_summary: bool = False,
    use_impression_timestamp: bool = False,
    pairwise_loss_weight: float = 0.5,  # lambda for the Bradley-Terry term (A7.3)
    full_impression_training: bool = False,
    candidate_chunk_size: int = 2,
):
    """Train with hybrid point-wise CE + pairwise Bradley-Terry objective."""
    if not math.isfinite(pairwise_loss_weight) or pairwise_loss_weight < 0:
        raise ValueError("Pairwise loss weight must be finite and nonnegative")
    if full_impression_training:
        if (int(os.environ.get("WORLD_SIZE", 1)) != 1 or micro_batch_size != 1
                or deepspeed_config or resume_from_checkpoint or group_by_length
                or sample > 0 or eval_behaviors_path or eval_news_path
                or neg_ratio != 1.0 or hard_neg_ratio != 0.5):
            raise ValueError(
                "Complete-source mode requires one process/impression per batch, fresh fixed "
                "epochs, no dev paths/sampling/ratio overrides or unqualified DeepSpeed/resume")
        if (not isinstance(candidate_chunk_size, int) or isinstance(candidate_chunk_size, bool)
                or candidate_chunk_size < 1 or num_epochs < 1
                or not isinstance(num_epochs, int) or isinstance(num_epochs, bool)
                or not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1):
            raise ValueError("Complete-source epochs, batch size and chunk size must be positive integers")
    if not output_dir:
        raise ValueError("Please specify a new --output_dir")
    if os.path.exists(os.path.join(output_dir, "final_checkpoint")):
        raise ValueError("Refusing to overwrite an existing final checkpoint; use a new output_dir")
    if os.path.isdir(output_dir) and os.listdir(output_dir) and not resume_from_checkpoint:
        raise ValueError("Use a new empty output_dir; do not reuse prior trial artifacts")
    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    set_seed(seed)

    if use_chat_template is None:
        use_chat_template = "instruct" in base_model.lower() or "chat" in base_model.lower()

    if wandb_run_id:
        os.environ['WANDB_RUN_ID'] = wandb_run_id
        os.environ['WANDB_RESUME'] = 'allow'
        print(f"Resuming WandB run: {wandb_run_id}")

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

    tokenizer = AutoTokenizer.from_pretrained(_base_model_arg, trust_remote_code=True, **_local_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    answer_tokens = [tokenizer.encode(answer, add_special_tokens=False) for answer in (" Yes", " No")]
    if (any(len(tokens) != 1 for tokens in answer_tokens)
            or answer_tokens[0] == answer_tokens[1]
            or any(tokens[0] in tokenizer.all_special_ids for tokens in answer_tokens)):
        raise ValueError("Yes/No answers must encode as distinct single non-special tokens")
    YES_ID, NO_ID = (tokens[0] for tokens in answer_tokens)
    full_data = None
    if full_impression_training:
        full_data = MINDImpressionSFTDataset(
            train_behaviors_path, train_news_path, tokenizer,
            max_len=cutoff_len, max_history=max_history, use_abstract=use_abstract,
            use_subcategory=use_subcategory, use_chat_template=use_chat_template,
            use_recency=use_recency, use_profile_summary=use_profile_summary,
            use_impression_timestamp=use_impression_timestamp)
    if pairwise_loss_weight > 0 and resume_from_checkpoint:
        resume_config = AutoConfig.from_pretrained(
            _Path(resume_from_checkpoint), local_files_only=True)
        if (getattr(resume_config, "mind_pairwise_objective", None) != PAIRWISE_OBJECTIVE
                or getattr(resume_config, "mind_pairwise_loss_weight", None) != pairwise_loss_weight
                or getattr(resume_config, "mind_pairwise_answer_token_ids", None) != [YES_ID, NO_ID]):
            raise ValueError("Cannot resume incompatible optimizer state under the repaired pairwise objective")

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

    if pairwise_loss_weight > 0 or full_impression_training:
        model.config.mind_pairwise_objective = PAIRWISE_OBJECTIVE
        model.config.mind_pairwise_score = "final_prompt_yes_minus_no"
        model.config.mind_pairwise_loss_weight = pairwise_loss_weight
        model.config.mind_pairwise_answer_token_ids = [YES_ID, NO_ID]
    if full_impression_training:
        model.config.mind_pairwise_objective = COMPLETE_OBJECTIVE
        model.config.mind_training_source = full_data.source_manifest
        model.config.mind_candidate_chunk_size = candidate_chunk_size
    print(f"Answer token ids: YES={YES_ID!r} NO={NO_ID!r}")

    train_data = full_data if full_impression_training else MINDPointwiseSFTDataset(
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
        return_meta=True,
    )

    val_data = None if full_impression_training else MINDPointwiseSFTDataset(
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
        return_meta=True,
    )

    print(f"\nTraining with Hybrid Point-wise + Pairwise (BT) SFT:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data) if val_data is not None else 0}")
    if full_impression_training:
        print(f"  Batch unit: complete impressions; candidate chunk size: {candidate_chunk_size}")
        print(f"  Full source: {train_data.source_manifest}")
    print(f"  Max history: {'unlimited' if max_history == 0 else max_history}")
    print("  Negative sampling: NONE" if full_impression_training else f"  Neg ratio: {neg_ratio}")
    print(f"  Cutoff length: {cutoff_len}")
    print(f"  Chat template: {'enabled' if use_chat_template else 'disabled (raw text)'}")
    print(f"  Pairwise (BT) loss weight lambda: {pairwise_loss_weight}")

    training_args_dict = {
        "remove_unused_columns": False,
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

    trainer_type = CompleteImpressionTrainer if full_impression_training else _PairwiseTrainer
    full_kwargs = ({"expected_source": train_data.source_manifest,
                    "candidate_chunk_size": candidate_chunk_size} if full_impression_training else {})
    trainer = trainer_type(
        yes_token_id=YES_ID,
        no_token_id=NO_ID,
        pairwise_loss_weight=pairwise_loss_weight,
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None,
        args=training_args,
        data_collator=(CompleteImpressionCollator(_Collator(tokenizer))
                       if full_impression_training else _Collator(tokenizer)),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if val_data else None,
        **full_kwargs,
    )

    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if full_impression_training:
        model.config.mind_training_epoch_coverage = trainer.epoch_coverage

    final_checkpoint_path = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)

    print(f"\n✓ Hybrid Point-wise + Pairwise training completed!")
    print(f"  Model saved to: {final_checkpoint_path}")
    print(f"\nTo evaluate:")
    print(f"  bash scripts/eval_mind_pointwise.sh {output_dir}/final_checkpoint dev")


if __name__ == "__main__":
    fire.Fire(train)
