"""
Train a model on MIND dataset using a HYBRID point-wise + pairwise objective
(Bradley-Terry) with DeepSpeed support.

This is a fork of `sft_mind_pointwise_ds.py`. It keeps the standard point-wise
CE over the Yes/No answer token, and ADDS a pairwise Bradley-Terry term so the
model also learns to score clicked candidates above non-clicked ones within the
same impression (A7.3, EXPERIMENT_PLAN.md):

    L_BT = -log(sigmoid(score_pos - score_neg))
    score(candidate) = logit of the " Yes" answer token

Total loss = L_CE + pairwise_loss_weight * L_BT

Rationale (arXiv:2312.16018, GLIMPSE 2409.17711): purely point-wise CE ignores
the ranking structure within an impression. Pairing positives against negatives
sharpens the relative score gap, which is what the AUC metric actually measures.
Expected gain ~ +0.2-0.5% AUC over the point-wise baseline (L1.3 settings).

Implementation notes:
- Uses `MINDPointwiseSFTDataset(return_meta=True)` so each sample carries its
  `impression_id` and `label`, letting the trainer pair pos/neg candidates from
  the same impression.
- The answer tokens are the only non -100 positions in `labels` (left-padded),
  so their logits can be recovered per-sample inside `compute_loss`.
- `score` is taken as the logit of the " Yes" token (the positive-class logit),
  matching how `evaluate_mind_pointwise.py` ranks candidates by P(yes).

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
if "LOCAL_RANK" in os.environ and not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(hours=2),
    )

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import MINDPointwiseSFTDataset


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
    """HF Trainer with CE + lambda*BT. The BT term ranks yes-logits pos>neg."""

    def __init__(self, yes_token_id, no_token_id, pairwise_loss_weight=0.5, **kwargs):
        self._yes_id = yes_token_id
        self._no_id = no_token_id
        self._pw = pairwise_loss_weight
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels", None)
        impressions = inputs.pop("impression_id", None)
        cls = inputs.pop("label", None)  # 1 = clicked, 0 = not clicked

        outputs = model(**inputs, labels=labels)
        ce_loss = outputs.loss

        loss = ce_loss
        if impressions is not None and cls is not None and self._pw > 0:
            logits = outputs.logits  # [B, L, V]
            # Answer tokens are the only non -100 positions (left-padded so
            # they are right-aligned). Grab each sample's yes-class logit as
            # its "score" proxy.
            valid = labels != -100  # [B, L]
            # Score = logit of the FIRST answer token at the YES token id.
            scores = []
            for i in range(logits.shape[0]):
                vpos = valid[i].nonzero(as_tuple=True)[0]
                if vpos.numel() == 0:
                    scores.append(torch.zeros((), device=logits.device))
                    continue
                pos = vpos[0].item()
                # P(yes) logit for the answer position.
                scores.append(logits[i, pos, self._yes_id])
            scores = torch.stack(scores)
            cls = cls.to(scores.device)

            bt = self._pairwise_term(scores, cls, impressions)
            if bt is not None:
                loss = loss + self._pw * bt

        return (loss, outputs) if return_outputs else loss

    def _pairwise_term(self, scores, cls, impressions):
        """Bradley-Terry: -log(sigmoid(score_pos - score_neg)).

        Pair candidates within the same impression; fall back to global pos/neg
        pairing when a batch has no impression-matched pairs.
        """
        pos_idx = (cls == 1).nonzero(as_tuple=True)[0]
        neg_idx = (cls == 0).nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return None

        # Prefer pairing within the same impression.
        imp_pos = neg_idx  # reuse
        paired_pos = []
        paired_neg = []
        used_neg = set()
        # Group negatives by impression for fast lookup.
        imp2neg = {}
        for ni in neg_idx.tolist():
            imp2neg.setdefault(impressions[ni], []).append(ni)
        for pi in pos_idx.tolist():
            imp = impressions[pi]
            cand = [n for n in imp2neg.get(imp, []) if n not in used_neg]
            if cand:
                ni = cand[0]
                used_neg.add(ni)
                paired_pos.append(pi)
                paired_neg.append(ni)

        if paired_pos:
            p = scores[torch.tensor(paired_pos, device=scores.device)]
            n = scores[torch.tensor(paired_neg, device=scores.device)]
        else:
            # Fall back to a random global pos/neg pairing.
            k = min(pos_idx.numel(), neg_idx.numel())
            perm = torch.randperm(neg_idx.numel(), device=scores.device)[:k]
            p = scores[pos_idx[:k]]
            n = scores[neg_idx[perm]]

        return -torch.log(torch.sigmoid(p - n) + 1e-8).mean()


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
):
    """Train with hybrid point-wise CE + pairwise Bradley-Terry objective."""
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

    # Answer tokens: the positive-class " Yes" logit is our ranking score.
    try:
        YES_ID = tokenizer.encode(" Yes", add_special_tokens=False)[0]
        NO_ID = tokenizer.encode(" No", add_special_tokens=False)[0]
    except Exception as e:
        raise RuntimeError(f"Could not resolve answer token ids: {e}")
    print(f"Answer token ids: YES={YES_ID!r} NO={NO_ID!r}")

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
        return_meta=True,
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
    )

    print(f"\nTraining with Hybrid Point-wise + Pairwise (BT) SFT:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Max history: {'unlimited' if max_history == 0 else max_history}")
    print(f"  Neg ratio: {neg_ratio}")
    print(f"  Cutoff length: {cutoff_len}")
    print(f"  Chat template: {'enabled' if use_chat_template else 'disabled (raw text)'}")
    print(f"  Pairwise (BT) loss weight lambda: {pairwise_loss_weight}")

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

    trainer = _PairwiseTrainer(
        yes_token_id=YES_ID,
        no_token_id=NO_ID,
        pairwise_loss_weight=pairwise_loss_weight,
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None,
        args=training_args,
        data_collator=_Collator(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if val_data else None,
    )

    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_checkpoint_path = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)

    print(f"\n✓ Hybrid Point-wise + Pairwise training completed!")
    print(f"  Model saved to: {final_checkpoint_path}")
    print(f"\nTo evaluate:")
    print(f"  bash scripts/eval_mind_pointwise.sh {output_dir}/final_checkpoint dev")


if __name__ == "__main__":
    fire.Fire(train)
