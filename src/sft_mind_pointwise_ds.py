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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import MINDPointwiseSFTDataset

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
    use_chat_template: bool = None,  # Auto-detect if None
    use_recency: bool = False,  # Mark 5 most recent history items with "(recent)"
    use_profile_summary: bool = False,  # Prepend top-3 category interest summary
    use_impression_timestamp: bool = False,  # Add day/time-of-day context from impression timestamp
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

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Load model
    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

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
        use_abstract=use_abstract,
        use_subcategory=use_subcategory,
        use_chat_template=use_chat_template,
        use_recency=use_recency,
        use_profile_summary=use_profile_summary,
        use_impression_timestamp=use_impression_timestamp,
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
        use_abstract=use_abstract,
        use_subcategory=use_subcategory,
        use_chat_template=use_chat_template,
        use_recency=use_recency,
        use_profile_summary=use_profile_summary,
        use_impression_timestamp=use_impression_timestamp,
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
        "group_by_length": group_by_length,
        "report_to": "wandb" if wandb_project else "none",
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

    # Initialize trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=30)] if val_data else None,
    )

    model.config.use_cache = False

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model (use trainer.save_model for proper DeepSpeed ZeRO handling)
    final_checkpoint_path = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)

    print(f"\n✓ Point-wise SFT training completed!")
    print(f"  Model saved to: {final_checkpoint_path}")
    print(f"\nTo evaluate:")
    print(f"  bash scripts/eval_mind_pointwise.sh {output_dir}/final_checkpoint dev")


if __name__ == "__main__":
    fire.Fire(train)
