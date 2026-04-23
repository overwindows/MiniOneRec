"""
Train a model on MIND dataset using Supervised Fine-Tuning (SFT).

This script is optimized for achieving SOTA results on MIND news recommendation.
Follows the same architecture as sft_text.py for consistency.
"""

import os
import sys
import random
import numpy as np
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset
import fire

# Add parent directory to path to import data module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import MINDTextSFTDataset


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
    max_history: int = 50,
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    train_from_scratch: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    set_seed(seed)

    if not base_model:
        raise ValueError("Please specify --base_model")

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    train_data = MINDTextSFTDataset(
        behaviors_path=train_behaviors_path,
        news_path=train_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        max_history=max_history,
        use_abstract=use_abstract,
    )

    val_data = MINDTextSFTDataset(
        behaviors_path=eval_behaviors_path,
        news_path=eval_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=min(sample, 5000) if sample > 0 else 5000,
        seed=seed,
        max_history=max_history,
        use_abstract=use_abstract,
    )

    train_samples = [train_data[i] for i in range(len(train_data))]
    val_samples = [val_data[i] for i in range(len(val_data))]

    if not train_samples:
        raise ValueError("No training samples available.")

    hf_train_dataset = HFDataset.from_dict(
        {k: [v[k] for v in train_samples] for k in train_samples[0].keys()}
    ).shuffle(seed=seed)

    hf_val_dataset = HFDataset.from_dict(
        {k: [v[k] for v in val_samples] for k in val_samples[0].keys()}
    ).shuffle(seed=seed)

    training_args = transformers.TrainingArguments(
        run_name=wandb_run_name if wandb_run_name else None,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=20,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=0.05,
        save_strategy="steps",
        save_steps=0.05,
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to=["wandb"] if wandb_project else [],
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)

    final_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    fire.Fire(train)
