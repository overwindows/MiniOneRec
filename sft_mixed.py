#!/usr/bin/env python3
"""
Mixed SFT: Combines recommendation-specific data with general instruction data
to prevent catastrophic forgetting of general LLM capabilities.
"""
import os
import sys
from typing import List
import numpy as np
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import (
    D3Dataset, SFTData, SidSFTDataset, SidItemFeatDataset,
    FusionSeqRecDataset, PreferenceSFTDataset,
    UserPreference2sidSFTDataset, TitleHistory2SidSFTDataset,
    InstructionJSONLDataset
)
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset


class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        if self.indices is None:
            self._load_data()

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / \
        float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str = "",
    eval_file: str = "",
    general_data_path: str = "",  # NEW: Path to general instruction data (JSONL format)
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,

    # NEW: Data mixing parameters
    general_data_ratio: float = 0.3,  # Ratio of general data in the training mix (0-1)
    general_data_sample: int = -1,  # Max number of general data examples to use (-1 = all)

    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    # either training checkpoint or final adapter
    resume_from_checkpoint: str = None,
    category: str = "",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
):
    set_seed(seed)
    os.environ['WANDB_PROJECT'] = wandb_project
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products",
                     "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(f"Category: {category}")
    category = category_dict.get(category, category)
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='Qwen/Qwen3-4B-Instruct-2507'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Load model
    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    original_vocab_size = len(tokenizer)

    # Add semantic IDs to tokenizer if provided
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0]
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

    # Freeze LLM parameters if required
    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad):
                    # grad shape: [vocab_size, hidden_dim]
                    grad[:original_vocab_size].zero_()
                    return grad

                embedding_layer.weight.register_hook(mask_grad)

                print(f"Unfrozen {len(new_tokens)} new token embeddings "
                      f"(indices {original_vocab_size} to {len(tokenizer)-1})")

        else:
            print(
                "Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        # Print the number of trainable parameters
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (with grad-mask): {trainable_params:,} / "
              f"{total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # ===========================
    # Build Training Datasets
    # ===========================
    train_datasets = []

    # Recommendation-specific datasets
    print("Loading recommendation datasets...")
    train_data1 = SidSFTDataset(train_file=train_file, tokenizer=tokenizer,
                                max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_datasets.append(train_data1)
    print(f"  - SidSFTDataset: {len(train_data1)} examples")

    train_data2 = SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path,
                                     tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_datasets.append(train_data2)
    print(f"  - SidItemFeatDataset: {len(train_data2)} examples")

    train_data3 = FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path,
                                      tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
    train_datasets.append(train_data3)
    print(f"  - FusionSeqRecDataset: {len(train_data3)} examples")

    # Load general instruction data if provided
    if general_data_path and os.path.exists(general_data_path):
        print(f"\n{'='*60}")
        print("LOADING GENERAL INSTRUCTION DATA")
        print(f"{'='*60}")
        print(f"General data file: {general_data_path}")
        print(f"Max samples to load: {general_data_sample if general_data_sample > 0 else 'all'}")
        print(f"Target ratio: {general_data_ratio:.1%} general data")

        general_data = InstructionJSONLDataset(
            jsonl_path=general_data_path,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=general_data_sample,
            seed=seed
        )
        print(f"\n✓ Successfully loaded {len(general_data)} general instruction examples")

        # Calculate how many times to include general data to achieve desired ratio
        rec_data_size = sum(len(d) for d in train_datasets)

        if general_data_ratio > 0:
            # Calculate target size for general data based on ratio
            # If ratio=0.3, we want: general_size / (rec_size + general_size) = 0.3
            # Solving: general_size = rec_size * ratio / (1 - ratio)
            target_general_size = int(rec_data_size * general_data_ratio / (1 - general_data_ratio))

            # Determine how many copies of general data we need
            num_copies = max(1, target_general_size // len(general_data))

            print(f"\n{'='*60}")
            print("DATA MIXING STRATEGY")
            print(f"{'='*60}")
            print(f"Recommendation data size: {rec_data_size:,} examples")
            print(f"  - SidSFTDataset: {len(train_datasets[0]):,} examples")
            print(f"  - SidItemFeatDataset: {len(train_datasets[1]):,} examples")
            print(f"  - FusionSeqRecDataset: {len(train_datasets[2]):,} examples")
            print(f"\nGeneral data size: {len(general_data):,} examples × {num_copies} copies = {len(general_data) * num_copies:,} examples")
            print(f"\nMixing ratios:")
            actual_ratio = len(general_data) * num_copies / (rec_data_size + len(general_data) * num_copies)
            print(f"  - Target: {general_data_ratio:.1%} general data / {1-general_data_ratio:.1%} recommendation data")
            print(f"  - Actual: {actual_ratio:.1%} general data / {1-actual_ratio:.1%} recommendation data")
            print(f"\nTotal training data after mixing: {rec_data_size + len(general_data) * num_copies:,} examples")
            print(f"{'='*60}\n")

            # Add general data (possibly multiple copies to achieve target ratio)
            for copy_idx in range(num_copies):
                train_datasets.append(general_data)
                print(f"✓ Added general data copy {copy_idx + 1}/{num_copies} to training datasets")
        else:
            print("\n⚠ WARNING: Skipping general data (ratio=0)")
            print("This will be pure recommendation-only training (may cause catastrophic forgetting)")
    else:
        print(f"\n{'='*60}")
        print("NO GENERAL DATA FOUND")
        print(f"{'='*60}")
        if general_data_path:
            print(f"⚠ WARNING: General data path specified but not found: {general_data_path}")
        else:
            print("No general data path specified")
        print("\n⚠ Training with recommendation data only")
        print("This may cause catastrophic forgetting of general LLM capabilities")
        print(f"{'='*60}\n")

    # Concatenate all datasets
    train_data = ConcatDataset(train_datasets)
    print(f"\nTotal training examples: {len(train_data)}")

    # Validation data (recommendation-only)
    val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer,
                             max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    print(f"Validation examples: {len(val_data)}")

    print("\n" + "="*60)
    print("DATA LOADING FINISHED")
    print("="*60 + "\n")

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # Convert to HuggingFace datasets and shuffle
    sample_frac = 1
    hf_train_dataset = HFDataset.from_dict(
        {k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(
        range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict(
        {k: [v[k] for v in val_data] for k in val_data[0].keys()}).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=42)

    print(hf_train_dataset)
    print(hf_val_dataset)
    eval_step = 0.05
    trainer = transformers.Trainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            run_name=wandb_run_name,
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
            eval_steps=eval_step,
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print(f"Model saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    fire.Fire(train)
