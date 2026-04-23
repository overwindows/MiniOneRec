import os
import sys
import random
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from datasets import Dataset as HFDataset
import fire

# Add parent directory to path to import data module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import SFTData, TextMetaSFTDataset, InstructionJSONLDataset


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
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    item_meta_path: str = "",
    general_jsonl: str = "",
    general_ratio: float = 0.0,
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    category: str = "",
    train_from_scratch: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    set_seed(seed)

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }
    if category not in category_dict:
        raise ValueError(f"Unknown category {category}")
    category_text = category_dict[category]

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

    if item_meta_path:
        train_data = TextMetaSFTDataset(
            train_file=train_file,
            tokenizer=tokenizer,
            item_meta_path=item_meta_path,
            max_len=cutoff_len,
            sample=sample,
            seed=seed,
            category=category_text,
        )
        val_data = TextMetaSFTDataset(
            train_file=eval_file,
            tokenizer=tokenizer,
            item_meta_path=item_meta_path,
            max_len=cutoff_len,
            sample=sample,
            seed=seed,
            category=category_text,
        )
    else:
        train_data = SFTData(
            train_file=train_file,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=sample,
            seed=seed,
            category=category_text,
        )
        val_data = SFTData(
            train_file=eval_file,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=sample,
            seed=seed,
            category=category_text,
        )

    rec_samples = [train_data[i] for i in range(len(train_data))]
    if general_jsonl and general_ratio > 0:
        general_data = InstructionJSONLDataset(
            jsonl_path=general_jsonl,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=-1,
            seed=seed,
        )
        general_samples = [general_data[i] for i in range(len(general_data))]
        target_general = int(len(rec_samples) * general_ratio / (1.0 - general_ratio))
        if general_samples:
            rng = random.Random(seed)
            mixed_general = [rng.choice(general_samples) for _ in range(target_general)]
        else:
            mixed_general = []
        train_samples = rec_samples + mixed_general
        rng = random.Random(seed)
        rng.shuffle(train_samples)
    else:
        train_samples = rec_samples

    if not train_samples:
        raise ValueError("No training samples available after mixing.")
    hf_train_dataset = HFDataset.from_dict(
        {k: [v[k] for v in train_samples] for k in train_samples[0].keys()}
    ).shuffle(seed=seed)
    hf_val_dataset = HFDataset.from_dict(
        {k: [v[k] for v in val_data] for k in val_data[0].keys()}
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
