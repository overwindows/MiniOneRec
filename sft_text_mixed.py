import os
import random
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from datasets import Dataset as HFDataset
import fire

from data import (
    SFTData,
    TextMetaSFTDataset,
    MINDTextSFTDataset,
    InstructionJSONLDataset,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sample_with_replacement(rng, items, target):
    if not items or target <= 0:
        return []
    return [rng.choice(items) for _ in range(target)]


def _build_train_samples(rng, datasets, ratios, base_size):
    total_ratio = sum(ratios)
    if total_ratio <= 0:
        raise ValueError("At least one ratio must be > 0")

    samples = []
    for items, ratio in zip(datasets, ratios):
        if ratio <= 0:
            continue
        target = int(base_size * (ratio / total_ratio))
        if len(items) >= target:
            samples.extend(rng.sample(items, target))
        else:
            samples.extend(_sample_with_replacement(rng, items, target))
    rng.shuffle(samples)
    return samples


def train(
    base_model: str = "",
    output_dir: str = "",
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    group_by_length: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,
    train_from_scratch: bool = False,
    # Amazon
    amazon_train_file: str = "",
    amazon_eval_file: str = "",
    amazon_category: str = "",
    amazon_item_meta_path: str = "",
    amazon_ratio: float = 0.7,
    # MIND
    mind_behaviors_path: str = "",
    mind_news_path: str = "",
    mind_ratio: float = 0.2,
    mind_max_history: int = 50,
    mind_use_abstract: bool = False,
    # General
    general_jsonl: str = "",
    general_ratio: float = 0.1,
    # Eval selection
    eval_source: str = "amazon",
):
    set_seed(seed)
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    if not base_model:
        raise ValueError("Please specify --base_model")

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }

    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    rng = random.Random(seed)

    amazon_samples = []
    if amazon_train_file:
        if amazon_category not in category_dict:
            raise ValueError(f"Unknown amazon_category {amazon_category}")
        category_text = category_dict[amazon_category]
        if amazon_item_meta_path:
            amazon_ds = TextMetaSFTDataset(
                train_file=amazon_train_file,
                tokenizer=tokenizer,
                item_meta_path=amazon_item_meta_path,
                max_len=cutoff_len,
                sample=-1,
                seed=seed,
                category=category_text,
            )
        else:
            amazon_ds = SFTData(
                train_file=amazon_train_file,
                tokenizer=tokenizer,
                max_len=cutoff_len,
                sample=-1,
                seed=seed,
                category=category_text,
            )
        amazon_samples = [amazon_ds[i] for i in range(len(amazon_ds))]

    mind_samples = []
    if mind_behaviors_path and mind_news_path:
        mind_ds = MINDTextSFTDataset(
            behaviors_path=mind_behaviors_path,
            news_path=mind_news_path,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=-1,
            seed=seed,
            max_history=mind_max_history,
            use_abstract=mind_use_abstract,
        )
        mind_samples = [mind_ds[i] for i in range(len(mind_ds))]

    general_samples = []
    if general_jsonl:
        general_ds = InstructionJSONLDataset(
            jsonl_path=general_jsonl,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=-1,
            seed=seed,
        )
        general_samples = [general_ds[i] for i in range(len(general_ds))]

    if not amazon_samples and not mind_samples and not general_samples:
        raise ValueError("No training data sources provided.")

    base_size = len(amazon_samples) or max(len(mind_samples), len(general_samples))
    datasets = [amazon_samples, mind_samples, general_samples]
    ratios = [amazon_ratio, mind_ratio, general_ratio]
    train_samples = _build_train_samples(rng, datasets, ratios, base_size)

    if not train_samples:
        raise ValueError("No training samples available after mixing.")

    hf_train_dataset = HFDataset.from_dict(
        {k: [v[k] for v in train_samples] for k in train_samples[0].keys()}
    ).shuffle(seed=seed)

    if eval_source == "amazon":
        if not amazon_eval_file or not amazon_train_file:
            raise ValueError("Amazon eval requires --amazon_eval_file and --amazon_train_file")
        if amazon_category not in category_dict:
            raise ValueError(f"Unknown amazon_category {amazon_category}")
        category_text = category_dict[amazon_category]
        if amazon_item_meta_path:
            eval_data = TextMetaSFTDataset(
                train_file=amazon_eval_file,
                tokenizer=tokenizer,
                item_meta_path=amazon_item_meta_path,
                max_len=cutoff_len,
                sample=-1,
                seed=seed,
                category=category_text,
            )
        else:
            eval_data = SFTData(
                train_file=amazon_eval_file,
                tokenizer=tokenizer,
                max_len=cutoff_len,
                sample=-1,
                seed=seed,
                category=category_text,
            )
    elif eval_source == "mind":
        if not mind_behaviors_path or not mind_news_path:
            raise ValueError("MIND eval requires --mind_behaviors_path and --mind_news_path")
        eval_data = MINDTextSFTDataset(
            behaviors_path=mind_behaviors_path,
            news_path=mind_news_path,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=-1,
            seed=seed,
            max_history=mind_max_history,
            use_abstract=mind_use_abstract,
        )
    else:
        raise ValueError("eval_source must be 'amazon' or 'mind'")

    hf_val_dataset = HFDataset.from_dict(
        {k: [v[k] for v in eval_data] for k in eval_data[0].keys()}
    ).shuffle(seed=seed)

    training_args = transformers.TrainingArguments(
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
        eval_steps=0.05,
        save_strategy="steps",
        save_steps=0.05,
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to=None,
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
