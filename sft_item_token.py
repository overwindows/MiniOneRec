import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire

from data import ItemTokenSFTDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ItemTokenCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []
        inject_positions = []
        inject_embeds = []
        target_positions = []
        target_embeds = []

        for i, f in enumerate(features):
            pad_left = max_len - len(f["input_ids"])
            input_ids.append([self.pad_id] * pad_left + f["input_ids"])
            attention_mask.append([0] * pad_left + f["attention_mask"])
            labels.append([-100] * pad_left + f["labels"])

            for pos, emb in zip(f["inject_positions"], f["inject_embeds"]):
                inject_positions.append([i, pos + pad_left])
                inject_embeds.append(emb)

            for pos, emb in zip(f["target_positions"], f["target_embeds"]):
                target_positions.append([i, pos + pad_left])
                target_embeds.append(emb)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if inject_positions:
            batch["inject_positions"] = torch.tensor(inject_positions, dtype=torch.long)
            batch["inject_embeds"] = torch.tensor(np.array(inject_embeds), dtype=torch.float16)
        else:
            batch["inject_positions"] = torch.empty((0, 2), dtype=torch.long)
            batch["inject_embeds"] = torch.empty((0, 1), dtype=torch.float16)

        if target_positions:
            batch["target_positions"] = torch.tensor(target_positions, dtype=torch.long)
            batch["target_embeds"] = torch.tensor(np.array(target_embeds), dtype=torch.float16)
        else:
            batch["target_positions"] = torch.empty((0, 2), dtype=torch.long)
            batch["target_embeds"] = torch.empty((0, 1), dtype=torch.float16)

        return batch


class ItemTokenModel(nn.Module):
    def __init__(self, base_model, item_emb_dim, item_loss_weight=1.0, freeze_llm=False):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
        )
        if freeze_llm:
            for param in self.lm.parameters():
                param.requires_grad = False
        self.item_projector = nn.Linear(item_emb_dim, self.lm.config.hidden_size, bias=False)
        self.item_pred_head = nn.Linear(self.lm.config.hidden_size, item_emb_dim, bias=False)
        self.item_loss_weight = item_loss_weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        inject_positions=None,
        inject_embeds=None,
        target_positions=None,
        target_embeds=None,
    ):
        inputs_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.to(self.lm.dtype)

        if inject_positions is not None and inject_positions.numel() > 0:
            inject_embeds = inject_embeds.to(inputs_embeds.device)
            projected = self.item_projector(inject_embeds.to(inputs_embeds.dtype))
            batch_idx = inject_positions[:, 0]
            pos_idx = inject_positions[:, 1]
            inputs_embeds[batch_idx, pos_idx] = projected

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        lm_loss = outputs.loss
        item_loss = None

        if target_positions is not None and target_positions.numel() > 0:
            target_embeds = target_embeds.to(inputs_embeds.device).float()
            batch_idx = target_positions[:, 0]
            pos_idx = target_positions[:, 1]
            hidden = outputs.hidden_states[-1][batch_idx, pos_idx]
            pred = self.item_pred_head(hidden.float())
            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target_embeds, dim=-1)
            item_loss = 1.0 - F.cosine_similarity(pred, target, dim=-1)
            item_loss = item_loss.mean()

        if lm_loss is None:
            total_loss = item_loss
        elif item_loss is None:
            total_loss = lm_loss
        else:
            total_loss = lm_loss + self.item_loss_weight * item_loss

        return {"loss": total_loss, "logits": outputs.logits}


def train(
    base_model: str = "",
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    item_emb_path: str = "",
    category: str = "",
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    item_token: str = "<item>",
    inject_target: bool = False,
    item_loss_weight: float = 1.0,
    freeze_llm: bool = False,
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

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if item_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [item_token]})

    emb_dim = int(np.load(item_emb_path).shape[1])

    model = ItemTokenModel(
        base_model=base_model,
        item_emb_dim=emb_dim,
        item_loss_weight=item_loss_weight,
        freeze_llm=freeze_llm,
    )
    model.lm.resize_token_embeddings(len(tokenizer))

    train_dataset = ItemTokenSFTDataset(
        train_file=train_file,
        tokenizer=tokenizer,
        item_emb_path=item_emb_path,
        item_token=item_token,
        max_len=cutoff_len,
        sample=-1,
        seed=seed,
        category=category_dict[category],
        inject_target=inject_target,
    )
    eval_dataset = ItemTokenSFTDataset(
        train_file=eval_file,
        tokenizer=tokenizer,
        item_emb_path=item_emb_path,
        item_token=item_token,
        max_len=cutoff_len,
        sample=-1,
        seed=seed,
        category=category_dict[category],
        inject_target=inject_target,
    )

    gradient_accumulation_steps = batch_size // micro_batch_size

    training_args = transformers.TrainingArguments(
        run_name="item_token_sft",
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
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
        report_to=None,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=ItemTokenCollator(tokenizer),
    )
    model.lm.config.use_cache = False

    trainer.train()
    trainer.save_model(output_dir)

    final_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    fire.Fire(train)
