import json
import os
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import fire

from data import EvalD3Dataset, EvalTextMetaDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(
    base_model: str = "",
    category: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    item_meta_path: str = "",
    batch_size: int = 4,
    max_new_tokens: int = 256,
    num_beams: int = 20,
    length_penalty: float = 0.0,
    seed: int = 42,
):
    set_seed(seed)
    if not base_model:
        raise ValueError("--base_model is required")
    if not test_data_path:
        raise ValueError("--test_data_path is required")
    if not result_json_data:
        raise ValueError("--result_json_data is required")

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }
    if category not in category_dict:
        raise ValueError(f"Unknown category {category}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if item_meta_path:
        val_dataset = EvalTextMetaDataset(
            train_file=test_data_path,
            tokenizer=tokenizer,
            item_meta_path=item_meta_path,
            max_len=2560,
            category=category_dict[category],
            test=True,
        )
    else:
        val_dataset = EvalD3Dataset(
            train_file=test_data_path,
            tokenizer=tokenizer,
            max_len=2560,
            category=category_dict[category],
            test=True,
        )

    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def evaluate(encodings, num_beams=10, max_new_tokens=64, length_penalty=1.0):
        max_len = max(len(enc["input_ids"]) for enc in encodings)

        padding_encodings = {"input_ids": []}
        attention_mask = []

        for enc in encodings:
            length = len(enc["input_ids"])
            padding_encodings["input_ids"].append(
                [tokenizer.pad_token_id] * (max_len - length) + enc["input_ids"]
            )
            attention_mask.append([0] * (max_len - length) + [1] * length)

        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
        )

        with torch.no_grad():
            generation_output = model.generate(
                torch.tensor(padding_encodings["input_ids"]).to(model.device),
                attention_mask=torch.tensor(attention_mask).to(model.device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        batched_completions = generation_output.sequences[:, max_len:]
        output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
        output = [out.split("Response:\n")[-1].strip() for out in output]
        real_outputs = [
            output[i * num_beams : (i + 1) * num_beams]
            for i in range(len(output) // num_beams)
        ]
        return real_outputs

    outputs = []
    blocks = (len(encodings) + batch_size - 1) // batch_size
    for i in range(blocks):
        chunk = encodings[i * batch_size : (i + 1) * batch_size]
        output = evaluate(
            chunk, max_new_tokens=max_new_tokens, num_beams=num_beams, length_penalty=length_penalty
        )
        outputs.extend(output)

    for i, sample in enumerate(test_data):
        sample["predict"] = outputs[i]

    with open(result_json_data, "w") as f:
        json.dump(test_data, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
