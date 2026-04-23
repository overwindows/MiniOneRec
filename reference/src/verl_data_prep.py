import argparse
import ast
import os

import pandas as pd
from datasets import Dataset


def _build_prompt(history_sids):
    history = ", ".join(history_sids)
    user_input = (
        f"The user has interacted with items {history} in chronological order. "
        "Can you predict the next possible item that the user may expect?"
    )
    prompt = f"### User Input:\n{user_input}\n\n### Response:\n"
    return [{"role": "user", "content": prompt}]


def _row_to_sample(row, data_source, ability):
    history_sids = ast.literal_eval(row["history_item_sid"])
    prompt = _build_prompt(history_sids)
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": ability,
        "reward_model": {"ground_truth": str(row["item_sid"])},
        "extra_info": {
            "user_id": str(row.get("user_id", "")),
            "item_sid": str(row["item_sid"]),
            "history_item_sid": history_sids,
        },
    }


def _convert_csv(path, data_source, ability, max_samples):
    df = pd.read_csv(path)
    if max_samples > 0:
        df = df.sample(max_samples, random_state=42)
    samples = [_row_to_sample(row, data_source, ability) for _, row in df.iterrows()]
    return Dataset.from_list(samples)


def main():
    parser = argparse.ArgumentParser(description="Prepare MiniOneRec data for VERL (parquet).")
    parser.add_argument("--train_file", required=True, help="CSV training file.")
    parser.add_argument("--eval_file", required=True, help="CSV validation file.")
    parser.add_argument("--output_dir", required=True, help="Output directory for parquet files.")
    parser.add_argument("--data_source", default="minionerec", help="Dataset name tag.")
    parser.add_argument("--ability", default="rec", help="Ability label for the dataset.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional cap on samples.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_ds = _convert_csv(args.train_file, args.data_source, args.ability, args.max_samples)
    eval_ds = _convert_csv(args.eval_file, args.data_source, args.ability, args.max_samples)

    train_path = os.path.join(args.output_dir, "train.parquet")
    eval_path = os.path.join(args.output_dir, "eval.parquet")
    train_ds.to_parquet(train_path)
    eval_ds.to_parquet(eval_path)
    print(f"Wrote train parquet: {train_path}")
    print(f"Wrote eval parquet: {eval_path}")


if __name__ == "__main__":
    main()
