"""
Ranking-Only Evaluation for MIND (A-Z Multiple-Choice Format)

This script evaluates MIND models that were trained with letter options (A-Z).
It matches the legacy training prompt and scores letter probabilities.

Usage:
    python eval_ranking_only_az.py \
        --model_path output_dir/sft_mind_ranking_*/final_checkpoint \
        --split dev \
        --max_impressions 100  # Optional: for testing
"""

import argparse
import math
import random
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


_OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_news(news_path: str, use_abstract: bool = False) -> dict:
    news = {}
    with open(news_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            news_id = parts[0]
            category = parts[1] if len(parts) > 1 else ""
            title = parts[3]
            abstract = parts[4] if len(parts) > 4 else ""
            text = f"{title} {abstract}" if use_abstract and abstract else title
            news[news_id] = {
                "text": text,
                "category": category,
            }
    return news


def build_ranking_prompt(history_items: List[dict], candidates: List[dict]) -> str:
    prompt = "Role: You are a news recommendation assistant.\n"
    prompt += "Task: Select the most relevant news article for the user based on their reading history.\n\n"

    prompt += "User History:\n"
    if history_items:
        for i, item in enumerate(history_items, 1):
            category = f" ({item.get('category', '')})" if item.get('category') else ""
            prompt += f"{i}. [Title] {item['text']}{category}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\n"
    prompt += "Candidate News Articles:\n"
    for i, cand in enumerate(candidates):
        letter = _OPTION_LETTERS[i]
        category = f" ({cand.get('category', '')})" if cand.get('category') else ""
        prompt += f"{letter}. [Title] {cand['text']}{category}\n"

    prompt += "\n"
    prompt += "Please analyze the user's interests and select the best article from the candidates above.\n"
    prompt += "Output only the option letter.\n\n"
    prompt += "Answer:"
    return prompt


def score_letter_probabilities(
    model,
    tokenizer,
    prompt: str,
    num_options: int,
    device
) -> List[float]:
    if num_options > len(_OPTION_LETTERS):
        raise ValueError(f"num_options={num_options} exceeds A-Z limit")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    candidate_tokens = []
    for i in range(num_options):
        letter_text = f" {_OPTION_LETTERS[i]}"
        token_ids = tokenizer.encode(letter_text, add_special_tokens=False)
        candidate_tokens.append(token_ids)

    scores = [0.0] * num_options

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        first_logits = outputs.logits[0, -1, :]
        first_log_probs = torch.nn.functional.log_softmax(first_logits, dim=-1)
        base_past_kv = outputs.past_key_values

        first_token_groups = {}
        for idx, token_seq in enumerate(candidate_tokens):
            first_tok = token_seq[0]
            if first_tok not in first_token_groups:
                first_token_groups[first_tok] = []
            first_token_groups[first_tok].append((idx, token_seq))

        for first_tok, group in first_token_groups.items():
            first_tok_log_prob = first_log_probs[first_tok].item()
            needs_second = [(idx, seq) for idx, seq in group if len(seq) > 1]
            single_token = [(idx, seq) for idx, seq in group if len(seq) == 1]

            for idx, _ in single_token:
                scores[idx] = first_tok_log_prob

            if not needs_second:
                continue

            first_tok_tensor = torch.tensor([[first_tok]], dtype=torch.long, device=device)
            outputs2 = model(input_ids=first_tok_tensor, past_key_values=base_past_kv, use_cache=True)
            second_logits = outputs2.logits[0, -1, :]
            second_log_probs = torch.nn.functional.log_softmax(second_logits, dim=-1)
            second_past_kv = outputs2.past_key_values

            needs_third = [(idx, seq) for idx, seq in needs_second if len(seq) > 2]
            two_token = [(idx, seq) for idx, seq in needs_second if len(seq) == 2]

            for idx, seq in two_token:
                scores[idx] = first_tok_log_prob + second_log_probs[seq[1]].item()

            if needs_third:
                second_token_groups = {}
                for idx, seq in needs_third:
                    second_tok = seq[1]
                    if second_tok not in second_token_groups:
                        second_token_groups[second_tok] = []
                    second_token_groups[second_tok].append((idx, seq))

                for second_tok, group3 in second_token_groups.items():
                    second_tok_log_prob = second_log_probs[second_tok].item()
                    second_tok_tensor = torch.tensor([[second_tok]], dtype=torch.long, device=device)
                    outputs3 = model(input_ids=second_tok_tensor, past_key_values=second_past_kv, use_cache=True)
                    third_logits = outputs3.logits[0, -1, :]
                    third_log_probs = torch.nn.functional.log_softmax(third_logits, dim=-1)

                    for idx, seq in group3:
                        score = first_tok_log_prob + second_tok_log_prob
                        if len(seq) > 2:
                            score += third_log_probs[seq[2]].item()
                        scores[idx] = score

    return scores


def compute_metrics(labels: List[int], scores: List[float]) -> dict:
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)
    pos_count = np.sum(labels_arr)
    if pos_count == 0 or pos_count == len(labels_arr):
        auc = 0.5
    else:
        auc = roc_auc_score(labels_arr, scores_arr)

    order = np.argsort(scores_arr)[::-1]
    y_true = np.take(labels_arr, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    mrr = np.sum(rr_score) / np.sum(y_true)

    def dcg_score(y_true, y_score, k=10):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def ndcg_score(y_true, y_score, k=10):
        best = dcg_score(y_true, y_true, k)
        actual = dcg_score(y_true, y_score, k)
        return actual / best if best > 0 else 0.0

    return {
        "AUC": float(auc),
        "MRR": float(mrr),
        "nDCG@5": float(ndcg_score(labels_arr, scores_arr, 5)),
        "nDCG@10": float(ndcg_score(labels_arr, scores_arr, 10)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--mind_root", default="../data/MIND")
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--max_history", type=int, default=0)
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    behaviors_path = f"{args.mind_root}/{args.split}/behaviors.tsv"
    news_path = f"{args.mind_root}/{args.split}/news.tsv"

    print(f"Loading news from: {news_path}")
    news = load_news(news_path, args.use_abstract)
    print(f"✓ Loaded {len(news)} news articles")

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"✓ Model loaded on device: {device}")

    aucs, mrrs, ndcg5, ndcg10 = [], [], [], []
    skipped_over_az = 0
    count = 0

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    with open(behaviors_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=None, desc="Evaluating impressions", unit="impression")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            history_ids = parts[3].split()
            if args.max_history > 0:
                history_ids = history_ids[-args.max_history:]
            impressions = parts[4].split()

            labels = []
            candidate_objs = []
            for imp in impressions:
                if "-" not in imp:
                    continue
                nid, label = imp.rsplit("-", 1)
                candidate_objs.append(news.get(nid, {"text": "[MISSING_NEWS]", "category": ""}))
                labels.append(int(label))

            if not candidate_objs:
                continue

            if len(candidate_objs) > len(_OPTION_LETTERS):
                skipped_over_az += 1
                continue

            history_objs = [news[nid] for nid in history_ids if nid in news]
            prompt = build_ranking_prompt(history_objs, candidate_objs)
            scores = score_letter_probabilities(
                model, tokenizer, prompt, len(candidate_objs), device
            )

            if sum(labels) > 0:
                metrics = compute_metrics(labels, scores)
                aucs.append(metrics["AUC"])
                mrrs.append(metrics["MRR"])
                ndcg5.append(metrics["nDCG@5"])
                ndcg10.append(metrics["nDCG@10"])

            count += 1
            pbar.update(1)
            pbar.set_postfix({
                "AUC": f"{_avg(aucs):.4f}",
                "MRR": f"{_avg(mrrs):.4f}",
                "nDCG@5": f"{_avg(ndcg5):.4f}",
                "nDCG@10": f"{_avg(ndcg10):.4f}",
            })
            if args.max_impressions and count >= args.max_impressions:
                break
        pbar.close()

    print("\nMIND Evaluation (Ranking-Only, A-Z Options)")
    print(f"Impressions processed: {count}")
    if skipped_over_az:
        print(f"Skipped impressions over 26 candidates: {skipped_over_az}")
    if aucs:
        print(f"AUC:  {_avg(aucs):.4f}")
        print(f"MRR:  {_avg(mrrs):.4f}")
        print(f"nDCG@5:  {_avg(ndcg5):.4f}")
        print(f"nDCG@10: {_avg(ndcg10):.4f}")
    else:
        print("No metrics computed (test set has no labels)")


if __name__ == "__main__":
    main()
