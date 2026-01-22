"""
Ranking-Only Evaluation for MIND (Multiple-Choice Format)

This script evaluates MIND models using ONLY the multiple-choice ranking format
that matches the training process.

Key Features:
- Multiple-choice prompt: "A. Title1\nB. Title2\nC. Title3\n..."
- Scores letter probabilities: P(" A"), P(" B"), P(" C"), ...
- Exactly matches training format
- No text generation, only selection

Usage:
    python eval_ranking_only.py \
        --model_path output_dir/sft_mind_ranking_*/final_checkpoint \
        --split dev \
        --max_impressions 100  # Optional: for testing
"""

import argparse
import math
import random
from typing import List, Tuple
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_news(news_path: str, use_abstract: bool = False) -> dict:
    """Load news articles from news.tsv"""
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

            if use_abstract and abstract:
                text = f"{title} {abstract}"
            else:
                text = title

            news[news_id] = {
                'text': text,
                'category': category
            }
    return news


def build_ranking_prompt(history_items: List[dict], candidates: List[dict]) -> str:
    """
    Build multiple-choice ranking prompt (EXACT training format).

    Returns:
        Prompt string ending with "Answer:"
    """
    prompt = "Role: You are a news recommendation assistant.\n"
    prompt += "Task: Select the most relevant news article for the user based on their reading history.\n\n"

    # User history
    prompt += "User History:\n"
    if history_items:
        for i, item in enumerate(history_items, 1):
            category = f" ({item['category']})" if item.get('category') else ""
            prompt += f"{i}. [Title] {item['text']}{category}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\n"

    # Candidate articles
    prompt += "Candidate News Articles:\n"
    option_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, cand in enumerate(candidates):
        letter = option_letters[i]
        category = f" ({cand['category']})" if cand.get('category') else ""
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
    """
    Score each option by measuring P(letter | prompt).

    For prompt ending with "Answer:", computes:
    - P(" A" | prompt)
    - P(" B" | prompt)
    - P(" C" | prompt)
    etc.

    Returns:
        List of log probabilities for each letter
    """
    option_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:num_options]

    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Get letter token IDs
    letter_token_ids = []
    for letter in option_letters:
        # Encode " A", " B", " C", ... (with leading space)
        letter_text = f" {letter}"
        tokens = tokenizer.encode(letter_text, add_special_tokens=False)

        # Handle tokenizers that split " A" into multiple tokens
        if len(tokens) > 1:
            letter_token_id = tokens[-1]  # Take last token (the letter)
        else:
            letter_token_id = tokens[0]

        letter_token_ids.append(letter_token_id)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]  # Last position logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Extract probabilities for each letter
    scores = []
    for token_id in letter_token_ids:
        score = log_probs[token_id].item()
        scores.append(score)

    return scores


def mrr_score(y_true, y_score):
    """
    MRR (Mean Reciprocal Rank) - Official MIND implementation.

    Source: https://github.com/msnews/MIND/blob/master/evaluate.py
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def dcg_score(y_true, y_score, k=10):
    """
    DCG (Discounted Cumulative Gain) - Official MIND implementation.

    Source: https://github.com/msnews/MIND/blob/master/evaluate.py
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """
    nDCG (Normalized Discounted Cumulative Gain) - Official MIND implementation.

    Source: https://github.com/msnews/MIND/blob/master/evaluate.py
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def compute_metrics(labels: List[int], scores: List[float]) -> dict:
    """
    Compute AUC, MRR, nDCG@5, nDCG@10 using official MIND metrics.

    All implementations match: https://github.com/msnews/MIND/blob/master/evaluate.py
    """
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    # AUC - using sklearn (same as official)
    pos_count = np.sum(labels_arr)
    if pos_count == 0 or pos_count == len(labels_arr):
        auc = 0.5
    else:
        auc = roc_auc_score(labels_arr, scores_arr)

    # MRR - official implementation
    mrr = mrr_score(labels_arr, scores_arr)

    # nDCG@5 - official implementation
    ndcg5 = ndcg_score(labels_arr, scores_arr, k=5)

    # nDCG@10 - official implementation
    ndcg10 = ndcg_score(labels_arr, scores_arr, k=10)

    return {
        'auc': auc,
        'mrr': mrr,
        'ndcg5': ndcg5,
        'ndcg10': ndcg10
    }


def main():
    parser = argparse.ArgumentParser(description="Ranking-only MIND evaluation")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--split", default="dev", choices=["dev", "test"], help="Evaluation split")
    parser.add_argument("--mind_root", default="../data/MIND", help="MIND dataset root")
    parser.add_argument("--use_abstract", action="store_true", help="Use news abstracts")
    parser.add_argument("--max_history", type=int, default=50, help="Max history items")
    parser.add_argument("--max_impressions", type=int, default=0, help="Limit impressions (0=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Paths
    behaviors_path = f"{args.mind_root}/{args.split}/behaviors.tsv"
    news_path = f"{args.mind_root}/{args.split}/news.tsv"

    print("=" * 80)
    print("MIND Ranking-Only Evaluation")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Split: {args.split}")
    print(f"Format: Multiple-choice (A/B/C/...)")
    print(f"Scoring: Letter probabilities only")
    print("=" * 80)
    print()

    # Load news
    print(f"Loading news from: {news_path}")
    news = load_news(news_path, args.use_abstract)
    print(f"✓ Loaded {len(news)} news articles")

    # Load model
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"✓ Model loaded on device: {device}")
    print()

    # Evaluate
    all_metrics = {
        'auc': [],
        'mrr': [],
        'ndcg5': [],
        'ndcg10': []
    }

    count = 0
    skipped_too_many = 0

    with open(behaviors_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total = min(len(lines), args.max_impressions) if args.max_impressions > 0 else len(lines)

        pbar = tqdm(lines[:total] if args.max_impressions > 0 else lines,
                   desc="Evaluating", unit="impression")

        for line in pbar:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            # Parse impression
            impression_id = parts[0]
            history_ids = parts[3].split()[-args.max_history:]
            impressions = parts[4].split()

            # Parse candidates and labels
            candidates = []
            labels = []
            for imp in impressions:
                if '-' not in imp:
                    continue
                nid, label = imp.rsplit('-', 1)
                if nid in news:
                    candidates.append(news[nid])
                    labels.append(int(label))

            # Skip if no candidates or no positives
            if not candidates or sum(labels) == 0:
                continue

            # Skip if too many candidates (>26, can't fit in A-Z)
            if len(candidates) > 26:
                skipped_too_many += 1
                continue

            # Build history
            history_items = [news[nid] for nid in history_ids if nid in news]

            # Build prompt
            prompt = build_ranking_prompt(history_items, candidates)

            # Score letters
            scores = score_letter_probabilities(
                model, tokenizer, prompt, len(candidates), device
            )

            # Compute metrics
            metrics = compute_metrics(labels, scores)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])

            count += 1

            # Update progress
            avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
            pbar.set_postfix({
                'AUC': f"{avg_metrics['auc']:.4f}",
                'MRR': f"{avg_metrics['mrr']:.4f}",
                'nDCG@5': f"{avg_metrics['ndcg5']:.4f}",
                'nDCG@10': f"{avg_metrics['ndcg10']:.4f}"
            })

    # Final results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Impressions processed: {count}")
    if skipped_too_many > 0:
        print(f"Skipped (>26 candidates): {skipped_too_many}")
    print()
    print(f"AUC:      {np.mean(all_metrics['auc']):.4f}")
    print(f"MRR:      {np.mean(all_metrics['mrr']):.4f}")
    print(f"nDCG@5:   {np.mean(all_metrics['ndcg5']):.4f}")
    print(f"nDCG@10:  {np.mean(all_metrics['ndcg10']):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
