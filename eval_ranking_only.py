"""
Ranking-Only Evaluation for MIND (Multiple-Choice Format)

This script evaluates MIND models using the multiple-choice ranking format
that EXACTLY matches the training process in sft_mind_ranking.py.

Key Features:
- Multiple-choice prompt: "1. Title1\n2. Title2\n3. Title3\n..." (NUMERIC)
- Scores number probabilities: P(" 1"), P(" 2"), P(" 3"), ...
- Exactly matches training format
- No text generation, only selection
- Uses official MIND evaluation metrics

Usage:
    python eval_ranking_only.py \
        --model_path output_dir/sft_mind_ranking_*/final_checkpoint \
        --split dev \
        --max_impressions 100  # Optional: for testing
"""

import argparse
import math
import random
from typing import List
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
    Build OPTIMIZED multiple-choice ranking prompt (EXACT training format).

    Based on Prompt4NR research (arXiv:2304.05263):
    - Concise format saves ~20 tokens
    - Category in [brackets] at start for better visibility
    - Natural language question
    - Limit to last 30 history items for focus

    Uses NUMERIC options (1, 2, 3, ...) to match sft_mind_ranking.py training.

    Returns:
        Prompt string ending with "Answer:"
    """
    prompt = "A user read these news articles:\n"

    # User history - limit to last 30 for token efficiency
    if history_items:
        recent_history = history_items[-30:] if len(history_items) > 30 else history_items
        for i, item in enumerate(recent_history, 1):
            cat = item.get('category', 'General')
            prompt += f"{i}. [{cat}] {item['text']}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\n"

    # Candidate articles - category first in brackets (NUMERIC options to match training)
    prompt += "Candidate articles:\n"
    for i, cand in enumerate(candidates):
        option_num = i + 1  # 1-indexed
        cat = cand.get('category', 'General')
        prompt += f"{option_num}. [{cat}] {cand['text']}\n"

    prompt += "\n"
    # Simple, direct question
    prompt += "Which article will this user read? Answer with the number.\n\n"
    prompt += "Answer:"

    return prompt


def score_number_probabilities(
    model,
    tokenizer,
    prompt: str,
    num_options: int,
    device
) -> List[float]:
    """
    Score each option by measuring P(number | prompt).

    For prompt ending with "Answer:", computes:
    - P(" 1" | prompt)
    - P(" 2" | prompt)
    - P(" 3" | prompt)
    etc.

    Handles multi-digit numbers (10+) by computing joint probability.
    Uses KV-cache for efficiency.

    Returns:
        List of log probabilities for each number
    """
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Pre-tokenize all candidate numbers to get their token sequences
    candidate_tokens = []
    for i in range(1, num_options + 1):
        # Tokenize " {number}" (with leading space)
        number_text = f" {i}"
        token_ids = tokenizer.encode(number_text, add_special_tokens=False)
        candidate_tokens.append(token_ids)

    scores = [0.0] * num_options

    with torch.no_grad():
        # Get initial logits with KV-cache
        outputs = model(input_ids=input_ids, use_cache=True)
        first_logits = outputs.logits[0, -1, :]
        first_log_probs = torch.nn.functional.log_softmax(first_logits, dim=-1)
        base_past_kv = outputs.past_key_values

        # Group candidates by first token to batch second-token scoring
        first_token_groups = {}
        for idx, token_seq in enumerate(candidate_tokens):
            first_tok = token_seq[0]
            if first_tok not in first_token_groups:
                first_token_groups[first_tok] = []
            first_token_groups[first_tok].append((idx, token_seq))

        for first_tok, group in first_token_groups.items():
            first_tok_log_prob = first_log_probs[first_tok].item()

            # Check if any in group need second token
            needs_second = [(idx, seq) for idx, seq in group if len(seq) > 1]
            single_token = [(idx, seq) for idx, seq in group if len(seq) == 1]

            # Handle single-token candidates (1-9)
            for idx, seq in single_token:
                scores[idx] = first_tok_log_prob

            if not needs_second:
                continue

            # For multi-token: run one forward pass with first token to get second token probs
            first_tok_tensor = torch.tensor([[first_tok]], dtype=torch.long, device=device)
            outputs2 = model(input_ids=first_tok_tensor, past_key_values=base_past_kv, use_cache=True)
            second_logits = outputs2.logits[0, -1, :]
            second_log_probs = torch.nn.functional.log_softmax(second_logits, dim=-1)
            second_past_kv = outputs2.past_key_values

            # Check if any need third token
            needs_third = [(idx, seq) for idx, seq in needs_second if len(seq) > 2]
            two_token = [(idx, seq) for idx, seq in needs_second if len(seq) == 2]

            # Handle two-token candidates (10-99)
            for idx, seq in two_token:
                scores[idx] = first_tok_log_prob + second_log_probs[seq[1]].item()

            # Handle three+ token candidates (100+)
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
    return actual / best if best > 0 else 0.0


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
    parser = argparse.ArgumentParser(description="Ranking-only MIND evaluation (numeric format)")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--split", default="dev", choices=["dev", "test"], help="Evaluation split")
    parser.add_argument("--mind_root", default="../data/MIND", help="MIND dataset root")
    parser.add_argument("--use_abstract", action="store_true", help="Use news abstracts")
    parser.add_argument("--max_history", type=int, default=0, help="Max history items (0=unlimited)")
    parser.add_argument("--max_impressions", type=int, default=0, help="Limit impressions (0=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention 2")
    parser.add_argument("--output_file", help="Output prediction file for MIND leaderboard")
    args = parser.parse_args()

    set_seed(args.seed)

    # Paths
    behaviors_path = f"{args.mind_root}/{args.split}/behaviors.tsv"
    news_path = f"{args.mind_root}/{args.split}/news.tsv"

    print("=" * 80)
    print("MIND Ranking-Only Evaluation (Official Metrics)")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Split: {args.split}")
    print(f"Format: Multiple-choice (1/2/3/...) - matches training")
    print(f"Scoring: Number probabilities")
    print(f"Max history: {'unlimited' if args.max_history == 0 else args.max_history}")
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

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto"
    }
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("✓ Using Flash Attention 2")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs
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
    predictions = []

    count = 0
    skipped_no_positive = 0

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
            history_ids = parts[3].split()
            if args.max_history > 0:
                history_ids = history_ids[-args.max_history:]
            impressions = parts[4].split()

            # Parse candidates and labels
            candidates = []
            candidate_ids = []
            labels = []
            for imp in impressions:
                if '-' not in imp:
                    continue
                nid, label = imp.rsplit('-', 1)
                if nid in news:
                    candidates.append(news[nid])
                    candidate_ids.append(nid)
                    labels.append(int(label))

            # Skip if no candidates or no positives
            if not candidates or sum(labels) == 0:
                skipped_no_positive += 1
                continue

            # Build history
            history_items = [news[nid] for nid in history_ids if nid in news]

            # Build prompt
            prompt = build_ranking_prompt(history_items, candidates)

            # Score numbers
            scores = score_number_probabilities(
                model, tokenizer, prompt, len(candidates), device
            )

            # Compute metrics
            metrics = compute_metrics(labels, scores)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])

            # Store predictions for leaderboard
            if args.output_file:
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                ranked_news_ids = [candidate_ids[i] for i in ranked_indices]
                predictions.append((impression_id, ranked_news_ids))

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
    print("RESULTS (Official MIND Metrics)")
    print("=" * 80)
    print(f"Impressions processed: {count}")
    if skipped_no_positive > 0:
        print(f"Skipped (no positive labels): {skipped_no_positive}")
    print()
    print(f"AUC:      {np.mean(all_metrics['auc']):.4f}")
    print(f"MRR:      {np.mean(all_metrics['mrr']):.4f}")
    print(f"nDCG@5:   {np.mean(all_metrics['ndcg5']):.4f}")
    print(f"nDCG@10:  {np.mean(all_metrics['ndcg10']):.4f}")
    print("=" * 80)

    # Write predictions
    if args.output_file and predictions:
        print(f"\nWriting predictions to: {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for impression_id, ranked_news_ids in predictions:
                f.write(f"{impression_id} {' '.join(ranked_news_ids)}\n")
        print(f"✓ Wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main()
