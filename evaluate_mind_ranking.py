"""
Evaluate MIND models trained with ranking-aware SFT (multiple-choice format).

This evaluation uses the SAME multiple-choice prompt format as training,
ensuring perfect alignment between training and evaluation.

Key difference from evaluate_mind.py:
- Uses multiple-choice format: "1. Title1\n2. Title2\n..." (numeric options)
- Scores option numbers (1, 2, 3, ...) instead of full text
- Matches training format exactly
- Uses KV-cache for efficient multi-token scoring
- Supports Flash Attention 2 for faster long-context processing

Usage:
    python evaluate_mind_ranking.py \
        --model_path output_dir/sft_mind_ranking_*/final_checkpoint \
        --behaviors_path ../data/MIND/dev/behaviors.tsv \
        --news_path ../data/MIND/dev/news.tsv \
        --flash_attn \
        --max_impressions 1000  # Optional: for quick testing
"""

import argparse
import json
import math
import random
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_news(news_path: str, use_abstract: bool) -> dict:
    news = {}
    with open(news_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
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
                'title': title,
                'text': text,
                'category': category
            }
    return news


def build_multiple_choice_prompt(history: List[dict], candidates: List[dict]) -> str:
    """
    Build multiple-choice ranking prompt matching training format.

    Uses numeric options (1, 2, 3, ...) instead of letters to support unlimited candidates.

    Returns the prompt without the "Answer:" part.
    """
    prompt = "Role: You are a news recommendation assistant.\n"
    prompt += "Task: Select the most relevant news article for the user based on their reading history.\n\n"

    # User history (must match training format exactly - includes category)
    prompt += "User History:\n"
    if history:
        for i, h in enumerate(history, 1):
            category = f" ({h['category']})" if h.get('category') else ""
            prompt += f"{i}. [Title] {h['text']}{category}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\n"

    # Candidate articles (use numbers instead of letters)
    prompt += "Candidate News Articles:\n"
    for i, cand in enumerate(candidates):
        option_num = i + 1  # 1-indexed
        category = f" ({cand['category']})" if cand.get('category') else ""
        prompt += f"{option_num}. [Title] {cand['text']}{category}\n"

    prompt += "\n"
    prompt += "Please analyze the user's interests and select the best article from the candidates above.\n"
    prompt += "Output only the option number.\n\n"
    prompt += "Answer:"

    return prompt


def score_candidates_multiple_choice(
    model,
    tokenizer,
    prompt: str,
    num_candidates: int,
    device,
) -> List[float]:
    """
    Score candidates using multiple-choice format with numeric options.

    Scores the probability of each option number (1, 2, 3, ...).
    Handles multi-digit numbers by computing the joint probability of all tokens.
    Uses KV-cache for efficiency.

    Returns:
        List of scores (log probabilities of each number)
    """
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

    # Pre-tokenize all candidate numbers to get their token sequences
    candidate_tokens = []
    for i in range(1, num_candidates + 1):
        # Tokenize " {number}" (with leading space)
        number_text = f" {i}"
        token_ids = tokenizer.encode(number_text, add_special_tokens=False)
        candidate_tokens.append(token_ids)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    scores = []
    with torch.no_grad():
        # Get initial logits with KV-cache
        outputs = model(input_ids=input_ids, use_cache=True)
        first_logits = outputs.logits[0, -1, :]
        first_log_probs = torch.nn.functional.log_softmax(first_logits, dim=-1)
        base_past_kv = outputs.past_key_values

        # Group candidates by first token to batch second-token scoring
        # This reduces forward passes for multi-digit numbers
        first_token_groups = {}
        for idx, token_seq in enumerate(candidate_tokens):
            first_tok = token_seq[0]
            if first_tok not in first_token_groups:
                first_token_groups[first_tok] = []
            first_token_groups[first_tok].append((idx, token_seq))

        # Initialize scores array
        scores = [0.0] * num_candidates

        for first_tok, group in first_token_groups.items():
            first_tok_log_prob = first_log_probs[first_tok].item()

            # Check if any in group need second token
            needs_second = [(idx, seq) for idx, seq in group if len(seq) > 1]
            single_token = [(idx, seq) for idx, seq in group if len(seq) == 1]

            # Handle single-token candidates
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

            # Handle two-token candidates
            for idx, seq in two_token:
                scores[idx] = first_tok_log_prob + second_log_probs[seq[1]].item()

            # Handle three+ token candidates (rare: numbers >= 100)
            if needs_third:
                # Group by second token
                second_token_groups = {}
                for idx, seq in needs_third:
                    second_tok = seq[1]
                    if second_tok not in second_token_groups:
                        second_token_groups[second_tok] = []
                    second_token_groups[second_tok].append((idx, seq))

                for second_tok, group3 in second_token_groups.items():
                    second_tok_log_prob = second_log_probs[second_tok].item()

                    # Run forward pass for third token
                    second_tok_tensor = torch.tensor([[second_tok]], dtype=torch.long, device=device)
                    outputs3 = model(input_ids=second_tok_tensor, past_key_values=second_past_kv, use_cache=True)
                    third_logits = outputs3.logits[0, -1, :]
                    third_log_probs = torch.nn.functional.log_softmax(third_logits, dim=-1)

                    for idx, seq in group3:
                        score = first_tok_log_prob + second_tok_log_prob
                        if len(seq) > 2:
                            score += third_log_probs[seq[2]].item()
                        # For 4+ tokens (1000+), just use first 3 tokens as approximation
                        scores[idx] = score

    return scores


def auc_score(labels: List[int], scores: List[float]) -> float:
    """Compute AUC score using sklearn's roc_auc_score."""
    pos = sum(labels)
    if pos == 0 or pos == len(labels):
        return 0.5
    return roc_auc_score(labels, scores)


def mrr_score(labels: List[int], scores: List[float]) -> float:
    """Compute MRR score (Mean Reciprocal Rank)."""
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rr_scores = []
    for rank, idx in enumerate(sorted_idx, start=1):
        if labels[idx] == 1:
            rr_scores.append(1.0 / rank)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def ndcg_score(labels: List[int], scores: List[float], k: int) -> float:
    """Compute nDCG@k score."""
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    dcg = 0.0
    for rank, idx in enumerate(sorted_idx[:k], start=1):
        if labels[idx] == 1:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(sum(labels), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--max_history", type=int, default=0, help="Max history items (0=unlimited)")
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", help="Output prediction file for MIND leaderboard")
    parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention 2 for faster inference")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading news from: {args.news_path}")
    news = load_news(args.news_path, args.use_abstract)
    print(f"✓ Loaded {len(news)} news articles")

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load model with optional Flash Attention 2
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("✓ Using Flash Attention 2")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, **model_kwargs
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"✓ Model loaded on device: {device}")

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    aucs = []
    mrrs = []
    ndcg5 = []
    ndcg10 = []
    predictions = []

    # Count total lines
    import subprocess
    total_lines = None
    if not args.max_impressions:
        try:
            total_lines = int(subprocess.check_output(['wc', '-l', args.behaviors_path]).split()[0])
        except:
            pass

    total_to_process = total_lines or args.max_impressions or None

    count = 0
    skipped_malformed = 0

    print(f"\nEvaluating with multiple-choice format (numeric options)...")
    print(f"Use abstract: {args.use_abstract}")
    print(f"Format: Candidate News Articles:\\n1. [Title] {'Title Abstract' if args.use_abstract else 'Title'} (Category)\\n...\\nAnswer: N")
    print()

    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=total_to_process, desc="Evaluating impressions", unit="impression")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                skipped_malformed += 1
                continue

            impression_id = parts[0]
            history_ids = parts[3].split()
            # Use all history if max_history is 0 or not set
            if args.max_history > 0:
                history_ids = history_ids[-args.max_history:]
            impressions = parts[4].split()

            labels = []
            candidate_objs = []
            candidate_ids = []
            missing_count = 0

            for imp in impressions:
                if "-" not in imp:
                    continue
                nid, label = imp.rsplit("-", 1)

                # Use placeholder for missing news
                if nid not in news:
                    candidate_objs.append({'text': '[MISSING_NEWS]', 'category': ''})
                    missing_count += 1
                else:
                    candidate_objs.append(news[nid])

                candidate_ids.append(nid)
                labels.append(int(label))

            # Skip if no candidates
            if not candidate_objs:
                continue

            # Build history (pass full news objects to include category)
            history_objs = [news[nid] for nid in history_ids if nid in news]

            # Build multiple-choice prompt and score
            prompt = build_multiple_choice_prompt(history_objs, candidate_objs)
            scores = score_candidates_multiple_choice(
                model, tokenizer, prompt, len(candidate_objs), device
            )

            # Compute metrics (only if we have positive labels)
            if sum(labels) > 0:
                aucs.append(auc_score(labels, scores))
                mrrs.append(mrr_score(labels, scores))
                ndcg5.append(ndcg_score(labels, scores, 5))
                ndcg10.append(ndcg_score(labels, scores, 10))

            # Generate ranked predictions
            if args.output_file:
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                ranked_news_ids = [candidate_ids[i] for i in ranked_indices]
                predictions.append((impression_id, ranked_news_ids))

            count += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'AUC': f'{_avg(aucs):.4f}',
                'MRR': f'{_avg(mrrs):.4f}',
                'nDCG@5': f'{_avg(ndcg5):.4f}',
                'nDCG@10': f'{_avg(ndcg10):.4f}'
            })

            if args.max_impressions and count >= args.max_impressions:
                break

        pbar.close()

    print("\nMIND Evaluation (Ranking-Aware, Numeric Options)")
    print(f"Impressions processed: {count}")
    if skipped_malformed > 0:
        print(f"⚠️  Skipped malformed lines: {skipped_malformed}")

    if aucs:
        print(f"AUC:  {_avg(aucs):.4f}")
        print(f"MRR:  {_avg(mrrs):.4f}")
        print(f"nDCG@5:  {_avg(ndcg5):.4f}")
        print(f"nDCG@10: {_avg(ndcg10):.4f}")
    else:
        print("No metrics computed (test set has no labels)")

    # Write predictions
    if args.output_file and predictions:
        print(f"\nWriting predictions to: {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for impression_id, ranked_news_ids in predictions:
                f.write(f"{impression_id} {' '.join(ranked_news_ids)}\n")
        print(f"✓ Wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main()
