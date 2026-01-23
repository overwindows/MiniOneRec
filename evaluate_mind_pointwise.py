"""
Evaluate MIND models trained with point-wise SFT (Yes/No classification).

This evaluation scores each candidate independently by computing P("Yes" | prompt)
and uses those scores to rank candidates within each impression.

Key Features:
- Point-wise scoring: Each candidate evaluated independently
- Scores P(" Yes") vs P(" No") for each (history, candidate) pair
- Ranks candidates by Yes probability
- Uses official MIND metrics (AUC, MRR, nDCG@5, nDCG@10)
- Supports Flash Attention 2 for faster inference

Usage:
    python evaluate_mind_pointwise.py \
        --model_path output_dir/sft_mind_pointwise_*/final_checkpoint \
        --behaviors_path ../data/MIND/dev/behaviors.tsv \
        --news_path ../data/MIND/dev/news.tsv \
        --flash_attn \
        --max_impressions 1000  # Optional: for quick testing
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
                'text': text,
                'category': category
            }
    return news


def build_pointwise_prompt(history: List[dict], candidate: dict) -> str:
    """
    Build point-wise prompt matching training format.

    Returns the prompt ending with "Answer:"
    """
    prompt = "Role: You are a news recommendation assistant.\n"
    prompt += "Task: Determine if the candidate article matches the user's interests.\n\n"

    # User history
    prompt += "User History:\n"
    if history:
        for i, h in enumerate(history, 1):
            category = f" ({h.get('category', '')})" if h.get('category') else ""
            prompt += f"{i}. [Title] {h['text']}{category}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\n"

    # Candidate article
    prompt += "Candidate Article:\n"
    category = f" ({candidate.get('category', '')})" if candidate.get('category') else ""
    prompt += f"[Title] {candidate['text']}{category}\n"

    prompt += "\n"
    prompt += "Based on the user's reading history, is this article relevant to them?\n"
    prompt += "Answer with Yes or No.\n\n"
    prompt += "Answer:"

    return prompt


def score_candidate_pointwise(
    model,
    tokenizer,
    prompt: str,
    device,
    yes_token_id: int,
    no_token_id: int,
) -> float:
    """
    Score a single candidate using point-wise Yes/No classification.

    Returns:
        Score = log P(" Yes") - log P(" No")
        Higher score = more likely relevant
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        yes_log_prob = log_probs[yes_token_id].item()
        no_log_prob = log_probs[no_token_id].item()

        # Return difference (log odds ratio)
        # This gives a score where higher = more likely Yes
        score = yes_log_prob - no_log_prob

    return score


def batch_score_candidates_pointwise(
    model,
    tokenizer,
    history: List[dict],
    candidates: List[dict],
    device,
    yes_token_id: int,
    no_token_id: int,
    batch_size: int = 8,
) -> List[float]:
    """
    Score multiple candidates in batches for efficiency.

    Returns:
        List of scores for each candidate
    """
    # Build all prompts
    prompts = [build_pointwise_prompt(history, cand) for cand in candidates]

    # Tokenize all prompts
    all_prompt_ids = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]

    scores = []

    # Process in batches
    for batch_start in range(0, len(all_prompt_ids), batch_size):
        batch_ids = all_prompt_ids[batch_start:batch_start + batch_size]

        # Pad to same length
        max_len = max(len(ids) for ids in batch_ids)
        padded_ids = []
        attention_masks = []

        for ids in batch_ids:
            pad_len = max_len - len(ids)
            # Left padding
            padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Get logits at last position (after "Answer:")
            logits = outputs.logits[:, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            yes_log_probs = log_probs[:, yes_token_id]
            no_log_probs = log_probs[:, no_token_id]

            batch_scores = (yes_log_probs - no_log_probs).cpu().tolist()
            scores.extend(batch_scores)

    return scores


def auc_score(labels: List[int], scores: List[float]) -> float:
    """Compute AUC score using sklearn's roc_auc_score."""
    pos = sum(labels)
    if pos == 0 or pos == len(labels):
        return 0.5
    return roc_auc_score(labels, scores)


def mrr_score(y_true, y_score):
    """
    MRR (Mean Reciprocal Rank) - Official MIND implementation.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0


def dcg_score(y_true, y_score, k=10):
    """
    DCG (Discounted Cumulative Gain) - Official MIND implementation.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """
    nDCG (Normalized Discounted Cumulative Gain) - Official MIND implementation.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--max_history", type=int, default=0, help="Max history items (0=unlimited)")
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for scoring candidates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", help="Output prediction file for MIND leaderboard")
    parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention 2")
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

    # Get Yes/No token IDs
    yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode(" No", add_special_tokens=False)

    # Handle tokenizers that might split differently
    yes_token_id = yes_tokens[0] if len(yes_tokens) == 1 else yes_tokens[-1]
    no_token_id = no_tokens[0] if len(no_tokens) == 1 else no_tokens[-1]

    print(f"Yes token ID: {yes_token_id} ('{tokenizer.decode([yes_token_id])}')")
    print(f"No token ID: {no_token_id} ('{tokenizer.decode([no_token_id])}')")

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

    print(f"\nEvaluating with point-wise format (Yes/No)...")
    print(f"Use abstract: {args.use_abstract}")
    print(f"Max history: {'unlimited' if args.max_history == 0 else args.max_history}")
    print(f"Batch size: {args.batch_size}")
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
            if args.max_history > 0:
                history_ids = history_ids[-args.max_history:]
            impressions = parts[4].split()

            labels = []
            candidate_objs = []
            candidate_ids = []

            for imp in impressions:
                if "-" not in imp:
                    continue
                nid, label = imp.rsplit("-", 1)

                if nid not in news:
                    candidate_objs.append({'text': '[MISSING_NEWS]', 'category': ''})
                else:
                    candidate_objs.append(news[nid])

                candidate_ids.append(nid)
                labels.append(int(label))

            # Skip if no candidates
            if not candidate_objs:
                continue

            # Build history
            history_objs = [news[nid] for nid in history_ids if nid in news]

            # Score all candidates using batched point-wise scoring
            scores = batch_score_candidates_pointwise(
                model, tokenizer, history_objs, candidate_objs, device,
                yes_token_id, no_token_id, args.batch_size
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

    print("\nMIND Evaluation (Point-wise, Yes/No)")
    print(f"Impressions processed: {count}")
    if skipped_malformed > 0:
        print(f"⚠️  Skipped malformed lines: {skipped_malformed}")

    if aucs:
        print(f"AUC:     {_avg(aucs):.4f}")
        print(f"MRR:     {_avg(mrrs):.4f}")
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
