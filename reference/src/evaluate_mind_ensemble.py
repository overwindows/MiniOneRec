"""
Evaluate MIND using Score Ensemble of Point-wise and Ranking-aware models.

This script combines scores from two models:
- Point-wise model: Scores P(" Yes") for each candidate independently
- Ranking model: Scores based on position logits for each candidate

Final score = alpha * pointwise_score + (1 - alpha) * ranking_score

Usage:
    python evaluate_mind_ensemble.py \
        --pointwise_model output_dir/sft_mind_pointwise_*/final_checkpoint \
        --ranking_model output_dir/sft_mind_ranking_*/final_checkpoint \
        --behaviors_path ../data/MIND/dev/behaviors.tsv \
        --news_path ../data/MIND/dev/news.tsv \
        --alpha 0.5
"""

import argparse
import random
from typing import List, Dict, Tuple

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
    """Load news articles from news.tsv"""
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


# ============== Point-wise scoring ==============

def build_pointwise_prompt(history: List[dict], candidate: dict) -> str:
    """
    Build OPTIMIZED point-wise prompt matching training format.

    Based on Prompt4NR research (arXiv:2304.05263):
    - Concise format saves ~20 tokens
    - Category in [brackets] at start for better visibility
    - Natural language question
    - Limit to last 30 history items for focus
    """
    prompt = "A user read these news articles:\n"

    # User history
    if history:
        for i, h in enumerate(history, 1):
            cat = h.get('category', 'General')
            prompt += f"{i}. [{cat}] {h['text']}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\n"

    # Candidate article - category first in brackets
    prompt += "Candidate article:\n"
    cat = candidate.get('category', 'General')
    prompt += f"[{cat}] {candidate['text']}\n"

    prompt += "\n"
    # Simple, natural question
    prompt += "Will this user read this article? Answer:"

    return prompt


def score_pointwise_batch(
    model,
    tokenizer,
    history: List[dict],
    candidates: List[dict],
    device,
    yes_token_id: int,
    no_token_id: int,
    batch_size: int = 8,
) -> List[float]:
    """Score candidates using point-wise Yes/No classification."""
    prompts = [build_pointwise_prompt(history, cand) for cand in candidates]
    all_prompt_ids = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]

    scores = []

    for batch_start in range(0, len(all_prompt_ids), batch_size):
        batch_ids = all_prompt_ids[batch_start:batch_start + batch_size]

        max_len = max(len(ids) for ids in batch_ids)
        padded_ids = []
        attention_masks = []

        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            yes_log_probs = log_probs[:, yes_token_id]
            no_log_probs = log_probs[:, no_token_id]

            # Log odds ratio
            batch_scores = (yes_log_probs - no_log_probs).cpu().tolist()
            scores.extend(batch_scores)

    return scores


# ============== Ranking-aware scoring ==============

def build_ranking_prompt(history: List[dict], candidates: List[dict]) -> str:
    """
    Build OPTIMIZED multiple-choice ranking prompt matching training format.

    Based on Prompt4NR research (arXiv:2304.05263):
    - Concise format saves ~20 tokens
    - Category in [brackets] at start for better visibility
    - Natural language question
    - Limit to last 30 history items for focus
    """
    prompt = "A user read these news articles:\n"

    # User history
    if history:
        for i, h in enumerate(history, 1):
            cat = h.get('category', 'General')
            prompt += f"{i}. [{cat}] {h['text']}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\n"

    # Candidate articles - category first in brackets
    prompt += "Candidate articles:\n"
    for i, cand in enumerate(candidates, 1):
        cat = cand.get('category', 'General')
        prompt += f"{i}. [{cat}] {cand['text']}\n"

    prompt += "\n"
    # Simple, direct question
    prompt += "Which article will this user read? Answer with the number.\n\n"
    prompt += "Answer:"

    return prompt


def score_ranking(
    model,
    tokenizer,
    history: List[dict],
    candidates: List[dict],
    device,
) -> List[float]:
    """Score candidates using ranking model's logits for each position."""
    prompt = build_ranking_prompt(history, candidates)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Get token IDs for numbers 1, 2, 3, ...
    num_candidates = len(candidates)
    number_token_ids = []
    for i in range(1, num_candidates + 1):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        number_token_ids.append(tokens[0] if tokens else tokenizer.unk_token_id)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]

        # Get log probabilities for each number
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        scores = [log_probs[tid].item() for tid in number_token_ids]

    return scores


# ============== Metrics ==============

def auc_score(labels: List[int], scores: List[float]) -> float:
    pos = sum(labels)
    if pos == 0 or pos == len(labels):
        return 0.5
    return roc_auc_score(labels, scores)


def mrr_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0


def dcg_score(y_true, y_score, k=10):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range."""
    scores = np.array(scores)
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-8:
        return [0.5] * len(scores)
    return ((scores - min_s) / (max_s - min_s)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise_model", required=True, help="Path to point-wise model")
    parser.add_argument("--ranking_model", required=True, help="Path to ranking model")
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--max_history", type=int, default=0)
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for pointwise scores (0-1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--output_file", help="Output prediction file")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 50)
    print("MIND Score Ensemble Evaluation")
    print("=" * 50)
    print(f"Point-wise model: {args.pointwise_model}")
    print(f"Ranking model: {args.ranking_model}")
    print(f"Alpha (pointwise weight): {args.alpha}")
    print(f"Use abstract: {args.use_abstract}")
    print()

    # Load news
    print(f"Loading news from: {args.news_path}")
    news = load_news(args.news_path, args.use_abstract)
    print(f"Loaded {len(news)} news articles")

    # Load models
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"\nLoading point-wise model...")
    pw_tokenizer = AutoTokenizer.from_pretrained(args.pointwise_model, trust_remote_code=True)
    pw_tokenizer.pad_token = pw_tokenizer.eos_token
    pw_tokenizer.pad_token_id = pw_tokenizer.eos_token_id
    pw_tokenizer.padding_side = "left"
    pw_model = AutoModelForCausalLM.from_pretrained(args.pointwise_model, **model_kwargs)
    pw_model.eval()
    pw_device = next(pw_model.parameters()).device

    # Get Yes/No token IDs
    yes_tokens = pw_tokenizer.encode(" Yes", add_special_tokens=False)
    no_tokens = pw_tokenizer.encode(" No", add_special_tokens=False)
    yes_token_id = yes_tokens[0] if len(yes_tokens) == 1 else yes_tokens[-1]
    no_token_id = no_tokens[0] if len(no_tokens) == 1 else no_tokens[-1]
    print(f"Yes token: {yes_token_id} ('{pw_tokenizer.decode([yes_token_id])}')")
    print(f"No token: {no_token_id} ('{pw_tokenizer.decode([no_token_id])}')")

    print(f"\nLoading ranking model...")
    rk_tokenizer = AutoTokenizer.from_pretrained(args.ranking_model, trust_remote_code=True)
    rk_tokenizer.pad_token = rk_tokenizer.eos_token
    rk_tokenizer.pad_token_id = rk_tokenizer.eos_token_id
    rk_tokenizer.padding_side = "left"
    rk_model = AutoModelForCausalLM.from_pretrained(args.ranking_model, **model_kwargs)
    rk_model.eval()
    rk_device = next(rk_model.parameters()).device

    print(f"\nModels loaded. Starting evaluation...")

    # Metrics
    aucs, mrrs, ndcg5, ndcg10 = [], [], [], []
    pw_aucs, pw_mrrs = [], []  # Track individual model performance
    rk_aucs, rk_mrrs = [], []
    predictions = []

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    count = 0
    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total = min(len(lines), args.max_impressions) if args.max_impressions else len(lines)

        for line in tqdm(lines[:total], desc="Evaluating"):
            parts = line.strip().split("\t")
            if len(parts) < 5:
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
                if nid in news:
                    candidate_objs.append(news[nid])
                else:
                    candidate_objs.append({'text': '[MISSING]', 'category': ''})
                candidate_ids.append(nid)
                labels.append(int(label))

            if not candidate_objs or sum(labels) == 0:
                continue

            history_objs = [news[nid] for nid in history_ids if nid in news]

            # Get point-wise scores
            pw_scores = score_pointwise_batch(
                pw_model, pw_tokenizer, history_objs, candidate_objs,
                pw_device, yes_token_id, no_token_id, args.batch_size
            )

            # Get ranking scores
            rk_scores = score_ranking(
                rk_model, rk_tokenizer, history_objs, candidate_objs, rk_device
            )

            # Normalize both to [0, 1]
            pw_norm = normalize_scores(pw_scores)
            rk_norm = normalize_scores(rk_scores)

            # Ensemble
            ensemble_scores = [
                args.alpha * pw + (1 - args.alpha) * rk
                for pw, rk in zip(pw_norm, rk_norm)
            ]

            # Track individual model performance
            if sum(labels) > 0:
                pw_aucs.append(auc_score(labels, pw_scores))
                pw_mrrs.append(mrr_score(labels, pw_scores))
                rk_aucs.append(auc_score(labels, rk_scores))
                rk_mrrs.append(mrr_score(labels, rk_scores))

            # Compute ensemble metrics
            aucs.append(auc_score(labels, ensemble_scores))
            mrrs.append(mrr_score(labels, ensemble_scores))
            ndcg5.append(ndcg_score(labels, ensemble_scores, 5))
            ndcg10.append(ndcg_score(labels, ensemble_scores, 10))

            if args.output_file:
                ranked_indices = sorted(range(len(ensemble_scores)), key=lambda i: ensemble_scores[i], reverse=True)
                ranked_news_ids = [candidate_ids[i] for i in ranked_indices]
                predictions.append((impression_id, ranked_news_ids))

            count += 1

    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    print(f"\nImpressions processed: {count}")
    print(f"Alpha (pointwise weight): {args.alpha}")

    print(f"\n--- Individual Model Performance ---")
    print(f"Point-wise only:  AUC={_avg(pw_aucs):.4f}, MRR={_avg(pw_mrrs):.4f}")
    print(f"Ranking only:     AUC={_avg(rk_aucs):.4f}, MRR={_avg(rk_mrrs):.4f}")

    print(f"\n--- Ensemble Performance ---")
    print(f"AUC:     {_avg(aucs):.4f}")
    print(f"MRR:     {_avg(mrrs):.4f}")
    print(f"nDCG@5:  {_avg(ndcg5):.4f}")
    print(f"nDCG@10: {_avg(ndcg10):.4f}")

    # Check if ensemble is better
    pw_auc_avg = _avg(pw_aucs)
    rk_auc_avg = _avg(rk_aucs)
    ens_auc_avg = _avg(aucs)

    if ens_auc_avg > max(pw_auc_avg, rk_auc_avg):
        print(f"\n[IMPROVEMENT] Ensemble AUC ({ens_auc_avg:.4f}) beats both individual models!")
    else:
        print(f"\n[INFO] Best individual model AUC: {max(pw_auc_avg, rk_auc_avg):.4f}")

    if args.output_file and predictions:
        print(f"\nWriting predictions to: {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for impression_id, ranked_news_ids in predictions:
                f.write(f"{impression_id} {' '.join(ranked_news_ids)}\n")


if __name__ == "__main__":
    main()
