"""
Cascade/Reranking Evaluation for MIND.

Two-stage evaluation:
1. Stage 1 (Filter): Point-wise model scores ALL candidates, select top-K
2. Stage 2 (Rerank): Ranking model reranks the top-K candidates

Benefits:
- Point-wise is fast for initial filtering (O(n) single forward passes)
- Ranking model sees fewer candidates, can make better comparisons
- Combines efficiency of point-wise with accuracy of ranking

Usage:
    python evaluate_mind_cascade.py \
        --pointwise_model output_dir/sft_mind_pointwise_*/final_checkpoint \
        --ranking_model output_dir/sft_mind_ranking_*/final_checkpoint \
        --behaviors_path ../data/MIND/dev/behaviors.tsv \
        --news_path ../data/MIND/dev/news.tsv \
        --top_k 10
"""

import argparse
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

            news[news_id] = {'text': text, 'category': category}
    return news


# ============== Point-wise scoring ==============

def build_pointwise_prompt(history: List[dict], candidate: dict) -> str:
    prompt = "Role: You are a news recommendation assistant.\n"
    prompt += "Task: Determine if the candidate article matches the user's interests.\n\n"

    prompt += "User History:\n"
    if history:
        for i, h in enumerate(history, 1):
            category = f" ({h.get('category', '')})" if h.get('category') else ""
            prompt += f"{i}. [Title] {h['text']}{category}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\nCandidate Article:\n"
    category = f" ({candidate.get('category', '')})" if candidate.get('category') else ""
    prompt += f"[Title] {candidate['text']}{category}\n"

    prompt += "\nBased on the user's reading history, is this article relevant to them?\n"
    prompt += "Answer with Yes or No.\n\nAnswer:"

    return prompt


def score_pointwise_batch(
    model, tokenizer, history, candidates, device,
    yes_token_id, no_token_id, batch_size=8
) -> List[float]:
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

            batch_scores = (yes_log_probs - no_log_probs).cpu().tolist()
            scores.extend(batch_scores)

    return scores


# ============== Ranking scoring ==============

def build_ranking_prompt(history: List[dict], candidates: List[dict]) -> str:
    prompt = "Role: You are a news recommendation assistant.\n"
    prompt += "Task: Select the article that best matches the user's reading interests.\n\n"

    prompt += "User History:\n"
    if history:
        for i, h in enumerate(history, 1):
            category = f" ({h.get('category', '')})" if h.get('category') else ""
            prompt += f"{i}. [Title] {h['text']}{category}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\nCandidate Articles:\n"
    for i, cand in enumerate(candidates, 1):
        category = f" ({cand.get('category', '')})" if cand.get('category') else ""
        prompt += f"{i}. [Title] {cand['text']}{category}\n"

    prompt += "\nWhich article number would this user most likely click on?\n"
    prompt += "Answer with the article number only.\n\nAnswer:"

    return prompt


def score_ranking(model, tokenizer, history, candidates, device) -> List[float]:
    """Score candidates using ranking model."""
    prompt = build_ranking_prompt(history, candidates)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    num_candidates = len(candidates)
    number_token_ids = []
    for i in range(1, num_candidates + 1):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        number_token_ids.append(tokens[0] if tokens else tokenizer.unk_token_id)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        scores = [log_probs[tid].item() for tid in number_token_ids]

    return scores


# ============== Metrics ==============

def auc_score(labels, scores):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise_model", required=True)
    parser.add_argument("--ranking_model", required=True)
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--max_history", type=int, default=0)
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=10, help="Number of candidates to pass to ranking stage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--output_file", help="Output prediction file")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 60)
    print("MIND Cascade/Reranking Evaluation")
    print("=" * 60)
    print(f"Point-wise model: {args.pointwise_model}")
    print(f"Ranking model: {args.ranking_model}")
    print(f"Top-K for reranking: {args.top_k}")
    print()

    # Load news
    print(f"Loading news...")
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

    yes_tokens = pw_tokenizer.encode(" Yes", add_special_tokens=False)
    no_tokens = pw_tokenizer.encode(" No", add_special_tokens=False)
    yes_token_id = yes_tokens[0] if len(yes_tokens) == 1 else yes_tokens[-1]
    no_token_id = no_tokens[0] if len(no_tokens) == 1 else no_tokens[-1]

    print(f"\nLoading ranking model...")
    rk_tokenizer = AutoTokenizer.from_pretrained(args.ranking_model, trust_remote_code=True)
    rk_tokenizer.pad_token = rk_tokenizer.eos_token
    rk_tokenizer.pad_token_id = rk_tokenizer.eos_token_id
    rk_tokenizer.padding_side = "left"
    rk_model = AutoModelForCausalLM.from_pretrained(args.ranking_model, **model_kwargs)
    rk_model.eval()
    rk_device = next(rk_model.parameters()).device

    print(f"\nStarting cascade evaluation...")

    # Metrics
    aucs, mrrs, ndcg5, ndcg10 = [], [], [], []
    # For comparison
    pw_only_aucs, pw_only_mrrs = [], []
    predictions = []

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    count = 0
    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total = min(len(lines), args.max_impressions) if args.max_impressions else len(lines)

        for line in tqdm(lines[:total], desc="Cascade evaluation"):
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

            # ===== STAGE 1: Point-wise filtering =====
            pw_scores = score_pointwise_batch(
                pw_model, pw_tokenizer, history_objs, candidate_objs,
                pw_device, yes_token_id, no_token_id, args.batch_size
            )

            # Track point-wise only performance
            pw_only_aucs.append(auc_score(labels, pw_scores))
            pw_only_mrrs.append(mrr_score(labels, pw_scores))

            # Select top-K indices based on point-wise scores
            top_k = min(args.top_k, len(candidate_objs))
            top_k_indices = sorted(range(len(pw_scores)), key=lambda i: pw_scores[i], reverse=True)[:top_k]

            # ===== STAGE 2: Ranking reranking =====
            top_k_candidates = [candidate_objs[i] for i in top_k_indices]
            top_k_labels = [labels[i] for i in top_k_indices]
            top_k_ids = [candidate_ids[i] for i in top_k_indices]

            # Get ranking scores for top-K
            rk_scores = score_ranking(rk_model, rk_tokenizer, history_objs, top_k_candidates, rk_device)

            # Build final scores: top-K get ranking scores, rest get low point-wise scores
            final_scores = []
            top_k_set = set(top_k_indices)

            # Assign scores: top-K items get reranked, others keep original PW scores (shifted down)
            min_rk_score = min(rk_scores) if rk_scores else 0
            for i in range(len(candidate_objs)):
                if i in top_k_set:
                    # Map index to top-K position
                    top_k_pos = top_k_indices.index(i)
                    final_scores.append(rk_scores[top_k_pos])
                else:
                    # Items not in top-K get their PW score minus offset to rank below top-K
                    final_scores.append(pw_scores[i] + min_rk_score - 100)

            # Compute metrics
            aucs.append(auc_score(labels, final_scores))
            mrrs.append(mrr_score(labels, final_scores))
            ndcg5.append(ndcg_score(labels, final_scores, 5))
            ndcg10.append(ndcg_score(labels, final_scores, 10))

            if args.output_file:
                ranked_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
                ranked_news_ids = [candidate_ids[i] for i in ranked_indices]
                predictions.append((impression_id, ranked_news_ids))

            count += 1

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\nImpressions processed: {count}")
    print(f"Top-K for reranking: {args.top_k}")

    print(f"\n--- Point-wise Only (Baseline) ---")
    print(f"AUC:     {_avg(pw_only_aucs):.4f}")
    print(f"MRR:     {_avg(pw_only_mrrs):.4f}")

    print(f"\n--- Cascade (PW Filter + Ranking Rerank) ---")
    print(f"AUC:     {_avg(aucs):.4f}")
    print(f"MRR:     {_avg(mrrs):.4f}")
    print(f"nDCG@5:  {_avg(ndcg5):.4f}")
    print(f"nDCG@10: {_avg(ndcg10):.4f}")

    # Check improvement
    if _avg(mrrs) > _avg(pw_only_mrrs):
        improvement = (_avg(mrrs) - _avg(pw_only_mrrs)) / _avg(pw_only_mrrs) * 100
        print(f"\n[IMPROVEMENT] MRR improved by {improvement:.2f}% with cascade!")

    if args.output_file and predictions:
        print(f"\nWriting predictions to: {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for impression_id, ranked_news_ids in predictions:
                f.write(f"{impression_id} {' '.join(ranked_news_ids)}\n")


if __name__ == "__main__":
    main()
