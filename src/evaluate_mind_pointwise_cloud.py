"""
Evaluate MIND models via cloud LLM (SambaNova) with point-wise Yes/No prompts.

This script does NOT require logits/logprobs. It asks the model to answer
Yes/No and uses 1.0/0.0 as scores for ranking within each impression.
"""

import argparse
import math
import os
import random
import sys
from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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

            news[news_id] = {"text": text, "category": category}
    return news


def build_pointwise_prompt(history: List[dict], candidate: dict) -> str:
    prompt = "A user read these news articles:\n"

    if history:
        for i, h in enumerate(history, 1):
            cat = h.get("category", "General")
            prompt += f"{i}. [{cat}] {h['text']}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\n"
    prompt += "Candidate article:\n"
    cat = candidate.get("category", "General")
    prompt += f"[{cat}] {candidate['text']}\n"

    prompt += "\n"
    prompt += "Will this user read this article? Answer:"
    return prompt


def _get_sambanova_client(api_key: str, base_url: str):
    try:
        from sambanova import SambaNova
    except Exception as exc:
        raise SystemExit("SambaNova client not available. Install with `pip install sambanova`.") from exc

    return SambaNova(api_key=api_key, base_url=base_url)


def score_candidate_pointwise_sambanova(
    client,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> float:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a news recommendation assistant. Based on a user's reading history, predict whether they will read a given article. Each article includes its category and title. Answer with Yes or No."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content.strip().lower()
    if content.startswith("yes"):
        return 1.0
    if content.startswith("no"):
        return 0.0
    print(f"Warning: unexpected response: {content}", file=sys.stderr)
    return 0.0


def score_candidates_pointwise_sambanova(
    client,
    model: str,
    history: List[dict],
    candidates: List[dict],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> List[float]:
    scores = []
    for cand in candidates:
        prompt = build_pointwise_prompt(history, cand)
        score = score_candidate_pointwise_sambanova(
            client,
            model,
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        scores.append(score)
    return scores


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="SambaNova model name")
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--max_history", type=int, default=0, help="Max history items (0=unlimited)")
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", help="Output prediction file for MIND leaderboard")
    parser.add_argument("--api_key", help="SambaNova API key (defaults to SAMBANOVA_API_KEY env var)")
    parser.add_argument("--base_url", default="https://api.sambanova.ai/v1")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=1)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("SAMBANOVA_API_KEY")
    if not api_key:
        raise SystemExit("Missing SambaNova API key. Set SAMBANOVA_API_KEY or pass --api_key.")

    set_seed(args.seed)

    print(f"Loading news from: {args.news_path}")
    news = load_news(args.news_path, args.use_abstract)
    print(f"✓ Loaded {len(news)} news articles")

    client = _get_sambanova_client(api_key, args.base_url)

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    aucs = []
    mrrs = []
    ndcg5 = []
    ndcg10 = []
    predictions = []

    import subprocess
    total_lines = None
    if not args.max_impressions:
        try:
            total_lines = int(subprocess.check_output(["wc", "-l", args.behaviors_path]).split()[0])
        except Exception:
            pass

    total_to_process = total_lines or args.max_impressions or None

    count = 0
    skipped_malformed = 0

    print("\nEvaluating with point-wise format (Yes/No) via SambaNova...")
    print(f"Use abstract: {args.use_abstract}")
    print(f"Max history: {'unlimited' if args.max_history == 0 else args.max_history}")
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
                    candidate_objs.append({"text": "[MISSING_NEWS]", "category": ""})
                else:
                    candidate_objs.append(news[nid])

                candidate_ids.append(nid)
                labels.append(int(label))

            if not candidate_objs:
                continue

            history_objs = [news[nid] for nid in history_ids if nid in news]

            scores = score_candidates_pointwise_sambanova(
                client,
                args.model,
                history_objs,
                candidate_objs,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )

            if sum(labels) > 0:
                aucs.append(auc_score(labels, scores))
                mrrs.append(mrr_score(labels, scores))
                ndcg5.append(ndcg_score(labels, scores, 5))
                ndcg10.append(ndcg_score(labels, scores, 10))

            if args.output_file:
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                ranked_news_ids = [candidate_ids[i] for i in ranked_indices]
                predictions.append((impression_id, ranked_news_ids))

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

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for impression_id, ranked_news_ids in predictions:
                ranked_str = " ".join(ranked_news_ids)
                f.write(f"{impression_id}\t[{ranked_str}]\n")
        print(f"\nSaved predictions to {args.output_file}")

    print("\nFinal Results:")
    if aucs:
        print(f"AUC:     {_avg(aucs):.4f}")
        print(f"MRR:     {_avg(mrrs):.4f}")
        print(f"nDCG@5:  {_avg(ndcg5):.4f}")
        print(f"nDCG@10: {_avg(ndcg10):.4f}")
    else:
        print("No metrics computed (test set has no labels)")

    if skipped_malformed > 0:
        print(f"Warning: Skipped {skipped_malformed} malformed lines")


if __name__ == "__main__":
    main()
