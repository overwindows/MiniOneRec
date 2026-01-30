"""
Evaluate MIND via cloud LLM (SambaNova) with prompt-based scoring.

Modes:
- pointwise: ask Yes/No for each candidate; score Yes=1, No=0.
- listwise: ask for a full ranked list of option numbers.

No logits/probs are used.
"""

import argparse
import json
import math
import os
import random
import re
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

            news[news_id] = {"title": title, "text": text, "category": category}
    return news


def build_pointwise_prompt(history: List[dict], candidate: dict) -> str:
    prompt = "A user read these news articles:\n"
    if history:
        recent_history = history[-30:] if len(history) > 30 else history
        for i, h in enumerate(recent_history, 1):
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


def build_listwise_prompt(history: List[dict], candidates: List[dict]) -> str:
    prompt = "A user read these news articles:\n"
    if history:
        recent_history = history[-30:] if len(history) > 30 else history
        for i, h in enumerate(recent_history, 1):
            cat = h.get("category", "General")
            prompt += f"{i}. [{cat}] {h['text']}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\n"
    prompt += "Candidate articles:\n"
    for i, cand in enumerate(candidates):
        option_num = i + 1
        cat = cand.get("category", "General")
        prompt += f"{option_num}. [{cat}] {cand['text']}\n"

    prompt += "\n"
    prompt += "Rank ALL candidate articles from most to least likely to be read by this user.\n\n"
    prompt += "Answer:"
    return prompt


def _get_sambanova_client(api_key: str, base_url: str):
    try:
        from sambanova import SambaNova
    except Exception as exc:
        raise SystemExit("SambaNova client not available. Install with `pip install sambanova`.") from exc

    return SambaNova(api_key=api_key, base_url=base_url)


def _parse_ranked_numbers(text: str, num_candidates: int) -> List[int]:
    order = []
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("ranking"), list):
            order = [int(n) for n in payload["ranking"]]
    except Exception:
        order = []

    if not order:
        order = [int(n) for n in re.findall(r"\d+", text)]

    parsed = order
    order = []
    seen = set()
    for n in parsed:
        if 1 <= n <= num_candidates and n not in seen:
            order.append(n)
            seen.add(n)
    missing = [i for i in range(1, num_candidates + 1) if i not in seen]
    if missing:
        print(
            f"Warning: {len(missing)} candidates missing from ranking; appending at end.",
            file=sys.stderr,
        )
    order.extend(missing)
    return order


def score_candidate_pointwise(
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
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt + "\nAnswer with Yes or No only."},
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


def score_candidates_pointwise(
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
        score = score_candidate_pointwise(
            client,
            model,
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        scores.append(score)
    return scores


def score_candidates_listwise(
    client,
    model: str,
    prompt: str,
    num_candidates: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> List[float]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that ranks news articles for users. "
                    "Always respond with valid JSON in the exact format: {\"ranking\":[...]} "
                    "where the list contains ALL option numbers ranked from most to least relevant."
                )
            },
            {
                "role": "user",
                "content": (
                    prompt
                    + f"\n\nYou MUST return valid JSON containing exactly {num_candidates} numbers (1 to {num_candidates}). "
                    "Format: {\"ranking\":[3,1,5,2,4,...]} where the first number is the BEST article. "
                    "Include every option number exactly once. Return ONLY the JSON, no explanations."
                ),
            },
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content.strip()

    # DEBUG: Print model responses to diagnose issues
    import os
    if os.getenv("DEBUG_CLOUD_EVAL") == "1":
        print(f"\n{'='*60}")
        print(f"Candidates: {num_candidates} | Response length: {len(content)}")
        print(f"Response: {content[:800]}")  # First 800 chars
        print(f"{'='*60}\n")

    order = _parse_ranked_numbers(content, num_candidates)

    scores = [0.0] * num_candidates
    for rank, option in enumerate(order):
        scores[option - 1] = float(num_candidates - rank)
    return scores


def auc_score(labels: List[int], scores: List[float]) -> float:
    pos = sum(labels)
    if pos == 0 or pos == len(labels):
        return 0.5
    return roc_auc_score(labels, scores)


def mrr_score(labels: List[int], scores: List[float]) -> float:
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rr_scores = []
    for rank, idx in enumerate(sorted_idx, start=1):
        if labels[idx] == 1:
            rr_scores.append(1.0 / rank)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def ndcg_score(labels: List[int], scores: List[float], k: int) -> float:
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    dcg = 0.0
    for rank, idx in enumerate(sorted_idx[:k], start=1):
        if labels[idx] == 1:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(sum(labels), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pointwise", "listwise"], required=True)
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
    parser.add_argument("--max_tokens", type=int, default=0, help="0=auto by mode")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("SAMBANOVA_API_KEY")
    if not api_key:
        raise SystemExit("Missing SambaNova API key. Set SAMBANOVA_API_KEY or pass --api_key.")

    if args.max_tokens == 0:
        args.max_tokens = 1 if args.mode == "pointwise" else 512

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

    print(f"\nEvaluating in {args.mode} mode via SambaNova...")
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

            if args.mode == "pointwise":
                scores = score_candidates_pointwise(
                    client,
                    args.model,
                    history_objs,
                    candidate_objs,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
            else:
                prompt = build_listwise_prompt(history_objs, candidate_objs)
                scores = score_candidates_listwise(
                    client,
                    args.model,
                    prompt,
                    len(candidate_objs),
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
        print(f"AUC:  {_avg(aucs):.4f}")
        print(f"MRR:  {_avg(mrrs):.4f}")
        print(f"nDCG@5:  {_avg(ndcg5):.4f}")
        print(f"nDCG@10: {_avg(ndcg10):.4f}")
    else:
        print("No metrics computed (test set has no labels)")

    if skipped_malformed > 0:
        print(f"Warning: Skipped {skipped_malformed} malformed lines")


if __name__ == "__main__":
    main()
