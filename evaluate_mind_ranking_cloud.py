"""
Evaluate MIND models via cloud LLM (SambaNova) with list-wise ranking prompts.

This script does NOT require logits/logprobs. It asks the model to return
an ordered list of option numbers and uses the order as ranking scores.
"""

import argparse
import json
import math
import os
import random
import re
import sys
from typing import List, Tuple

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


def build_multiple_choice_prompt(history: List[dict], candidates: List[dict]) -> str:
    """Standard prompt for ranking (asks for full ranking)."""
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
    prompt += "Which article will this user read? Answer with the number.\n\n"
    prompt += "Answer:"

    return prompt


def build_cot_prompt(history: List[dict], candidates: List[dict]) -> str:
    """
    Chain-of-Thought prompt that asks for reasoning before the answer.
    Typically produces better results with cloud LLMs.
    """
    lines = []

    lines.append("You are a news recommendation assistant.")
    lines.append("Your task is to predict which article a user will click based on their reading history.")
    lines.append("")

    # User history
    lines.append("=== User Reading History ===")
    if history:
        recent_history = history[-30:] if len(history) > 30 else history
        for i, h in enumerate(recent_history, 1):
            cat = f"[{h.get('category', 'General')}]" if h.get('category') else ""
            lines.append(f"{i}. {cat} {h['text']}")
    else:
        lines.append("(No reading history available)")
    lines.append("")

    # Candidates
    lines.append("=== Candidate Articles ===")
    for i, cand in enumerate(candidates, 1):
        cat = f"[{cand.get('category', 'General')}]" if cand.get('category') else ""
        lines.append(f"{i}. {cat} {cand['text']}")
    lines.append("")

    # CoT instruction
    lines.append("=== Instructions ===")
    lines.append("Think step by step:")
    lines.append("1. What topics/categories does this user seem interested in?")
    lines.append("2. Which candidate articles match these interests?")
    lines.append("3. Which ONE article is the user most likely to click?")
    lines.append("")
    lines.append("After your analysis, provide your final answer in this exact format:")
    lines.append("Answer: <number>")

    return "\n".join(lines)


def build_top1_prompt(history: List[dict], candidates: List[dict]) -> str:
    """
    Simple prompt that just asks for the top-1 choice.
    Easier task than full ranking.
    """
    prompt = "A user has read these news articles:\n"

    if history:
        recent_history = history[-30:] if len(history) > 30 else history
        for i, h in enumerate(recent_history, 1):
            cat = h.get("category", "General")
            prompt += f"{i}. [{cat}] {h['text']}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\nNow, given these candidate articles:\n"
    for i, cand in enumerate(candidates, 1):
        cat = cand.get("category", "General")
        prompt += f"{i}. [{cat}] {cand['text']}\n"

    prompt += "\nWhich ONE article (number only) will this user most likely click?\n"
    prompt += "Just respond with the number, nothing else."

    return prompt


def _get_sambanova_client(api_key: str, base_url: str):
    try:
        from sambanova import SambaNova
    except Exception as exc:
        raise SystemExit("SambaNova client not available. Install with `pip install sambanova`.") from exc

    return SambaNova(api_key=api_key, base_url=base_url)


def _parse_ranked_numbers(text: str, num_candidates: int) -> List[int]:
    nums = [int(n) for n in re.findall(r"\d+", text)]
    order = []
    seen = set()
    for n in nums:
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


def score_candidates_multiple_choice_sambanova(
    client,
    model: str,
    prompt: str,
    num_candidates: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> List[float]:
    """Original: asks for full ranking."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    prompt
                    + "\nReturn a ranked list of all option numbers (best to worst), "
                    "separated by spaces."
                ),
            },
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content.strip()
    order = _parse_ranked_numbers(content, num_candidates)

    scores = [0.0] * num_candidates
    for rank, option in enumerate(order):
        scores[option - 1] = float(num_candidates - rank)
    return scores


def _extract_answer_number(text: str, num_candidates: int) -> int:
    """Extract the answer number from CoT or top-1 response."""
    # Pattern 1: "Answer: X"
    match = re.search(r'[Aa]nswer\s*:\s*(\d+)', text)
    if match:
        ans = int(match.group(1))
        if 1 <= ans <= num_candidates:
            return ans

    # Pattern 2: "The answer is X"
    match = re.search(r'[Tt]he\s+answer\s+is\s+(\d+)', text)
    if match:
        ans = int(match.group(1))
        if 1 <= ans <= num_candidates:
            return ans

    # Pattern 3: Just a number (for top-1 mode)
    match = re.search(r'^(\d+)$', text.strip())
    if match:
        ans = int(match.group(1))
        if 1 <= ans <= num_candidates:
            return ans

    # Pattern 4: Last number in text
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        ans = int(numbers[-1])
        if 1 <= ans <= num_candidates:
            return ans

    # Fallback: return 1
    return 1


def score_candidates_cot_sambanova(
    client,
    model: str,
    prompt: str,
    num_candidates: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Tuple[List[float], str]:
    """
    CoT mode: asks for reasoning, then extracts top-1 answer.
    Returns (scores, generated_text) for debugging.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful news recommendation assistant. Think step by step before giving your answer."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content.strip()

    # Extract the answer
    answer = _extract_answer_number(content, num_candidates)

    # Create scores: answer gets 1.0, rest get 0.0
    scores = [0.0] * num_candidates
    scores[answer - 1] = 1.0

    return scores, content


def score_candidates_top1_sambanova(
    client,
    model: str,
    prompt: str,
    num_candidates: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> List[float]:
    """
    Top-1 mode: just asks for the single best choice.
    Simpler task than full ranking.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=16,  # Only need a number
    )
    content = response.choices[0].message.content.strip()

    answer = _extract_answer_number(content, num_candidates)

    scores = [0.0] * num_candidates
    scores[answer - 1] = 1.0

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
    parser.add_argument("--model", required=True, help="SambaNova model name")
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--max_history", type=int, default=30, help="Max history items (default: 30)")
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", help="Output prediction file for MIND leaderboard")
    parser.add_argument("--api_key", help="SambaNova API key (defaults to SAMBANOVA_API_KEY env var)")
    parser.add_argument("--base_url", default="https://api.sambanova.ai/v1")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=128)
    # New options for different evaluation modes
    parser.add_argument("--mode", choices=["ranking", "cot", "top1"], default="cot",
                        help="Evaluation mode: 'ranking' (full ranking), 'cot' (chain-of-thought), 'top1' (simple top-1)")
    parser.add_argument("--save_responses", help="Save model responses to this file (for debugging)")
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

    print(f"\nEvaluating with mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Use abstract: {args.use_abstract}")
    print(f"Max history: {args.max_history}")
    if args.mode == "cot":
        print("Mode: Chain-of-Thought (reasoning before answer)")
    elif args.mode == "top1":
        print("Mode: Top-1 only (simple single choice)")
    else:
        print("Mode: Full ranking (asks for ordered list)")
    print()

    # For saving responses (debugging)
    saved_responses = []

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

            # Build prompt based on mode
            if args.mode == "cot":
                prompt = build_cot_prompt(history_objs, candidate_objs)
                scores, response_text = score_candidates_cot_sambanova(
                    client,
                    args.model,
                    prompt,
                    len(candidate_objs),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                if args.save_responses:
                    saved_responses.append({
                        "impression_id": impression_id,
                        "response": response_text,
                        "predicted": int(scores.index(max(scores)) + 1),
                        "labels": labels
                    })
            elif args.mode == "top1":
                prompt = build_top1_prompt(history_objs, candidate_objs)
                scores = score_candidates_top1_sambanova(
                    client,
                    args.model,
                    prompt,
                    len(candidate_objs),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
            else:  # ranking mode
                prompt = build_multiple_choice_prompt(history_objs, candidate_objs)
                scores = score_candidates_multiple_choice_sambanova(
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
