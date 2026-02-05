"""
Evaluate MIND via cloud LLM (SambaNova) with prompt-based scoring.

Modes:
- pointwise: ask Yes/No for each candidate; score Yes=1, No=0.
- listwise: ask for a full ranked list of option numbers.
- selection: ask LLM to pick 1-3 articles the user would click (LLM decides count).

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
    prompt = "User's reading history:\n"
    if history:
        recent_history = history[-30:] if len(history) > 30 else history
        for i, h in enumerate(recent_history, 1):
            cat = h.get("category", "General")
            prompt += f"{i}. [{cat}] {h['text']}\n"
    else:
        prompt += "(No history)\n"

    prompt += "\nCandidate article:\n"
    cat = candidate.get("category", "General")
    prompt += f"[{cat}] {candidate['text']}\n"

    prompt += "\nWould this user click the candidate article?\n\n"
    prompt += "Format:\n"
    prompt += "THINKING: [Analyze user interests, then match with candidate]\n"
    prompt += "ANSWER: Yes or No\n"

    return prompt


def build_listwise_prompt(history: List[dict], candidates: List[dict]) -> str:
    # User history section
    prompt = "User's reading history:\n"
    if history:
        recent_history = history[-30:] if len(history) > 30 else history
        for i, h in enumerate(recent_history, 1):
            cat = h.get("category", "General")
            prompt += f"{i}. [{cat}] {h['text']}\n"
    else:
        prompt += "(No history)\n"

    # Candidates section
    prompt += f"\nCandidate articles:\n"
    for i, cand in enumerate(candidates):
        option_num = i + 1
        cat = cand.get("category", "General")
        prompt += f"{option_num}. [{cat}] {cand['text']}\n"

    prompt += f"\nRank ALL {len(candidates)} articles by likelihood the user would click (most likely first).\n\n"
    prompt += "Format:\n"
    prompt += "THINKING: [Analyze user interests from history]\n"
    prompt += "RANKING: "
    if len(candidates) <= 5:
        example_ranking = list(range(1, len(candidates) + 1))
        prompt += f'{{\"ranking\": {example_ranking}}}\n'
    else:
        prompt += f'{{\"ranking\": [3,1,5,2,4,...]}} (all {len(candidates)} numbers)\n'

    return prompt


def build_selection_prompt(history: List[dict], candidates: List[dict]) -> str:
    """Build selection prompt: predict 1-3 articles the user would click."""
    prompt = "User's reading history:\n"
    if history:
        recent_history = history[-30:] if len(history) > 30 else history
        for i, h in enumerate(recent_history, 1):
            cat = h.get("category", "General")
            prompt += f"{i}. [{cat}] {h['text']}\n"
    else:
        prompt += "(No history)\n"

    prompt += f"\nCandidate articles:\n"
    for i, cand in enumerate(candidates):
        option_num = i + 1
        cat = cand.get("category", "General")
        prompt += f"{option_num}. [{cat}] {cand['text']}\n"

    prompt += "\nBased on the user's reading interests, which article(s) would they likely click?\n"
    prompt += "Select 1-3 articles that clearly match the user's interests. Be selective - only pick articles with strong relevance.\n\n"
    prompt += "Format:\n"
    prompt += "THINKING: [First identify user's main interests from history, then find matching candidates]\n"
    prompt += "PICKS: <number>, <number>, ... (1-3 numbers ordered by likelihood, most likely first)\n"

    return prompt


def _get_sambanova_client(api_key: str, base_url: str):
    try:
        from sambanova import SambaNova
    except Exception as exc:
        raise SystemExit("SambaNova client not available. Install with `pip install sambanova`.") from exc

    return SambaNova(api_key=api_key, base_url=base_url)


def _parse_ranked_numbers(text: str, num_candidates: int) -> List[int]:
    order = []

    # First, try to find the RANKING: section
    ranking_section = text
    if "RANKING:" in text.upper():
        idx = text.upper().find("RANKING:")
        ranking_section = text[idx + 8:]

    # Try multiple parsing strategies
    # 1. Try to parse JSON with "ranking" key
    try:
        start = ranking_section.find("{")
        end = ranking_section.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = ranking_section[start : end + 1]
            payload = json.loads(json_str)
            if isinstance(payload, dict) and isinstance(payload.get("ranking"), list):
                order = [int(n) for n in payload["ranking"]]
    except Exception:
        pass

    # 2. Try to parse array directly [1,2,3,...]
    if not order:
        try:
            start = ranking_section.find("[")
            end = ranking_section.rfind("]")
            if start != -1 and end != -1 and end > start:
                array_str = ranking_section[start : end + 1]
                parsed = json.loads(array_str)
                if isinstance(parsed, list):
                    order = [int(n) for n in parsed]
        except Exception:
            pass

    # 3. Fallback: extract numbers (limit to reasonable range to avoid parsing history)
    if not order:
        # Only look at first 500 chars of ranking section to avoid picking up history numbers
        sample = ranking_section[:500]
        order = [int(n) for n in re.findall(r"\b\d+\b", sample)]

    # Filter valid numbers and remove duplicates
    filtered = []
    seen = set()
    for n in order:
        if 1 <= n <= num_candidates and n not in seen:
            filtered.append(n)
            seen.add(n)

    # Add missing candidates at the end
    missing = [i for i in range(1, num_candidates + 1) if i not in seen]
    if missing:
        print(
            f"Warning: {len(missing)} candidates missing from ranking; appending at end.",
            file=sys.stderr,
        )
    filtered.extend(missing)
    return filtered


def score_candidate_pointwise(
    client,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> float:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You predict whether a user would click a news article based on their reading history."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if not response.choices or not response.choices[0].message.content:
            print("Warning: Empty API response", file=sys.stderr)
            return 0.0
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: API error ({e})", file=sys.stderr)
        return 0.0

    if os.getenv("DEBUG_CLOUD_EVAL") == "1":
        # Truncate THINKING section, show only ANSWER
        debug_content = content
        if "ANSWER:" in content.upper():
            idx = content.upper().find("ANSWER:")
            debug_content = content[idx:]
        print(f"[Pointwise] {debug_content[:100].strip()}")

    # Look for the ANSWER: section
    answer_text = content.lower()
    if "answer:" in answer_text:
        idx = answer_text.find("answer:")
        answer_text = answer_text[idx + 7:]

    # Look for Yes/No in the answer section
    if "yes" in answer_text:
        return 1.0
    if "no" in answer_text:
        return 0.0

    # Fallback: check entire response
    if "yes" in content.lower():
        return 1.0
    if "no" in content.lower():
        return 0.0

    print(f"Warning: unexpected response: {content[:100]}", file=sys.stderr)
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
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You rank news articles based on user reading history."
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if not response.choices or not response.choices[0].message.content:
            print("Warning: Empty API response, using default order", file=sys.stderr)
            return [float(num_candidates - i) for i in range(num_candidates)]
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: API error ({e}), using default order", file=sys.stderr)
        return [float(num_candidates - i) for i in range(num_candidates)]

    if os.getenv("DEBUG_CLOUD_EVAL") == "1":
        # Truncate THINKING section, show only RANKING
        debug_content = content
        if "RANKING:" in content.upper():
            idx = content.upper().find("RANKING:")
            debug_content = content[idx:]
        print(f"[Listwise] {debug_content[:200].strip()}")

    order = _parse_ranked_numbers(content, num_candidates)

    scores = [0.0] * num_candidates
    for rank, option in enumerate(order):
        scores[option - 1] = float(num_candidates - rank)
    return scores


def _parse_selection_picks(text: str, num_candidates: int) -> List[int]:
    """Parse 1-3 picked numbers from selection response."""
    picks = []

    # Find PICKS: section
    picks_section = text
    if "PICKS:" in text.upper():
        idx = text.upper().find("PICKS:")
        picks_section = text[idx + 6:]

    # Try to parse array [1,2,3]
    try:
        start = picks_section.find("[")
        end = picks_section.find("]")
        if start != -1 and end != -1 and end > start:
            array_str = picks_section[start : end + 1]
            parsed = json.loads(array_str)
            if isinstance(parsed, list):
                picks = [int(n) for n in parsed]
    except Exception:
        pass

    # Fallback: extract numbers from first line after PICKS:
    if not picks:
        # Take first line or first 100 chars
        first_line = picks_section.split("\n")[0][:100]
        picks = [int(n) for n in re.findall(r"\b\d+\b", first_line)]

    # Filter valid numbers, remove duplicates, limit to 3
    filtered = []
    seen = set()
    for n in picks:
        if 1 <= n <= num_candidates and n not in seen:
            filtered.append(n)
            seen.add(n)
        if len(filtered) >= 3:
            break

    return filtered


def score_candidates_selection(
    client,
    model: str,
    prompt: str,
    num_candidates: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> List[float]:
    """Score candidates using selection mode (1-3 picks)."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You predict which news articles a user would click based on their reading history. Analyze the user's interests carefully and only recommend articles that clearly match. Be selective and confident in your picks."
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if not response.choices or not response.choices[0].message.content:
            print("Warning: Empty API response, using random scores", file=sys.stderr)
            return [float(num_candidates - i) for i in range(num_candidates)]
        content = response.choices[0].message.content.strip()

        # Check for potential truncation
        if os.getenv("DEBUG_CLOUD_EVAL") == "1" and response.choices[0].finish_reason == "length":
            print(f"[Selection] WARNING: Response truncated (finish_reason=length)", file=sys.stderr)
    except Exception as e:
        print(f"Warning: API error ({e}), using random scores", file=sys.stderr)
        return [float(num_candidates - i) for i in range(num_candidates)]

    if os.getenv("DEBUG_CLOUD_EVAL") == "1":
        # Show PICKS section and parsing result
        debug_content = content
        if "PICKS:" in content.upper():
            idx = content.upper().find("PICKS:")
            debug_content = content[idx:]
        print(f"[Selection] {debug_content[:200].strip()}")

    picks = _parse_selection_picks(content, num_candidates)

    if os.getenv("DEBUG_CLOUD_EVAL") == "1":
        print(f"[Selection] Parsed picks: {picks} (from {num_candidates} candidates)")
        if not picks:
            print(f"[Selection] WARNING: No picks parsed! Full response:\n{content[:500]}", file=sys.stderr)

    # Convert picks to full ranking scores for MIND metrics
    # Picked items get highest scores in pick order
    # Unpicked items get lower scores (random order among them)
    scores = [0.0] * num_candidates
    picked_set = set(picks)

    # Picked items: highest scores (num_candidates, num_candidates-1, ...)
    for rank, option in enumerate(picks):
        scores[option - 1] = float(num_candidates - rank)

    # Unpicked items: assign same low score (treat equally to avoid index bias)
    remaining_score = float(num_candidates - len(picks))
    for i in range(num_candidates):
        if (i + 1) not in picked_set:
            scores[i] = remaining_score

    return scores


def auc_score(labels: List[int], scores: List[float]) -> float:
    pos = sum(labels)
    if pos == 0 or pos == len(labels):
        return 0.5
    return roc_auc_score(labels, scores)


def mrr_score(labels: List[int], scores: List[float]) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant item."""
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    for rank, idx in enumerate(sorted_idx, start=1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


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
    parser.add_argument("--mode", choices=["pointwise", "listwise", "selection"], required=True)
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
        # Increased defaults to allow for reasoning + answer
        # 4096 for listwise to handle large candidate sets (up to ~80 items)
        # 512 for selection (reasoning + 1-3 numbers)
        if args.mode == "pointwise":
            args.max_tokens = 128
        elif args.mode == "selection":
            args.max_tokens = 512
        else:
            args.max_tokens = 4096

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
    selection_stats = []  # Track number of picks per impression (for selection mode)

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
            elif args.mode == "selection":
                prompt = build_selection_prompt(history_objs, candidate_objs)
                scores = score_candidates_selection(
                    client,
                    args.model,
                    prompt,
                    len(candidate_objs),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                # Track how many items were picked (have score above minimum)
                num_picked = sum(1 for s in scores if s > min(scores))
                selection_stats.append(num_picked)
            else:  # listwise
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

    if selection_stats and args.mode == "selection":
        print(f"\nSelection Statistics:")
        print(f"Average picks per impression: {_avg(selection_stats):.2f}")
        print(f"Min picks: {min(selection_stats)}, Max picks: {max(selection_stats)}")
        if 0 in selection_stats:
            print(f"WARNING: {selection_stats.count(0)} impressions had 0 picks (model too conservative)")

    if skipped_malformed > 0:
        print(f"Warning: Skipped {skipped_malformed} malformed lines")


if __name__ == "__main__":
    main()
