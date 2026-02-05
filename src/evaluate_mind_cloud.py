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
    prompt = "User's reading history (most recent):\n"
    if history:
        # Use last 25 articles for better category preference understanding
        recent_history = history[-25:] if len(history) > 25 else history
        for i, h in enumerate(recent_history, 1):
            cat = h.get("category", "General")
            prompt += f"{i}. [{cat}] {h['text']}\n"
    else:
        prompt += "(No history)\n"

    prompt += "\nCandidate article:\n"
    cat = candidate.get("category", "General")
    prompt += f"[{cat}] {candidate['text']}\n"

    prompt += "\nTask: Predict if the user would click this candidate article.\n\n"
    prompt += "Instructions:\n"
    prompt += "FIRST, analyze step-by-step in THINKING section:\n"
    prompt += "THINKING:\n"
    prompt += "1. User Interests: [What topics/themes does the user read about?]\n"
    prompt += "2. Candidate Relevance: [How does this candidate match user interests?]\n"
    prompt += "3. Decision: [Would they click? Why/why not?]\n\n"
    prompt += "THEN, provide your final answer:\n"
    prompt += "ANSWER: Yes or No\n"

    return prompt


def build_listwise_prompt(history: List[dict], candidates: List[dict], top_k: int = 5) -> str:
    """Build optimized listwise ranking prompt with detailed instructions.

    Args:
        history: User's reading history
        candidates: Candidate articles to rank
        top_k: Number of top articles to identify (default=5, 0=all candidates)
    """
    # Pre-compute category statistics from history
    from collections import Counter
    category_counts = Counter()
    if history:
        recent_history = history[-25:] if len(history) > 25 else history
        for h in recent_history:
            cat = h.get("category", "General")
            category_counts[cat] += 1

    # User history section with pre-computed stats
    prompt = "You are analyzing a user's news reading preferences.\n\n"

    # Show category summary FIRST (most important signal)
    prompt += "=== USER'S CATEGORY PREFERENCES (Pre-computed) ===\n"
    if category_counts:
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        for cat, count in sorted_cats[:5]:  # Show top 5 categories
            prompt += f"  {cat}: {count} articles\n"
        prompt += f"\nTop categories: {', '.join([c for c, _ in sorted_cats[:3]])}\n"
    else:
        prompt += "(No history)\n"

    prompt += "\n=== USER'S RECENT READING HISTORY ===\n"
    if history:
        recent_history = history[-25:] if len(history) > 25 else history
        for i, h in enumerate(recent_history, 1):
            cat = h.get("category", "General")
            prompt += f"{i}. [{cat}] {h['text']}\n"
    else:
        prompt += "(No history available)\n"

    # Candidates section
    prompt += f"\n=== CANDIDATE ARTICLES TO RANK ===\n"
    for i, cand in enumerate(candidates):
        option_num = i + 1
        cat = cand.get("category", "General")
        prompt += f"{option_num}. [{cat}] {cand['text']}\n"

    # Task and instructions
    prompt += f"\n=== TASK ===\n"
    # If top_k=0, rank all candidates; otherwise use specified top_k
    num_to_rank = len(candidates) if top_k == 0 else min(top_k, len(candidates))

    # Adjust task description based on whether ranking all or top-K
    if num_to_rank == len(candidates):
        prompt += f"Rank ALL {num_to_rank} articles by click likelihood (most likely first).\n"
    else:
        prompt += f"Identify the TOP {num_to_rank} articles (from {len(candidates)} candidates) that the user is most likely to click.\n"
        prompt += "Rank only these top articles by likelihood. The rest will remain in original order.\n"
    prompt += "Consider topic relevance, category alignment, and content similarity to the user's reading patterns.\n\n"

    # Structured reasoning process - optimized for news recommendation
    prompt += "=== INSTRUCTIONS ===\n"
    prompt += "FIRST, reason through the ranking using this process:\n\n"
    prompt += "THINKING:\n"
    prompt += "Step 1 - Use Pre-computed Category Preferences (ALREADY PROVIDED ABOVE):\n"
    prompt += "  - The category counts are ALREADY computed for you\n"
    prompt += "  - Focus on the top 2-3 categories from the summary\n"
    prompt += "  - These are the user's strongest preferences\n\n"

    prompt += "Step 2 - Match Candidates to Top Categories:\n"
    prompt += "  - For each candidate, check: does its category match user's top categories?\n"
    prompt += "  - STRONG match = candidate category is #1 or #2 in user's preferences\n"
    prompt += "  - MEDIUM match = candidate category is #3 in user's preferences\n"
    prompt += "  - WEAK match = candidate category not in user's top 3\n"
    if num_to_rank == len(candidates):
        prompt += f"  - Rank all {num_to_rank} candidates by category match strength\n\n"
    else:
        prompt += f"  - SELECT ONLY candidates with strong category matches (top {num_to_rank})\n\n"

    prompt += "Step 3 - Final Ranking:\n"
    prompt += "  - PRIORITIZE: Candidates matching user's most frequent categories\n"
    prompt += "  - Among same-category candidates, prefer those with more specific topic overlap\n"
    if num_to_rank < len(candidates):
        prompt += f"  - Output your top {num_to_rank} picks only\n\n"
    else:
        prompt += f"  - Output complete ranking of all {num_to_rank} candidates\n\n"

    # Output format - EXTREMELY EXPLICIT with example
    prompt += "THEN, output your ranking IMMEDIATELY:\n\n"
    prompt += "CRITICAL OUTPUT RULES:\n"
    prompt += "1. Complete your THINKING section (this is required reasoning)\n"
    prompt += "2. After THINKING ends, the VERY NEXT LINE must be RANKING:\n"
    prompt += "3. Do NOT add any transition text between THINKING and RANKING\n"
    prompt += "4. Do NOT add a separate 'Ranking:' section with bullet points\n"
    prompt += "5. Output ONLY the JSON on a single line, nothing after it\n\n"

    prompt += "EXACT FORMAT:\n"
    if num_to_rank == 1:
        prompt += f'RANKING: {{\"ranking\": [X]}}\n\n'
        prompt += f"Where X is your single top pick (1-{len(candidates)})\n\n"
    elif num_to_rank <= 5:
        example_ranking = list(range(1, min(num_to_rank + 1, 6)))
        prompt += f'RANKING: {{\"ranking\": {example_ranking}}}\n\n'
    else:
        prompt += f'RANKING: {{\"ranking\": [3, 1, 5, 2, 4, ...]}}\n\n'

    prompt += "EXAMPLE OUTPUT:\n"
    prompt += "THINKING:\n"
    prompt += "Step 1 - Top categories (from summary): finance (#1), tech (#2)\n"
    prompt += "Step 2 - Candidate matches: #4=finance (STRONG), #7=tech (STRONG), #2=sports (WEAK)\n"
    prompt += "Step 3 - Ranking: #4 first (matches #1 category), #7 second (matches #2 category)\n\n"
    if num_to_rank == 1:
        prompt += 'RANKING: {"ranking": [4]}\n\n'
    else:
        prompt += 'RANKING: {"ranking": [4, 7]}\n\n'

    # Add explicit WRONG vs RIGHT examples
    prompt += "WRONG OUTPUTS (DO NOT DO THIS):\n"
    prompt += "❌ Adding bullet list before JSON:\n"
    prompt += "   Ranking:\n   - Top 1: Candidate 4\n   RANKING: {...}\n\n"
    prompt += "❌ Adding transition text:\n"
    prompt += "   Based on the analysis, RANKING: {\"ranking\": [4]}\n\n"
    prompt += "❌ Incomplete JSON:\n"
    prompt += "   RANKING: {\"ranking\": [4\n\n"

    prompt += "RIGHT OUTPUT (Keep THINKING, then immediately output JSON):\n"
    if num_to_rank == 1:
        prompt += '✓ RANKING: {"ranking": [4]}\n\n'
    else:
        prompt += '✓ RANKING: {"ranking": [4, 7, 2]}\n\n'

    prompt += "REQUIREMENTS:\n"
    prompt += "- Complete your THINKING section first (required)\n"
    prompt += f"- Then output exactly: RANKING: {{\"ranking\": [...]}}\n"
    if num_to_rank == len(candidates):
        prompt += f"- Include ALL {num_to_rank} numbers (complete ranking)\n"
    else:
        prompt += f"- Include exactly {num_to_rank} numbers\n"
    prompt += "- Valid JSON on ONE line\n"
    prompt += "- NO transition text between THINKING and RANKING\n"
    prompt += "- NO text after the JSON line\n"
    prompt += f"- Numbers must be in range 1-{len(candidates)}\n"

    return prompt


def build_selection_prompt(history: List[dict], candidates: List[dict]) -> str:
    """Build selection prompt: predict 1-3 articles the user would click."""
    prompt = "User's reading history (most recent):\n"
    if history:
        # Use last 25 articles for better category preference understanding
        recent_history = history[-25:] if len(history) > 25 else history
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

    prompt += "\nTask: Predict which article(s) the user would click based on their reading patterns.\n"
    prompt += "Be selective - only pick articles with strong relevance to the user's demonstrated interests.\n\n"
    prompt += "Instructions:\n"
    prompt += "FIRST, analyze step-by-step in THINKING section:\n"
    prompt += "THINKING:\n"
    prompt += "1. User Interests: [Identify 2-3 main topics/themes from reading history]\n"
    prompt += "2. Candidate Analysis: [For each candidate, assess relevance to user interests]\n"
    prompt += "3. Final Selection: [Explain which 1-3 articles best match and why]\n\n"
    prompt += "THEN, provide your picks:\n"
    prompt += "PICKS: <number>, <number>, ... (1-3 numbers ordered by likelihood)\n"

    return prompt


def _get_sambanova_client(api_key: str, base_url: str):
    try:
        from sambanova import SambaNova
    except Exception as exc:
        raise SystemExit("SambaNova client not available. Install with `pip install sambanova`.") from exc

    return SambaNova(api_key=api_key, base_url=base_url)


def _parse_ranked_numbers(text: str, num_candidates: int) -> List[int]:
    order = []

    # First, try to find the RANKING: section (case-insensitive)
    ranking_section = text
    if "RANKING:" in text.upper():
        idx = text.upper().find("RANKING:")
        ranking_section = text[idx + 8:]

    # Remove any "Ranking:" explanatory text that sometimes appears before RANKING:
    # This handles cases where LLM adds extra "Ranking:" section with explanations
    # Look for patterns like "Ranking:\n  - Top 1: ..." and remove everything up to RANKING:
    if "RANKING:" not in ranking_section.upper():
        # If no RANKING: marker in this section, it might be all explanatory text
        # Clear any lines starting with "Ranking:" or containing bullet points
        ranking_section = re.sub(r'^Ranking:.*?$', '', ranking_section, flags=re.IGNORECASE | re.MULTILINE)
        ranking_section = re.sub(r'^\s*[-•]\s*Top\s+\d+:.*?$', '', ranking_section, flags=re.IGNORECASE | re.MULTILINE)

    # Clean up markdown code blocks if present
    ranking_section = re.sub(r'```json\s*', '', ranking_section)
    ranking_section = re.sub(r'```\s*', '', ranking_section)
    ranking_section = ranking_section.strip()

    # Try multiple parsing strategies
    # 1. Try to parse JSON with "ranking" key
    try:
        start = ranking_section.find("{")
        end = ranking_section.rfind("}")
        if start != -1:
            if end != -1 and end > start:
                json_str = ranking_section[start : end + 1]
            else:
                # No closing brace found, take everything after opening brace
                json_str = ranking_section[start:]

            # Fix common malformed JSON: missing closing brackets
            # Need to close in proper order: inner brackets first
            open_square = json_str.count('[')
            close_square = json_str.count(']')
            open_curly = json_str.count('{')
            close_curly = json_str.count('}')

            # Close square brackets first (they're inner), then curly braces
            if open_square > close_square:
                json_str += ']' * (open_square - close_square)
            if open_curly > close_curly:
                json_str += '}' * (open_curly - close_curly)

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
                    "content": "You are an expert at predicting news article clicks. Analyze the user's reading patterns to identify their interests, then determine if the candidate article matches those interests well enough for them to click."
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


def _build_system_message_with_examples(top_k: int, num_candidates: int) -> str:
    """Build system message optimized for news recommendation quality."""
    if top_k == 0 or top_k >= num_candidates:
        base = f"You are a news recommendation system. Match articles to user's category preferences."
        count = num_candidates
    else:
        base = f"You are a news recommendation system. Find the TOP {top_k} articles that match user's top categories."
        count = top_k

    msg = base + "\n\n"
    msg += "KEY PRINCIPLE: Match candidates to user's TOP categories!\n"
    msg += "- Category counts are PRE-COMPUTED for you (no need to count)\n"
    msg += "- Simply match candidate categories to user's top 2-3 categories\n"
    msg += "- Prioritize candidates matching category #1, then #2, then #3\n\n"

    # Concrete example with pre-computed stats
    msg += "EXAMPLE:\n"
    msg += "User category preferences: finance(8), tech(2)\n"
    msg += "Candidates: 1=finance, 2=sports, 3=finance, 4=tech, 5=health\n"
    msg += "THINKING:\n"
    msg += "Top categories: finance (#1), tech (#2). Match: #1,#3=finance (STRONG), #4=tech (MEDIUM)\n\n"
    if count == 1:
        msg += 'RANKING: {"ranking": [1]}\n\n'
    else:
        msg += 'RANKING: {"ranking": [1, 3, 4]}\n\n'

    msg += f"OUTPUT: THINKING section, then RANKING: {{\"ranking\": [...]}} with {count} numbers. "
    msg += "NO extra text after JSON."

    return msg


def score_candidates_listwise(
    client,
    model: str,
    prompt: str,
    num_candidates: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    top_k: int = 5,
) -> List[float]:
    # Build enhanced system message with few-shot examples (Option A + B)
    system_msg = _build_system_message_with_examples(top_k, num_candidates)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_msg
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

    # Parse top-K ranking
    ranked_items = _parse_ranked_numbers(content, num_candidates)

    # Determine expected count based on top_k parameter
    expected_k = num_candidates if (top_k == 0 or top_k >= num_candidates) else top_k

    # Option C: Retry with simpler prompt if parsing completely failed
    if not ranked_items:
        if os.getenv("DEBUG_CLOUD_EVAL") == "1":
            print(f"[Listwise] WARNING: Parsing failed, retrying with simpler prompt...", file=sys.stderr)

        # Build ultra-simple retry prompt
        retry_prompt = f"Rank these {num_candidates} articles for the user. Output ONLY: {{\"ranking\": [list of {expected_k} numbers]}}\n\n"
        retry_prompt += prompt.split("=== TASK ===")[0] if "=== TASK ===" in prompt else prompt[:500]

        try:
            retry_response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Output ONLY valid JSON: {{\"ranking\": [...]}} with {expected_k} numbers. NO text before or after."
                    },
                    {
                        "role": "user",
                        "content": retry_prompt,
                    },
                ],
                temperature=0.0,  # More deterministic for retry
                top_p=top_p,
                max_tokens=100,  # Very short - just need the JSON
            )
            if retry_response.choices and retry_response.choices[0].message.content:
                retry_content = retry_response.choices[0].message.content.strip()
                ranked_items = _parse_ranked_numbers(retry_content, num_candidates)
                if os.getenv("DEBUG_CLOUD_EVAL") == "1":
                    print(f"[Listwise] Retry result: {ranked_items}", file=sys.stderr)
        except Exception as e:
            if os.getenv("DEBUG_CLOUD_EVAL") == "1":
                print(f"[Listwise] Retry failed: {e}", file=sys.stderr)

    # Limit parsed items to expected count
    ranked_items = ranked_items[:expected_k] if len(ranked_items) > expected_k else ranked_items

    # Build scores
    scores = [0.0] * num_candidates
    ranked_set = set(ranked_items)

    # Assign scores to ranked items based on their position
    for rank, option in enumerate(ranked_items):
        scores[option - 1] = float(num_candidates - rank)

    # If not all candidates were ranked (top-K < N), assign remaining scores in original order
    if len(ranked_items) < num_candidates:
        remaining_score = float(num_candidates - len(ranked_items))
        for i in range(num_candidates):
            if (i + 1) not in ranked_set:
                scores[i] = remaining_score
                remaining_score -= 1

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
                    "content": "You are an expert at predicting news article clicks based on user reading patterns. Follow a structured reasoning process: (1) identify the user's main interests from their history, (2) analyze how each candidate matches those interests, (3) select only the 1-3 most relevant articles. Be selective and precise."
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
    parser.add_argument("--top_k", type=int, default=5, help="Top-K for listwise ranking (0=rank all, default=5)")
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
                prompt = build_listwise_prompt(history_objs, candidate_objs, top_k=args.top_k)
                scores = score_candidates_listwise(
                    client,
                    args.model,
                    prompt,
                    len(candidate_objs),
                    top_k=args.top_k,
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
