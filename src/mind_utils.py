"""
Shared utilities for MIND benchmark operations.

Consolidates functions duplicated across evaluation, data preparation,
and training scripts into a single module.

Functions:
    load_news() - Load news articles from news.tsv
    build_pointwise_prompt() - Build Yes/No classification prompt
    parse_behaviors_line() - Parse a single line from behaviors.tsv
    auc_score() - Compute AUC metric
    mrr_score() - Compute MRR metric
    dcg_score() - Compute DCG metric
    ndcg_score() - Compute nDCG metric
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def _parse_timestamp(ts: Optional[str]):
    """Parse MIND timestamp string into (day_of_week, time_period) or (None, None)."""
    if not ts:
        return None, None
    try:
        from datetime import datetime
        dt = datetime.strptime(ts.strip(), "%m/%d/%Y %I:%M:%S %p")
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day = days[dt.weekday()]
        h = dt.hour
        if 6 <= h < 12:
            period = "morning"
        elif 12 <= h < 18:
            period = "afternoon"
        elif 18 <= h < 24:
            period = "evening"
        else:
            period = "night"
        return day, period
    except Exception:
        return None, None


def _build_profile_summary(history: List[Dict[str, str]]) -> str:
    """Build a brief category frequency summary from reading history."""
    cats = [h.get("category", "") for h in history if h.get("category", "")]
    if not cats:
        return ""
    top = Counter(cats).most_common(3)
    return "User interests: " + ", ".join(f"{c} ({n})" for c, n in top)


def load_news(news_path: str, use_abstract: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Load news articles from MIND news.tsv file.

    Args:
        news_path: Path to news.tsv
        use_abstract: Whether to append abstract to title text

    Returns:
        Dict mapping news_id -> {'text': str, 'category': str, 'title': str}
    """
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
                "text": text,
                "category": category,
                "subcategory": parts[2] if len(parts) > 2 else "",
                "title": title,
            }
    return news


def build_pointwise_prompt(
    history: List[Dict[str, str]],
    candidate: Dict[str, str],
    tokenizer=None,
    use_chat_template: bool = False,
    use_recency: bool = False,
    use_profile_summary: bool = False,
    impression_timestamp: Optional[str] = None,
) -> str:
    """
    Build pointwise Yes/No classification prompt matching SFT training format.

    Based on Prompt4NR research (arXiv:2304.05263):
    - Concise format saves ~20 tokens
    - Category in [brackets] for visibility
    - Limits to last 30 history items

    Args:
        history: List of news dicts with 'text' and 'category' keys
        candidate: Single candidate news dict
        tokenizer: Tokenizer object (required if use_chat_template=True)
        use_chat_template: Whether to format using chat template for instruct models
        use_recency: Mark the 5 most recent history items with "(recent)"
        use_profile_summary: Prepend top-3 category frequency summary
        impression_timestamp: MIND timestamp string (e.g. "11/15/2019 1:00:00 PM")
                              Adds day/time-of-day context when provided

    Returns:
        Prompt string ending with "Answer:" (raw) or chat-formatted prompt
    """
    # Build the base content
    content = ""

    # Optional: time-of-day / day-of-week context
    if impression_timestamp:
        day, period = _parse_timestamp(impression_timestamp)
        if day and period:
            content += f"Reading time: {day} {period}\n"

    # Optional: user interest profile summary
    if use_profile_summary and history:
        summary = _build_profile_summary(history)
        if summary:
            content += summary + "\n"

    content += "A user read these news articles:\n"

    if history:
        recent_history = history[-30:] if len(history) > 30 else history
        recency_cutoff = max(0, len(recent_history) - 5) if use_recency else len(recent_history)
        for i, h in enumerate(recent_history, 1):
            cat = h.get("category", "General")
            tag = " (recent)" if use_recency and (i - 1) >= recency_cutoff else ""
            content += f"{i}. [{cat}] {h['text']}{tag}\n"
    else:
        content += "(No reading history)\n"

    content += "\n"
    content += "Candidate article:\n"
    cat = candidate.get("category", "General")
    content += f"[{cat}] {candidate['text']}\n"
    content += "\n"
    content += "Will this user read this article? Answer:"

    # Apply chat template if requested
    if use_chat_template:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when use_chat_template=True")
        system_prompt = (
            "You are a news recommendation assistant. "
            "Based on a user's reading history, predict whether they will read a given article. "
            "Each article includes its category and title. "
            "Answer with Yes or No."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    return content


def build_pointwise_prompt_subcategory(
    history: List[Dict[str, str]],
    candidate: Dict[str, str],
    tokenizer=None,
    use_chat_template: bool = False
) -> str:
    """
    Build pointwise Yes/No prompt with [category/subcategory] format.

    Identical to build_pointwise_prompt but exposes subcategory for finer-grained
    user interest signals (e.g., [Sports/NBA] instead of [Sports]).
    """
    def _fmt(item: Dict[str, str]) -> str:
        cat = item.get("category", "General")
        subcat = item.get("subcategory", "")
        return f"{cat}/{subcat}" if subcat else cat

    content = "A user read these news articles:\n"
    if history:
        recent_history = history[-30:] if len(history) > 30 else history
        for i, h in enumerate(recent_history, 1):
            content += f"{i}. [{_fmt(h)}] {h['text']}\n"
    else:
        content += "(No reading history)\n"

    content += "\n"
    content += "Candidate article:\n"
    content += f"[{_fmt(candidate)}] {candidate['text']}\n"
    content += "\n"
    content += "Will this user read this article? Answer:"

    if use_chat_template:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when use_chat_template=True")
        system_prompt = (
            "You are a news recommendation assistant. "
            "Based on a user's reading history, predict whether they will read a given article. "
            "Each article includes its category, subcategory and title. "
            "Answer with Yes or No."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    return content


def build_ranking_prompt(
    history: List[Dict[str, str]],
    candidates: List[Dict[str, str]],
    tokenizer=None,
    use_chat_template: bool = False
) -> str:
    """
    Build ranking selection prompt matching SFT training format.

    Args:
        history: List of news dicts with 'text' and 'category' keys
        candidates: List of candidate news dicts
        tokenizer: Tokenizer object (required if use_chat_template=True)
        use_chat_template: Whether to format using chat template for instruct models

    Returns:
        Prompt string ending with "Answer:" (raw) or chat-formatted prompt
    """
    # Build the base content
    content = "A user read these news articles:\n"

    if history:
        recent_history = history[-30:] if len(history) > 30 else history
        for i, h in enumerate(recent_history, 1):
            cat = h.get("category", "General")
            content += f"{i}. [{cat}] {h['text']}\n"
    else:
        content += "(No reading history)\n"

    content += "\nCandidate articles:\n"
    for i, cand in enumerate(candidates, 1):
        cat = cand.get("category", "General")
        content += f"{i}. [{cat}] {cand['text']}\n"

    content += "\nWhich article will the user read? Answer:"

    # Apply chat template if requested
    if use_chat_template:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when use_chat_template=True")
        system_prompt = (
            "You are a news recommendation assistant. "
            "Based on a user's reading history, select the article they are most likely to read. "
            "Each article includes its category and title. "
            "Answer with the article number."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    return content


def parse_behaviors_line(line: str) -> Optional[Tuple[str, str, str, List[str], List[Tuple[str, int]]]]:
    """
    Parse a single line from MIND behaviors.tsv.

    Args:
        line: Raw line from behaviors.tsv

    Returns:
        Tuple of (impression_id, user_id, timestamp, history_ids, impressions)
        where impressions is a list of (news_id, label) tuples.
        Returns None if the line is malformed.
    """
    parts = line.strip().split("\t")
    if len(parts) < 5:
        return None

    impression_id = parts[0]
    user_id = parts[1]
    timestamp = parts[2]
    history_ids = parts[3].split()

    impressions = []
    for imp in parts[4].split():
        if "-" not in imp:
            # Test split: no labels, treat as label=0
            impressions.append((imp, 0))
        else:
            news_id, label = imp.rsplit("-", 1)
            impressions.append((news_id, int(label)))

    return impression_id, user_id, timestamp, history_ids, impressions


# ---------------------------------------------------------------------------
# Metrics - Official MIND evaluation formulas
# ---------------------------------------------------------------------------

def auc_score(labels: List[int], scores: List[float]) -> float:
    """Compute AUC score using sklearn. Returns 0.5 for degenerate cases."""
    pos = sum(labels)
    if pos == 0 or pos == len(labels):
        return 0.5
    return roc_auc_score(labels, scores)


def mrr_score(y_true, y_score) -> float:
    """
    MRR (Mean Reciprocal Rank) - Official MIND implementation.
    Averages reciprocal rank over all positive items.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return float(np.sum(rr_score) / np.sum(y_true)) if np.sum(y_true) > 0 else 0.0


def dcg_score(y_true, y_score, k: int = 10) -> float:
    """DCG (Discounted Cumulative Gain) - Official MIND implementation."""
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return float(np.sum(gains / discounts))


def ndcg_score(y_true, y_score, k: int = 10) -> float:
    """nDCG (Normalized DCG) - Official MIND implementation."""
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0
