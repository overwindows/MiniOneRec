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

import json
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def _parse_timestamp_dt(ts: Optional[str]):
    """Parse a MIND timestamp ('11/15/2019 1:00:00 PM') to a datetime, or None."""
    if not ts:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(ts.strip(), "%m/%d/%Y %I:%M:%S %p")
    except Exception:
        return None


def _parse_timestamp(ts: Optional[str]):
    """Parse MIND timestamp string into (day_of_week, time_period) or (None, None)."""
    dt = _parse_timestamp_dt(ts)
    if dt is None:
        return None, None
    try:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day = days[dt.weekday()]
        h = dt.hour
        if 6 <= h < 12:  # noqa: PLR2004
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


def load_news(news_path: str, use_abstract: bool = False, use_entity: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Load news articles from MIND news.tsv file.

    Args:
        news_path: Path to news.tsv
        use_abstract: Whether to append abstract to title text
        use_entity: Whether to append Wikidata entity labels ("Topics: ...") to text

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
            entity_json = parts[6] if len(parts) > 6 else ""

            if use_abstract and abstract:
                text = f"{title} {abstract}"
            else:
                text = title

            # Optional: append Wikidata entity labels ("Topics: A, B, C")
            if use_entity and entity_json and entity_json != "[]":
                try:
                    entities = [
                        e.get("Label", "") for e in json.loads(entity_json)
                        if e.get("Label")
                    ]
                    if entities:
                        text += " [Topics: " + ", ".join(entities[:5]) + "]"
                except (ValueError, AttributeError):
                    pass

            news[news_id] = {
                "text": text,
                "category": category,
                "subcategory": parts[2] if len(parts) > 2 else "",
                "title": title,
            }
    return news


_system_role_supported_cache = {}


def _chat_template_supports_system(tokenizer) -> bool:
    """Probe once whether tokenizer.apply_chat_template accepts a 'system' role.

    Some instruct chat templates (e.g. gemma-2-2b-it) raise a jinja2
    TemplateError "System role not supported". Mirrors the pre-detect done in
    data.py:MINDPointwiseSFTDataset so eval prompts use the SAME folding as
    training and do not crash at inference time.
    """
    key = id(tokenizer)
    if key not in _system_role_supported_cache:
        supported = True
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "x"},
                 {"role": "user", "content": "y"}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            supported = False
        _system_role_supported_cache[key] = supported
    return _system_role_supported_cache[key]


def _render_history(history: List[Dict[str, str]], history_format: str = "flat",
                    use_recency: bool = False) -> str:
    """Render the user's history list into prompt text.

    Formats (wave-11 history-representation study):
      flat            - numbered list, one item per line (baseline)
      grouped_profile - items grouped by category with counts, most recent
                        first within each group ("Sports (12): ...")
      two_block       - "Recently read:" (last 10, recency-tagged) then
                        "Longer-term interests:" (the rest)
      title_only      - bare titles, no [category] tag, no abstract text
    """
    if not history:
        return "(No reading history)\n"

    if history_format == "grouped_profile":
        # Group by category preserving first-seen order; within a group keep
        # chronological order (most recent last, same as the flat list).
        groups: Dict[str, List[Dict[str, str]]] = {}
        for h in history:
            groups.setdefault(h.get("category", "General"), []).append(h)
        # Order groups by size descending (dominant interest first)
        ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        lines = []
        for cat, items in ordered:
            lines.append(f"{cat} ({len(items)}):")
            for h in items:
                lines.append(f"  - {h['text']}")
        return "\n".join(lines) + "\n"

    if history_format == "two_block":
        recent = history[-10:]
        older = history[:-10]
        lines = ["Recently read:"]
        for h in recent:
            lines.append(f"  - {h['text']} (recent)")
        if older:
            lines.append("Longer-term interests:")
            for h in older:
                lines.append(f"  - {h['text']}")
        return "\n".join(lines) + "\n"

    if history_format == "title_only":
        lines = [f"{i}. {h['title']}" for i, h in enumerate(history, 1)]
        return "\n".join(lines) + "\n"

    # flat (baseline)
    recency_cutoff = max(0, len(history) - 5) if use_recency else len(history)
    lines = []
    for i, h in enumerate(history, 1):
        cat = h.get("category", "General")
        tag = " (recent)" if use_recency and (i - 1) >= recency_cutoff else ""
        lines.append(f"{i}. [{cat}] {h['text']}{tag}")
    return "\n".join(lines) + "\n"


def dedup_history_ids(history_ids: List[str], max_history: int = 0,
                      backfill: bool = True) -> List[str]:
    """Drop repeated news IDs from a history id list (chronological order and
    first occurrence kept).

    backfill=True (max_history>0): dedup the WHOLE list, then take the last
      max_history uniques — older unique items backfill the window (recovers
      ~21k items on MIND_small train).
    backfill=False: slice the last-max_history window FIRST, then dedup within
      it — the window shrinks, nothing older is pulled in. (Deduping the whole
      list before slicing would be identical to backfill=True.)
    """
    src = history_ids if (backfill or max_history <= 0) else history_ids[-max_history:]
    seen = set()
    out = []
    for nid in src:
        if nid not in seen:
            seen.add(nid)
            out.append(nid)
    if backfill and max_history > 0:
        return out[-max_history:]
    return out


def catcap_history_ids(history_ids: List[str], news: Dict[str, Dict],
                       max_history: int = 0, per_cat_cap: int = 10) -> List[str]:
    """Cap per-category items in the recent window, filling freed slots with
    the next most-recent items from other categories (no candidate info used).
    Selected items are re-rendered in chronological order.
    """
    window = history_ids[-max_history:] if max_history > 0 else list(history_ids)
    counts: Counter = Counter()
    kept = []
    for nid in window:
        cat = news.get(nid, {}).get("category", "")
        if counts[cat] < per_cat_cap:
            counts[cat] += 1
            kept.append(nid)
    if max_history > 0 and len(kept) < max_history and len(history_ids) > len(window):
        # backfill from older items under the same cap
        older = [nid for nid in history_ids[:-max_history] if nid not in set(window)]
        for nid in reversed(older):
            if len(kept) >= max_history:
                break
            cat = news.get(nid, {}).get("category", "")
            if counts[cat] < per_cat_cap:
                counts[cat] += 1
                kept.append(nid)
        kept.sort(key=lambda n: history_ids.index(n))
    return kept


def build_pointwise_prompt(
    history: List[Dict[str, str]],
    candidate: Dict[str, str],
    tokenizer=None,
    use_chat_template: bool = False,
    use_recency: bool = False,
    use_profile_summary: bool = False,
    impression_timestamp: Optional[str] = None,
    history_format: str = "flat",
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

    content += _render_history(history, history_format=history_format, use_recency=use_recency)

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
        if _chat_template_supports_system(tokenizer):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        else:
            # gemma-2-it and similar reject a 'system' role; fold it into the
            # user message exactly as data.py does at train time.
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{content}"}
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
        for i, h in enumerate(history, 1):
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
        if _chat_template_supports_system(tokenizer):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        else:
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{content}"}
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
        for i, h in enumerate(history, 1):
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
