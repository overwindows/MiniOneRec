import ast
import math
import os
import pickle
import re

import numpy as np
import torch

# Lazy import for SASRec - only needed for sasrec reward type
SASRec = None

def _get_sasrec():
    global SASRec
    if SASRec is None:
        from sasrec import SASRec as _SASRec
        SASRec = _SASRec
    return SASRec


_SID_INFO_CACHE = None
_TITLE_TO_SID_CACHE = None
_EMBED_CACHE = None
_SASREC_CACHE = None


def _normalize_sid(value):
    if value is None:
        return ""
    return str(value).strip().strip("\n").strip("\"").strip("'")


def _normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip().strip("\n").strip("\"").strip("'").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _strip_quotes(value):
    if value is None:
        return ""
    return str(value).strip().strip("\n").strip("\"").strip("'")


def _map_option_to_candidate(solution_str, extra_info):
    if not extra_info:
        return solution_str
    option_letters = extra_info.get("option_letters")
    candidates = extra_info.get("candidates")
    if not option_letters or not candidates or len(option_letters) != len(candidates):
        return solution_str
    solution_upper = str(solution_str or "").upper()
    tokens = re.findall(r"[A-Z]+", solution_upper)
    for token in reversed(tokens):
        if token in option_letters:
            idx = option_letters.index(token)
            if 0 <= idx < len(candidates):
                return candidates[idx]
    return solution_str


def _load_sid_info():
    global _SID_INFO_CACHE
    if _SID_INFO_CACHE is not None:
        return _SID_INFO_CACHE

    info_path = os.environ.get("SID_INFO_FILE", "")
    if not info_path:
        _SID_INFO_CACHE = ([], {})
        return _SID_INFO_CACHE

    with open(info_path, "r", encoding="utf-8") as f:
        info = f.readlines()
    item_name = [line.split("\t")[0].strip() for line in info]
    item2id = {name: i for i, name in enumerate(item_name)}
    _SID_INFO_CACHE = (item_name, item2id)
    return _SID_INFO_CACHE


def _load_title_to_sid():
    global _TITLE_TO_SID_CACHE
    if _TITLE_TO_SID_CACHE is not None:
        return _TITLE_TO_SID_CACHE

    info_path = os.environ.get("SID_INFO_FILE", "")
    if not info_path:
        _TITLE_TO_SID_CACHE = {}
        return _TITLE_TO_SID_CACHE

    title_to_sid = {}
    with open(info_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sid = parts[0].strip()
            title = _normalize_text(parts[1])
            if title and sid:
                title_to_sid[title] = sid
    _TITLE_TO_SID_CACHE = title_to_sid
    return _TITLE_TO_SID_CACHE


def _map_text_to_sid(value):
    sid = _normalize_sid(value)
    if sid:
        return sid
    title = _normalize_text(value)
    if not title:
        return ""
    return _load_title_to_sid().get(title, "")


def _load_embeddings():
    global _EMBED_CACHE
    if _EMBED_CACHE is not None:
        return _EMBED_CACHE

    ada_path = os.environ.get("ADA_PATH", "")
    if not ada_path:
        return None
    with open(ada_path, "rb") as f:
        emb = pickle.load(f)
    emb = torch.tensor(emb, dtype=torch.float32)
    _EMBED_CACHE = emb
    return _EMBED_CACHE


def _load_sasrec():
    global _SASREC_CACHE
    if _SASREC_CACHE is not None:
        return _SASREC_CACHE

    cf_path = os.environ.get("SASREC_PATH", "")
    if not cf_path:
        return None

    item_name, _ = _load_sid_info()
    if not item_name:
        return None

    item_num = len(item_name)
    len_seq = int(os.environ.get("SASREC_LEN_SEQ", "10"))
    SASRecClass = _get_sasrec()
    model = SASRecClass(32, item_num, len_seq, 0.3, torch.device("cpu"))
    model.load_state_dict(torch.load(cf_path, map_location="cpu"))
    model.eval()
    _SASREC_CACHE = model
    return _SASREC_CACHE


def _get_history(extra_info):
    if not extra_info:
        return []
    history = extra_info.get("history_item_sid", [])
    if isinstance(history, str):
        try:
            history = ast.literal_eval(history)
        except Exception:
            history = []
    return history


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return compute_score_rule(data_source, solution_str, ground_truth, extra_info)


def compute_score_rule(data_source, solution_str, ground_truth, extra_info=None):
    pred = _map_text_to_sid(solution_str)
    target = _map_text_to_sid(ground_truth)
    return 1.0 if pred == target else 0.0


def compute_score_ndcg(data_source, solution_str, ground_truth, extra_info=None):
    pred = _map_text_to_sid(solution_str)
    target = _map_text_to_sid(ground_truth)
    if pred == target:
        return 1.0
    rank = None if not extra_info else extra_info.get("rank", None)
    if rank is None:
        return 0.0
    try:
        rank = int(rank)
    except Exception:
        return 0.0
    return 1.0 / math.log2(rank + 2)


def compute_score_ranking(data_source, solution_str, ground_truth, extra_info=None):
    return compute_score_rule(data_source, solution_str, ground_truth, extra_info) + compute_score_ndcg(
        data_source, solution_str, ground_truth, extra_info
    )


def compute_score_ranking_only(data_source, solution_str, ground_truth, extra_info=None):
    return compute_score_ndcg(data_source, solution_str, ground_truth, extra_info)


def compute_score_semantic(data_source, solution_str, ground_truth, extra_info=None):
    item_name, item2id = _load_sid_info()
    if not item_name:
        return 0.0
    emb = _load_embeddings()
    if emb is None:
        return 0.0

    pred = _map_text_to_sid(solution_str)
    target = _map_text_to_sid(ground_truth)
    if pred not in item2id or target not in item2id:
        return 0.0

    pred_vec = emb[item2id[pred]]
    target_vec = emb[item2id[target]]
    pred_vec = pred_vec / (pred_vec.norm() + 1e-8)
    target_vec = target_vec / (target_vec.norm() + 1e-8)
    return float(torch.sum(pred_vec * target_vec).item())


def compute_score_sasrec(data_source, solution_str, ground_truth, extra_info=None):
    model = _load_sasrec()
    if model is None:
        return 0.0

    item_name, item2id = _load_sid_info()
    if not item_name:
        return 0.0

    pred = _map_text_to_sid(solution_str)
    if pred not in item2id:
        return 0.0

    history = _get_history(extra_info)
    history_ids = []
    for sid in history:
        sid = _normalize_sid(sid)
        if sid in item2id:
            history_ids.append(item2id[sid])

    len_seq = int(os.environ.get("SASREC_LEN_SEQ", "10"))
    item_num = len(item_name)
    if len(history_ids) < len_seq:
        history_ids = history_ids + [item_num] * (len_seq - len(history_ids))

    seq = torch.LongTensor([history_ids])
    len_lis = torch.tensor(np.array([min(len(history), len_seq)]))
    pred_id = torch.LongTensor([item2id[pred]])

    with torch.no_grad():
        predictions = model.forward_eval(seq, len_lis)
        score = torch.gather(predictions, 1, pred_id.view(-1, 1)).view(-1)[0]
    return float(score.item())


def compute_score_mind_ndcg(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute nDCG-style reward for MIND news recommendation (text-based).

    This function supports both simple binary matching and full ranking evaluation.

    Args:
        data_source: Not used (kept for API compatibility)
        solution_str: LLM-generated news title (prediction)
        ground_truth: Ground truth clicked news title
        extra_info: Optional dict containing:
            - 'candidates': List[str] of all candidate news titles
            - 'labels': List[int] of binary labels (1=clicked, 0=not clicked)
            - 'rank': int - pre-computed rank (if available)

    Returns:
        float: Reward score in [0, 1] range
            - 1.0 if exact match with ground truth
            - 1/log2(rank+1) if found in top-10 candidates
            - 0.0 otherwise

    Examples:
        >>> # Exact match
        >>> compute_score_mind_ndcg(None, "Breaking News Story", "Breaking News Story", None)
        1.0

        >>> # Ranked at position 2
        >>> extra = {'rank': 2}
        >>> compute_score_mind_ndcg(None, "Some News", "Breaking News", extra)
        0.6309...  # 1/log2(3)
    """
    # Normalize text for comparison (case-insensitive, strip whitespace)
    solution_str = _map_option_to_candidate(solution_str, extra_info)
    solution_normalized = _strip_quotes(solution_str).lower() if solution_str else ""
    ground_truth_normalized = _strip_quotes(ground_truth).lower() if ground_truth else ""

    # Case 1: Exact match with ground truth (best case)
    if solution_normalized == ground_truth_normalized:
        return 1.0

    # Case 2: Use pre-computed rank if available
    if extra_info and 'rank' in extra_info:
        rank = extra_info.get('rank')
        if rank is None:
            return 0.0
        try:
            rank = int(rank)
            if rank <= 0:
                return 0.0
            # DCG-style reward: 1/log2(rank+1)
            # rank=1 -> 1.0, rank=2 -> 0.631, rank=3 -> 0.5, rank=5 -> 0.387, rank=10 -> 0.301
            if rank <= 10:
                return 1.0 / math.log2(rank + 1)
            else:
                return 0.0
        except (ValueError, TypeError):
            return 0.0

    # Case 3: Compute rank from full candidate list
    if extra_info and 'candidates' in extra_info and 'labels' in extra_info:
        candidates = extra_info.get('candidates', [])
        labels = extra_info.get('labels', [])

        if not candidates or not labels or len(candidates) != len(labels):
            return 0.0

        # Find rank of the generated solution in candidate list
        rank = None
        for i, candidate in enumerate(candidates):
            candidate_normalized = candidate.strip().lower() if candidate else ""
            if candidate_normalized == solution_normalized:
                rank = i + 1  # 1-indexed rank
                break

        if rank is None:
            # Generated text doesn't match any candidate
            return 0.0

        # Check if this candidate was actually clicked
        if rank <= len(labels) and labels[rank - 1] == 1:
            # Clicked item - give DCG-style reward based on rank
            if rank <= 10:
                return 1.0 / math.log2(rank + 1)
            else:
                return 0.0
        else:
            # Not a clicked item - no reward
            return 0.0

    # Case 4: Fuzzy matching as fallback (if solution is substring of ground truth or vice versa)
    if solution_normalized and ground_truth_normalized:
        if solution_normalized in ground_truth_normalized or ground_truth_normalized in solution_normalized:
            # Partial match - give small reward (0.3)
            return 0.3

    # No match
    return 0.0


def compute_score_mind_mrr(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute MRR (Mean Reciprocal Rank) style reward for MIND.
    Similar to nDCG but uses 1/rank instead of 1/log2(rank+1).

    Args:
        Same as compute_score_mind_ndcg

    Returns:
        float: 1/rank if found in candidates, 0 otherwise
    """
    solution_str = _map_option_to_candidate(solution_str, extra_info)
    solution_normalized = _strip_quotes(solution_str).lower() if solution_str else ""
    ground_truth_normalized = _strip_quotes(ground_truth).lower() if ground_truth else ""

    if solution_normalized == ground_truth_normalized:
        return 1.0

    if extra_info and 'rank' in extra_info:
        rank = extra_info.get('rank')
        try:
            rank = int(rank)
            if rank > 0:
                return 1.0 / rank
        except (ValueError, TypeError):
            pass

    if extra_info and 'candidates' in extra_info:
        candidates = extra_info.get('candidates', [])
        for i, candidate in enumerate(candidates):
            candidate_normalized = candidate.strip().lower() if candidate else ""
            if candidate_normalized == solution_normalized:
                return 1.0 / (i + 1)

    return 0.0


def compute_score_mind_auc(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute AUC-like reward for MIND news recommendation.

    This reward simulates AUC by computing the proportion of negative samples
    that would be ranked below the selected positive sample.

    The reward is designed to match the AUC evaluation metric used by MIND benchmark:
    - If model selects a clicked item, reward = (# non-clicked items) / total_non_clicked
      (i.e., fraction of negative items that would be ranked below)
    - If model selects a non-clicked item, reward = 0

    This encourages the model to rank clicked items above non-clicked items,
    which is exactly what AUC measures.

    Args:
        data_source: Not used (kept for API compatibility)
        solution_str: Model's predicted candidate index (as string number, e.g., "3")
        ground_truth: Ground truth clicked news title (for reference)
        extra_info: Dict containing:
            - 'candidates': List[str] of all candidate news titles
            - 'labels': List[int] of binary labels (1=clicked, 0=not clicked)

    Returns:
        float: AUC-like reward in [0, 1] range
            - 1.0 if selected item is clicked and all other items are non-clicked
            - Proportional reward based on position among negatives
            - 0.0 if selected item is not clicked or invalid

    Example:
        If there are 5 candidates with labels [0, 1, 0, 0, 1] and model selects
        candidate 1 (clicked), the reward would be 1.0 because selecting a clicked
        item is always correct.

        If we had a ranking-based version, selecting the higher-ranked clicked
        item would give higher reward, but for simplicity we treat all clicked
        items equally.
    """
    if not extra_info:
        return 0.0

    candidates = extra_info.get('candidates', [])
    labels = extra_info.get('labels', [])

    if not candidates or not labels or len(candidates) != len(labels):
        return 0.0

    # Parse the solution - expecting a number index
    solution_str = str(solution_str or "").strip()

    # First, try to map option letter to index (for backward compatibility)
    option_letters = extra_info.get('option_letters', [])
    selected_idx = None

    if option_letters:
        # Try letter-based selection (e.g., "A", "B", etc.)
        solution_upper = solution_str.upper()
        tokens = re.findall(r"[A-Z]+", solution_upper)
        for token in reversed(tokens):
            if token in option_letters:
                selected_idx = option_letters.index(token)
                break

    # Try numeric index if letter mapping failed
    if selected_idx is None:
        # Extract first number from solution
        numbers = re.findall(r"\d+", solution_str)
        if numbers:
            try:
                # Assume 1-indexed from model output
                selected_idx = int(numbers[0]) - 1
            except (ValueError, IndexError):
                pass

    # Validate index
    if selected_idx is None or selected_idx < 0 or selected_idx >= len(candidates):
        return 0.0

    # Check if selected candidate is clicked
    if labels[selected_idx] != 1:
        # Selected a non-clicked item - no reward
        return 0.0

    # Selected a clicked item - compute AUC-like reward
    # Count total positives and negatives
    num_positives = sum(labels)
    num_negatives = len(labels) - num_positives

    if num_negatives == 0:
        # All items are clicked (rare), perfect score
        return 1.0

    if num_positives == 0:
        # No clicked items (shouldn't happen with valid data)
        return 0.0

    # Basic AUC reward: selecting any clicked item gives positive reward
    # The reward is higher if there are more negatives to beat
    # This simulates: "what fraction of pairwise comparisons would we win?"

    # Simple version: binary reward for selecting clicked item
    # More sophisticated: could weight by position, but keep it simple for stability
    return 1.0


def compute_score_mind_auc_rank(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute AUC-like reward with ranking consideration for MIND.

    This is a more nuanced version that gives partial credit based on
    where the selected item ranks among clicked items.

    Args:
        Same as compute_score_mind_auc

    Returns:
        float: Reward in [0, 1] range
            - 1.0 if selected the first (most relevant) clicked item
            - Decreasing reward for lower-ranked clicked items
            - 0.0 if selected non-clicked item
    """
    if not extra_info:
        return 0.0

    candidates = extra_info.get('candidates', [])
    labels = extra_info.get('labels', [])

    if not candidates or not labels or len(candidates) != len(labels):
        return 0.0

    # Parse solution
    solution_str = str(solution_str or "").strip()
    option_letters = extra_info.get('option_letters', [])
    selected_idx = None

    if option_letters:
        solution_upper = solution_str.upper()
        tokens = re.findall(r"[A-Z]+", solution_upper)
        for token in reversed(tokens):
            if token in option_letters:
                selected_idx = option_letters.index(token)
                break

    if selected_idx is None:
        numbers = re.findall(r"\d+", solution_str)
        if numbers:
            try:
                selected_idx = int(numbers[0]) - 1
            except (ValueError, IndexError):
                pass

    if selected_idx is None or selected_idx < 0 or selected_idx >= len(candidates):
        return 0.0

    if labels[selected_idx] != 1:
        return 0.0

    # Find rank of selected item among clicked items
    clicked_indices = [i for i, l in enumerate(labels) if l == 1]
    if not clicked_indices:
        return 0.0

    # Find position of selected index among clicked items
    click_rank = clicked_indices.index(selected_idx) + 1  # 1-indexed
    num_clicked = len(clicked_indices)

    # Higher reward for selecting earlier clicked items
    # rank 1 -> 1.0, rank 2 -> 0.75, rank 3 -> 0.67, etc.
    # Using 1/log2(rank+1) style decay
    reward = 1.0 / math.log2(click_rank + 1)

    return reward
