import ast
import math
import os
import pickle

import numpy as np
import torch

from sasrec import SASRec


_SID_INFO_CACHE = None
_EMBED_CACHE = None
_SASREC_CACHE = None


def _normalize_sid(value):
    if value is None:
        return ""
    return str(value).strip().strip("\n").strip("\"").strip("'")


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
    model = SASRec(32, item_num, len_seq, 0.3, torch.device("cpu"))
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
    pred = _normalize_sid(solution_str)
    target = _normalize_sid(ground_truth)
    return 1.0 if pred == target else 0.0


def compute_score_ndcg(data_source, solution_str, ground_truth, extra_info=None):
    pred = _normalize_sid(solution_str)
    target = _normalize_sid(ground_truth)
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

    pred = _normalize_sid(solution_str)
    target = _normalize_sid(ground_truth)
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

    pred = _normalize_sid(solution_str)
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
