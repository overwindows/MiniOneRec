"""Independent raw-score metrics and CPU score replay for the frozen selection."""

import numpy as np
import torch
import torch.nn.functional as F


SCORE_ATOL = 1e-4
SCORE_RTOL = 1e-5


def metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (labels.ndim != 1 or scores.shape != labels.shape or not len(labels)
            or not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all()):
        raise ValueError("Every candidate needs an aligned binary target and finite score")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Never silently exclude a single-class impression")
    ascending = np.argsort(scores)
    ordered = scores[ascending]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    counts = np.diff(np.r_[starts, len(scores)])
    positive_counts = np.add.reduceat(labels[ascending].astype(np.int64), starts)
    negative_counts = counts - positive_counts
    preceding_negatives = np.cumsum(negative_counts) - negative_counts
    auc = np.sum(positive_counts * (preceding_negatives + .5 * negative_counts)) / (positives * negatives)
    ranked = labels[ascending[::-1]]
    mrr = np.sum(ranked / np.arange(1, len(ranked) + 1)) / positives
    discounts = 1 / np.log2(np.arange(min(10, len(ranked))) + 2)
    ndcgs = [
        np.sum(ranked[:k] * discounts[:k]) / np.sum(discounts[:min(positives, k)])
        for k in (5, 10)
    ]
    return np.asarray((auc, mrr, *ndcgs), dtype=np.float64)


def check_scores(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, rtol=SCORE_RTOL, atol=SCORE_ATOL)):
        raise ValueError("Complete CPU/source-order score replay differs beyond registered FP32 tolerance")
    return float(np.max(np.abs(actual - expected))) if len(actual) else 0.


class CachedReplay:
    """Project every article once, then independently reconstruct history scores."""

    def __init__(self, features, state, mode):
        if mode not in ("mean", "candidate_attention"):
            raise ValueError("Only the two fixed recipes may be replayed")
        expected = {
            "project.0.weight": (128, features.shape[1]), "project.0.bias": (128,),
            "project.2.weight": (128,), "project.2.bias": (128,),
            "article_bias.weight": (1, features.shape[1]), "article_bias.bias": (1,),
        }
        if mode == "candidate_attention":
            expected["recency"] = (30,)
        if set(state) != set(expected) or any(
                tuple(state[name].shape) != shape or state[name].dtype != torch.float32
                or not torch.isfinite(state[name]).all() for name, shape in expected.items()):
            raise ValueError("Expected finite complete head parameters without a frozen-feature payload")
        if (features.ndim != 2 or not torch.isfinite(features).all() or torch.any(features[0] != 0)
                or features.device.type != "cpu"):
            raise ValueError("Independent replay requires finite CPU features with zero padding")
        with torch.inference_mode():
            projected = F.linear(features, state["project.0.weight"], state["project.0.bias"])
            projected = F.gelu(projected)
            projected = F.layer_norm(projected, (128,), state["project.2.weight"], state["project.2.bias"])
            self.projected = F.normalize(projected, dim=-1)
            self.bias = F.linear(features, state["article_bias.weight"], state["article_bias.bias"]).squeeze(-1)
        self.mode = mode
        self.recency = state.get("recency")

    @torch.inference_mode()
    def __call__(self, histories, candidates, groups):
        mask = histories != 0
        history = self.projected[histories]
        candidate = self.projected[candidates]
        if self.mode == "mean":
            pooled = (history * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            user = F.normalize(pooled, dim=-1)[groups]
        else:
            history = history[groups]
            selected_mask = mask[groups]
            logits = 4 * (candidate.unsqueeze(1) * history).sum(dim=-1) + self.recency
            safe_mask = selected_mask.clone()
            safe_mask[~selected_mask.any(dim=-1), -1] = True
            weights = logits.masked_fill(~safe_mask, -torch.inf).softmax(dim=-1) * selected_mask
            user = F.normalize((history * weights.unsqueeze(-1)).sum(dim=1), dim=-1)
        return (10 * (candidate * user).sum(dim=-1) + self.bias[candidates]).numpy()
