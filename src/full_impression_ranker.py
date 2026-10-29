"""Frozen article representations with masked history scoring and all-click loss."""

import torch
from torch import nn
import torch.nn.functional as F


def full_impression_loss(scores, labels, groups, impressions):
    if (scores.ndim != 1 or labels.shape != scores.shape or groups.shape != scores.shape
            or not len(scores) or impressions < 1 or groups.dtype != torch.long
            or groups.min() < 0 or groups.max() >= impressions
            or not torch.isfinite(scores).all() or not torch.all((labels == 0) | (labels == 1))):
        raise ValueError("Full-impression loss requires aligned finite scores, binary labels and valid groups")
    counts = torch.bincount(groups, minlength=impressions)
    positives = scores.new_zeros(impressions).scatter_add(0, groups, labels.to(scores.dtype))
    if torch.any(positives <= 0) or torch.any(positives >= counts):
        raise ValueError("Every complete training impression must contain both classes")
    maxima = scores.new_full((impressions,), -torch.inf).scatter_reduce(
        0, groups, scores.detach(), reduce="amax", include_self=True,
    )
    centered = scores - maxima[groups]
    denominator = scores.new_zeros(impressions).scatter_add(0, groups, centered.exp())
    positive_loss = -labels.to(scores.dtype) * (centered - denominator.log()[groups])
    total = scores.new_zeros(impressions).scatter_add(0, groups, positive_loss)
    loss = (total / positives).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Full-impression objective is nonfinite")
    return loss


class FrozenArticleRanker(nn.Module):
    def __init__(self, features, mode, dimension=128, history_limit=30):
        super().__init__()
        if (features.ndim != 2 or features.shape[0] < 2 or not features.is_floating_point()
                or features.requires_grad or not torch.isfinite(features).all()
                or torch.any(features[0] != 0) or mode not in ("mean", "candidate_attention")
                or dimension < 1 or history_limit < 1):
            raise ValueError("Expected finite frozen article features, zero padding and a registered scorer")
        self.register_buffer("features", features, persistent=False)
        self.mode = mode
        self.history_limit = history_limit
        self.project = nn.Sequential(nn.Linear(features.shape[1], dimension), nn.GELU(), nn.LayerNorm(dimension))
        self.article_bias = nn.Linear(features.shape[1], 1)
        self.recency = nn.Parameter(torch.zeros(history_limit)) if mode == "candidate_attention" else None

    def _validate_inputs(self, histories, candidates, groups):
        if (histories.ndim != 2 or histories.shape[1] != self.history_limit
                or candidates.ndim != 1 or groups.shape != candidates.shape or not len(candidates)
                or any(value.dtype != torch.long for value in (histories, candidates, groups))
                or torch.any(candidates <= 0) or torch.any(candidates >= len(self.features))
                or torch.any(histories < 0) or torch.any(histories >= len(self.features))
                or torch.any(groups < 0) or torch.any(groups >= len(histories))):
            raise ValueError("Ranker inputs must cover valid article indices, histories and candidate groups")
        mask = histories != 0
        if torch.any(mask[:, :-1] & ~mask[:, 1:]):
            raise ValueError("History padding must be on the left")

    def forward(self, histories, candidates, groups):
        self._validate_inputs(histories, candidates, groups)
        mask = histories != 0
        candidate_features = F.embedding(candidates, self.features)
        candidate = F.normalize(self.project(candidate_features), dim=-1)
        history = F.normalize(self.project(F.embedding(histories, self.features)), dim=-1)
        if self.mode == "mean":
            pooled = (history * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            user = F.normalize(pooled, dim=-1)[groups]
        else:
            selected = history[groups]
            selected_mask = mask[groups]
            logits = 4 * (candidate.unsqueeze(1) * selected).sum(dim=-1) + self.recency
            empty = ~selected_mask.any(dim=-1)
            safe_mask = selected_mask.clone()
            safe_mask[empty, -1] = True
            weights = logits.masked_fill(~safe_mask, -torch.inf).softmax(dim=-1)
            weights = weights * selected_mask
            user = F.normalize((selected * weights.unsqueeze(-1)).sum(dim=1), dim=-1)
        scores = 10 * (candidate * user).sum(dim=-1) + self.article_bias(candidate_features).squeeze(-1)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Masked full-impression ranker produced nonfinite scores")
        return scores
