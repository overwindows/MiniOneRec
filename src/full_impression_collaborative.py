"""Semantic-plus-collaborative article readout; every impression/click is retained."""

import torch
from torch import nn
import torch.nn.functional as F

from full_impression_ranker import FrozenArticleRanker


class CollaborativeArticleRanker(FrozenArticleRanker):
    def __init__(self, features, mode="collaborative_mean"):
        nn.Module.__init__(self)
        if (features.ndim != 2 or features.shape[0] < 2 or features.dtype != torch.float32
                or features.requires_grad or not torch.isfinite(features).all()
                or torch.any(features[0] != 0) or mode != "collaborative_mean"):
            raise ValueError("Finite fixed FP32 semantics,zero padding and registered collaborative mode required")
        self.register_buffer("features", features, persistent=False)
        self.mode, self.history_limit = mode, 30
        self.residual = nn.Embedding.from_pretrained(torch.zeros_like(features), freeze=False, padding_idx=0)
        self.article_bias = nn.Embedding(len(features), 1, padding_idx=0)
        nn.init.zeros_(self.article_bias.weight)
        self.recency = nn.Parameter(torch.zeros(self.history_limit))

    def forward(self, histories, candidates, groups):
        self._validate_inputs(histories, candidates, groups)
        history = F.normalize(F.embedding(histories, self.features) + self.residual(histories), dim=-1)
        candidate = F.normalize(F.embedding(candidates, self.features) + self.residual(candidates), dim=-1)
        mask = histories != 0
        safe = mask.clone()
        safe[~mask.any(dim=-1), -1] = True
        weights = self.recency.expand_as(histories).masked_fill(~safe, -torch.inf).softmax(dim=-1) * mask
        user = F.normalize((history * weights.unsqueeze(-1)).sum(dim=1), dim=-1)
        scores = 10 * (candidate * user[groups]).sum(dim=-1) + self.article_bias(candidates).squeeze(-1)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Collaborative full-impression scores must remain finite")
        return scores


class CollaborativeReplay:
    def __init__(self, features, state):
        expected = {"residual.weight": tuple(features.shape), "article_bias.weight": (len(features), 1),
                    "recency": (30,)}
        if (set(state) != set(expected) or features.device.type != "cpu"
                or features.dtype != torch.float32 or not torch.isfinite(features).all()
                or torch.any(features[0])
                or any(tuple(state[key].shape) != shape or state[key].dtype != torch.float32
                       or not torch.isfinite(state[key]).all() for key, shape in expected.items())
                or torch.any(state["residual.weight"][0]) or torch.any(state["article_bias.weight"][0])):
            raise ValueError("Finite exact collaborative checkpoint without a semantic-feature payload required")
        self.articles = F.normalize(features + state["residual.weight"], dim=-1)
        self.bias = state["article_bias.weight"].squeeze(-1)
        self.recency = state["recency"]

    @torch.inference_mode()
    def __call__(self, histories, candidates, groups):
        mask = histories != 0
        logits = self.recency.expand_as(histories)
        safe = mask.clone()
        safe[~mask.any(dim=-1), -1] = True
        weights = torch.exp(logits.masked_fill(~safe, -torch.inf) -
                            logits.masked_fill(~safe, -torch.inf).max(dim=-1, keepdim=True).values)
        weights = weights / weights.sum(dim=-1, keepdim=True) * mask
        user = F.normalize((self.articles[histories] * weights.unsqueeze(-1)).sum(dim=1), dim=-1)
        return (10 * (self.articles[candidates] * user[groups]).sum(dim=-1) + self.bias[candidates]).numpy()
