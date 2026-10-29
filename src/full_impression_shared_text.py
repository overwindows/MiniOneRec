"""Shared TF-IDF term embeddings trained directly by full-impression choices."""

import numpy as np
from scipy import sparse
import torch
from torch import nn
import torch.nn.functional as F

from full_impression_ranker import FrozenArticleRanker


MODE = "shared_text_multiinterest"


class SharedTextRanker(FrozenArticleRanker):
    def __init__(self, features, mode, terms, initial):
        nn.Module.__init__(self)
        if (mode != MODE or features.device.type != "cpu" or features.shape != (terms.shape[0], 1)
                or not sparse.isspmatrix_csr(terms) or terms.dtype != np.float32
                or terms[0].nnz or not np.isfinite(terms.data).all() or np.any(terms.data < 0)
                or initial.shape != (terms.shape[1], 64) or initial.dtype != np.float32
                or not np.isfinite(initial).all()):
            raise ValueError("CPU-only fixed finite TF-IDF CSR,zero padding and shared64 term initialization required")
        self.register_buffer("features", features, persistent=False)
        self.mode, self.history_limit, self.terms = mode, 30, terms
        self.words = nn.Parameter(torch.from_numpy(initial.copy()))
        self.dimension = 64
        self.queries = nn.Parameter(torch.randn(4, 64) * .1)
        self.recency = nn.Parameter(torch.zeros(30))

    def encode(self, identities):
        block = self.terms[identities.numpy()]
        encoded = F.embedding_bag(torch.from_numpy(block.indices.astype(np.int64)), self.words,
                                  torch.from_numpy(block.indptr.astype(np.int64)), mode="sum",
                                  per_sample_weights=torch.from_numpy(block.data), include_last_offset=True)
        return F.normalize(encoded, dim=-1)

    def forward(self, histories, candidates, groups):
        self._validate_inputs(histories, candidates, groups)
        identities, inverse = torch.unique(torch.cat((histories.flatten(), candidates)), return_inverse=True)
        encoded = self.encode(identities)
        history = encoded[inverse[:histories.numel()]].reshape(*histories.shape, self.dimension)
        candidate = encoded[inverse[histories.numel():]]
        mask = histories != 0
        safe = mask.clone()
        safe[~mask.any(dim=-1), -1] = True
        logits = 4 * (history @ self.queries.T).transpose(1, 2) + self.recency
        weights = logits.masked_fill(~safe[:, None], -torch.inf).softmax(dim=-1) * mask[:, None]
        user = F.normalize(weights @ history, dim=-1)
        interests = 10 * (candidate[:, None] * user[groups]).sum(dim=-1)
        scores = interests.logsumexp(dim=-1) - np.log(4.)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Shared-text full-impression scores must remain finite")
        return scores


class SharedTextReplay:
    def __init__(self, terms, state):
        if (set(state) != {"words", "queries", "recency"} or state["words"].shape != (terms.shape[1], 64)
                or state["queries"].shape != (4, 64)
                or state["recency"].shape != (30,) or terms[0].nnz
                or any(value.dtype != torch.float32 or not torch.isfinite(value).all()
                       for value in state.values())):
            raise ValueError("Exact finite shared-text checkpoint and zero-padding term matrix required")
        encoded = np.asarray(terms @ state["words"].numpy(), dtype=np.float32)
        self.articles = encoded / np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-12)
        self.recency = state["recency"].numpy()
        self.queries = state["queries"].numpy()

    def __call__(self, histories, candidates, groups):
        h, c, g = (value.numpy() for value in (histories, candidates, groups))
        history, candidate = self.articles[h], self.articles[c]
        mask = h != 0
        safe = mask.copy()
        safe[~mask.any(axis=-1), -1] = True
        logits = 4 * np.einsum("bhd,kd->bkh", history, self.queries) + self.recency
        logits = np.where(safe[:, None], logits, -np.inf)
        weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True) * mask[:, None]
        user = np.einsum("bkh,bhd->bkd", weights, history)
        user /= np.maximum(np.linalg.norm(user, axis=-1, keepdims=True), 1e-12)
        interests = 10 * np.sum(candidate[:, None] * user[g], axis=-1)
        maximum = interests.max(axis=-1)
        return maximum + np.log(np.exp(interests - maximum[:, None]).mean(axis=-1))
