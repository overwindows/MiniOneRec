"""Joint semantic/text learning without article-ID residuals or global item biases."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from full_impression_shared_text import SharedTextRanker, SharedTextReplay


MODE = "joint_semantic_text_multiinterest"


class MultiViewRanker(SharedTextRanker):
    def __init__(self, features, mode, terms, initial):
        if (mode != MODE or features.device.type != "cpu" or features.shape != (terms.shape[0], 128)
                or features.dtype != torch.float32 or not torch.isfinite(features).all()
                or features.requires_grad or features[0].any()):
            raise ValueError("Finite fixed128 semantic features with zero padding required")
        super().__init__(torch.zeros((len(features), 1)), "shared_text_multiinterest", terms, initial)
        self.features, self.mode, self.dimension = features, mode, 128
        self.semantic_project = nn.Linear(128, 64, bias=False)
        self.queries = nn.Parameter(torch.randn(4, 128) * .1)

    def encode(self, identities):
        lexical = super().encode(identities)
        semantic = F.normalize(self.semantic_project(self.features[identities]), dim=-1)
        return F.normalize(torch.cat((semantic, lexical), dim=-1), dim=-1)


class MultiViewReplay(SharedTextReplay):
    def __init__(self, features, terms, state):
        expected = {"words": (terms.shape[1], 64), "queries": (4, 128),
                    "semantic_project.weight": (64, 128), "recency": (30,)}
        if (set(state) != set(expected) or features.device.type != "cpu"
                or features.shape != (terms.shape[0], 128)
                or features.dtype != torch.float32 or not torch.isfinite(features).all()
                or features[0].any() or terms[0].nnz
                or any(tuple(state[key].shape) != shape or state[key].dtype != torch.float32
                       or not torch.isfinite(state[key]).all() for key, shape in expected.items())):
            raise ValueError("Exact finite multiview checkpoint and aligned fixed source features required")
        lexical = np.asarray(terms @ state["words"].numpy(), dtype=np.float32)
        lexical /= np.maximum(np.linalg.norm(lexical, axis=1, keepdims=True), 1e-12)
        semantic = features.numpy() @ state["semantic_project.weight"].numpy().T
        semantic /= np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-12)
        self.articles = np.concatenate((semantic, lexical), axis=1)
        self.articles /= np.maximum(np.linalg.norm(self.articles, axis=1, keepdims=True), 1e-12)
        self.recency, self.queries = state["recency"].numpy(), state["queries"].numpy()
