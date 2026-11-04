"""Frozen contextual titles with newly trained candidate-conditioned history interest."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from full_impression_training import epoch
from mind_cpu_ranker import Priors, digest, write_json
from mind_public_contextual import CORPUS_SHA, NEWS_SHA, PublicEncoder, REVISION
from mind_pretrained_training import main


EPOCHS = 2
RSS_LIMIT = 3758096384


def conditional_context(history, candidate, groups, mask, recency):
    counts = torch.bincount(groups, minlength=len(history))
    offsets = torch.cumsum(counts, dim=0) - counts
    order = torch.argsort(groups, stable=True)
    positions = torch.empty_like(groups)
    positions[order] = torch.arange(len(groups)) - offsets[groups[order]]
    indices_parts, context_parts = [], []
    width = 1
    while width <= 1 << (int(counts.max()) - 1).bit_length():
        selected = torch.nonzero((counts > width // 2) & (counts <= width), as_tuple=True)[0]
        if len(selected):
            mapping = torch.full((len(history),), -1, dtype=torch.long)
            mapping[selected] = torch.arange(len(selected))
            indices = torch.nonzero(mapping[groups] >= 0, as_tuple=True)[0]
            row, column = mapping[groups[indices]], positions[indices]
            padded = torch.zeros((len(selected), width, candidate.shape[1]), dtype=candidate.dtype)
            padded[row, column] = candidate[indices]
            values, valid = history[selected], mask[selected]
            logits = 8 * torch.bmm(padded, values.transpose(1, 2)) + recency
            logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
            logits = torch.where(valid.any(dim=1)[:, None, None], logits, torch.zeros_like(logits))
            attention = torch.softmax(logits, dim=-1) * valid[:, None, :]
            context = F.normalize(torch.bmm(attention, values), dim=-1)
            indices_parts.append(indices)
            context_parts.append(context[row, column])
        width *= 2
    permutation = torch.argsort(torch.cat(indices_parts))
    return torch.cat(context_parts)[permutation]


def load_vectors(path, corpus):
    proof = json.loads(path.with_suffix(".json").read_bytes())
    if (proof["vectors_sha256"] != digest(path) or proof["revision"] != REVISION
            or proof["dimensions"] != 384 or proof["maximum_tokens"] != 64
            or proof["text"] != "title_only" or proof["dtype"] != "float32"
            or proof["pooling"] != "masked_mean_then_l2_normalize"
            or proof["source_sha256"] != digest(Path(__file__).with_name("mind_public_contextual.py"))
            or proof["news_sha256"] != NEWS_SHA or proof["corpus_sha256"] != CORPUS_SHA
            or proof["click_labels_read"] or proof["dev_or_test_contents_read"]):
        raise ValueError("Pinned label-free contextual representation required")
    with np.load(path, allow_pickle=False) as arrays:
        if arrays["names"].tolist() != corpus.names:
            raise ValueError("Contextual article identities differ")
        return arrays["vectors"]


class CandidateInterest(nn.Module):
    def __init__(self, corpus, vectors, priors, public_directory):
        super().__init__()
        if (vectors.dtype != np.float32 or vectors.shape != (len(corpus.names) + 1, 384)
                or not np.isfinite(vectors).all() or np.any(vectors[0])
                or not np.allclose(np.linalg.norm(vectors[1:], axis=1), 1, rtol=1e-4, atol=1e-4)
                or priors.pooled.shape != (len(corpus.names) + 1, 3)
                or not np.isfinite(priors.pooled).all()):
            raise ValueError("Aligned finite contextual vectors/priors and zero padding required")
        self.corpus, self.priors, self.public_directory = corpus, priors, Path(public_directory)
        self.prior_fold, self.cached = None, None
        self.mode = "public_contextual_candidate_conditioned_interest"
        self.register_buffer("features", torch.from_numpy(vectors), persistent=False)
        self.register_buffer("categories", torch.from_numpy(corpus.category), persistent=False)
        self.register_buffer("subcategories", torch.from_numpy(corpus.subcategory), persistent=False)
        self.register_buffer("pooled_logits", torch.from_numpy(priors.pooled[:, 0]), persistent=False)
        self.register_buffer("fold_logits", torch.from_numpy(np.stack(
            [table[:, 0] for table in priors.tables])) if hasattr(priors, "tables") else None, persistent=False)
        self.projection = nn.Linear(384, 64, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.category = nn.Embedding(len(corpus.categories) + 1, 16, padding_idx=0)
        self.subcategory = nn.Embedding(len(corpus.subcategories) + 1, 16, padding_idx=0)
        nn.init.zeros_(self.category.weight)
        nn.init.zeros_(self.subcategory.weight)
        self.taxonomy_projection = nn.Linear(32, 64, bias=False)
        self.dropout = nn.Dropout(.1)
        self.recency = nn.Parameter(torch.zeros(30))
        self.semantic_scale = nn.Parameter(torch.tensor(4.))
        self.head = nn.Sequential(nn.Linear(192, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode=True):
        if mode:
            self.cached = None
        return super().train(mode)

    def encode(self, news):
        taxonomy = torch.cat((self.category(self.categories[news]), self.subcategory(self.subcategories[news])), dim=1)
        value = self.projection(self.dropout(self.features[news])) + self.taxonomy_projection(taxonomy)
        return F.normalize(value, dim=1)

    @torch.no_grad()
    def prepare_inference(self):
        if self.training:
            raise ValueError("Only frozen evaluation representations may be cached")
        self.cached = torch.cat([self.encode(torch.arange(start, min(start + 512, len(self.features))))
                                 for start in range(0, len(self.features), 512)])

    def forward(self, histories, candidates, groups):
        if (histories.ndim != 2 or histories.shape[1] != 30 or candidates.ndim != 1 or not len(candidates)
                or groups.shape != candidates.shape
                or any(value.dtype != torch.long for value in (histories, candidates, groups))
                or torch.any(histories < 0) or torch.any(histories >= len(self.features))
                or torch.any(candidates < 1) or torch.any(candidates >= len(self.features))
                or torch.any(groups < 0) or torch.any(groups >= len(histories))):
            raise ValueError("Complete aligned history/candidate identities required")
        if self.training and (self.prior_fold not in (0, 1) or self.fold_logits is None):
            raise ValueError("Explicit opposite-fold training priors required")
        if self.cached is None or self.training:
            unique, inverse = torch.unique(torch.cat((histories.reshape(-1), candidates)), return_inverse=True)
            encoded = self.encode(unique)
            history = encoded[inverse[:histories.numel()]].reshape(*histories.shape, 64)
            candidate = encoded[inverse[histories.numel():]]
        else:
            history, candidate = self.cached[histories], self.cached[candidates]
        context = conditional_context(history, candidate, groups, histories != 0, self.recency)
        interest = (candidate * context).sum(dim=1)
        residual = self.head(torch.cat((candidate, context, candidate * context), dim=1)).squeeze(-1)
        prior = (self.fold_logits[1 - self.prior_fold, candidates] if self.training
                 else self.pooled_logits[candidates])
        result = prior + self.semantic_scale * interest + residual
        if not torch.isfinite(result).all():
            raise FloatingPointError("Candidate-conditioned contextual scores must remain finite")
        return result

    @torch.no_grad()
    def scores(self, histories, candidates, groups):
        histories = torch.from_numpy(histories.astype(np.int64))
        candidates, groups = torch.from_numpy(candidates.astype(np.int64)), torch.from_numpy(groups.astype(np.int64))
        return torch.cat([self(histories, candidates[start:start + 8192], groups[start:start + 8192])
                          for start in range(0, len(candidates), 8192)]).numpy()

    def extend(self, news):
        if self.training:
            raise ValueError("Only frozen inference may extend article identities")
        old = len(self.corpus.names)
        self.corpus.extend(news)
        if len(self.corpus.names) > old:
            names = self.corpus.names[old:]
            encoder = PublicEncoder(self.public_directory)
            added = np.concatenate([encoder.encode([news[name]["title"] for name in names[start:start + 128]])
                                    for start in range(0, len(names), 128)])
            self.features = torch.cat((self.features, torch.from_numpy(added)))
            self.priors.extend(self.corpus)
            self.categories = torch.from_numpy(self.corpus.category)
            self.subcategories = torch.from_numpy(self.corpus.subcategory)
            self.pooled_logits = torch.from_numpy(self.priors.pooled[:, 0])
            self.cached = None
            self.prepare_inference()


def fit(source, arrays, corpus, public_encoder, groups, directory, deadline):
    torch.manual_seed(42)
    priors = Priors(source, corpus, groups, deadline, RSS_LIMIT)
    model = CandidateInterest(corpus, load_vectors(public_encoder, corpus), priors, public_encoder.parent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0001)
    expected, coverage = source.counts(groups), []
    selected = np.zeros(len(source.days), dtype=bool)
    selected[groups] = True
    bitmap = hashlib.sha256(np.packbits(selected).tobytes()).hexdigest()
    for index in range(EPOCHS):
        parts = []
        for fold in (index % 2, 1 - index % 2):
            model.prior_fold = fold
            fitting = groups[source.folds[groups] == fold]
            if not len(fitting):
                raise ValueError("Both nonempty fitting folds required")
            print(json.dumps({"stage": "contextual_interest_fold", "epoch": index + 1,
                              "fold": fold, "impressions": len(fitting)}), flush=True)
            parts.append(epoch(model, fitting, arrays, optimizer, 42 + index, deadline))
        record = {key: sum(part[key] for part in parts)
                  for key in ("impressions", "exposures", "positives", "optimizer_steps")}
        record["mean_loss"] = sum(part["mean_loss"] * part["impressions"] for part in parts) / len(groups)
        if any(record[key] != value for key, value in expected.items()):
            raise ValueError("Incomplete contextual all-candidate epoch")
        coverage.append(dict(record, epoch=index + 1, consumed_source_positions_sha256=bitmap))
        print(json.dumps({"stage": "contextual_complete_epoch", **coverage[-1]}), flush=True)
    directory.mkdir()
    torch.save(model.state_dict(), directory / "model.pt")
    priors.save(directory)
    write_json(directory / "epoch_coverage.json", coverage)
    write_json(directory / "model_config.json", {
        "hypothesis": "frozen_minilm_trainable_candidate_conditioned_history_v1",
        "public_encoder_sha256": digest(public_encoder), "public_revision": REVISION,
        "epochs": EPOCHS, "seed": 42, "history_length": 30, "public_dimensions": 384,
        "learned_dimensions": 64, "attention_temperature": 8., "initial_semantic_scale": 4.,
        "dropout": .1, "optimizer": "AdamW", "learning_rate": .0005, "weight_decay": .0001,
        "loss": "complete_impression_mean_positive_log_softmax", "candidate_sampling": "none",
        "click_prior_training": "opposite_blake2b_impression_fold_only", "source_coverage": expected,
        "ranking_tie_dtype": "float32"})
    return model


def restore(directory, corpus, public_encoder):
    config = json.loads((directory / "model_config.json").read_bytes())
    if (config["public_encoder_sha256"] != digest(public_encoder) or config["epochs"] != EPOCHS
            or config["public_revision"] != REVISION or config["candidate_sampling"] != "none"):
        raise ValueError("Fixed contextual source or recipe changed")
    model = CandidateInterest(corpus, load_vectors(public_encoder, corpus), Priors.load(directory), public_encoder.parent)
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True, map_location="cpu"), strict=True)
    model.eval()
    model.prepare_inference()
    return model


if __name__ == "__main__":
    main(fitter=fit)
