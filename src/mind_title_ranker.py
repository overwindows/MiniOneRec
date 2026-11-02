"""Jointly train title-feature embeddings and history ranking on complete groups."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from full_impression_contract import read_news
from full_impression_training import MANIFEST_SHA, PROOF_SHA, load_contract
from mind_click_graph import digest_bytes
from mind_cpu_ranker import (
    Corpus, Source, candidate_rows, complete_loss, digest, encode_behaviors,
    limits, validate_groups, write_json,
)
from mind_lexical_glm import ENCODER_HASHES
from mind_utils import mrr_score, ndcg_score


DIMENSIONS = 32
TITLE_FEATURES = 64
RSS_LIMIT = 3758096384


def title_features(corpus, news, names=None):
    names = corpus.names if names is None else names
    tokens = np.zeros((len(names) + 1, TITLE_FEATURES), dtype=np.int32)
    weights = np.zeros_like(tokens, dtype=np.float32)
    for start in range(0, len(names), 2048):
        selected = names[start:start + 2048]
        if any(corpus._metadata_hash(news[name]) != corpus.metadata_sha256[name] for name in selected):
            raise ValueError("Title metadata differs from frozen source identities")
        matrix = corpus.vectorizer.transform([news[name]["title"] for name in selected]).tocsr()
        for row in range(len(selected)):
            lo, hi = matrix.indptr[row:row + 2]
            columns, values = matrix.indices[lo:hi], matrix.data[lo:hi]
            retain = np.lexsort((columns, -values))[:TITLE_FEATURES]
            retain = retain[np.argsort(columns[retain])]
            size = len(retain)
            tokens[start + row + 1, :size] = columns[retain] + 1
            weights[start + row + 1, :size] = values[retain]
    return tokens, weights


class TitleHistoryRanker(nn.Module):
    def __init__(self, corpus, tokens, weights, global_ratio):
        super().__init__()
        if (tokens.shape != weights.shape or tokens.shape != (len(corpus.names) + 1, TITLE_FEATURES)
                or tokens.dtype != np.int32 or weights.dtype != np.float32
                or np.any(tokens < 0) or np.any(tokens > len(corpus.vectorizer.vocabulary_))
                or not np.isfinite(weights).all() or np.any(weights < 0)
                or np.any((tokens == 0) & (weights != 0))
                or np.any(tokens[0]) or np.any(weights[0]) or not 0 < global_ratio < 1):
            raise ValueError("Exact bounded title features and mixed-class fitting ratio required")
        self.corpus = corpus
        dimensions = min(DIMENSIONS, corpus.svd.components_.shape[0])
        initial = torch.from_numpy(np.ascontiguousarray(corpus.svd.components_[:dimensions].T))
        initial = torch.cat((torch.zeros(1, dimensions), initial))
        self.words = nn.EmbeddingBag.from_pretrained(
            initial, freeze=False, mode="sum", padding_idx=0, include_last_offset=True)
        self.category = nn.Embedding(len(corpus.categories) + 1, dimensions, padding_idx=0)
        self.subcategory = nn.Embedding(len(corpus.subcategories) + 1, dimensions, padding_idx=0)
        nn.init.zeros_(self.category.weight)
        nn.init.zeros_(self.subcategory.weight)
        self.intrinsic = nn.Sequential(nn.Linear(dimensions, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.intrinsic[-1].weight)
        nn.init.zeros_(self.intrinsic[-1].bias)
        self.bias = nn.Parameter(torch.tensor(math.log(global_ratio / (1 - global_ratio)), dtype=torch.float32))
        self.recency = nn.Parameter(torch.zeros(30))
        self.register_buffer("tokens", torch.from_numpy(tokens.astype(np.int64)), persistent=False)
        self.register_buffer("weights", torch.from_numpy(weights), persistent=False)
        self.register_buffer("categories", torch.from_numpy(corpus.category), persistent=False)
        self.register_buffer("subcategories", torch.from_numpy(corpus.subcategory), persistent=False)
        self.cached = None

    def train(self, mode=True):
        if mode:
            self.cached = None
        return super().train(mode)

    def encode(self, news):
        ids, weights = self.tokens[news], self.weights[news]
        offsets = torch.arange(len(news) + 1) * TITLE_FEATURES
        vectors = self.words(ids.reshape(-1), offsets, per_sample_weights=weights.reshape(-1))
        vectors = vectors + self.category(self.categories[news]) + self.subcategory(self.subcategories[news])
        return F.normalize(vectors, dim=1)

    @torch.no_grad()
    def prepare_inference(self):
        if self.training:
            raise ValueError("Only a frozen evaluation model may cache article representations")
        self.cached = torch.cat([self.encode(torch.arange(start, min(start + 4096, len(self.tokens))))
                                 for start in range(0, len(self.tokens), 4096)])

    def forward(self, histories, candidates, groups):
        if (histories.ndim != 2 or histories.shape[1] != 30 or candidates.ndim != 1
                or groups.shape != candidates.shape
                or any(value.dtype != torch.int64 for value in (histories, candidates, groups))
                or torch.any(histories < 0) or torch.any(histories >= len(self.tokens))
                or torch.any(candidates < 1) or torch.any(candidates >= len(self.tokens))
                or torch.any(groups < 0) or torch.any(groups >= len(histories))):
            raise ValueError("Exact complete-group history/candidate identities required")
        if self.cached is not None and not self.training:
            history, candidate = self.cached[histories], self.cached[candidates]
        else:
            unique, inverse = torch.unique(torch.cat((histories.reshape(-1), candidates)), return_inverse=True)
            vectors = self.encode(unique)
            history = vectors[inverse[:histories.numel()]].reshape(*histories.shape, -1)
            candidate = vectors[inverse[histories.numel():]]
        mask = histories != 0
        logits = self.recency.expand_as(histories).masked_fill(~mask, -torch.inf)
        logits = torch.where(mask.any(dim=1, keepdim=True), logits, torch.zeros_like(logits))
        attention = torch.softmax(logits, dim=1) * mask
        user = F.normalize((history * attention[:, :, None]).sum(dim=1), dim=1)
        score = 10 * (candidate * user[groups]).sum(dim=1) + self.intrinsic(candidate).squeeze(-1) + self.bias
        if not torch.isfinite(score).all():
            raise FloatingPointError("Joint title/history scores must remain finite")
        return score

    @torch.no_grad()
    def scores(self, histories, candidates, groups):
        return self(torch.from_numpy(histories.astype(np.int64)),
                    torch.from_numpy(candidates.astype(np.int64)),
                    torch.from_numpy(groups.astype(np.int64))).numpy()

    def save(self, directory, coverage):
        directory.mkdir()
        torch.save(self.state_dict(), directory / "model.pt")
        np.savez_compressed(directory / "titles.npz", tokens=self.tokens.numpy().astype(np.int32),
                            weights=self.weights.numpy(), categories=self.categories.numpy(),
                            subcategories=self.subcategories.numpy())
        write_json(directory / "fit_coverage.json", coverage)
        write_json(directory / "model_config.json", {
            "hypothesis": "joint_trainable_title_embeddings_history_ranker_v1",
            "dimensions": self.words.weight.shape[1], "title_features": TITLE_FEATURES,
            "history_limit": 30, "epochs": 1, "seed": 42, "batch_impressions": 256,
            "objective": "complete_group_binary_ce_plus_0.5_all_positive_negative_bt",
        })

    @classmethod
    def load(cls, directory, corpus):
        config = json.loads((directory / "model_config.json").read_bytes())
        if (config["epochs"] != 1 or config["title_features"] != TITLE_FEATURES
                or config["dimensions"] != min(DIMENSIONS, corpus.svd.components_.shape[0])):
            raise ValueError("Saved joint title recipe differs")
        with np.load(directory / "titles.npz", allow_pickle=False) as arrays:
            model = cls(corpus, arrays["tokens"], arrays["weights"], 0.5)
            if not np.array_equal(arrays["categories"], corpus.category) or not np.array_equal(
                    arrays["subcategories"], corpus.subcategory):
                raise ValueError("Saved article taxonomy differs")
        model.load_state_dict(torch.load(directory / "model.pt", weights_only=True, map_location="cpu"), strict=True)
        model.eval()
        model.prepare_inference()
        return model

    def extend(self, news):
        old = len(self.corpus.names)
        self.corpus.extend(news)
        if len(self.corpus.names) > old:
            tokens, weights = title_features(self.corpus, news, self.corpus.names[old:])
            self.tokens = torch.cat((self.tokens, torch.from_numpy(tokens[1:].astype(np.int64))))
            self.weights = torch.cat((self.weights, torch.from_numpy(weights[1:])))
            self.categories = torch.from_numpy(self.corpus.category)
            self.subcategories = torch.from_numpy(self.corpus.subcategory)
            self.cached = None
            self.prepare_inference()


def fit_epoch(model, source, groups, deadline):
    validate_groups(source, groups)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    order = np.random.default_rng(42).permutation(groups)
    seen = np.zeros(len(source.offsets) - 1, dtype=bool)
    expected = source.counts(groups)
    exposures = positives = steps = 0
    for start in range(0, len(order), 256):
        limits(deadline, RSS_LIMIT)
        selected = order[start:start + 256]
        if seen[selected].any():
            raise ValueError("Duplicate training impression")
        seen[selected] = True
        rows, local, sizes = candidate_rows(source.offsets, selected)
        histories = torch.from_numpy(source.histories[selected].astype(np.int64))
        candidates = torch.from_numpy(source.candidates[rows].astype(np.int64))
        labels = torch.from_numpy(source.labels[rows].astype(np.float32))
        local = torch.from_numpy(local)
        optimizer.zero_grad(set_to_none=True)
        loss = complete_loss(model(histories, candidates, local), labels, local, torch.from_numpy(sizes))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        exposures += len(rows)
        positives += int(labels.sum())
        steps += 1
        if steps % 512 == 0:
            print(json.dumps({"optimizer_steps": steps, "impressions": int(seen.sum()),
                              "exposures": exposures, "positives": positives}), flush=True)
    if (int(seen.sum()) != expected["impressions"] or not seen[groups].all()
            or exposures != expected["exposures"] or positives != expected["positives"]):
        raise ValueError("Incomplete fixed full-impression training epoch")
    return dict(expected, epochs=1, optimizer_steps=steps,
                consumed_source_positions_sha256=digest_bytes(np.packbits(seen).tobytes()))


def exact_auc(labels, scores):
    if (labels.shape != scores.shape or not np.isfinite(scores).all()
            or not np.all((labels == 0) | (labels == 1))):
        raise ValueError("Finite complete binary group scores required")
    positive, negative = scores[labels == 1], np.sort(scores[labels == 0])
    if not len(positive) or not len(negative):
        raise ValueError("Undefined group AUC must not be discarded")
    lower = np.searchsorted(negative, positive, side="left")
    upper = np.searchsorted(negative, positive, side="right")
    return float((lower + upper).sum()) / (2 * len(positive) * len(negative))


def evaluate(model, source, groups, path, deadline):
    validate_groups(source, groups)
    model.eval()
    model.prepare_inference()
    metrics, exposures, positives = [], 0, 0
    with path.open("x") as handle:
        for start in range(0, len(groups), 256):
            limits(deadline, RSS_LIMIT)
            selected = groups[start:start + 256]
            rows, local, sizes = candidate_rows(source.offsets, selected)
            scores = model.scores(source.histories[selected], source.candidates[rows], local)
            labels, offset = source.labels[rows], 0
            for index, size in zip(selected.tolist(), sizes.tolist(), strict=True):
                target, values = labels[offset:offset + size], scores[offset:offset + size]
                metrics.append((exact_auc(target, values), mrr_score(target, values),
                                ndcg_score(target, values, 5), ndcg_score(target, values, 10)))
                identity = source.impression_ids[index] if hasattr(source, "impression_ids") else index
                json.dump([identity, values.tolist()], handle, allow_nan=False)
                handle.write("\n")
                offset += size
            exposures += len(rows)
            positives += int(labels.sum())
    if source.counts(groups) != {"impressions": len(metrics), "exposures": exposures, "positives": positives}:
        raise ValueError("Incomplete scored population")
    return {"impressions": len(metrics), "exposures": exposures, "positives": positives,
            "metrics": {name: math.fsum(row[col] for row in metrics) / len(metrics)
                        for col, name in enumerate(("auc", "mrr", "ndcg5", "ndcg10"))},
            "scores_sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "encoder", "train-news", "dev-news", "dev-behaviors", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.seconds < 1:
        raise ValueError("Fresh output and bounded execution required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    deadline = time.monotonic() + args.seconds
    manifest, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    if manifest["implementation_sha256"] != digest(Path(__file__).with_name("full_impression_contract.py")):
        raise ValueError("Pinned complete-source producer changed")
    for name, expected in ENCODER_HASHES.items():
        if digest(args.encoder / name) != expected:
            raise ValueError("Frozen label-free encoder changed")
    if digest(args.train_news) != manifest["input_sha256"]["news.tsv"]:
        raise ValueError("Complete training-news bytes changed")
    corpus, news = Corpus.load(args.encoder), read_news(args.train_news)
    if sorted(news) != corpus.names:
        raise ValueError("Full article identities differ")
    tokens, weights = title_features(corpus, news)
    source = object.__new__(Source)
    for field, filename in (("offsets", "candidate_offsets.npy"), ("candidates", "candidate_news.npy"),
                            ("labels", "candidate_labels.npy"), ("histories", "histories.npy"),
                            ("days", "training_days.npy")):
        setattr(source, field, arrays[filename])
    args.output.mkdir()
    write_json(args.output / "registration.json", {
        "registered_at": datetime.now(timezone.utc).isoformat(), "source_sha256": digest(Path(__file__)),
        "manifest_sha256": MANIFEST_SHA, "encoder_sha256": ENCODER_HASHES,
        "epochs": 1, "title_features": TITLE_FEATURES, "dimensions": DIMENSIONS,
        "gate_auc": 0.69, "seconds": args.seconds, "old_model_weights_or_priors_read": False,
        "official_dev_or_test_contents_read": False, "new_gpu_allocations": 0,
    })
    earlier = np.flatnonzero(source.days < source.days.max())
    heldout = np.flatnonzero(source.days == source.days.max())
    if (source.counts(earlier) != {"impressions": 1801231, "exposures": 66107268, "positives": 2712612}
            or source.counts(heldout) != {"impressions": 431517, "exposures": 17400106, "positives": 671044}):
        raise ValueError("Exact earlier/heldout training populations differ")
    expected = source.counts(earlier)
    model = TitleHistoryRanker(corpus, tokens, weights, expected["positives"] / expected["exposures"])
    coverage = fit_epoch(model, source, earlier, deadline)
    model.save(args.output / "internal_selection", coverage)
    internal = evaluate(model, source, heldout, args.output / "internal_scores.jsonl", deadline)
    load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    if internal["metrics"]["auc"] < 0.69:
        write_json(args.output / "result.json", {
            "stage": "rejected_internal_gate", "internal_validation": internal,
            "final_all_training_fit": False, "official_dev_contents_read": False,
            "competition_test_goal_achieved": False, "model_fits": 1})
        return
    torch.manual_seed(42)
    expected = source.counts(np.arange(len(source.days)))
    if expected != {"impressions": 2232748, "exposures": 83507374, "positives": 3383656}:
        raise ValueError("Final fitting requires every original training candidate")
    model = TitleHistoryRanker(corpus, tokens, weights, expected["positives"] / expected["exposures"])
    coverage = fit_epoch(model, source, np.arange(len(source.days)), deadline)
    model.save(args.output / "all_training_fit", coverage)
    if (digest(args.dev_news) != "03ba5f02160672500bf5f1af6dd18e8f99830a845b08ca9fd9fbbc9bdd72baff"
            or digest(args.dev_behaviors) != "7031ced7b8de5183c1bf2c4a303018df8a49d858714597421d9489e571d87b45"):
        raise ValueError("Pinned official-dev bytes differ")
    model.eval()
    model.extend(news | read_news(args.dev_news))
    dev = encode_behaviors(args.dev_behaviors, corpus, 376471, 14085557)
    if int(dev.labels.sum()) != 574845:
        raise ValueError("Pinned dev click population differs")
    official = evaluate(model, dev, np.arange(376471), args.output / "official_dev_scores.jsonl", deadline)
    if (digest(args.dev_news) != "03ba5f02160672500bf5f1af6dd18e8f99830a845b08ca9fd9fbbc9bdd72baff"
            or digest(args.dev_behaviors) != "7031ced7b8de5183c1bf2c4a303018df8a49d858714597421d9489e571d87b45"):
        raise ValueError("Official-dev inputs changed while scoring")
    load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    write_json(args.output / "result.json", {
        "stage": "complete_all_training_fit_and_official_dev_scored",
        "internal_validation": internal, "official_dev": official,
        "final_all_training_fit": True, "official_dev_contents_read": True,
        "competition_test_contents_read": False, "competition_test_goal_achieved": False, "model_fits": 2})


if __name__ == "__main__":
    main()
