"""CPU candidate-aware ranking with complete impressions and train-only priors."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from full_impression_contract import read_news


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def limits(deadline, rss_limit):
    if time.monotonic() >= deadline:
        raise TimeoutError("CPU ranking trial reached its immutable wall-clock cutoff")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > rss_limit:
        raise MemoryError("CPU ranking trial exceeded its registered peak-RSS limit")


def candidate_rows(offsets, groups):
    starts = offsets[groups].astype(np.int64)
    sizes = (offsets[groups + 1] - offsets[groups]).astype(np.int64)
    if not len(groups) or np.any(sizes < 1):
        raise ValueError("Complete source groups must all contain candidates")
    local_starts = np.cumsum(sizes) - sizes
    rows = np.repeat(starts - local_starts, sizes) + np.arange(int(sizes.sum()))
    group_index = np.repeat(np.arange(len(groups)), sizes)
    return rows, group_index, sizes


def validate_groups(source, groups):
    if (not isinstance(groups, np.ndarray) or groups.ndim != 1
            or not np.issubdtype(groups.dtype, np.integer) or not len(groups)
            or np.any(groups < 0) or np.any(groups >= len(source.offsets) - 1)
            or len(np.unique(groups)) != len(groups)):
        raise ValueError("Source groups must be nonempty, unique, in-range integer positions")


def complete_loss(scores, labels, group_index, sizes, pairwise_weight=0.5):
    if (scores.ndim != 1 or scores.shape != labels.shape
            or group_index.shape != labels.shape or int(sizes.sum()) != len(labels)
            or not torch.all((labels == 0) | (labels == 1))
            or not torch.isfinite(scores).all() or torch.any(sizes < 1)):
        raise ValueError("Complete ranking loss needs finite scores and exact binary group coverage")
    expected = torch.repeat_interleave(torch.arange(len(sizes)), sizes)
    if not torch.equal(group_index, expected):
        raise ValueError("Candidate groups must retain contiguous complete source rows")
    if not math.isfinite(pairwise_weight) or pairwise_weight < 0:
        raise ValueError("Pairwise weight must be finite and nonnegative")
    per_candidate = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
    ce = scores.new_zeros(len(sizes)).index_add(0, group_index, per_candidate) / sizes
    positive = labels == 1
    positive_counts = torch.bincount(group_index[positive], minlength=len(sizes))
    negative_counts = sizes - positive_counts
    eligible = (positive_counts > 0) & (negative_counts > 0)
    loss = ce.mean()
    if pairwise_weight and torch.any(eligible):
        positive_groups = group_index[positive]
        repeats = negative_counts[positive_groups]
        positive_indices = torch.repeat_interleave(torch.arange(int(positive.sum())), repeats)
        pair_groups = positive_groups[positive_indices]
        negative_starts = torch.cumsum(negative_counts, 0) - negative_counts
        pair_starts = torch.cumsum(repeats, 0) - repeats
        negative_indices = (negative_starts[pair_groups]
                            + torch.arange(len(positive_indices)) - pair_starts[positive_indices])
        pair_losses = F.softplus(
            scores[~positive][negative_indices] - scores[positive][positive_indices])
        denominators = (positive_counts * negative_counts).clamp_min(1)
        bt = scores.new_zeros(len(sizes)).index_add(0, pair_groups, pair_losses) / denominators
        loss = loss + pairwise_weight * bt[eligible].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("CPU ranking loss is not finite")
    return loss


class Corpus:
    def __init__(self, news, seed=42, dimensions=64):
        if not news:
            raise ValueError("Ranking requires a nonempty source news corpus")
        self.names = sorted(news)
        self.lookup = {name: index + 1 for index, name in enumerate(self.names)}
        self.categories = {name: index + 1 for index, name in enumerate(
            sorted({item["category"] for item in news.values()}))}
        self.subcategories = {name: index + 1 for index, name in enumerate(
            sorted({item["subcategory"] for item in news.values()}))}
        records = [news[name] for name in self.names]
        self.metadata_sha256 = {name: self._metadata_hash(news[name]) for name in self.names}
        texts = [item["title"] + " " + item["abstract"] for item in records]
        self.vectorizer = TfidfVectorizer(
            max_features=32768, ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
        matrix = self.vectorizer.fit_transform(texts)
        rank = min(dimensions, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if rank < 1:
            raise ValueError("News lexical projection needs at least two records and features")
        self.svd = TruncatedSVD(n_components=rank, n_iter=3, random_state=seed)
        vectors = self.svd.fit_transform(matrix).astype(np.float32)
        self.lexical = self._normalize(vectors)
        self.category = np.asarray(
            [0] + [self.categories[item["category"]] for item in records], dtype=np.int64)
        self.subcategory = np.asarray(
            [0] + [self.subcategories[item["subcategory"]] for item in records], dtype=np.int64)

    @staticmethod
    def _normalize(vectors):
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        return np.concatenate((np.zeros((1, vectors.shape[1]), dtype=np.float32), vectors))

    @staticmethod
    def _metadata_hash(item):
        fields = [item[key] for key in ("category", "subcategory", "title", "abstract")]
        return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode("utf-8")).hexdigest()

    def extend(self, news):
        for name in set(news) & set(self.lookup):
            if self._metadata_hash(news[name]) != self.metadata_sha256[name]:
                raise ValueError("Shared news identity changed its frozen metadata")
        additions = sorted(set(news) - set(self.lookup))
        if not additions:
            return
        records = [news[name] for name in additions]
        matrix = self.vectorizer.transform(
            [item["title"] + " " + item["abstract"] for item in records])
        vectors = self._normalize(self.svd.transform(matrix).astype(np.float32))[1:]
        self.lexical = np.concatenate((self.lexical, vectors))
        self.category = np.concatenate((self.category, [
            self.categories.get(item["category"], 0) for item in records]))
        self.subcategory = np.concatenate((self.subcategory, [
            self.subcategories.get(item["subcategory"], 0) for item in records]))
        for name in additions:
            self.lookup[name] = len(self.names) + 1
            self.names.append(name)
            self.metadata_sha256[name] = self._metadata_hash(news[name])

    def save(self, output):
        vocabulary = sorted(self.vectorizer.vocabulary_, key=self.vectorizer.vocabulary_.get)
        np.savez_compressed(
            output / "lexical_encoder.npz", vocabulary=np.asarray(vocabulary),
            idf=self.vectorizer.idf_, components=self.svd.components_,
            lexical=self.lexical, category=self.category, subcategory=self.subcategory)
        write_json(output / "corpus.json", {
            "names": self.names, "categories": self.categories,
            "subcategories": self.subcategories, "metadata_sha256": self.metadata_sha256,
        })

    @classmethod
    def load(cls, output):
        document = json.loads((output / "corpus.json").read_bytes())
        corpus = object.__new__(cls)
        for key in ("names", "categories", "subcategories", "metadata_sha256"):
            setattr(corpus, key, document[key])
        corpus.lookup = {name: index + 1 for index, name in enumerate(corpus.names)}
        with np.load(output / "lexical_encoder.npz", allow_pickle=False) as arrays:
            vocabulary = {name: index for index, name in enumerate(arrays["vocabulary"].tolist())}
            corpus.vectorizer = TfidfVectorizer(
                vocabulary=vocabulary, ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
            corpus.vectorizer.vocabulary_ = vocabulary
            corpus.vectorizer.fixed_vocabulary_ = True
            corpus.vectorizer._tfidf = TfidfTransformer(sublinear_tf=True)
            corpus.vectorizer._tfidf.n_features_in_ = len(vocabulary)
            corpus.vectorizer._tfidf.idf_ = arrays["idf"]
            corpus.svd = TruncatedSVD(n_components=arrays["components"].shape[0])
            corpus.svd.components_ = arrays["components"]
            corpus.svd.n_features_in_ = len(vocabulary)
            for key in ("lexical", "category", "subcategory"):
                setattr(corpus, key, arrays[key])
        return corpus


class Source:
    def __init__(self, contract, raw_source, deadline, rss_limit):
        self.contract = Path(contract)
        self.raw_source = Path(raw_source)
        self.manifest = json.loads((self.contract / "manifest.json").read_bytes())
        if (self.manifest["stage"] != "all_training_candidates_encoded"
                or self.manifest["history_limit"] != 30
                or self.manifest["sampling"] != "none;every_source_candidate_in_original_order"
                or not self.manifest["final_training_requires_all_impressions"]):
            raise ValueError("CPU ranker requires the lossless full-candidate source contract")
        if self.manifest["implementation_sha256"] != digest(
                Path(__file__).with_name("full_impression_contract.py")):
            raise ValueError("Full-source contract producer implementation differs")
        paths = {self.contract / name: expected for name, expected in self.manifest["sha256"].items()}
        paths.update({self.raw_source / name: expected
                      for name, expected in self.manifest["input_sha256"].items()})
        self.fingerprints = {}
        manifest_stat = (self.contract / "manifest.json").stat()
        self.fingerprints[self.contract / "manifest.json"] = (
            manifest_stat.st_size, manifest_stat.st_mtime_ns)
        for path, expected in paths.items():
            limits(deadline, rss_limit)
            stat = path.stat()
            self.fingerprints[path] = (stat.st_size, stat.st_mtime_ns)
            if digest(path) != expected:
                raise ValueError(f"Immutable training input differs: {path.name}")
        self.news = read_news(self.raw_source / "news.tsv")
        if json.loads((self.contract / "news.ids.json").read_bytes()) != sorted(self.news):
            raise ValueError("Encoded news identities differ from the strict raw corpus")
        arrays = {
            "candidates": ("candidate_news.npy", np.int32),
            "labels": ("candidate_labels.npy", np.uint8),
            "offsets": ("candidate_offsets.npy", np.uint64),
            "histories": ("histories.npy", np.int32),
            "days": ("training_days.npy", np.int32),
        }
        for field, (filename, dtype) in arrays.items():
            array = np.load(self.contract / filename, allow_pickle=False)
            if array.dtype != dtype:
                raise ValueError(f"Unexpected source dtype: {field}")
            setattr(self, field, array)
        count, exposures = self.manifest["impressions"], self.manifest["exposures"]
        if (self.candidates.shape != (exposures,) or self.labels.shape != (exposures,)
                or self.offsets.shape != (count + 1,) or self.offsets[0] != 0
                or self.offsets[-1] != exposures or self.histories.shape != (count, 30)
                or self.days.shape != (count,)):
            raise ValueError("Complete source array dimensions differ")
        with (self.contract / "impression_ids.txt").open("rb") as handle:
            self.folds = np.fromiter(
                (hashlib.blake2b(line.rstrip(b"\r\n"), digest_size=1).digest()[0] & 1
                 for line in handle), dtype=np.uint8, count=count)
            if handle.read(1):
                raise ValueError("Extra source impression IDs")

    def unchanged(self):
        for path, expected in self.fingerprints.items():
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != expected:
                raise ValueError("Bound training input changed during the trial")

    def counts(self, groups):
        sizes = self.offsets[groups + 1] - self.offsets[groups]
        positives = 0
        for start in range(0, len(groups), 16384):
            rows, _, _ = candidate_rows(self.offsets, groups[start:start + 16384])
            positives += int(self.labels[rows].sum(dtype=np.uint64))
        return {"impressions": len(groups), "exposures": int(sizes.sum()),
                "positives": positives}


class Priors:
    def __init__(self, source, corpus, groups, deadline, rss_limit):
        validate_groups(source, groups)
        width = len(corpus.names) + 1
        exposures = np.zeros((2, width), dtype=np.float64)
        positives = np.zeros_like(exposures)
        self.known = np.zeros(width, dtype=bool)
        for start in range(0, len(groups), 16384):
            limits(deadline, rss_limit)
            batch_groups = groups[start:start + 16384]
            rows, local_groups, _ = candidate_rows(source.offsets, batch_groups)
            news = source.candidates[rows]
            folds = source.folds[batch_groups][local_groups]
            for fold in (0, 1):
                selected = folds == fold
                exposures[fold] += np.bincount(news[selected], minlength=width)
                positives[fold] += np.bincount(
                    news[selected], weights=source.labels[rows[selected]], minlength=width)
            self.known[news] = True
            self.known[source.histories[batch_groups].ravel()] = True
        self.known[0] = False
        self.tables = [self._table(exposures[fold], positives[fold], corpus.category)
                       for fold in (0, 1)]
        self.pooled = self._table(exposures.sum(0), positives.sum(0), corpus.category)
        self.coverage = {"exposures": int(exposures.sum()), "positives": int(positives.sum())}

    @staticmethod
    def _table(exposures, positives, categories):
        ratio = positives.sum() / exposures.sum()
        if not 0 < ratio < 1:
            raise ValueError("Each cross-fit source pool requires both click classes")
        category_exposures = np.bincount(categories, weights=exposures)
        category_positives = np.bincount(categories, weights=positives)
        category_probability = (category_positives + 100 * ratio) / (category_exposures + 100)
        probability = (positives + 20 * category_probability[categories]) / (exposures + 20)
        return np.stack((np.log(probability / (1 - probability)),
                         np.log(category_probability[categories] / (1 - category_probability[categories])),
                         np.log1p(exposures)), axis=1).astype(np.float32)

    def features(self, news, folds=None):
        if np.any(news >= len(self.pooled)):
            raise ValueError("Extend inference priors explicitly before scoring unseen news")
        if folds is None:
            return self.pooled[news]
        if not hasattr(self, "tables"):
            raise ValueError("Saved pooled priors support inference only, not resumed training")
        if folds.shape != news.shape or not np.all((folds == 0) | (folds == 1)):
            raise ValueError("Cross-fit features require one valid fold per source candidate")
        return np.where(folds[:, None] == 0, self.tables[1][news], self.tables[0][news])

    def extend(self, corpus):
        missing = len(corpus.names) + 1 - len(self.pooled)
        if missing < 0:
            raise ValueError("Inference corpus cannot shrink")
        if missing:
            # Unseen documents have no training clicks: use the pooled category prior.
            category_logit = {}
            for index, category in enumerate(corpus.category[:len(self.pooled)]):
                category_logit.setdefault(int(category), float(self.pooled[index, 1]))
            global_logit = float(self.pooled[0, 1])
            added = np.asarray([
                [category_logit.get(int(category), global_logit)] * 2 + [0.0]
                for category in corpus.category[len(self.pooled):]], dtype=np.float32)
            self.pooled = np.concatenate((self.pooled, added))
            self.known = np.concatenate((self.known, np.zeros(missing, dtype=bool)))

    def save(self, output):
        np.savez_compressed(output / "training_priors.npz", pooled=self.pooled, known=self.known)

    @classmethod
    def load(cls, output):
        priors = object.__new__(cls)
        with np.load(output / "training_priors.npz", allow_pickle=False) as arrays:
            priors.pooled, priors.known = arrays["pooled"], arrays["known"]
        return priors


class CandidateHistoryRanker(nn.Module):
    def __init__(self, corpus, known, dimensions=64, id_dropout=0.5):
        super().__init__()
        if not 0 <= id_dropout < 1:
            raise ValueError("ID-feature dropout must be in [0,1)")
        self.id_dropout = id_dropout
        self.dimensions = dimensions
        self.register_buffer("lexical", torch.from_numpy(corpus.lexical))
        self.register_buffer("categories", torch.from_numpy(corpus.category))
        self.register_buffer("subcategories", torch.from_numpy(corpus.subcategory))
        self.register_buffer("id_map", torch.arange(len(corpus.names) + 1) * torch.from_numpy(known))
        self.id_embedding = nn.Embedding(len(corpus.names) + 1, dimensions, padding_idx=0, sparse=True)
        nn.init.normal_(self.id_embedding.weight, std=0.01)
        with torch.no_grad():
            self.id_embedding.weight[0].zero_()
        self.lexical_projection = nn.Linear(corpus.lexical.shape[1], dimensions, bias=False)
        self.category_embedding = nn.Embedding(len(corpus.categories) + 1, dimensions, padding_idx=0)
        self.subcategory_embedding = nn.Embedding(len(corpus.subcategories) + 1, dimensions, padding_idx=0)
        for embedding in (self.category_embedding, self.subcategory_embedding):
            nn.init.normal_(embedding.weight, std=0.05)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.positions = nn.Parameter(torch.zeros(30, dimensions))
        self.query = nn.Linear(dimensions, dimensions, bias=False)
        self.key = nn.Linear(dimensions, dimensions, bias=False)
        self.value = nn.Linear(dimensions, dimensions, bias=False)
        self.head = nn.Sequential(nn.Linear(dimensions * 3 + 8, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def news_vectors(self, news):
        identifiers = self.id_map[news]
        id_features = self.id_embedding(identifiers)
        if self.training and self.id_dropout:
            keep = torch.rand((*identifiers.shape, 1)) >= self.id_dropout
            id_features = id_features * keep / (1 - self.id_dropout)
        return (self.lexical_projection(self.lexical[news]) + id_features
                + self.category_embedding(self.categories[news])
                + self.subcategory_embedding(self.subcategories[news]))

    def _chunk(self, news, groups, priors, sizes, histories, mean_histories, keys, values):
        candidate = self.news_vectors(news)
        mask = histories[groups] != 0
        query = self.query(candidate)
        scores = torch.bmm(keys[groups], query[:, :, None]).squeeze(-1) / math.sqrt(self.dimensions)
        scores = scores.masked_fill(~mask, -torch.inf)
        scores = torch.where(mask.any(1)[:, None], scores, torch.zeros_like(scores))
        attention = torch.softmax(scores, dim=1) * mask
        context = torch.bmm(attention[:, None, :], values[groups]).squeeze(1)
        count = mask.sum(1).clamp_min(1)
        category_match = ((self.categories[histories[groups]] == self.categories[news, None])
                          & mask & (self.categories[news, None] != 0)).sum(1) / count
        subcategory_match = ((self.subcategories[histories[groups]] == self.subcategories[news, None])
                             & mask & (self.subcategories[news, None] != 0)).sum(1) / count
        cosine = F.cosine_similarity(candidate, mean_histories[groups], dim=1)
        extra = torch.cat((priors, category_match[:, None], subcategory_match[:, None],
                           cosine[:, None], torch.log1p(mask.sum(1).float())[:, None],
                           torch.log1p(sizes[groups].float())[:, None]), dim=1)
        features = torch.cat((candidate, context, candidate * context, extra), dim=1)
        return priors[:, 0] + self.head(features).squeeze(-1)

    def forward(self, histories, news, groups, sizes, priors, chunk_size=2048):
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("Candidate chunk size must be a positive integer")
        history_vectors = self.news_vectors(histories)
        history_mask = histories != 0
        mean_histories = ((history_vectors * history_mask[:, :, None]).sum(1)
                          / history_mask.sum(1).clamp_min(1)[:, None])
        keys = self.key(history_vectors + self.positions[None, :, :])
        values = self.value(history_vectors)
        chunks = []
        for start in range(0, len(news), chunk_size):
            args = (news[start:start + chunk_size], groups[start:start + chunk_size],
                    priors[start:start + chunk_size], sizes, histories, mean_histories, keys, values)
            if torch.is_grad_enabled():
                logits = checkpoint(self._chunk, *args, use_reentrant=False, preserve_rng_state=True)
            else:
                logits = self._chunk(*args)
            chunks.append(logits)
        return torch.cat(chunks)

    def extend_features(self, corpus, known):
        old = len(self.id_map)
        self.lexical = torch.from_numpy(corpus.lexical)
        self.categories = torch.from_numpy(corpus.category)
        self.subcategories = torch.from_numpy(corpus.subcategory)
        self.id_map = torch.cat((self.id_map[:old], torch.zeros(len(known) - old, dtype=torch.long)))


def batch_tensors(source, corpus, priors, groups, training):
    rows, local, sizes = candidate_rows(source.offsets, groups)
    news = source.candidates[rows].astype(np.int64)
    folds = source.folds[groups][local] if training else None
    return (
        torch.from_numpy(source.histories[groups].astype(np.int64)),
        torch.from_numpy(news), torch.from_numpy(local),
        torch.from_numpy(sizes), torch.from_numpy(priors.features(news, folds)),
        torch.from_numpy(source.labels[rows].astype(np.float32)),
    )


def train_epochs(source, corpus, priors, groups, output, epochs, seed, deadline, rss_limit):
    validate_groups(source, groups)
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("Training requires a positive fixed epoch count")
    torch.manual_seed(seed)
    model = CandidateHistoryRanker(corpus, priors.known)
    sparse_parameters = [model.id_embedding.weight]
    dense_parameters = [parameter for parameter in model.parameters()
                        if parameter is not model.id_embedding.weight]
    dense = torch.optim.AdamW(dense_parameters, lr=0.001, weight_decay=0.001)
    sparse = torch.optim.SparseAdam(sparse_parameters, lr=0.001)
    expected = source.counts(groups)
    if any(expected[key] != priors.coverage[key] for key in ("exposures", "positives")):
        raise ValueError("Cross-fitted priors do not cover every selected training exposure")
    ledgers = []
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(groups)
        seen = np.zeros(source.manifest["impressions"], dtype=bool)
        exposures = positives = steps = 0
        for start in range(0, len(order), 256):
            limits(deadline, rss_limit)
            batch = order[start:start + 256]
            if seen[batch].any():
                raise ValueError("Source impression consumed twice in one epoch")
            tensors = batch_tensors(source, corpus, priors, batch, True)
            histories, news, group_index, sizes, prior_features, labels = tensors
            dense.zero_grad(set_to_none=True)
            sparse.zero_grad(set_to_none=True)
            scores = model(histories, news, group_index, sizes, prior_features)
            loss = complete_loss(scores, labels, group_index, sizes)
            loss.backward()
            for parameter in model.parameters():
                gradient = parameter.grad
                if gradient is not None:
                    values = gradient.coalesce().values() if gradient.is_sparse else gradient
                    if not torch.isfinite(values).all():
                        raise FloatingPointError("Nonfinite CPU ranker gradient")
            torch.nn.utils.clip_grad_norm_(dense_parameters, 1.0, error_if_nonfinite=True)
            dense.step()
            sparse.step()
            seen[batch] = True
            exposures += len(news)
            positives += int(labels.sum())
            steps += 1
            if steps % 256 == 0:
                print(json.dumps({"phase": output.name, "epoch": epoch + 1, "steps": steps,
                                  "impressions": int(seen.sum()), "exposures": exposures,
                                  "loss": float(loss.detach())}), flush=True)
        if (seen.sum() != expected["impressions"] or not seen[groups].all()
                or exposures != expected["exposures"] or positives != expected["positives"]):
            raise ValueError("CPU training epoch did not cover every selected source row")
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError("CPU ranker parameters became nonfinite")
        source.unchanged()
        ledger = {"epoch": epoch + 1, **expected, "optimizer_steps": steps,
                  "consumed_source_positions_sha256": hashlib.sha256(np.packbits(seen).tobytes()).hexdigest()}
        ledgers.append(ledger)
        print(json.dumps({"verified_epoch": ledger}), flush=True)
    write_json(output / "epoch_coverage.json", ledgers)
    torch.save(model.state_dict(), output / "model.pt")
    priors.save(output)
    corpus.save(output)
    write_json(output / "model_config.json", {
        "dimensions": model.dimensions, "id_dropout": model.id_dropout,
        "objective": "complete_impression_binary_ce_plus_0.5_bt", "source_coverage": expected,
    })
    return model, ledgers


def evaluate(source, corpus, priors, groups, model, deadline, rss_limit, score_path=None):
    validate_groups(source, groups)
    expected = source.counts(groups)
    model.eval()
    aucs = []
    seen = np.zeros(len(source.offsets) - 1, dtype=bool)
    exposures = positives = 0
    with torch.no_grad(), (score_path.open("x", encoding="utf-8") if score_path else nullcontext()) as score_file:
        for start in range(0, len(groups), 512):
            limits(deadline, rss_limit)
            selected = groups[start:start + 512]
            if seen[selected].any():
                raise ValueError("Duplicate impression in evaluation")
            tensors = batch_tensors(source, corpus, priors, selected, False)
            histories, news, group_index, sizes, prior_features, labels = tensors
            scores = model(histories, news, group_index, sizes, prior_features).numpy()
            if not np.isfinite(scores).all():
                raise FloatingPointError("Nonfinite evaluation scores")
            label_array = labels.numpy()
            offset = 0
            for index, size in enumerate(sizes.tolist()):
                row = scores[offset:offset + size]
                target = label_array[offset:offset + size]
                positive, negative = row[target == 1], row[target == 0]
                if not len(positive) or not len(negative):
                    raise ValueError("AUC undefined; no silent impression dropping")
                wins = 2 * np.count_nonzero(positive[:, None] > negative)
                wins += np.count_nonzero(positive[:, None] == negative)
                aucs.append(wins / (2 * len(positive) * len(negative)))
                if score_file is not None:
                    json.dump([source.impression_ids[int(selected[index])], row.tolist()],
                              score_file, allow_nan=False)
                    score_file.write("\n")
                offset += size
            seen[selected] = True
            exposures += len(news)
            positives += int(labels.sum())
    if (not seen[groups].all() or int(seen.sum()) != expected["impressions"]
            or exposures != expected["exposures"] or positives != expected["positives"]):
        raise ValueError("Evaluation did not cover every source impression")
    return {"auc": math.fsum(aucs) / len(aucs), "impressions": len(groups),
            "exposures": exposures, "positives": positives}


def dev_source(behaviors, news_path, corpus, priors, model, output):
    if (digest(behaviors) != "7031ced7b8de5183c1bf2c4a303018df8a49d858714597421d9489e571d87b45"
            or digest(news_path) != "03ba5f02160672500bf5f1af6dd18e8f99830a845b08ca9fd9fbbc9bdd72baff"):
        raise ValueError("Official-dev raw bytes differ from the independently pinned population")
    news = read_news(Path(news_path))
    corpus.extend(news)
    priors.extend(corpus)
    model.extend_features(corpus, priors.known)
    source = object.__new__(Source)
    offsets = [0]
    histories, candidates, labels = [], [], []
    impression_ids = []
    seen = set()
    with Path(behaviors).open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 5 or not fields[0] or fields[0] in seen:
                raise ValueError("Malformed or duplicate official-dev impression")
            seen.add(fields[0])
            impression_ids.append(fields[0])
            history = [corpus.lookup[name] for name in fields[3].split()[-30:]]
            histories.append([0] * (30 - len(history)) + history)
            for token in fields[4].split():
                if not token.endswith(("-0", "-1")):
                    raise ValueError("Official-dev candidate lacks a valid label")
                identifier, label = token.rsplit("-", 1)
                candidates.append(corpus.lookup[identifier])
                labels.append(int(label))
            if len(candidates) == offsets[-1]:
                raise ValueError("Empty official-dev impression")
            offsets.append(len(candidates))
    source.offsets = np.asarray(offsets, dtype=np.uint64)
    source.histories = np.asarray(histories, dtype=np.int32)
    source.candidates = np.asarray(candidates, dtype=np.int32)
    source.labels = np.asarray(labels, dtype=np.uint8)
    source.folds = np.zeros(len(histories), dtype=np.uint8)
    source.impression_ids = impression_ids
    if (len(histories), len(candidates), int(source.labels.sum())) != (376471, 14085557, 574845):
        raise ValueError("Official-dev source population differs")
    write_json(output / "dev_source_coverage.json", {
        "impressions": len(histories), "exposures": len(candidates),
        "positives": int(source.labels.sum()), "raw_behaviors_sha256": digest(behaviors),
        "raw_news_sha256": digest(news_path), "source_candidate_order_retained": True,
    })
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-source", type=Path, required=True)
    parser.add_argument("--dev-behaviors", type=Path, required=True)
    parser.add_argument("--dev-news", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.output.exists() or args.seconds < 1:
        raise ValueError("Use a fresh output path and positive immutable duration")
    if digest(args.contract / "manifest.json") != "4daa3377b4601aca4ae5b72e45ed2931becb8dd648dd23998f17fb3e1342a5f4":
        raise ValueError("CPU experiment source manifest differs from its pinned full-training contract")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise ValueError("The distinct CPU trial must explicitly disable CUDA")
    started = time.monotonic()
    deadline, rss_limit = started + args.seconds, int(3.5 * 2**30)
    args.output.mkdir()
    write_json(args.output / "registration.json", {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "cpu_lexical_category_id_candidate_history_attention_v1",
        "source_revision_sha256": digest(Path(__file__)),
        "source_contract_sha256": digest(args.contract / "manifest.json"),
        "source_population": {"impressions": 2232748, "exposures": 83507374, "positives": 3383656},
        "epochs": 2, "seed": 42, "batch_impressions": 256, "candidate_chunk": 2048,
        "objective": "impression_balanced_binary_ce_plus_0.5_all_within_impression_bt",
        "crossfit": "two BLAKE2b impression-ID folds;own fold excluded from training click priors",
        "lexical_fit_scope": "full_training_news_metadata_only;no heldout click targets",
        "temporary_validation_day": "2019-11-14", "temporary_gate_auc": 0.69,
        "final_refit_requires_all_training": True, "official_dev_only_after_gate_and_complete_refit": True,
        "wall_clock_seconds": args.seconds, "cpu_threads": 4, "rss_limit_bytes": rss_limit,
        "new_gpu_allocations": 0, "cloud_jobs": 0, "competition_test_contents_read": False,
        "previous_deadlines_or_gates_changed": False, "competition_test_goal_achieved": False,
    })
    source = Source(args.contract, args.raw_source, deadline, rss_limit)
    if tuple(source.manifest[key] for key in ("impressions", "exposures", "positives")) != (2232748, 83507374, 3383656):
        raise ValueError("New CPU trial requires the complete configured MIND-large training population")
    corpus = Corpus(source.news)
    limits(deadline, rss_limit)
    all_groups = np.arange(source.manifest["impressions"])
    heldout_day = source.days.max()
    if datetime.fromordinal(int(heldout_day)).date().isoformat() != "2019-11-14":
        raise ValueError("Frozen internal validation day differs")
    train_groups, validation_groups = all_groups[source.days < heldout_day], all_groups[source.days == heldout_day]
    if (len(train_groups), len(validation_groups)) != (1801231, 431517):
        raise ValueError("Frozen source-day selection population differs")
    selection = args.output / "internal_selection"
    selection.mkdir()
    priors = Priors(source, corpus, train_groups, deadline, rss_limit)
    print("Beginning real complete-impression CPU training on earlier training days", flush=True)
    model, ledgers = train_epochs(source, corpus, priors, train_groups, selection, 2, 42, deadline, rss_limit)
    internal = evaluate(source, corpus, priors, validation_groups, model, deadline, rss_limit)
    write_json(selection / "validation.json", internal)
    print(json.dumps({"scope": "heldout_training_day_not_official_dev", "metrics": internal}), flush=True)
    if internal["auc"] < 0.69:
        write_json(args.output / "result.json", {
            "stage": "rejected_internal_gate", "internal_validation": internal,
            "final_all_training_refit": False, "official_dev_contents_read": False,
            "model_fits": 1, "new_gpu_allocations": 0, "competition_test_goal_achieved": False,
        })
        return
    del model, priors
    final = args.output / "all_training_refit"
    final.mkdir()
    priors = Priors(source, corpus, all_groups, deadline, rss_limit)
    model, ledgers = train_epochs(source, corpus, priors, all_groups, final, 2, 42, deadline, rss_limit)
    source.unchanged()
    official = dev_source(args.dev_behaviors, args.dev_news, corpus, priors, model, args.output)
    score_path = args.output / "official_dev_scores.jsonl"
    metrics = evaluate(official, corpus, priors, np.arange(376471), model, deadline, rss_limit, score_path)
    write_json(args.output / "result.json", {
        "stage": "complete_all_training_refit_and_official_dev_scored",
        "internal_validation": internal, "official_large_dev": metrics,
        "final_epoch_coverage": ledgers, "model_fits": 2, "new_gpu_allocations": 0,
        "official_dev_scores_sha256": digest(score_path),
        "official_dev_used_for_fitting": False, "competition_test_contents_read": False,
        "competition_test_goal_achieved": False,
        "elapsed_seconds": time.monotonic() - started,
    })
    print(json.dumps({"scope": "official_MIND_large_dev_not_competition_test", "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
