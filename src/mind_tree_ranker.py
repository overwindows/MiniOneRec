"""All-candidate query ranking with streamed LightGBM features and cross-fit priors."""

from datetime import datetime
import argparse
import gc
import hashlib
import inspect
import json
import math
import mmap
from pathlib import Path
import re
import resource
import time

import lightgbm as lgb
import numpy as np


MANIFEST_SHA = "4daa3377b4601aca4ae5b72e45ed2931becb8dd648dd23998f17fb3e1342a5f4"
PROOF_SHA = "de60c130dcd7ea853ce8d2b7fbe81b02b83d6b604fbbba99b96bbfa1ec841b25"
ENCODER_HASHES = {
    "corpus.json": "1a678cbb88976656001fe868c85edb58786c3db1a299d34520cc860b04262467",
    "lexical_encoder.npz": "823f5fc617506b210ca1e6f072eb8e7fe7e75357109f36febf92213ca3782de5"}
RSS_LIMIT = 3758096384
FEATURES = (
    "article_logit", "category_logit", "log_exposures", "category", "subcategory",
    "history_length", "category_fraction", "subcategory_fraction", "read_again",
    "mean_history_cosine", "max_history_cosine", "top_three_history_cosine",
    "latest_history_cosine", "age_hours", "title_words", "recent_category_fraction")
PARAMETERS = {
    "objective": "rank_xendcg", "metric": "None", "learning_rate": .1,
    "num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 2000, "max_bin": 15,
    "num_threads": 4, "device_type": "cpu", "seed": 42, "data_random_seed": 42, "deterministic": True,
    "force_col_wise": True, "histogram_pool_size": 32, "feature_fraction": 1.,
    "bagging_fraction": 1., "bagging_freq": 0, "verbosity": -1}
ROUNDS = 64


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4194304), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def limits(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Tree trial reached its original cutoff")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > RSS_LIMIT:
        raise MemoryError("Tree trial exceeded the unchanged peak-RSS ceiling")


def evict_pages(arrays):
    for array in arrays:
        if isinstance(array, np.memmap):
            array._mmap.madvise(mmap.MADV_DONTNEED)


def load_contract(directory, deadline):
    directory = Path(directory)
    if digest(directory / "manifest.json") != MANIFEST_SHA or digest(directory / "verification.json") != PROOF_SHA:
        raise ValueError("Pinned complete-source provenance changed")
    manifest = json.loads((directory / "manifest.json").read_bytes())
    for name, expected in manifest["sha256"].items():
        limits(deadline)
        if digest(directory / name) != expected:
            raise ValueError("Pinned complete-source file changed: " + name)
    arrays = {name: np.load(directory / name, mmap_mode="r", allow_pickle=False) for name in (
        "candidate_offsets.npy", "candidate_news.npy", "candidate_labels.npy", "histories.npy", "training_days.npy")}
    source = object.__new__(Source)
    for field, name in (("offsets", "candidate_offsets.npy"), ("candidates", "candidate_news.npy"),
                        ("labels", "candidate_labels.npy"), ("histories", "histories.npy"),
                        ("days", "training_days.npy")):
        setattr(source, field, arrays[name])
    with (directory / "impression_ids.txt").open("rb") as handle:
        source.folds = np.fromiter(
            (hashlib.blake2b(line.rstrip(b"\r\n"), digest_size=1).digest()[0] & 1 for line in handle),
            dtype=np.uint8, count=len(source.days))
        if handle.read(1):
            raise ValueError("Extra source impression identity")
    source.arrays = list(arrays.values())
    return manifest, source


class Source:
    def rows(self, groups):
        sizes = (self.offsets[groups + 1] - self.offsets[groups]).astype(np.int64)
        positions = np.repeat(self.offsets[groups].astype(np.int64) - (np.cumsum(sizes) - sizes), sizes)
        return positions + np.arange(int(sizes.sum())), np.repeat(groups, sizes)

    def counts(self, groups):
        exposures = positives = 0
        for start in range(0, len(groups), 8192):
            rows, _ = self.rows(groups[start:start + 8192])
            exposures += len(rows)
            positives += int(self.labels[rows].sum(dtype=np.uint64))
        return {"impressions": len(groups), "exposures": exposures, "positives": positives}


class Metadata:
    @classmethod
    def load(cls, directory):
        for name, expected in ENCODER_HASHES.items():
            if digest(Path(directory) / name) != expected:
                raise ValueError("Frozen label-free metadata encoder changed")
        document = json.loads((Path(directory) / "corpus.json").read_bytes())
        metadata = cls()
        for key in ("names", "categories", "subcategories", "metadata_sha256"):
            setattr(metadata, key, document[key])
        metadata.lookup = {name: index + 1 for index, name in enumerate(metadata.names)}
        with np.load(Path(directory) / "lexical_encoder.npz", allow_pickle=False) as arrays:
            metadata.lexical, metadata.category, metadata.subcategory = (
                arrays[name] for name in ("lexical", "category", "subcategory"))
            metadata.vocabulary, metadata.idf, metadata.components = (
                arrays[name] for name in ("vocabulary", "idf", "components"))
        return metadata

    def bind_titles(self, path):
        lengths, seen = np.zeros(len(self.names) + 1, dtype=np.float32), set()
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 8 or fields[0] in seen or fields[0] not in self.lookup:
                    raise ValueError("Exact training article identities required")
                name = fields[0]
                fingerprint = hashlib.sha256(json.dumps(
                    [fields[index] for index in (1, 2, 3, 4)], ensure_ascii=False).encode("utf-8")).hexdigest()
                if fingerprint != self.metadata_sha256[name]:
                    raise ValueError("Training article metadata changed")
                seen.add(name)
                lengths[self.lookup[name]] = len(re.findall(r"\w+", fields[3]))
        if seen != set(self.names):
            raise ValueError("Training article metadata omitted")
        self.title_words = lengths

    def extend(self, path):
        from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

        additions, seen = {}, set()
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 8 or fields[0] in seen:
                    raise ValueError("Malformed/duplicate inference article")
                name = fields[0]
                seen.add(name)
                fingerprint = hashlib.sha256(json.dumps(
                    [fields[index] for index in (1, 2, 3, 4)], ensure_ascii=False).encode("utf-8")).hexdigest()
                if name in self.lookup:
                    if fingerprint != self.metadata_sha256[name]:
                        raise ValueError("Shared inference article metadata changed")
                else:
                    additions[name] = fields
                    self.metadata_sha256[name] = fingerprint
        names = sorted(additions)
        if not names:
            return
        vectorizer = TfidfVectorizer(
            vocabulary={name: index for index, name in enumerate(self.vocabulary.tolist())},
            ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
        vectorizer.vocabulary_ = vectorizer.vocabulary
        vectorizer.fixed_vocabulary_ = True
        vectorizer._tfidf = TfidfTransformer(sublinear_tf=True)
        vectorizer._tfidf.n_features_in_ = len(self.vocabulary)
        vectorizer._tfidf.idf_ = self.idf
        matrix = vectorizer.transform([additions[name][3] + " " + additions[name][4] for name in names])
        vectors = np.asarray(matrix @ self.components.T, dtype=np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        self.lexical = np.concatenate((self.lexical, vectors))
        self.category = np.r_[self.category, [self.categories.get(additions[name][1], 0) for name in names]]
        self.subcategory = np.r_[self.subcategory, [self.subcategories.get(additions[name][2], 0) for name in names]]
        self.title_words = np.r_[self.title_words, [len(re.findall(r"\w+", additions[name][3])) for name in names]]
        for name in names:
            self.names.append(name)
            self.lookup[name] = len(self.names)


def prepare_times(source, contract, raw_behaviors, deadline):
    times = np.empty(len(source.days), dtype=np.int32)
    dates = {}
    first_day = int(source.days.min())
    seen = 0
    with Path(raw_behaviors).open("rb") as handle, (Path(contract) / "impression_ids.txt").open("rb") as identities:
        for index, pair in enumerate(zip(handle, identities, strict=True)):
            if index >= len(times):
                raise ValueError("Extra raw training impression")
            fields = pair[0].split(b"\t", 3)
            if len(fields) != 4 or fields[0] != pair[1].rstrip(b"\r\n"):
                raise ValueError("Raw time/source impression identity changed")
            day, clock, ampm = fields[2].split()
            if day not in dates:
                dates[day] = datetime.strptime(day.decode("ascii"), "%m/%d/%Y").date().toordinal()
            hour, minute, second = map(int, clock.split(b":"))
            if (dates[day] != source.days[index] or ampm not in (b"AM", b"PM")
                    or not 1 <= hour <= 12 or not 0 <= minute < 60 or not 0 <= second < 60):
                raise ValueError("Raw training time/day changed")
            times[index] = ((dates[day] - first_day) * 86400
                            + ((hour % 12) + 12 * (ampm == b"PM")) * 3600 + minute * 60 + second)
            seen += 1
            if index % 16384 == 0:
                limits(deadline)
        if seen != len(times):
            raise ValueError("Omitted raw training impression")
    source.times = times


def prior_table(exposures, clicks, categories):
    rate = clicks.sum() / exposures.sum()
    if not 0 < rate < 1:
        raise ValueError("Each opposite-fold pool requires both classes")
    category_p = (np.bincount(categories, weights=clicks) + 100 * rate) / (
        np.bincount(categories, weights=exposures) + 100)
    probability = (clicks + 20 * category_p[categories]) / (exposures + 20)
    return np.stack((np.log(probability / (1 - probability)),
                     np.log(category_p[categories] / (1 - category_p[categories])),
                     np.log1p(exposures)), axis=1).astype(np.float32)


class Statistics:
    @classmethod
    def fit(cls, source, metadata, groups, deadline):
        stats = cls()
        width = len(metadata.names) + 1
        exposures, clicks = np.zeros((2, width)), np.zeros((2, width))
        first = np.full(width, np.iinfo(np.int32).max, dtype=np.int32)
        for start in range(0, len(groups), 8192):
            limits(deadline)
            rows, queries = source.rows(groups[start:start + 8192])
            news = source.candidates[rows]
            folds = source.folds[queries]
            for fold in (0, 1):
                selected = folds == fold
                exposures[fold] += np.bincount(news[selected], minlength=width)
                clicks[fold] += np.bincount(news[selected], weights=source.labels[rows[selected]], minlength=width)
            np.minimum.at(first, news, source.times[queries])
        stats.tables = np.stack([prior_table(exposures[i], clicks[i], metadata.category) for i in (0, 1)])
        stats.pooled = prior_table(exposures.sum(0), clicks.sum(0), metadata.category)
        stats.first = first
        return stats

    def extend(self, metadata):
        old = len(self.pooled)
        missing = len(metadata.names) + 1 - old
        if missing < 0:
            raise ValueError("Inference catalog cannot shrink")
        if missing:
            category_logits = {}
            for index, category in enumerate(metadata.category[:old]):
                category_logits.setdefault(int(category), float(self.pooled[index, 1]))
            added = np.asarray([
                [category_logits.get(int(category), float(self.pooled[0, 1]))] * 2 + [0.]
                for category in metadata.category[old:]], dtype=np.float32)
            self.pooled = np.concatenate((self.pooled, added))
            self.first = np.r_[self.first, np.full(missing, np.iinfo(np.int32).max, dtype=np.int32)]


class FeatureSequence(lgb.Sequence):
    batch_size = 8192

    def __init__(self, source, metadata, statistics, groups, training, deadline):
        groups = np.asarray(groups)
        if (groups.ndim != 1 or groups.dtype.kind not in "iu" or not len(groups)
                or np.any(groups < 0) or np.any(groups >= len(source.offsets) - 1)
                or len(np.unique(groups)) != len(groups)):
            raise ValueError("Distinct valid source impressions required")
        self.source, self.metadata, self.statistics = source, metadata, statistics
        self.groups, self.training, self.deadline = groups, training, deadline
        self.sizes = (source.offsets[groups + 1] - source.offsets[groups]).astype(np.int32)
        if not len(groups) or np.any(self.sizes < 2) or np.any(self.sizes > 8192):
            raise ValueError("Complete nonempty ranking queries required")
        self.offsets = np.r_[np.int64(0), np.cumsum(self.sizes, dtype=np.int64)]
        self.consumed = 0

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        scalar = isinstance(index, (int, np.integer))
        if isinstance(index, slice):
            start, end, step = index.indices(len(self))
            if step != 1 or start != self.consumed:
                raise ValueError("Dataset construction must stream each complete row exactly once")
            positions = np.arange(start, end)
            self.consumed = end
        else:
            positions = np.atleast_1d(np.asarray(index, dtype=np.int64))
        if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= len(self)):
            raise ValueError("Invalid streamed exposure position")
        query = np.searchsorted(self.offsets, positions, side="right") - 1
        groups = self.groups[query]
        rows = self.source.offsets[groups].astype(np.int64) + positions - self.offsets[query]
        result = self.features(groups, self.source.candidates[rows])
        if not isinstance(index, slice):
            result = result.astype(np.float64)
        return result[0] if scalar else result

    def features(self, groups, news):
        limits(self.deadline)
        source, metadata, stats = self.source, self.metadata, self.statistics
        histories = source.histories[groups]
        mask = histories != 0
        count = mask.sum(axis=1)
        denominator = np.maximum(count, 1)
        category, subcategory = metadata.category[news], metadata.subcategory[news]
        same_category = (metadata.category[histories] == category[:, None]) & mask
        same_subcategory = (metadata.subcategory[histories] == subcategory[:, None]) & mask
        vectors = metadata.lexical[histories]
        similarities = np.einsum("nd,nhd->nh", metadata.lexical[news], vectors)
        similarities[~mask] = 0
        means = vectors.sum(axis=1) / denominator[:, None]
        means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
        top = np.partition(np.where(mask, similarities, -np.inf), -3, axis=1)[:, -3:]
        top[~np.isfinite(top)] = 0
        maximum = np.max(np.where(mask, similarities, -np.inf), axis=1)
        maximum[count == 0] = 0
        prior = (stats.tables[1 - source.folds[groups], news] if self.training else stats.pooled[news])
        first = stats.first[news]
        age = np.where(first != np.iinfo(np.int32).max, (source.times[groups] - first.astype(np.int64)) / 3600, -1)
        result = np.column_stack((
            prior, category, subcategory, count, same_category.sum(axis=1) / denominator,
            same_subcategory.sum(axis=1) / denominator, ((histories == news[:, None]) & mask).any(axis=1),
            np.einsum("nd,nd->n", metadata.lexical[news], means), maximum,
            top.sum(axis=1) / np.maximum(np.minimum(count, 3), 1), similarities[:, -1],
            np.clip(age, -1, 168), metadata.title_words[news], same_category[:, -5:].sum(axis=1) / 5,
        )).astype(np.float32)
        if result.shape != (len(news), len(FEATURES)) or not np.isfinite(result).all():
            raise ValueError("Exact finite tree feature schema required")
        return result


def fit(source, metadata, groups, directory, deadline):
    if lgb.__version__ != "4.6.0" or Path(directory).exists():
        raise ValueError("Pinned LightGBM4.6.0 and fresh fit destination required")
    directory = Path(directory)
    directory.mkdir()
    statistics = Statistics.fit(source, metadata, groups, deadline)
    sequence = FeatureSequence(source, metadata, statistics, groups, True, deadline)
    labels = np.memmap(directory / "labels.float32", dtype=np.float32, mode="w+", shape=(len(sequence),))
    cursor = positives = 0
    for start in range(0, len(groups), 8192):
        rows, _ = source.rows(groups[start:start + 8192])
        labels[cursor:cursor + len(rows)] = source.labels[rows]
        cursor += len(rows)
        positives += int(source.labels[rows].sum(dtype=np.uint64))
    labels.flush()
    if cursor != len(sequence):
        raise ValueError("Training labels omitted complete source rows")
    dataset = lgb.Dataset(
        sequence, label=labels, group=sequence.sizes, feature_name=list(FEATURES),
        categorical_feature=["category", "subcategory"], params=PARAMETERS, free_raw_data=True)
    dataset.construct()
    if dataset.num_data() != cursor or sequence.consumed != cursor:
        raise ValueError("Native dataset omitted streamed exposures")
    evict_pages([*source.arrays, labels])
    gc.collect()

    def callback(environment):
        limits(deadline)
        if environment.iteration % 8 == 0:
            print(json.dumps({"stage": "all_candidate_tree_fit", "iteration": environment.iteration + 1,
                              "impressions": len(groups), "exposures": cursor, "positives": positives}), flush=True)

    model = lgb.train(PARAMETERS, dataset, num_boost_round=ROUNDS, callbacks=[callback])
    if model.current_iteration() != ROUNDS:
        raise ValueError("Incomplete fixed-round tree recipe")
    model.save_model(str(directory / "model.txt"))
    np.savez_compressed(directory / "statistics.npz", pooled=statistics.pooled, first=statistics.first)
    mask = np.zeros(len(source.days), dtype=bool)
    mask[groups] = True
    coverage = {"impressions": len(groups), "exposures": cursor, "positives": positives}
    write_json(directory / "fit_coverage.json", {
        **coverage, "rounds": ROUNDS, "every_candidate_streamed_once": True,
        "source_positions_sha256": hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
        "training_candidate_sampling": "none", "bin_cutpoints_sample_feature_values_only": True})
    write_json(directory / "model_config.json", {
        "lightgbm": lgb.__version__, "parameters": PARAMETERS, "features": FEATURES, "rounds": ROUNDS,
        "priors": "entire_opposite_impression_fold_only", "age_source": "fitting_exposures_only",
        "source_coverage": coverage})
    model.free_dataset()
    del dataset, labels
    evict_pages(source.arrays)
    gc.collect()
    return model, statistics


def evaluate(model, source, metadata, statistics, groups, path, deadline):
    sequence = FeatureSequence(source, metadata, statistics, groups, False, deadline)
    metrics, exposures, positives = [], 0, 0
    with Path(path).open("x") as handle:
        start = 0
        while start < len(groups):
            end, size = start, 0
            while end < len(groups) and end - start < 256 and size + sequence.sizes[end] <= 8192:
                size += int(sequence.sizes[end])
                end += 1
            selected = groups[start:end]
            rows, queries = source.rows(selected)
            scores = model.predict(sequence.features(queries, source.candidates[rows]), num_threads=4)
            cursor = 0
            for index in selected:
                count = int(source.offsets[index + 1] - source.offsets[index])
                labels = source.labels[rows[cursor:cursor + count]]
                values = scores[cursor:cursor + count]
                positive, negative = values[labels == 1], values[labels == 0]
                if not len(positive) or not len(negative) or not np.isfinite(values).all():
                    raise ValueError("Undefined/nonfinite group cannot be discarded")
                auc = (np.count_nonzero(positive[:, None] > negative) +
                       .5 * np.count_nonzero(positive[:, None] == negative)) / (len(positive) * len(negative))
                target = labels[np.argsort(values)[::-1]]
                mrr = float(np.sum(target / np.arange(1, count + 1)) / target.sum())
                ndcg = []
                for k in (5, 10):
                    length = min(k, count)
                    discounts = np.log2(np.arange(2, length + 2))
                    ndcg.append(float(np.sum(target[:length] / discounts) /
                                      np.sum(np.sort(labels)[::-1][:length] / discounts)))
                metrics.append((auc, mrr, *ndcg))
                identity = source.impression_ids[index] if hasattr(source, "impression_ids") else int(index)
                json.dump([identity, values.tolist()], handle, allow_nan=False)
                handle.write("\n")
                cursor += count
            exposures += size
            positives += int(source.labels[rows].sum(dtype=np.uint64))
            start = end
    coverage = {"impressions": len(metrics), "exposures": exposures, "positives": positives}
    if source.counts(groups) != coverage:
        raise ValueError("Incomplete tree evaluation population")
    return {**coverage, "metrics": {name: math.fsum(row[col] for row in metrics) / len(metrics)
                                   for col, name in enumerate(("auc", "mrr", "ndcg5", "ndcg10"))},
            "scores_sha256": digest(path), "ranking_tie_dtype": "float64"}


def dev_source(behaviors, metadata, first_training_day):
    source = object.__new__(Source)
    source.histories = np.zeros((376471, 30), dtype=np.int32)
    source.offsets = np.zeros(376472, dtype=np.uint64)
    source.candidates = np.empty(14085557, dtype=np.int32)
    source.labels = np.empty(14085557, dtype=np.uint8)
    source.times = np.empty(376471, dtype=np.int32)
    source.impression_ids = []
    identities, cursor = set(), 0
    with Path(behaviors).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            fields = line.rstrip("\r\n").split("\t")
            if index >= 376471 or len(fields) != 5 or not fields[0] or fields[0] in identities:
                raise ValueError("Pinned complete dev identity population changed")
            identities.add(fields[0])
            source.impression_ids.append(fields[0])
            stamp = datetime.strptime(fields[2], "%m/%d/%Y %I:%M:%S %p")
            source.times[index] = (stamp.date().toordinal() - first_training_day) * 86400 + (
                stamp.hour * 3600 + stamp.minute * 60 + stamp.second)
            history = [metadata.lookup[name] for name in fields[3].split()[-30:]]
            if history:
                source.histories[index, -len(history):] = history
            for token in fields[4].split():
                if cursor >= 14085557 or not token.endswith(("-0", "-1")):
                    raise ValueError("Pinned complete dev candidate population changed")
                name, label = token.rsplit("-", 1)
                source.candidates[cursor], source.labels[cursor] = metadata.lookup[name], int(label)
                cursor += 1
            source.offsets[index + 1] = cursor
    if len(identities) != 376471 or cursor != 14085557 or int(source.labels.sum()) != 574845:
        raise ValueError("Incomplete pinned official dev")
    return source


def main(fitter=fit, parameters=PARAMETERS):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "encoder", "raw-source", "dev-news", "dev-behaviors", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.seconds < 1:
        raise ValueError("Fresh bounded tree experiment required")
    deadline = time.monotonic() + args.seconds
    manifest, source = load_contract(args.contract, deadline)
    news, behaviors = args.raw_source / "news.tsv", args.raw_source / "behaviors.tsv"
    if any(digest(args.raw_source / name) != expected for name, expected in manifest["input_sha256"].items()):
        raise ValueError("Pinned raw training source changed")
    metadata = Metadata.load(args.encoder)
    metadata.bind_titles(news)
    prepare_times(source, args.contract, behaviors, deadline)
    earlier = np.flatnonzero(source.days < source.days.max())
    heldout = np.flatnonzero(source.days == source.days.max())
    if (source.counts(earlier) != {"impressions": 1801231, "exposures": 66107268, "positives": 2712612}
            or source.counts(heldout) != {"impressions": 431517, "exposures": 17400106, "positives": 671044}):
        raise ValueError("Pinned fitting/screening populations changed")
    args.output.mkdir()
    write_json(args.output / "registration.json", {
        "source_sha256": digest(Path(__file__)), "manifest_sha256": MANIFEST_SHA,
        "fitter_source_sha256": digest(inspect.getsourcefile(fitter)),
        "parameters": parameters, "rounds": ROUNDS, "gate_auc": .69,
        "seconds": args.seconds, "candidate_sampling": "none",
        "old_fitted_models_or_priors_read": False, "dev_or_test_contents_read": False})
    model, statistics = fitter(source, metadata, earlier, args.output / "internal_selection", deadline)
    internal = evaluate(model, source, metadata, statistics, heldout, args.output / "internal_scores.jsonl", deadline)
    write_json(args.output / "internal_validation.json", internal)
    load_contract(args.contract, deadline)
    if any(digest(args.raw_source / name) != expected for name, expected in manifest["input_sha256"].items()):
        raise ValueError("Pinned raw training source changed while fitting")
    if internal["metrics"]["auc"] < .69:
        write_json(args.output / "result.json", {
            "stage": "rejected_internal_gate", "internal_validation": internal,
            "final_all_training_fit": False, "official_dev_contents_read": False,
            "competition_test_goal_achieved": False})
        return
    del model, statistics
    gc.collect()
    all_groups = np.arange(len(source.days))
    if source.counts(all_groups) != {"impressions": 2232748, "exposures": 83507374, "positives": 3383656}:
        raise ValueError("ALL training candidates required")
    model, statistics = fitter(source, metadata, all_groups, args.output / "all_training_fit", deadline)
    expected_dev = {
        args.dev_news: "03ba5f02160672500bf5f1af6dd18e8f99830a845b08ca9fd9fbbc9bdd72baff",
        args.dev_behaviors: "7031ced7b8de5183c1bf2c4a303018df8a49d858714597421d9489e571d87b45"}
    if any(digest(path) != expected for path, expected in expected_dev.items()):
        raise ValueError("Pinned complete official dev changed")
    metadata.extend(args.dev_news)
    statistics.extend(metadata)
    dev = dev_source(args.dev_behaviors, metadata, int(source.days.min()))
    official = evaluate(model, dev, metadata, statistics, np.arange(376471),
                        args.output / "official_dev_scores.jsonl", deadline)
    if any(digest(path) != expected for path, expected in expected_dev.items()):
        raise ValueError("Pinned complete official dev changed during evaluation")
    load_contract(args.contract, deadline)
    write_json(args.output / "result.json", {
        "stage": "complete_all_training_fit_and_official_dev_scored",
        "internal_validation": internal, "official_dev": official,
        "final_all_training_fit": True, "official_dev_contents_read": True,
        "competition_test_contents_read": False, "competition_test_goal_achieved": False})


if __name__ == "__main__":
    main()
