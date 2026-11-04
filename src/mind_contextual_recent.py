"""Frozen verified contextual scores plus a complete-impression recency residual."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from torch import nn

import mind_contextual_interest as contextual
from full_impression_ranker import full_impression_loss
from full_impression_training import batches
from mind_cpu_ranker import Priors, candidate_rows, digest, limits, validate_groups, write_json
from mind_pretrained_training import main


RSS_LIMIT = 3758096384
CACHE_DTYPE = np.dtype([("base", "<f4"), ("features", "<f2", (8,))])
BASE_HASHES = {
    "internal_selection": {
        "model.pt": "b5288183a35add2fe9c1b72cbc406a0da7dfdf9f1a6ccee2eb509ca1b49294a2",
        "training_priors.npz": "03ad105105876c6dae09fcb1e6188ce668ccecb6fe8e6a8ae22798be71a341ee",
        "model_config.json": "198905a32bb3e888948868de39e1398678fdb69120c167d89480df580aefe0ef"},
    "all_training_fit": {
        "model.pt": "8aa22656c7df837dfdbab2b39d80d91e33b8b43f35f63f18c4fd922cefef49fb",
        "training_priors.npz": "839d8513d70a0689e97d2a3fbaa9a81bca8633f8615a0b6871d3c280ab5ececd",
        "model_config.json": "8fe11a7c6b760d137488c44cc80300f8ecbba54b90ee4f9ab1c6cab65f91ab94"}}
FEATURES = ["contextual_nonprior_score", "article_prior_logit", "last_fitting_day_ctr_logit",
            "last_day_log_exposures_over_10", "all_days_log_exposures_over_10",
            "first_exposure_age_at_fitting_cutoff_over_7", "history_category_share", "history_subcategory_share"]


def load_base(phase, corpus, public_encoder):
    if (phase not in BASE_HASHES or digest(Path(contextual.__file__)) !=
            "aaa2213dd7f685f6329c63ead79d9dd273b35ca20a988a6477924f725af4a84d"):
        raise ValueError("Only the independently verified contextual base is eligible")
    directory = public_encoder.parent.parent / "jev-contextual-interest-01-20261104" / phase
    for name, expected in BASE_HASHES[phase].items():
        if digest(directory / name) != expected:
            raise ValueError("Frozen accepted base changed: " + name)
    return contextual.restore(directory, corpus, public_encoder).requires_grad_(False)


def phase_for(source, groups):
    counts = source.counts(groups)
    if counts == {"impressions": 1801231, "exposures": 66107268, "positives": 2712612}:
        return "internal_selection"
    if counts == {"impressions": 2232748, "exposures": 83507374, "positives": 3383656}:
        return "all_training_fit"
    raise ValueError("A base must have exactly the same fitting population, never heldout-trained weights")


def recent_statistics(source, corpus, fitting, priors, deadline):
    validate_groups(source, fitting)
    width = len(corpus.names) + 1
    maximum = int(source.days[fitting].max())
    first = np.full(width, maximum + 1, dtype=np.int32)
    exposures, clicks = np.zeros((2, width)), np.zeros((2, width))
    for start in range(0, len(fitting), 16384):
        limits(deadline, RSS_LIMIT)
        batch = fitting[start:start + 16384]
        rows, mapping, _ = candidate_rows(source.offsets, batch)
        articles, days = source.candidates[rows], source.days[batch][mapping]
        np.minimum.at(first, articles, days)
        fold_ids = source.folds[batch][mapping]
        for fold in (0, 1):
            selected = (days == maximum) & (fold_ids == fold)
            exposures[fold] += np.bincount(articles[selected], minlength=width)
            clicks[fold] += np.bincount(articles[selected], weights=source.labels[rows[selected]], minlength=width)
    age = np.where(first <= maximum, np.minimum(maximum - first, 14) / 7., 0)
    tables = []
    for index, prior in enumerate([priors.pooled, *priors.tables]):
        current_exposure = exposures.sum(0) if index == 0 else exposures[index - 1]
        current_clicks = clicks.sum(0) if index == 0 else clicks[index - 1]
        category_probability = 1 / (1 + np.exp(-prior[:, 1].astype(np.float64)))
        probability = (current_clicks + 20 * category_probability) / (current_exposure + 20)
        tables.append(np.stack((np.log(probability / (1 - probability)),
                                np.log1p(current_exposure) / 10, prior[:, 2] / 10, age), axis=1))
    result = np.asarray(tables, dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Finite opposite-fold recency features required")
    return result


class RecentResidual(nn.Module):
    def __init__(self, base, priors, statistics):
        super().__init__()
        self.base, self.priors, self.statistics = base.requires_grad_(False), priors, statistics
        self.corpus = base.corpus
        if statistics.shape != (3, len(self.corpus.names) + 1, 4) or not np.isfinite(statistics).all():
            raise ValueError("Aligned finite pooled/opposite-fold recency tables required")
        if not np.array_equal(priors.pooled, base.priors.pooled):
            raise ValueError("Frozen base and residual must share the exact fitting population")
        self.head = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode=True):
        self.training = mode
        self.head.train(mode)
        self.base.eval()
        return self

    def prepare_inference(self):
        self.base.prepare_inference()

    def inputs(self, histories, candidates, groups, folds=None):
        base = self.base.scores(histories, candidates, groups)
        prior = self.priors.features(candidates, folds)
        index = np.zeros(len(candidates), dtype=np.int64) if folds is None else 2 - folds
        stats = self.statistics[index, candidates]
        valid = histories != 0
        denominator = np.maximum(valid.sum(1), 1)[groups]
        shares = []
        for metadata in (self.corpus.category, self.corpus.subcategory):
            shares.append(((metadata[histories][groups] == metadata[candidates, None]) &
                           valid[groups]).sum(1) / denominator)
        semantic = base - self.base.priors.pooled[candidates, 0]
        features = np.column_stack((semantic, prior[:, 0], stats, *shares)).astype(np.float16)
        if folds is not None:
            base = semantic + prior[:, 0]
        if not np.isfinite(features).all() or not np.isfinite(base).all():
            raise FloatingPointError("Quantized residual features and float32 base scores must be finite")
        return base.astype(np.float32), features

    @torch.no_grad()
    def scores(self, histories, candidates, groups):
        output = []
        for start in range(0, len(candidates), 8192):
            base, features = self.inputs(histories, candidates[start:start + 8192], groups[start:start + 8192])
            output.append(base + self.head(torch.from_numpy(features.astype(np.float32))).squeeze(1).numpy())
        return np.concatenate(output)

    def extend(self, news):
        if self.training:
            raise ValueError("Only frozen inference may extend articles")
        self.base.extend(news)
        self.priors.extend(self.corpus)
        missing = len(self.corpus.names) + 1 - self.statistics.shape[1]
        if missing:
            additions = np.zeros((3, missing, 4), dtype=np.float32)
            additions[:, :, 0] = self.priors.pooled[-missing:, 1]
            self.statistics = np.concatenate((self.statistics, additions), axis=1)


def cache_inputs(model, source, arrays, fitting, cache, deadline):
    seen = np.zeros(len(source.days), dtype=bool)
    target_hash = hashlib.sha256()
    exposures = positives = 0
    for batch in batches(fitting, arrays):
        limits(deadline, RSS_LIMIT)
        if seen[batch].any():
            raise ValueError("Duplicate cached source impression")
        seen[batch] = True
        rows, groups, _ = candidate_rows(source.offsets, batch)
        base, features = model.inputs(source.histories[batch], source.candidates[rows], groups,
                                      source.folds[batch][groups])
        cache["base"][rows], cache["features"][rows] = base, features
        exposures += len(rows)
        positives += int(source.labels[rows].sum())
        target_hash.update(batch.astype("<i8").tobytes())
        target_hash.update(source.candidates[rows].astype("<i4").tobytes())
        target_hash.update(source.labels[rows].astype("u1").tobytes())
    expected = source.counts(fitting)
    if (int(seen.sum()) != expected["impressions"] or exposures != expected["exposures"]
            or positives != expected["positives"]):
        raise ValueError("Incomplete all-candidate feature cache")
    cache.flush()
    return dict(expected, consumed_source_positions_sha256=hashlib.sha256(np.packbits(seen).tobytes()).hexdigest(),
                target_sha256=target_hash.hexdigest(), feature_storage="8float16_plus_float32_base",
                physical_cache_rows=len(cache), row_bytes=CACHE_DTYPE.itemsize)


def fit(source, arrays, corpus, public_encoder, groups, directory, deadline):
    validate_groups(source, groups)
    phase = phase_for(source, groups)
    base = load_base(phase, corpus, public_encoder)
    priors = Priors(source, corpus, groups, deadline, RSS_LIMIT)
    statistics = recent_statistics(source, corpus, groups, priors, deadline)
    torch.manual_seed(42)
    model = RecentResidual(base, priors, statistics)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.001, weight_decay=.0001)
    if shutil.disk_usage("/tmp").free < 10000000000:
        raise OSError("At least10GB local disk required before complete feature caching")
    directory.mkdir()
    expected, ledgers = source.counts(groups), []
    with tempfile.TemporaryDirectory(prefix="jev-contextual-recent-", dir="/tmp") as temporary:
        cache = np.memmap(Path(temporary) / "features.bin", mode="w+", dtype=CACHE_DTYPE,
                          shape=(len(source.candidates),))
        try:
            coverage = cache_inputs(model, source, arrays, groups, cache, deadline)
            write_json(directory / "cache_coverage.json", coverage)
            for index in range(2):
                seen = np.zeros(len(source.days), dtype=bool)
                counts = dict(impressions=0, exposures=0, positives=0, optimizer_steps=0)
                total_loss = 0.
                for fold in (index % 2, 1 - index % 2):
                    fitting = groups[source.folds[groups] == fold]
                    if not len(fitting):
                        raise ValueError("Both nonempty fitting folds required")
                    for batch in batches(np.random.default_rng(42 + index).permutation(fitting), arrays):
                        limits(deadline, RSS_LIMIT)
                        if seen[batch].any():
                            raise ValueError("Duplicate trained source impression")
                        seen[batch] = True
                        rows, mapping, _ = candidate_rows(source.offsets, batch)
                        values = cache[rows]
                        features = torch.from_numpy(values["features"].astype(np.float32))
                        initial = torch.from_numpy(values["base"].copy())
                        labels = torch.from_numpy(source.labels[rows].astype(np.float32))
                        optimizer.zero_grad(set_to_none=True)
                        scores = initial + model.head(features).squeeze(1)
                        loss = full_impression_loss(scores, labels, torch.from_numpy(mapping), len(batch))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        total_loss += float(loss.detach()) * len(batch)
                        counts["impressions"] += len(batch)
                        counts["exposures"] += len(rows)
                        counts["positives"] += int(labels.sum())
                        counts["optimizer_steps"] += 1
                if any(counts[key] != value for key, value in expected.items()) or seen.sum() != len(groups):
                    raise ValueError("Incomplete residual fitting epoch")
                ledgers.append(dict(counts, epoch=index + 1, mean_loss=total_loss / len(groups),
                                    consumed_source_positions_sha256=hashlib.sha256(
                                        np.packbits(seen).tobytes()).hexdigest()))
                print(json.dumps({"stage": "contextual_recent_complete_epoch", **ledgers[-1]}), flush=True)
        finally:
            cache._mmap.close()
    torch.save(model.head.state_dict(), directory / "model.pt")
    priors.save(directory)
    np.savez(directory / "recent_statistics.npz", tables=statistics)
    write_json(directory / "epoch_coverage.json", ledgers)
    write_json(directory / "model_config.json", {
        "hypothesis": "verified_contextual_base_last_day_recency_residual_v1", "base_phase": phase,
        "base_sha256": BASE_HASHES[phase], "public_encoder_sha256": digest(public_encoder),
        "epochs": 2, "seed": 42, "hidden_dimensions": 16, "features": FEATURES,
        "feature_dtype": "float16", "base_score_dtype": "float32", "optimizer": "AdamW",
        "learning_rate": .001, "weight_decay": .0001, "candidate_sampling": "none",
        "loss": "complete_impression_mean_positive_log_softmax", "source_coverage": expected,
        "click_prior_training": "opposite_blake2b_impression_fold_only",
        "frozen_supervised_base_is_previously_verified_not_rejected": True})
    return model


def restore(directory, corpus, public_encoder):
    config = json.loads((directory / "model_config.json").read_bytes())
    if (config["public_encoder_sha256"] != digest(public_encoder) or config["epochs"] != 2
            or config["base_sha256"] != BASE_HASHES[config["base_phase"]]
            or config["features"] != FEATURES or config["candidate_sampling"] != "none"):
        raise ValueError("Frozen residual recipe/base changed")
    base = load_base(config["base_phase"], corpus, public_encoder)
    with np.load(directory / "recent_statistics.npz", allow_pickle=False) as arrays:
        statistics = arrays["tables"]
    model = RecentResidual(base, Priors.load(directory), statistics)
    model.head.load_state_dict(torch.load(directory / "model.pt", weights_only=True, map_location="cpu"), strict=True)
    return model.eval()


if __name__ == "__main__":
    main(fitter=fit)
