import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

import mind_contextual_recent as module
from mind_cpu_ranker import Priors, candidate_rows
from test_mind_contextual_interest import build
from test_mind_lexical_glm import inputs


def example(inputs):
    source, corpus, _, _ = inputs
    prototype = build(inputs)
    source.days = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    groups = np.arange(4)
    priors = Priors(source, corpus, groups, time.monotonic() + 30, module.RSS_LIMIT)
    base = module.contextual.CandidateInterest(corpus, prototype.features.numpy(), priors, Path("public")).eval()
    base.prepare_inference()
    stats = module.recent_statistics(source, corpus, groups, priors, time.monotonic() + 30)
    return module.RecentResidual(base, priors, stats).eval()


def test_zero_residual_preserves_verified_base_and_duplicate_order(inputs):
    model = example(inputs)
    candidates, groups = np.array([2, 1, 2]), np.array([0, 0, 0])
    expected = model.base.scores(inputs[0].histories[:1], candidates, groups)
    np.testing.assert_array_equal(model.scores(inputs[0].histories[:1], candidates, groups), expected)
    model.train()
    assert not model.base.training
    assert all(not parameter.requires_grad for parameter in model.base.parameters())


def test_opposite_fold_priors_and_empty_history_features(inputs):
    model = example(inputs)
    candidates, groups, folds = np.array([1, 2]), np.zeros(2, dtype=np.int64), np.array([0, 1])
    histories = np.zeros((1, 30), dtype=np.int32)
    score, features = model.inputs(histories, candidates, groups, folds)
    np.testing.assert_array_equal(features[:, 1], model.priors.features(candidates, folds)[:, 0].astype(np.float16))
    np.testing.assert_array_equal(features[:, 2:6], model.statistics[2 - folds, candidates].astype(np.float16))
    assert not features[:, 6:].any() and np.isfinite(score).all()


def test_recency_statistics_are_opposite_fold_and_not_pooled(inputs):
    model = example(inputs)
    source, corpus, _, _ = inputs
    width = len(corpus.names) + 1
    for index, fold in ((1, 0), (2, 1)):
        selected = np.flatnonzero((source.days == 1) & (source.folds == fold))
        rows, _, _ = candidate_rows(source.offsets, selected)
        exposures = np.bincount(source.candidates[rows], minlength=width)
        clicks = np.bincount(source.candidates[rows], weights=source.labels[rows], minlength=width)
        category = 1 / (1 + np.exp(-model.priors.tables[fold][:, 1].astype(np.float64)))
        probability = (clicks + 20 * category) / (exposures + 20)
        np.testing.assert_array_equal(model.statistics[index, :, 0],
                                      np.log(probability / (1 - probability)).astype(np.float32))


def test_tiny_complete_fit_restore_ledgers_and_base_freeze(inputs, tmp_path, monkeypatch):
    initial = example(inputs)
    source, corpus, _, _ = inputs
    public = tmp_path / "vectors.npz"
    public.write_bytes(b"synthetic-pinned-public")
    monkeypatch.setattr(module, "phase_for", lambda *_: "internal_selection")
    monkeypatch.setattr(module, "load_base", lambda *args: initial.base)
    arrays = {name: getattr(source, field) for field, name in (
        ("offsets", "candidate_offsets.npy"), ("candidates", "candidate_news.npy"),
        ("labels", "candidate_labels.npy"), ("histories", "histories.npy"))}
    before = {name: value.clone() for name, value in initial.base.state_dict().items()}
    directory = tmp_path / "fit"
    model = module.fit(source, arrays, corpus, public, np.arange(4), directory, time.monotonic() + 60).eval()
    ledgers = json.loads((directory / "epoch_coverage.json").read_bytes())
    assert len(ledgers) == 2 and all(row["impressions"] == 4 and row["exposures"] == 11 for row in ledgers)
    assert all(row["positives"] == 4 for row in ledgers)
    coverage = json.loads((directory / "cache_coverage.json").read_bytes())
    assert coverage["row_bytes"] == 20 and coverage["exposures"] == 11
    for name, value in initial.base.state_dict().items():
        assert torch.equal(before[name], value)
    rows, mapping, _ = candidate_rows(source.offsets, np.arange(4))
    expected = model.scores(source.histories, source.candidates[rows], mapping)
    restored = module.restore(directory, corpus, public)
    np.testing.assert_array_equal(restored.scores(source.histories, source.candidates[rows], mapping), expected)
    assert model.head[-1].weight.count_nonzero()


def test_population_guard_rejects_unqualified_base(inputs):
    with pytest.raises(ValueError, match="same fitting population"):
        module.phase_for(inputs[0], np.arange(4))


def test_population_guard_checks_exact_source_positions_not_just_counts(inputs, monkeypatch):
    source = inputs[0]
    source.days = np.zeros(len(source.histories), dtype=np.int32)
    monkeypatch.setattr(source, "counts", lambda _: {
        "impressions": 1801231, "exposures": 66107268, "positives": 2712612})
    with pytest.raises(ValueError, match="same fitting population"):
        module.phase_for(source, np.arange(4))


def test_cold_article_extension_preserves_frozen_parameters_and_prior_fallback(inputs, monkeypatch):
    model = example(inputs)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    class Encoder:
        def __init__(self, _):
            pass

        def encode(self, texts):
            values = np.zeros((len(texts), 384), dtype=np.float32)
            values[:, 0] = 1
            return values

    monkeypatch.setattr(module.contextual, "PublicEncoder", Encoder)
    model.extend({"N5": {"category": "never", "subcategory": "never",
                          "title": "Novel title", "abstract": "Novel abstract"}})
    assert model.statistics.shape == (3, 6, 4)
    np.testing.assert_array_equal(model.statistics[:, 5, 0],
                                  np.repeat(model.priors.pooled[5, 1], 3))
    assert not model.statistics[:, 5, 1:].any()
    assert not model.priors.known[5]
    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value)
    histories = np.zeros((1, 30), dtype=np.int32)
    scores = model.scores(histories, np.array([5, 5]), np.array([0, 0]))
    np.testing.assert_array_equal(scores, model.priors.pooled[[5, 5], 0])
    model.train()
    with pytest.raises(ValueError, match="frozen inference"):
        model.extend({})


def test_float16_feature_overflow_fails_explicitly(inputs):
    model = example(inputs)
    model.statistics[0, 1, 0] = 1e20
    with pytest.warns(RuntimeWarning, match="overflow"), pytest.raises(FloatingPointError, match="Quantized"):
        model.inputs(np.zeros((1, 30), dtype=np.int32), np.array([1]), np.array([0]))
