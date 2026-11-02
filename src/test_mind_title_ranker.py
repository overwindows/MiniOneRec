from pathlib import Path
import json
import sys
import time

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mind_cpu_ranker import candidate_rows, complete_loss
from mind_title_ranker import TitleHistoryRanker, evaluate, exact_auc, fit_epoch, title_features
from test_mind_lexical_glm import inputs


def make_model(inputs):
    _, corpus, _, news = inputs
    torch.manual_seed(42)
    return TitleHistoryRanker(corpus, *title_features(corpus, news), 0.04)


def test_title_weighted_embedding_matches_dense_reference(inputs):
    model = make_model(inputs)
    ids = torch.arange(len(model.tokens))
    actual = model.encode(ids)
    expected = (model.words.weight[model.tokens] * model.weights[:, :, None]).sum(dim=1)
    expected = torch.nn.functional.normalize(expected, dim=1)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[0], torch.zeros_like(actual[0]))


def test_joint_word_embeddings_receive_finite_nonzero_click_gradients(inputs):
    source = inputs[0]
    model = make_model(inputs)
    rows, groups, sizes = candidate_rows(source.offsets, np.arange(4))
    scores = model(torch.from_numpy(source.histories[:4].astype(np.int64)),
                   torch.from_numpy(source.candidates[rows].astype(np.int64)), torch.from_numpy(groups))
    complete_loss(scores, torch.from_numpy(source.labels[rows].astype(np.float32)),
                  torch.from_numpy(groups), torch.from_numpy(sizes)).backward()
    assert torch.isfinite(model.words.weight.grad).all()
    assert model.words.weight.grad[1:].abs().sum() > 0
    assert torch.equal(model.words.weight.grad[0], torch.zeros_like(model.words.weight.grad[0]))


def test_all_training_candidates_consumed_once_and_heldout_excluded(inputs):
    source = inputs[0]
    first = make_model(inputs)
    coverage = fit_epoch(first, source, np.arange(4), time.monotonic() + 60)
    source.labels[-3:] = [1, 1, 0]
    source.candidates[-3:] = [4, 4, 3]
    second = make_model(inputs)
    changed = fit_epoch(second, source, np.arange(4), time.monotonic() + 60)
    assert coverage == changed
    assert (coverage["impressions"], coverage["exposures"], coverage["positives"]) == (4, 11, 4)
    assert coverage["epochs"] == coverage["optimizer_steps"] == 1
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_cached_frozen_scores_and_serialization_match(inputs, tmp_path):
    source, corpus, _, _ = inputs
    model = make_model(inputs)
    coverage = fit_epoch(model, source, np.arange(4), time.monotonic() + 60)
    model.eval()
    candidates, groups = np.array([2, 1, 2]), np.zeros(3, dtype=np.int64)
    direct = model.scores(source.histories[:1], candidates, groups)
    model.prepare_inference()
    cached = model.scores(source.histories[:1], candidates, groups)
    np.testing.assert_allclose(cached, direct, atol=1e-6, rtol=0)
    model.save(tmp_path / "model", coverage)
    restored = TitleHistoryRanker.load(tmp_path / "model", corpus)
    np.testing.assert_array_equal(cached, restored.scores(source.histories[:1], candidates, groups))
    assert cached[0] == cached[2]
    model.train()
    assert model.cached is None
    with pytest.raises(ValueError, match="frozen"):
        model.prepare_inference()


def test_empty_histories_and_oov_article_are_finite(inputs):
    model = make_model(inputs)
    model.eval()
    model.extend({"N5": {"category": "unknown", "subcategory": "unknown",
                          "title": "zzzoovterm", "abstract": "zzzoovterm"}})
    score = model.scores(np.zeros((1, 30), dtype=np.int32), np.array([5]), np.array([0]))
    assert np.isfinite(score).all()
    assert score[0] == model.bias.item()


def test_shared_metadata_cannot_change(inputs):
    model = make_model(inputs)
    with pytest.raises(ValueError, match="metadata"):
        model.extend({"N1": dict(inputs[3]["N1"], title="Changed title")})


def test_exact_auc_matches_independent_sklearn_with_ties():
    rng = np.random.default_rng(92)
    for _ in range(150):
        labels = rng.integers(0, 2, 73)
        labels[:2] = [0, 1]
        scores = rng.integers(-5, 6, 73).astype(np.float32)
        assert abs(exact_auc(labels, scores) - roc_auc_score(labels, scores)) < 1e-14


def test_persisted_evaluation_preserves_every_impression_candidate_and_auc(inputs, tmp_path):
    source = inputs[0]
    result = evaluate(make_model(inputs), source, np.arange(5), tmp_path / "scores.jsonl",
                      time.monotonic() + 60)
    records = [json.loads(line) for line in (tmp_path / "scores.jsonl").read_text().splitlines()]
    assert [row[0] for row in records] == list(range(5))
    assert sum(len(row[1]) for row in records) == len(source.labels)
    assert (result["impressions"], result["exposures"], result["positives"]) == (5, 14, 5)
    expected = [roc_auc_score(source.labels[int(source.offsets[i]):int(source.offsets[i + 1])], scores)
                for i, scores in records]
    assert abs(result["metrics"]["auc"] - np.mean(expected)) < 1e-14


@pytest.mark.parametrize("kind", ["constant_labels", "nonfinite", "nonbinary"])
def test_invalid_metric_population_rejected(kind):
    labels, scores = np.array([0, 1]), np.array([0.0, 1.0])
    if kind == "constant_labels":
        labels[:] = 0
    elif kind == "nonfinite":
        scores[0] = np.nan
    else:
        labels[0] = 3
    with pytest.raises(ValueError):
        exact_auc(labels, scores)


@pytest.mark.parametrize("kind", ["deadline", "duplicate_groups", "invalid_label"])
def test_invalid_training_rejected(inputs, kind):
    source = inputs[0]
    model = make_model(inputs)
    if kind == "invalid_label":
        source.labels[0] = 3
    with pytest.raises((ValueError, TimeoutError)):
        fit_epoch(model, source, np.array([0, 0]) if kind == "duplicate_groups" else np.arange(4),
                  time.monotonic() - 1 if kind == "deadline" else time.monotonic() + 60)
