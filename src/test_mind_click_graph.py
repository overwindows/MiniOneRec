import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from mind_click_graph import (
    ClickGraph, GRAPH_SHRINKAGE, GRAPH_WEIGHT, PRIOR_STRENGTH,
    Source, evaluate, history_matrix,
)


@pytest.fixture
def source():
    result = object.__new__(Source)
    result.offsets = np.array([0, 3, 5, 8, 11, 14], dtype=np.uint64)
    result.candidates = np.array([1, 2, 2, 1, 3, 2, 3, 4, 1, 3, 4, 1, 2, 4], dtype=np.int32)
    result.labels = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=np.uint8)
    result.histories = np.zeros((5, 30), dtype=np.int32)
    result.histories[0, -3:] = [4, 4, 2]
    result.histories[1, -2:] = [1, 2]
    result.histories[2, -2:] = [1, 2]
    result.histories[3, -2:] = [2, 1]
    result.histories[4, -1:] = [3]
    return result


def fit(source, groups=None, k=256):
    return ClickGraph.fit(source, np.arange(4) if groups is None else groups, 5,
                          time.monotonic() + 60, 3758096384, top_k=k)


def reference(source, groups, width, k):
    pairs = np.zeros((width, width), dtype=np.float64)
    exposures, clicks, left = (np.zeros(width) for _ in range(3))
    for row in groups:
        lo, hi = source.offsets[row:row + 2].astype(int)
        for news, label in zip(source.candidates[lo:hi], source.labels[lo:hi], strict=True):
            exposures[news] += 1
            clicks[news] += label
            if label:
                for history in source.histories[row]:
                    if history:
                        pairs[history, news] += 1
                        left[history] += 1
    expected = left[:, None] * clicks[None, :] / clicks.sum()
    weights = np.maximum(np.log((pairs + GRAPH_SHRINKAGE) / (expected + GRAPH_SHRINKAGE)), 0)
    weights[:, 0] = weights[0] = 0
    np.fill_diagonal(weights, 0)
    for row in weights:
        columns = np.flatnonzero(row > 0)
        retain = columns[np.lexsort((columns, -row[columns]))[:k]]
        row[np.setdiff1d(columns, retain)] = 0
    ratio = clicks.sum() / exposures.sum()
    probability = (clicks + PRIOR_STRENGTH * ratio) / (exposures + PRIOR_STRENGTH)
    return weights.astype(np.float32), np.log(probability / (1 - probability))


@pytest.mark.parametrize("k", [1, 2, 256])
def test_sparse_fit_matches_independent_complete_pair_reference(source, k):
    model = fit(source, k=k)
    graph, logits = reference(source, np.arange(4), 5, k)
    np.testing.assert_array_equal(model.graph.toarray(), graph)
    np.testing.assert_allclose(model.logits, logits, rtol=0, atol=1e-14)
    assert model.coverage["impressions"] == 4
    assert model.coverage["exposures"] == 11
    assert model.coverage["positives"] == 4
    assert model.coverage["complete_aggregation_passes"] == 1
    assert np.all(np.diff(model.graph.indptr) <= k)


def test_duplicate_exposures_and_history_occurrences_not_deduplicated(source):
    model = fit(source)
    assert model.logits[2] < model.logits[4]
    history = history_matrix(source.histories, 5)
    assert history[0, 4] == 2
    assert history[0].sum() == 3


def test_heldout_labels_and_candidates_cannot_change_fit(source):
    original = fit(source)
    source.labels[-3:] = [1, 0, 0]
    source.candidates[-3:] = [4, 4, 2]
    changed = fit(source)
    np.testing.assert_array_equal(original.graph.toarray(), changed.graph.toarray())
    np.testing.assert_array_equal(original.logits, changed.logits)
    assert original.coverage == changed.coverage


def test_repeated_positive_candidate_exposures_preserve_pair_counts(source):
    source.labels[:3] = [0, 1, 1]
    model = fit(source)
    graph, logits = reference(source, np.arange(4), 5, 256)
    np.testing.assert_array_equal(model.graph.toarray(), graph)
    np.testing.assert_allclose(model.logits, logits, rtol=0, atol=1e-14)
    assert model.coverage["positives"] == 5


def test_scores_match_dense_reference_and_retain_order_duplicates(source):
    model = fit(source)
    news = np.array([2, 4, 2, 3, 1], dtype=np.int32)
    groups = np.array([0, 0, 0, 1, 1])
    scores = model.scores(source.histories[:2], news, groups)
    expected = []
    for item, group in zip(news, groups, strict=True):
        history = source.histories[group]
        history = history[history != 0]
        signal = sum(float(model.graph[h, item]) for h in history) / max(len(history), 1)
        expected.append(model.logits[item] + GRAPH_WEIGHT * signal)
    np.testing.assert_allclose(scores, expected, atol=2e-7, rtol=0)
    assert scores[0] == scores[2]


def test_empty_history_is_exact_prior_and_unknown_extension_is_global_prior(source):
    model = fit(source)
    original = model.graph.toarray()
    model.extend(7)
    np.testing.assert_array_equal(model.graph[:5, :5].toarray(), original)
    assert model.logits[5] == model.logits[6] == model.logits[0]
    scores = model.scores(np.zeros((1, 30), dtype=np.int32), np.array([1, 6]), np.array([0, 0]))
    np.testing.assert_array_equal(scores, model.logits[[1, 6]])
    with pytest.raises(ValueError, match="shrink"):
        model.extend(4)


def test_serialization_preserves_all_scores(source, tmp_path):
    model = fit(source)
    path = tmp_path / "model"
    model.save(path)
    loaded = ClickGraph.load(path)
    np.testing.assert_array_equal(model.graph.toarray(), loaded.graph.toarray())
    np.testing.assert_array_equal(model.logits, loaded.logits)
    assert loaded.coverage == model.coverage
    with pytest.raises(FileExistsError):
        model.save(path)


def test_complete_evaluation_and_persisted_tie_correct_auc(source, tmp_path):
    model = fit(source)
    path = tmp_path / "scores.jsonl"
    result = evaluate(model, source, np.array([4]), path, time.monotonic() + 60, 3758096384)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0][0] == 4 and len(rows[0][1]) == 3
    assert result["impressions"] == 1 and result["exposures"] == 3 and result["positives"] == 1
    assert result["metrics"]["auc"] == roc_auc_score(source.labels[-3:], rows[0][1])
    with pytest.raises(FileExistsError):
        evaluate(model, source, np.array([4]), path, time.monotonic() + 60, 3758096384)


@pytest.mark.parametrize("groups", [np.array([], dtype=int), np.array([0, 0]), np.array([-1]),
                                     np.array([5]), np.array([0.0])])
def test_invalid_population_rejected(source, groups):
    with pytest.raises(ValueError):
        fit(source, groups)


@pytest.mark.parametrize("kind", ["labels", "empty_history", "candidate", "deadline"])
def test_invalid_fit_or_resource_bound_rejected(source, kind):
    if kind == "labels":
        source.labels[:3] = 0
    elif kind == "empty_history":
        source.histories[0, -1] = 5
    elif kind == "candidate":
        source.candidates[0] = 0
    with pytest.raises((ValueError, TimeoutError)):
        ClickGraph.fit(source, np.arange(4), 5,
                       time.monotonic() - 1 if kind == "deadline" else time.monotonic() + 60,
                       3758096384)


def test_single_class_evaluation_does_not_return_success_shaped_auc(source, tmp_path):
    model = fit(source)
    source.labels[-3:] = 0
    with pytest.raises(ValueError, match="both binary"):
        evaluate(model, source, np.array([4]), tmp_path / "bad.jsonl",
                 time.monotonic() + 60, 3758096384)


def test_invalid_graph_and_score_inputs_rejected(source):
    model = fit(source)
    graph = sparse.eye(5, format="csr")
    with pytest.raises(ValueError):
        ClickGraph(graph, model.logits, {})
    for candidates, groups in ((np.array([0]), np.array([0])),
                                (np.array([1]), np.array([1])),
                                (np.array([1.0]), np.array([0]))):
        with pytest.raises(ValueError):
            model.scores(source.histories[:1], candidates, groups)
