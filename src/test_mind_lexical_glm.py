from pathlib import Path
import sys
import time

import numpy as np
import pytest
from scipy import sparse
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).parent))
from mind_cpu_ranker import Corpus, Source
from mind_lexical_glm import LexicalClickModel, binomial_objective, lexical_matrix, posterior_logits


@pytest.fixture
def inputs():
    news = {
        "N1": {"category": "sport", "subcategory": "ball", "title": "Football team wins match", "abstract": "Great football players"},
        "N2": {"category": "tech", "subcategory": "computer", "title": "Computer chip launch", "abstract": "New laptop processor"},
        "N3": {"category": "sport", "subcategory": "ball", "title": "Basketball team scores", "abstract": "Sports players game"},
        "N4": {"category": "tech", "subcategory": "computer", "title": "Software computer update", "abstract": "Laptop software news"},
    }
    corpus = Corpus(news, dimensions=3)
    source = object.__new__(Source)
    source.offsets = np.array([0, 3, 5, 8, 11, 14], dtype=np.uint64)
    source.candidates = np.array([1, 2, 2, 1, 3, 2, 3, 4, 1, 3, 4, 1, 2, 4], dtype=np.int32)
    source.labels = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=np.uint8)
    source.histories = np.zeros((5, 30), dtype=np.int32)
    source.histories[0, -2:] = [1, 2]
    source.histories[1, -1:] = [3]
    return source, corpus, lexical_matrix(corpus, news), news


def test_aggregate_binomial_objective_equals_every_candidate_loss_and_gradient():
    matrix = sparse.csr_matrix(np.array([[0.2, 0.7], [0.4, -0.3], [0.9, 0.1]]))
    news = np.array([0, 1, 1, 2, 0, 2, 2])
    labels = np.array([1, 0, 1, 0, 0, 1, 0])
    exposures = np.bincount(news, minlength=3)
    clicks = np.bincount(news, weights=labels, minlength=3)
    parameters = np.array([-0.5, 0.2, -0.1])
    objective, gradient = binomial_objective(parameters, matrix, exposures, clicks, 0)
    dense = matrix.toarray()[news]
    scores = dense @ parameters[1:] + parameters[0]
    expected = (np.logaddexp(0, scores) - labels * scores).mean()
    residual = expit(scores) - labels
    expected_gradient = np.concatenate(([residual.mean()], dense.T @ residual / len(labels)))
    np.testing.assert_allclose(objective, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=1e-14)


def test_analytic_gradient_matches_finite_difference():
    matrix = sparse.csr_matrix(np.array([[0.2, 0.7], [0.4, -0.3], [0.9, 0.1]]))
    parameters = np.array([-0.5, 0.2, -0.1])
    exposures, clicks = np.array([3, 5, 7]), np.array([1, 2, 1])
    _, analytic = binomial_objective(parameters, matrix, exposures, clicks, 0.01)
    for index in range(len(parameters)):
        delta = np.eye(len(parameters))[index] * 1e-6
        plus = binomial_objective(parameters + delta, matrix, exposures, clicks, 0.01)[0]
        minus = binomial_objective(parameters - delta, matrix, exposures, clicks, 0.01)[0]
        assert abs((plus - minus) / 2e-6 - analytic[index]) < 1e-9


def test_cold_content_logits_and_warm_posterior_are_numerically_stable():
    logits = np.array([-1000.0, 1000.0, -2.0])
    exposures, clicks = np.array([0.0, 0.0, 30.0]), np.array([0.0, 0.0, 4.0])
    posterior = posterior_logits(logits, exposures, clicks)
    np.testing.assert_allclose(posterior[:2], logits[:2], atol=1e-12, rtol=0)
    wanted = (4 + 20 * expit(-2)) / 50
    assert abs(posterior[2] - np.log(wanted / (1 - wanted))) < 1e-14
    assert np.isfinite(posterior).all()


def fit(inputs):
    source, corpus, matrix, _ = inputs
    return LexicalClickModel.fit(source, np.arange(4), corpus, matrix,
                                 time.monotonic() + 60, 3758096384)


def test_complete_aggregation_and_heldout_target_exclusion(inputs):
    source, _, _, _ = inputs
    original = fit(inputs)
    source.labels[-3:] = [1, 1, 0]
    source.candidates[-3:] = [4, 4, 3]
    changed = fit(inputs)
    np.testing.assert_array_equal(original.parameters, changed.parameters)
    np.testing.assert_array_equal(original.logits, changed.logits)
    assert original.coverage["impressions"] == 4
    assert original.coverage["exposures"] == 11
    assert original.coverage["positives"] == 4
    assert original.coverage["optimizer_converged"]


def test_empty_history_and_duplicate_candidate_order(inputs):
    model = fit(inputs)
    candidates = np.array([2, 1, 2])
    scores = model.scores(np.zeros((1, 30), dtype=np.int32), candidates, np.zeros(3, dtype=int))
    np.testing.assert_array_equal(scores, model.logits[candidates])
    assert scores[0] == scores[2]


def test_serialization_preserves_parameters_logits_and_scores(inputs, tmp_path):
    source, corpus, _, _ = inputs
    original = fit(inputs)
    original.save(tmp_path / "model")
    loaded = LexicalClickModel.load(tmp_path / "model", corpus)
    np.testing.assert_array_equal(original.parameters, loaded.parameters)
    np.testing.assert_array_equal(original.logits, loaded.logits)
    np.testing.assert_array_equal(original.scores(source.histories[:1], np.array([1, 2]), np.array([0, 0])),
                                  loaded.scores(source.histories[:1], np.array([1, 2]), np.array([0, 0])))


def test_unseen_article_gets_content_prediction_without_any_new_labels(inputs):
    _, _, _, news = inputs
    model = fit(inputs)
    original = model.logits.copy()
    new = {"N5": {"category": "tech", "subcategory": "computer",
                  "title": "New computer processor", "abstract": "Laptop chip"}}
    model.extend(new)
    np.testing.assert_array_equal(model.logits[:5], original)
    matrix = lexical_matrix(model.corpus, new, ["N5"])[1:]
    assert model.logits[-1] == (matrix @ model.parameters[1:] + model.parameters[0]).item()
    changed = dict(news["N1"], title="Changed article identity")
    with pytest.raises(ValueError, match="metadata"):
        model.extend({"N1": changed})


@pytest.mark.parametrize("kind", ["labels", "groups", "deadline"])
def test_invalid_fit_rejected(inputs, kind):
    source, corpus, matrix, _ = inputs
    if kind == "labels":
        source.labels[0] = 3
    with pytest.raises((ValueError, TimeoutError)):
        LexicalClickModel.fit(source, np.array([0, 0]) if kind == "groups" else np.arange(4),
                              corpus, matrix, time.monotonic() - 1 if kind == "deadline"
                              else time.monotonic() + 60, 3758096384)
