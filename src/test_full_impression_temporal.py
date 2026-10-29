import time

import numpy as np
import pytest

from full_impression_temporal_training import behavior, design, feature_batch, fit, population, score, NAMES
from jev_content_ranker import content_features
from full_impression_temporal_verification import independent_population, numeric_tree_scores
from sklearn.ensemble import HistGradientBoostingClassifier


def fixture():
    arrays = {
        "candidate_offsets.npy": np.asarray([0, 3, 5, 8], dtype=np.uint64),
        "candidate_news.npy": np.asarray([1, 2, 3, 2, 3, 1, 2, 3], dtype=np.int32),
        "candidate_labels.npy": np.asarray([1, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8),
        "training_days.npy": np.asarray([10, 11, 12], dtype=np.int32),
        "histories.npy": np.asarray([[0, 0, 0], [0, 1, 2], [1, 2, 3]], dtype=np.int32),
    }
    identifiers = ["a", "b", "c"]
    vectors = np.asarray([[0, 0], [1, 0], [0, 1], [.6, .8]], dtype=np.float32)
    table = ({name: index + 1 for index, name in enumerate(identifiers)}, vectors,
             np.asarray([0, 1, 2, 1]), np.asarray([0, 1, 3, 2]), np.asarray([0, 3, 4, 5]))
    margins = np.asarray([0, .1, -.2, .3], dtype=np.float32)
    return arrays, identifiers, table, margins


def test_complete_counts_and_strict_past_day_no_target_leakage():
    arrays, names, table, margins = fixture()
    days, counts = population(np.arange(3), arrays, 3, time.monotonic() + 60)
    assert counts[:, :, 0].sum() == 8 and counts[:, :, 1].sum() == 3
    actual = feature_batch(np.arange(3), arrays, names, table, margins, days, counts)
    changed = counts.copy()
    changed[2, 1, 1] += 1
    np.testing.assert_array_equal(actual, feature_batch(np.arange(3), arrays, names, table, margins, days, changed))
    changed[1, 2, 1] = 0
    changed[1, 3, 1] = 1
    updated = feature_batch(np.arange(3), arrays, names, table, margins, days, changed)
    np.testing.assert_array_equal(actual[:5], updated[:5])
    assert not np.array_equal(actual[5:], updated[5:])
    assert actual.shape == (8, len(NAMES)) and len(NAMES) == 27
    np.testing.assert_array_equal(actual[:3, -8:], 0)


def test_context_features_preserve_the_existing_content_encoder_and_no_padding():
    arrays, names, table, margins = fixture()
    days, counts = population(np.asarray([0, 1]), arrays, 3, time.monotonic() + 60)
    actual = feature_batch(np.asarray([1]), arrays, names, table, margins, days, counts)
    _, reference = content_features([{"impression_id": "1", "history": ("a", "b"),
                                     "items": [("b", 0), ("c", 0)]}], table)
    np.testing.assert_array_equal(actual[:, 1:19], reference[:, 2:])
    np.testing.assert_allclose(actual[:, 19:23], behavior(np.asarray([2, 3]), counts[0]), atol=0, rtol=0)


def test_full_support_design_has_every_positive_negative_and_unit_impression_weight(tmp_path):
    arrays, names, table, margins = fixture()
    days, counts = population(np.arange(2), arrays, 3, time.monotonic() + 60)
    features = feature_batch(np.arange(3), arrays, names, table, margins, days, counts)
    x, y, weights, report = design(np.asarray([0, 2]), arrays, features, tmp_path / "design",
                                    time.monotonic() + 60)
    np.testing.assert_array_equal(y, arrays["candidate_labels.npy"][[0, 1, 2, 5, 6, 7]])
    np.testing.assert_array_equal(x, features[[0, 1, 2, 5, 6, 7]])
    assert report["impressions"] == 2 and report["exposures"] == 6 and report["positives"] == 2
    np.testing.assert_allclose(weights, 1 / 3, atol=0, rtol=0)


def test_invalid_nonempty_population_cannot_be_silently_defaulted():
    with pytest.raises(ValueError, match="both training classes"):
        behavior(np.asarray([1]), np.asarray([[0, 0], [3, 0]], dtype=np.int64))


def test_independent_population_excludes_all_last_day_targets():
    arrays, _, _, _ = fixture()
    expected_days, expected_counts = population(np.arange(2), arrays, 3, time.monotonic() + 60)
    days, counts = independent_population(arrays, 3, False)
    np.testing.assert_array_equal(days, expected_days)
    np.testing.assert_array_equal(counts, expected_counts)
    arrays["candidate_labels.npy"][5:] = [1, 0, 1]
    altered_days, altered_counts = independent_population(arrays, 3, False)
    np.testing.assert_array_equal(days, altered_days)
    np.testing.assert_array_equal(counts, altered_counts)


def test_independent_numeric_tree_traversal_reproduces_every_raw_logit():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(500, 7))
    y = (x[:, 0] + x[:, 1] > 0).astype(np.uint8)
    model = HistGradientBoostingClassifier(max_iter=7, min_samples_leaf=5, categorical_features=None,
                                           early_stopping=False, random_state=42).fit(x, y)
    unseen = rng.normal(size=(123, 7))
    np.testing.assert_allclose(numeric_tree_scores(model, unseen), model.decision_function(unseen),
                               atol=1e-12, rtol=0)


def test_fixed_recipe_design_fit_score_and_raw_tree_replay(tmp_path):
    arrays, names, table, margins = fixture()
    deadline = time.monotonic() + 60
    days, counts = population(np.arange(2), arrays, 3, deadline)
    features = feature_batch(np.arange(3), arrays, names, table, margins, days, counts)
    inputs = design(np.arange(3), arrays, features, tmp_path / "design", deadline)
    model, report = fit(inputs, tmp_path / "head.joblib", deadline)
    measured = score(model, np.arange(3), arrays, features, tmp_path / "scores.f64", deadline)
    assert report["impressions"] == 3 and report["exposures"] == 8 and report["iterations"] == 160
    assert measured["impressions"] == 3 and measured["exposures"] == 8
    raw = np.fromfile(tmp_path / "scores.f64", dtype=np.float64)
    np.testing.assert_allclose(raw, numeric_tree_scores(model, features), atol=1e-12, rtol=0)


@pytest.mark.parametrize("seconds", [0, -1, 17401])
def test_invalid_training_budget_fails_before_any_data_read(seconds):
    from types import SimpleNamespace
    from full_impression_temporal_training import run

    with pytest.raises(ValueError, match="at most17400"):
        run(SimpleNamespace(max_seconds=seconds))
