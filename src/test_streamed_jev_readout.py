import time

import numpy as np
import pytest
from scipy.optimize import check_grad

from probe_jev_readout import fit_head, predict
from streamed_jev_readout import (
    RECIPES, StreamedObjective, fit_streamed, predict_streamed,
    select_joint, validation_impressions,
)


def fixture():
    random = np.random.default_rng(42)
    features, extras = random.normal(size=(24, 3)), random.normal(size=(24, 2))
    labels = np.tile([0, 1, 0], 8)
    groups = [str(index // 3) for index in range(24)]
    margins = random.normal(size=24)
    mask = np.arange(24) < 18
    return features, extras, margins, labels, groups, mask


@pytest.mark.parametrize("recipe", RECIPES)
def test_streamed_analytic_gradient(recipe):
    data = fixture()
    objective = StreamedObjective(*data, recipe)
    parameters = np.linspace(-.2, .2, 6)
    error = check_grad(
        lambda value: objective.value_gradient(value)[0],
        lambda value: objective.value_gradient(value)[1], parameters,
    )
    assert error < 1e-6


def test_binary_matches_existing_dense_head():
    features, extras, margins, labels, groups, mask = fixture()
    head, report = fit_streamed(*fixture(), RECIPES[0])
    dense = np.column_stack((features, extras))
    expected, _ = fit_head(
        dense[mask], margins[mask], labels[mask],
        [group for group, keep in zip(groups, mask) if keep], .01, "binary",
    )
    np.testing.assert_allclose(head["mean"], expected["mean"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(head["scale"], expected["scale"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        predict_streamed(head, features, extras, margins),
        predict(expected, dense, margins), rtol=0, atol=1e-4,
    )
    assert report["converged"] and report["fitting_rows"] == mask.sum()


def test_validation_features_and_labels_do_not_enter_fit():
    data = list(fixture())
    first, _ = fit_streamed(*data, RECIPES[3])
    data[0] = data[0].copy()
    data[1] = data[1].copy()
    data[3] = data[3].copy()
    data[0][~data[-1]] = 1e8
    data[1][~data[-1]] = -1e8
    data[3][~data[-1]] = 1 - data[3][~data[-1]]
    second, _ = fit_streamed(*data, RECIPES[3])
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_impression_weights_and_unpaired_binary_coverage():
    features, extras, margins, labels, groups, mask = fixture()
    labels[:3] = 1
    objective = StreamedObjective(features, extras, margins, labels, groups, mask, RECIPES[3])
    np.testing.assert_allclose(objective.weights.sum(), 1.)
    assert len(objective.indices) == 18
    assert not np.isin(objective.pairs[0], [0, 1, 2]).any()
    assert not np.isin(objective.pairs[1], [0, 1, 2]).any()
    assert (objective.weights[:3] > 0).all()


def test_deadline_and_misalignment_fail():
    with pytest.raises(TimeoutError):
        fit_streamed(*fixture(), RECIPES[0], deadline=time.monotonic() - 1)
    data = list(fixture())
    data[1] = data[1][:-1]
    with pytest.raises(ValueError, match="aligned"):
        fit_streamed(*data, RECIPES[0])


def test_original_fitting_and_selection_cohorts_are_both_excluded():
    records = [
        {"impression_id": name, "day": day}
        for name, day in (("a", "1"), ("b", "2"), ("c", "2"), ("d", "2"))
    ]
    day, selected = validation_impressions(
        records, [{"impression_id": "b"}], [{"impression_id": "c"}],
    )
    assert day == "2" and selected == {"d"}
    with pytest.raises(ValueError, match="nonempty"):
        validation_impressions(records, [{"impression_id": name} for name in "bcd"], [])


def test_gate_requires_improvement_over_both_strong_controls():
    def arm(name, auc, mrr=.4):
        return {"name": name, "metrics": {"auc": auc, "mrr": mrr, "ndcg5": .4, "ndcg10": .4}}
    controls = [arm("retained", .7), arm("binary", .71)]
    best, opens = select_joint([arm("a", .712), arm("b", .716)], controls)
    assert best["name"] == "b" and opens
    assert not select_joint([arm("a", .714)], controls)[1]
    assert not select_joint([arm("a", .72, .39)], controls)[1]
    assert select_joint([arm("first", .72), arm("second", .72)], controls)[0]["name"] == "first"
