import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import check_grad

import streamed_jev_readout
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


def test_optimizer_history_matches_dense_prior_without_relaxing_limits(monkeypatch):
    original = streamed_jev_readout.minimize
    captured = {}

    def optimizer(*args, **kwargs):
        captured.update(kwargs["options"])
        return original(*args, **kwargs)

    monkeypatch.setattr(streamed_jev_readout, "minimize", optimizer)
    _, report = fit_streamed(*fixture(), RECIPES[0])
    assert captured == {
        "maxiter": 100, "maxcor": 100, "maxls": 30, "gtol": 1e-6, "ftol": 1e-10,
    }
    assert report["history_size"] == 100


def test_nonconvergence_is_not_accepted(monkeypatch):
    monkeypatch.setattr(
        streamed_jev_readout, "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, message="iteration limit", nit=100, nfev=120, fun=.3, jac=np.ones(6),
        ),
    )
    with pytest.raises(RuntimeError, match="iterations=100.*maximum_gradient=1"):
        fit_streamed(*fixture(), RECIPES[0])


@pytest.mark.parametrize("recipe", (*RECIPES, {
    "name": "pair_only_fixture", "balanced": True, "pair_fraction": 1., "l2": .01,
}))
def test_streamed_hessian_product_matches_gradient_differences(recipe, monkeypatch):
    monkeypatch.setattr(streamed_jev_readout, "BATCH_SIZE", 4)
    data = list(fixture())
    data[3][:3] = 1
    objective = StreamedObjective(*data, recipe)
    parameters = np.linspace(-.2, .2, 6)
    direction = np.linspace(.4, -.3, 6)
    epsilon = 1e-5
    expected = (objective.value_gradient(parameters + epsilon * direction)[1]
                - objective.value_gradient(parameters - epsilon * direction)[1]) / (2 * epsilon)
    actual = objective.hessian_product(parameters, direction)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
    assert direction @ actual >= 0
    other = np.linspace(-.5, .1, 6)
    assert other @ actual == pytest.approx(direction @ objective.hessian_product(parameters, other))
    if recipe["pair_fraction"] == 1.:
        assert actual[-1] == 0.


@pytest.mark.parametrize("recipe", RECIPES)
def test_newton_matches_default_solver_and_checks_stationarity(recipe):
    features, extras, margins, _, _, _ = fixture()
    expected, _ = fit_streamed(*fixture(), recipe)
    head, report = fit_streamed(*fixture(), recipe, solver="newton-cg")
    np.testing.assert_allclose(
        predict_streamed(head, features, extras, margins),
        predict_streamed(expected, features, extras, margins), rtol=0, atol=1e-4,
    )
    assert report["converged"] and report["solver"] == "newton-cg"
    assert report["maximum_gradient"] <= 1e-6 and report["hessian_vector_products"] > 0
    assert report["iterations"] <= 100 and "history_size" not in report


def test_newton_validation_invariance_and_deadline(monkeypatch):
    data = list(fixture())
    first, _ = fit_streamed(*data, RECIPES[3], solver="newton-cg")
    data[0][~data[-1]] = 1e8
    data[1][~data[-1]] = -1e8
    data[3][~data[-1]] = 1 - data[3][~data[-1]]
    second, _ = fit_streamed(*data, RECIPES[3], solver="newton-cg")
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    objective = StreamedObjective(*data, RECIPES[3])
    objective.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        objective.hessian_product(np.zeros(6), np.ones(6))
    with pytest.raises(TimeoutError):
        fit_streamed(*data, RECIPES[3], solver="newton-cg", deadline=time.monotonic() - 1)


@pytest.mark.parametrize("parameters,vector", [
    (np.zeros(5), np.ones(6)), (np.zeros(6), np.ones(5)),
    (np.full(6, np.nan), np.ones(6)), (np.zeros(6), np.full(6, np.inf)),
])
def test_hessian_invalid_inputs_fail_closed(parameters, vector):
    objective = StreamedObjective(*fixture(), RECIPES[3])
    with pytest.raises(ValueError, match="aligned finite"):
        objective.hessian_product(parameters, vector)


def test_newton_false_success_does_not_bypass_stationarity(monkeypatch):
    monkeypatch.setattr(
        streamed_jev_readout, "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=True, message="small step", nit=1, nfev=2, fun=.3,
            jac=np.zeros(6), x=np.zeros(6),
        ),
    )
    with pytest.raises(RuntimeError, match="stationarity.*maximum_gradient"):
        fit_streamed(*fixture(), RECIPES[0], solver="newton-cg")


@pytest.mark.parametrize("solver", ["lbfgs", "newton-cg"])
def test_iteration_exhaustion_never_returns_a_head(solver):
    with pytest.raises(RuntimeError, match="failed to converge"):
        fit_streamed(*fixture(), RECIPES[0], solver=solver, max_iter=1)


def test_unknown_solver_fails_before_fitting():
    with pytest.raises(ValueError, match="registered"):
        fit_streamed(*fixture(), RECIPES[0], solver="unregistered")


def test_newton_converges_with_correlated_full_width_features():
    random = np.random.default_rng(42)
    latent = random.normal(size=(48, 2))
    features = latent[:, :1] + .01 * random.normal(size=(48, 2560))
    extras = np.column_stack((latent, np.ones((48, 2))))
    labels = np.tile([0, 1, 0], 16)
    groups = [str(index // 3) for index in range(48)]
    mask = np.arange(48) < 36
    head, report = fit_streamed(
        features, extras, np.zeros(48), labels, groups, mask, RECIPES[0],
        deadline=time.monotonic() + 60, solver="newton-cg",
    )
    assert len(head["weight"]) == 2564
    assert report["iterations"] <= 100 and report["maximum_gradient"] <= 1e-6
    assert np.isfinite(predict_streamed(head, features, extras, np.zeros(48))).all()
