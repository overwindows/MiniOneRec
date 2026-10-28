from types import SimpleNamespace

import numpy as np
import pytest

import streamed_jev_readout as readout
from probe_jev_readout import predict


def data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 5))
    extras = rng.normal(size=(24, 2))
    base = rng.normal(size=24)
    labels = np.tile([0, 1], 12)
    groups = [str(index // 2) for index in range(24)]
    mask = np.arange(24) < 20
    recipe = {"name": "uniform_binary", "l2": .01, "balanced": False, "pair_fraction": 0.}
    head = {
        "mean": rng.normal(size=5), "scale": rng.uniform(.1, 2., size=5),
        "weight": rng.normal(size=5), "bias": np.asarray(.7),
    }
    return x, extras, base, labels, groups, mask, recipe, head


def test_warm_start_preserves_all_predictions_with_new_normalization_and_extra_columns():
    x, extras, base, labels, groups, mask, recipe, head = data()
    objective = readout.StreamedObjective(x, extras, base, labels, groups, mask, recipe)
    parameters = readout.initial_parameters(objective, head)
    expected = predict(head, x, base)
    actual = base + objective.standardized(np.arange(len(x))) @ parameters[:-1] + parameters[-1]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(parameters[5:-1], np.zeros(2))
    np.testing.assert_array_equal(objective.mean, np.column_stack((x, extras))[mask].mean(axis=0))


def test_old_default_is_still_100_and_uniform_budget_is_explicit(monkeypatch):
    x, extras, base, labels, groups, mask, recipe, head = data()
    seen = []

    def fake_minimize(function, initial, **kwargs):
        seen.append((initial.copy(), kwargs["options"].copy()))
        return SimpleNamespace(success=True, x=initial, fun=function(initial)[0], nit=0, nfev=1)

    monkeypatch.setattr(readout, "minimize", fake_minimize)
    _, default_report = readout.fit_streamed(x, extras, base, labels, groups, mask, recipe)
    _, uniform_report = readout.fit_streamed(
        x, extras, base, labels, groups, mask, recipe, max_iter=250, initial_head=head,
    )
    assert seen[0][1] == {"maxiter": 100, "maxcor": 100, "maxls": 30, "gtol": 1e-6, "ftol": 1e-10}
    assert seen[1][1] == {**seen[0][1], "maxiter": 250}
    assert default_report["iteration_limit"] == 100 and not default_report["warm_start"]
    assert uniform_report["iteration_limit"] == 250 and uniform_report["warm_start"]
    np.testing.assert_array_equal(seen[0][0], np.zeros(8))
    assert np.abs(seen[1][0]).sum() > 0


def test_invalid_warm_start_and_iteration_limits_fail_explicitly():
    x, extras, base, labels, groups, mask, recipe, head = data()
    objective = readout.StreamedObjective(x, extras, base, labels, groups, mask, recipe)
    for changed in (
        {**head, "scale": np.zeros(5)}, {**head, "weight": np.ones(8)},
        {**head, "bias": np.asarray([1.])}, {**head, "weight": np.full(5, np.nan)},
    ):
        with pytest.raises(ValueError, match="Warm start"):
            readout.initial_parameters(objective, changed)
    for limit in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="iteration"):
            readout.fit_streamed(x, extras, base, labels, groups, mask, recipe, max_iter=limit)


def test_real_uniform_fit_converges_and_saved_head_replays():
    x, extras, base, labels, groups, mask, recipe, head = data()
    fitted, report = readout.fit_streamed(
        x, extras, base, labels, groups, mask, recipe, max_iter=250, initial_head=head,
    )
    assert report["converged"] and report["iterations"] < 250
    scores = readout.predict_streamed(fitted, x, extras, base)
    np.testing.assert_allclose(scores, predict(fitted, np.column_stack((x, extras)), base),
                               rtol=0, atol=1e-13)
