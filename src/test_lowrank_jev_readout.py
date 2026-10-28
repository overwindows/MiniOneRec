import numpy as np
import pytest

from lowrank_jev_readout import (
    fit_whitening, gaussian_projection, predict_lowrank, project, transform,
    save_model,
)
from streamed_jev_readout import RECIPES, fit_streamed, predict_streamed


def test_projection_is_fixed_and_dimension_checked():
    np.testing.assert_array_equal(gaussian_projection(8, 3), gaussian_projection(8, 3))
    values = np.arange(32).reshape(4, 8)
    projection = gaussian_projection(8, 3)
    np.testing.assert_allclose(project(values, projection), values @ projection)
    with pytest.raises(ValueError):
        gaussian_projection(3, 4)


def test_whitening_has_identity_fitting_covariance_and_drops_constants():
    values = np.random.default_rng(42).normal(size=(100, 4))
    values[:, 3] = 1.
    mask = np.arange(100) < 70
    encoding = fit_whitening(values, mask)
    compact = transform(values, encoding)
    assert compact.shape == (100, 3)
    np.testing.assert_allclose(compact[mask].mean(axis=0), 0., atol=1e-14)
    np.testing.assert_allclose(compact[mask].T @ compact[mask] / mask.sum(), np.eye(3), atol=1e-12)


def test_validation_values_do_not_change_whitening():
    values = np.random.default_rng(42).normal(size=(40, 4))
    mask = np.arange(40) < 30
    first = fit_whitening(values, mask)
    values[~mask] = 1e12
    second = fit_whitening(values, mask)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_zero_residual_preserves_the_rich_text_baseline():
    features = np.random.default_rng(42).normal(size=(40, 8))
    extras = np.random.default_rng(43).normal(size=(40, 2))
    projection = gaussian_projection(8, 3)
    values = np.column_stack((project(features, projection), extras))
    encoding = {**fit_whitening(values, np.ones(40, dtype=bool)), "projection": projection}
    dimension = encoding["whitening"].shape[1]
    head = {"mean": np.zeros(dimension), "scale": np.ones(dimension),
            "weight": np.zeros(dimension), "bias": np.asarray(0.)}
    baseline = np.linspace(-3., 3., 40)
    np.testing.assert_array_equal(predict_lowrank(encoding, head, features, extras, baseline), baseline)


def test_constant_or_nonfinite_whitening_is_rejected():
    with pytest.raises(ValueError, match="constant"):
        fit_whitening(np.ones((4, 2)), np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="finite"):
        fit_whitening(np.full((4, 2), np.nan), np.ones(4, dtype=bool))


@pytest.mark.parametrize("recipe", RECIPES[1:])
def test_compact_training_converges_and_saved_model_replays(recipe, tmp_path):
    features = np.random.default_rng(42).normal(size=(80, 8))
    extras = np.random.default_rng(43).normal(size=(80, 2))
    labels = np.tile([0, 1], 40)
    groups = [str(index // 4) for index in range(80)]
    baseline = np.linspace(-2., 2., 80)
    mask = np.arange(80) < 60
    projection = gaussian_projection(8, 3)
    values = np.column_stack((project(features, projection), extras))
    encoding = {**fit_whitening(values, mask), "projection": projection}
    compact = transform(values, encoding)
    head, report = fit_streamed(
        compact, np.empty((80, 0)), baseline, labels, groups, mask, recipe,
    )
    assert report["converged"] and report["iterations"] < 100
    path = tmp_path / "model.npz"
    save_model(path, encoding, head)
    with np.load(path, allow_pickle=False) as archive:
        restored_encoding = {key: archive[key] for key in ("projection", "center", "whitening")}
        restored_head = {key: archive["head_" + key] for key in head}
    np.testing.assert_allclose(
        predict_lowrank(restored_encoding, restored_head, features, extras, baseline),
        predict_streamed(head, compact, np.empty((80, 0)), baseline),
        rtol=0, atol=1e-12,
    )
