import time
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import numpy as np
import pytest

from full_impression_centered_training import epoch_coverage, projection, transform, unit_rows
from full_impression_centered_verification import verify_basis


def test_training_only_centering_whitening_and_zero_padding(tmp_path):
    raw = np.random.default_rng(42).normal(size=(21, 4)).astype(np.float32)
    raw[0] = 0
    basis, encoded = tmp_path / "basis.npz", tmp_path / "features.npy"
    before = raw.copy()
    report = projection(raw, basis, time.monotonic() + 60, dimension=2)
    transform(raw, basis, encoded, time.monotonic() + 60)
    with np.load(basis) as stored:
        x = unit_rows(raw[1:])
        np.testing.assert_allclose(stored["mean"], x.mean(axis=0), rtol=0, atol=1e-14)
        axes = stored["axes"]
        np.testing.assert_allclose(axes.T @ axes, np.eye(2), rtol=0, atol=1e-14)
        assert np.all(axes[np.argmax(np.abs(axes), axis=0), np.arange(2)] > 0)
        covariance = (x - x.mean(axis=0)).T @ (x - x.mean(axis=0)) / len(x)
        np.testing.assert_allclose(covariance @ axes, axes * stored["variance"], atol=1e-14, rtol=0)
        expected = ((x - stored["mean"]) @ axes) / np.sqrt(stored["variance"] + stored["ridge"])
    actual = np.load(encoded)
    assert actual.dtype == np.float32 and actual.shape == (21, 2)
    np.testing.assert_array_equal(actual[0], 0)
    np.testing.assert_allclose(actual[1:], expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(raw, before)
    assert report["articles"] == 20 and report["ridge_fraction"] == .01
    basis.rename(tmp_path / "projection.npz")
    encoded.rename(tmp_path / "article.centered.npy")
    evidence = verify_basis(raw, tmp_path, time.monotonic() + 60, dimension=2)
    assert evidence["all_transformed_bits_replayed"] and evidence["top_eigensystem_replayed"]
    corrupted = np.load(tmp_path / "article.centered.npy")
    corrupted[1, 0] += .001
    np.save(tmp_path / "article.centered.npy", corrupted)
    with pytest.raises(ValueError, match="Every transformed article bit"):
        verify_basis(raw, tmp_path, time.monotonic() + 60, dimension=2)


@pytest.mark.parametrize("kind", ["zero_article", "nan", "padding", "dtype"])
def test_bad_frozen_training_features_cannot_produce_a_basis(tmp_path, kind):
    raw = np.random.default_rng(42).normal(size=(21, 4)).astype(np.float32)
    raw[0] = 0
    if kind == "zero_article":
        raw[1] = 0
    elif kind == "nan":
        raw[1, 0] = np.nan
    elif kind == "padding":
        raw[0, 0] = 1.
    else:
        raw = raw.astype(np.float64)
    with pytest.raises(ValueError):
        projection(raw, tmp_path / "basis.npz", time.monotonic() + 60, dimension=2)
    assert not (tmp_path / "basis.npz").exists()


def test_new_articles_use_the_same_training_basis_without_refitting(tmp_path):
    rng = np.random.default_rng(42)
    training = rng.normal(size=(21, 4)).astype(np.float32)
    training[0] = 0
    basis = tmp_path / "basis.npz"
    projection(training, basis, time.monotonic() + 60, dimension=2)
    before = basis.read_bytes()
    unseen = np.vstack([np.zeros((1, 4), dtype=np.float32), rng.normal(size=(3, 4)).astype(np.float32)])
    transform(unseen, basis, tmp_path / "unseen.npy", time.monotonic() + 60)
    assert basis.read_bytes() == before and np.load(tmp_path / "unseen.npy").shape == (4, 2)


def test_partial_or_extra_epoch_coverage_cannot_claim_full_fitting():
    row = {"impressions": 3, "exposures": 8, "positives": 4, "mean_loss": 1., "optimizer_steps": 1}
    assert epoch_coverage({"epochs": [row.copy() for _ in range(3)]}, 3, 8, 4)
    assert not epoch_coverage({"epochs": [row]}, 3, 8, 4)
    bad = [row.copy() for _ in range(3)]
    bad[1]["positives"] = 3
    assert not epoch_coverage({"epochs": bad}, 3, 8, 4)


def test_frozen_basis_and_transformed_features_cannot_be_overwritten(tmp_path):
    raw = np.random.default_rng(42).normal(size=(21, 4)).astype(np.float32)
    raw[0] = 0
    basis, encoded = tmp_path / "basis.npz", tmp_path / "features.npy"
    deadline = time.monotonic() + 60
    projection(raw, basis, deadline, dimension=2)
    transform(raw, basis, encoded, deadline)
    before = basis.read_bytes(), encoded.read_bytes()
    with pytest.raises(FileExistsError):
        projection(raw, basis, deadline, dimension=2)
    with pytest.raises(FileExistsError):
        transform(raw, basis, encoded, deadline)
    assert before == (basis.read_bytes(), encoded.read_bytes())


@pytest.mark.parametrize("kind", ["scalar_variance", "empty_axes", "wrong_ridge", "nan", "wrong_dtype"])
def test_corrupt_frozen_basis_is_rejected_before_output_creation(tmp_path, kind):
    raw = np.random.default_rng(42).normal(size=(21, 4)).astype(np.float32)
    raw[0] = 0
    basis, encoded = tmp_path / "basis.npz", tmp_path / "features.npy"
    deadline = time.monotonic() + 60
    projection(raw, basis, deadline, dimension=2)
    with np.load(basis) as stored:
        values = {name: stored[name] for name in stored.files}
    if kind == "scalar_variance":
        values["variance"] = values["variance"][0]
    elif kind == "empty_axes":
        values["variance"] = values["variance"][:0]
        values["axes"] = values["axes"][:, :0]
    elif kind == "wrong_ridge":
        values["ridge"] *= 2
    elif kind == "nan":
        values["axes"][0, 0] = np.nan
    else:
        values["axes"] = values["axes"].astype(np.float32)
    np.savez(basis, **values)
    with pytest.raises(ValueError):
        transform(raw, basis, encoded, deadline)
    assert not encoded.exists()


def test_rejected_training_gate_never_refits_or_scores_dev(tmp_path, monkeypatch):
    import full_impression_centered_training as module

    class RegisteredTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 29, 15, tzinfo=timezone.utc)

    monkeypatch.setattr(module, "datetime", RegisteredTime)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(module.torch, "get_num_threads", lambda: 4)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.np, "__version__", "1.26.4")
    for key, value in {"AVX2": True, "AVX512F": False, "AVX512_SKX": False, "AVX512_ICL": False}.items():
        monkeypatch.setitem(module.np.core._multiarray_umath.__cpu_features__, key, value)
    baseline = {"auc": .6831950387658544, "mrr": .3, "ndcg5": .35, "ndcg10": .4}
    previous = {"feature_extraction": {"features_sha256": "fixed"},
                "recipes": {"candidate_attention": {"metrics": baseline}}}
    days = np.r_[np.zeros(1801231, dtype=np.uint8), np.ones(431517, dtype=np.uint8)]
    monkeypatch.setattr(module, "load_gate", lambda *args: previous)
    monkeypatch.setattr(module, "load_contract", lambda *args: ({"news": 101527}, {"training_days.npy": days}))
    monkeypatch.setattr(module, "digest", lambda path: "fixed")
    raw = np.broadcast_to(np.zeros((1, 2560), dtype=np.float32), (101528, 2560))
    monkeypatch.setattr(module.np, "load", lambda path, **kwargs:
                        raw if path.name == "article.features.npy" else np.zeros((2, 128), dtype=np.float32))
    monkeypatch.setattr(module, "projection", lambda *args: {"sha256": "fixed"})
    monkeypatch.setattr(module, "transform", lambda *args: None)
    calls = []

    def selection_only(features, indices, arrays, mode, output, deadline):
        calls.append(len(indices))
        assert len(indices) == 1801231, "A rejected gate must never fit the final population"
        row = dict(impressions=1801231, exposures=66107268, positives=2712612,
                   mean_loss=1., optimizer_steps=1)
        return object(), {"epochs": [row.copy() for _ in range(3)]}

    monkeypatch.setattr(module, "fit", selection_only)
    monkeypatch.setattr(module, "ModelPredictor", lambda model: model)
    monkeypatch.setattr(module, "evaluate", lambda *args:
                        {"metrics": {**baseline, "auc": baseline["auc"] + .004}})
    output = tmp_path / "run"
    module.run(SimpleNamespace(max_seconds=21000, source_revision="a" * 40, output=output,
                              training_artifacts=tmp_path, training_proof=tmp_path / "proof",
                              contract=tmp_path))
    result = json.loads((output / "result.json").read_bytes())
    assert calls == [1801231] and not result["opens_dev_gate"]
    assert not result["dev_or_test_contents_read"] and "final_refit" not in result
    assert not (output / "selected-full.pt").exists()


@pytest.mark.parametrize("seconds", [0, -1, 21001])
def test_invalid_training_budget_fails_before_reading_any_data(seconds):
    from full_impression_centered_training import run

    with pytest.raises(ValueError, match="at most21000"):
        run(SimpleNamespace(max_seconds=seconds))
