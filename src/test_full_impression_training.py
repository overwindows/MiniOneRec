import json
import time

import numpy as np
import pytest
import torch

from full_impression_contract import digest, prepare, verify, write_json
from full_impression_ranker import FrozenArticleRanker
from full_impression_training import (
    ControlPredictor, article_features, batch_inputs, batches, clipped_article_state, epoch, evaluate,
    load_contract, parity_differences, popularity,
)
from test_full_impression_contract import source


@pytest.fixture
def contract(source, tmp_path):
    root, sha = source
    directory = tmp_path / "contract"
    prepare(root, sha, directory, time.monotonic() + 60)
    write_json(directory / "verification.json", verify(root, directory, time.monotonic() + 60))
    _, arrays = load_contract(directory, digest(directory / "manifest.json"),
                              digest(directory / "verification.json"), time.monotonic() + 60)
    return directory, arrays


def test_ragged_batches_preserve_all_candidates_and_final_partial_batch(contract):
    _, arrays = contract
    chosen = list(batches(np.arange(3), arrays, candidate_limit=5, impression_limit=2))
    assert [batch.tolist() for batch in chosen] == [[0], [1, 2]]
    ids, targets = [], []
    for batch in chosen:
        (history, candidates, labels, groups), lengths = batch_inputs(batch, arrays, "cpu")
        assert history.shape == (len(batch), 30)
        assert len(candidates) == sum(lengths) and int(groups.max()) == len(batch) - 1
        ids.extend(candidates.tolist())
        targets.extend(labels.tolist())
    assert ids == [2, 3, 1, 4, 4, 3, 1, 2]
    assert targets == [1, 0, 1, 0, 0, 1, 1, 0]
    with pytest.raises(ValueError, match="truncate"):
        list(batches(np.arange(3), arrays, candidate_limit=3))


def test_every_click_and_exposure_contributes_to_a_complete_epoch(contract):
    _, arrays = contract
    torch.manual_seed(42)
    features = torch.randn(5, 8)
    features[0] = 0
    model = FrozenArticleRanker(features, "candidate_attention", dimension=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = epoch(model, np.arange(3), arrays, optimizer, 42, time.monotonic() + 60)
    assert result["impressions"] == 3 and result["exposures"] == 8 and result["positives"] == 4
    assert result["optimizer_steps"] == 1 and np.isfinite(result["mean_loss"])
    assert model.features.grad is None


def test_training_only_popularity_cannot_see_heldout_targets(contract):
    _, original = contract
    arrays = dict(original)
    before = popularity(np.array([0, 1]), arrays, 4, time.monotonic() + 60)
    arrays["candidate_labels.npy"] = np.array(arrays["candidate_labels.npy"], copy=True)
    arrays["candidate_labels.npy"][6:] = [0, 1]
    after = popularity(np.array([0, 1]), arrays, 4, time.monotonic() + 60)
    np.testing.assert_array_equal(before, after)
    assert before[1] > before[3] > before[4]


def test_all_heldout_predictions_and_official_multi_positive_metrics_are_persisted(contract, tmp_path):
    _, arrays = contract
    features = torch.zeros(5, 8)
    control = ControlPredictor(features, "popularity", ctr=torch.tensor([0., 1., 1., .5, -1.]))
    path = tmp_path / "scores.f64"
    result = evaluate(np.arange(3), arrays, control, path, time.monotonic() + 60)
    assert result["impressions"] == 3 and result["exposures"] == 8
    assert path.stat().st_size == 8 * 8
    assert result["metrics"]["auc"] == pytest.approx(5 / 6)
    assert result["metrics"]["mrr"] == pytest.approx(.75)
    assert np.load(path.with_suffix(".metrics.npy")).shape == (3, 4)


def test_rehashed_complete_contract_is_required_before_loading_arrays(contract):
    directory, _ = contract
    manifest_sha, proof_sha = digest(directory / "manifest.json"), digest(directory / "verification.json")
    labels = np.load(directory / "candidate_labels.npy", mmap_mode="r+")
    labels[2] = 0
    labels.flush()
    del labels
    with pytest.raises(ValueError, match="input changed"):
        load_contract(directory, manifest_sha, proof_sha, time.monotonic() + 60)


class LengthScorer:
    def _encode(self, state):
        return range(200 + len(state))


def test_only_article_text_is_clipped_and_the_fixed_state_is_preserved():
    article = {"title": "headline", "abstract": "long abstract " * 1000, "category": "world"}
    state, clipped = clipped_article_state(LengthScorer(), article)
    assert clipped and len(LengthScorer()._encode(state)) <= 512
    assert state.startswith("A user read these news articles:\n(No reading history)\n\nCandidate article:\n[world] headline")
    article["abstract"] = ""
    short, clipped = clipped_article_state(LengthScorer(), article)
    assert not clipped and short.endswith("[world] headline")


def test_parity_diagnostics_report_float_errors_without_relaxing_bit_exactness():
    from uniform_jev_features import check_parity
    expected = np.array([0., 1.], dtype=np.float32)
    actual = expected.copy()
    actual[0] = -0.
    report = parity_differences(actual, expected)
    assert report["different_bit_patterns"] == 1
    assert report["maximum_float_absolute_error"] == 0.
    with pytest.raises(AssertionError):
        check_parity(actual, actual, expected, expected)
    actual[1] = np.nextafter(expected[1], np.float32(2.))
    assert parity_differences(actual, expected)["different_bit_patterns"] == 2
    assert parity_differences(actual, expected)["maximum_float_absolute_error"] == 2 ** -23
    assert not parity_differences(actual.astype(np.float64), expected)["shape_and_dtype_match"]
    actual[1] = np.nan
    report = parity_differences(actual, expected)
    assert not report["all_values_finite"] and report["maximum_float_absolute_error"] is None
    json.dumps(report, allow_nan=False)


def test_failed_original16_parity_persists_evidence_and_never_extracts_new_articles(tmp_path, monkeypatch):
    import evaluate_mind_jev
    import full_jev_readout
    import mind_utils
    import probe_jev_readout
    import uniform_jev_features

    model_root, output = tmp_path / "model", tmp_path / "output"
    model_root.mkdir()
    output.mkdir()
    (model_root / "model.safetensors").write_bytes(b"fixture weights")
    expected = np.zeros((16, 2560), dtype=np.float32)
    actual = expected.copy()
    actual[0, 0] = 1.
    margins = np.zeros(16, dtype=np.float32)

    class Scorer:
        def __init__(self, *args, **kwargs):
            self.model = torch.nn.Linear(1, 1)

        def _encode(self, state):
            return [1, 2, 3]

    calls = []

    def extract(scorer, states, *args, **kwargs):
        assert not any(parameter.requires_grad for parameter in scorer.model.parameters())
        calls.append(len(states))
        return actual, margins

    monkeypatch.setattr(full_jev_readout, "DATA_SHA256", {})
    monkeypatch.setattr(mind_utils, "load_news", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluate_mind_jev, "JevScorer", Scorer)
    monkeypatch.setattr(uniform_jev_features, "load_parity",
                        lambda *args: ([{}] * 16, expected, margins))
    monkeypatch.setattr(uniform_jev_features, "training_states",
                        lambda *args: [{"state": "fixture"}] * 16)
    monkeypatch.setattr(probe_jev_readout, "extract", extract)
    with pytest.raises(AssertionError):
        article_features(tmp_path, str(model_root), tmp_path, tmp_path, "fixture-parity",
                         output, time.monotonic() + 60)
    assert calls == [16]
    report = json.loads((output / "parity.comparison.json").read_bytes())
    assert report["features"]["different_bit_patterns"] == 1
    assert report["features"]["maximum_float_absolute_error"] == 1.
    assert report["margins"]["different_bit_patterns"] == 0
    assert digest(output / "parity.observed.npz") == report["observed_sha256"]
    assert digest(output / "parity.runtime.json") == report["runtime_sha256"]
    assert (output / "model.first-observed-sha256.json").exists()
    assert len(json.loads((output / "parity.input-tokens.json").read_bytes())) == 16
    with np.load(output / "parity.observed.npz", allow_pickle=False) as observed:
        np.testing.assert_array_equal(observed["features"], actual)
    assert not (output / "article.features.npy").exists()
