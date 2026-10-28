import hashlib
import json
from pathlib import Path

import pytest

import evaluate_uniform_jev as evaluation


def report(tmp_path, **changes):
    value = {
        "stage": "full_training_fitted", "opens_dev_gate": True, "training_only": True,
        "training_rows": 472626, "training_impressions": 156965,
        "final_refit_holdout_impressions": 0, "feature_source_revision": evaluation.FEATURE_REVISION,
        "evaluator_sha256": evaluation.digest(Path(evaluation.__file__)),
        **changes,
    }
    raw = json.dumps(value).encode()
    (tmp_path / "result.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def run(tmp_path, sha):
    return evaluation.evaluate(tmp_path, sha, tmp_path / "missing-dev-cache",
                               tmp_path / "missing-dev-data", tmp_path / "missing-retained",
                               tmp_path / "missing-head", tmp_path / "evaluation")


def test_closed_or_partial_training_gate_never_reads_dev(tmp_path):
    sha = report(tmp_path, stage="train_only_gate_closed", opens_dev_gate=False)
    with pytest.raises(ValueError, match="gate"):
        run(tmp_path, sha)
    sha = report(tmp_path, training_rows=469264)
    with pytest.raises(ValueError, match="gate"):
        run(tmp_path, sha)
    assert not (tmp_path / "evaluation").exists()


def test_frozen_training_checksum_and_evaluator_cannot_change(tmp_path):
    report(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        run(tmp_path, "0" * 64)
    sha = report(tmp_path, evaluator_sha256="0" * 64)
    with pytest.raises(ValueError, match="evaluator"):
        run(tmp_path, sha)
    assert not (tmp_path / "evaluation").exists()


def test_selection_gate_is_recomputed_before_any_dev_access(tmp_path):
    metrics = {"auc": .7, "mrr": .4, "ndcg5": .5, "ndcg10": .6}
    selection = {
        "selected": "uniform_binary", "opens_dev_gate": True,
        "candidates": [{"name": "uniform_binary", "metrics": metrics}],
        "controls": [{"name": "retained", "metrics": metrics}],
    }
    raw = json.dumps(selection).encode()
    (tmp_path / "train-selection.json").write_bytes(raw)
    sha = report(tmp_path, selection_sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="selection gate"):
        run(tmp_path, sha)
    assert not (tmp_path / "evaluation").exists()


def test_previous_evaluation_is_not_overwritten(tmp_path):
    (tmp_path / "evaluation").mkdir()
    sentinel = tmp_path / "evaluation" / "result.json"
    sentinel.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(tmp_path, "0" * 64)
    assert sentinel.read_text() == "original"
