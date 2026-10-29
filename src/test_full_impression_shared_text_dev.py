import json

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

import full_impression_shared_text_dev as evaluation
from jev_content_ranker import digest


def test_closed_gate_refuses_before_any_dev_access(tmp_path, monkeypatch):
    proof = tmp_path / "proof.json"
    proof.write_text("{}")
    touched = []
    monkeypatch.setattr(evaluation, "verify_source", lambda *args: touched.append(True))
    with pytest.raises(ValueError, match="BEFORE any dev access"):
        evaluation.load_shared_gate(tmp_path, proof, "bad-proof", "bad-result")
    assert not touched


def test_worker_gate_without_complete_independent_full_fit_cannot_pass(tmp_path):
    proof = tmp_path / "proof.json"
    (tmp_path / "result.json").write_text(json.dumps({
        "source_revision": evaluation.TRAINING_SOURCE, "opens_dev_gate": True}))
    proof.write_text(json.dumps({
        "result_sha256": digest(tmp_path / "result.json"),
        "source_revision": evaluation.TRAINING_SOURCE, "opens_dev_gate": True,
        "full_training_epoch_source_and_coverage_verified": False}))
    with pytest.raises(ValueError, match="BEFORE dev"):
        evaluation.load_shared_gate(tmp_path, proof, digest(proof), digest(tmp_path / "result.json"))


def test_text_cap_never_changes_candidate_population_and_has_exact_stable_ties():
    text = " ".join(f"token{index}" for index in range(60))
    vectorizer = TfidfVectorizer(dtype=np.float32).fit([text, "other context"])
    matrix = evaluation.capped_titles(vectorizer, [text, "", "other context"])
    assert matrix.shape == (3, len(vectorizer.vocabulary_))
    assert np.diff(matrix.indptr).tolist() == [32, 0, 2]
    expected = vectorizer.transform([text]).tocsr()
    chosen = sorted(zip(expected.indices, expected.data), key=lambda item: (-item[1], item[0]))[:32]
    assert matrix.indices[:32].tolist() == sorted(int(column) for column, _ in chosen)


def test_joint_worker_success_cannot_bypass_complete_independent_two_epoch_fit(tmp_path):
    proof = tmp_path / "proof.json"
    (tmp_path / "result.json").write_text(json.dumps({
        "source_revision": evaluation.MULTIVIEW_SOURCE, "opens_dev_gate": True}))
    proof.write_text(json.dumps({
        "result_sha256": digest(tmp_path / "result.json"),
        "source_revision": evaluation.MULTIVIEW_SOURCE, "opens_dev_gate": True,
        "full_training_epoch_source_and_coverage_verified": False}))
    with pytest.raises(ValueError, match="BEFORE dev"):
        evaluation.load_multiview_gate(tmp_path, proof, digest(proof), digest(tmp_path / "result.json"))
    with pytest.raises(ValueError, match="BEFORE dev"):
        evaluation.load_shared_gate(tmp_path, proof, digest(proof), digest(tmp_path / "result.json"))
