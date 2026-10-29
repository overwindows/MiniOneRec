import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

import run_jev_large_dev as subject


@pytest.mark.parametrize("opens", [True, False])
def test_single_frozen_full_dev_worker_requires_independent_training_gate(tmp_path, monkeypatch, opens):
    contract, parity, proof = [tmp_path / name for name in ("contract", "parity", "proof")]
    for directory in (contract, parity, proof):
        directory.mkdir()
    manifest, receipt, parity_bytes = b"manifest", b"receipt", b"parity"
    (contract / "manifest.json").write_bytes(manifest)
    (contract / "verification.json").write_bytes(receipt)
    (parity / "manifest.json").write_bytes(parity_bytes)
    training = {
        "result_sha256": subject.RESULT_SHA, "opens_dev_gate": opens,
        "original16_bit_exact_parity_independently_verified": True,
        "full_training_epoch_source_and_coverage_verified": True,
        "selected_recipe": "candidate_attention", "official_dev_allocations": 0,
        "parent_status": "Completed", "worker_status": "Completed",
    }
    proof_bytes = json.dumps(training).encode()
    (proof / "verification.json").write_bytes(proof_bytes)
    sha = lambda value: hashlib.sha256(value).hexdigest()
    monkeypatch.setattr(subject, "CONTRACT_MANIFEST", sha(manifest))
    monkeypatch.setattr(subject, "CONTRACT_PROOF", sha(receipt))
    monkeypatch.setattr(sys, "argv", [
        "run_jev_large_dev", "--contract", str(contract), "--parity-cache", str(parity),
        "--training-proof", str(proof), "--training-proof-sha256", sha(proof_bytes),
        "--parity-sha256", sha(parity_bytes), "--source-revision", "a" * 40,
    ])
    jobs = []

    def capture(job):
        jobs.append(job)
        return SimpleNamespace(name="fixture_only_no_submission")

    monkeypatch.setattr(subject.ml_client.jobs, "create_or_update", capture)
    if not opens:
        with pytest.raises(SystemExit):
            subject.main()
        assert not jobs
        return
    subject.main()
    assert len(jobs) == 1 and len(jobs[0].jobs) == 1
    assert jobs[0].name == subject.RUN_NAME
    node = next(iter(jobs[0].jobs.values()))
    assert node.resources.instance_count == 1
    assert node.resources.instance_type == "Singularity.ND12am_A100_v4"
    assert node.limits.timeout == 21600
    inputs = node._to_dict()["inputs"]
    assert inputs["max_seconds"] == 21600
    assert inputs["training_result_sha256"] == subject.RESULT_SHA
    assert inputs["training_proof_sha256"] == sha(proof_bytes)
    assert inputs["output_root"] == subject.OUTPUT
    assert inputs["training_proof"]["mode"] == "download"
