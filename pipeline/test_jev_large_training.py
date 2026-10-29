import hashlib
import json
import sys
from types import SimpleNamespace

import run_jev_large_training as subject


def test_registered_graph_has_exactly_one_bounded_training_only_worker(tmp_path, monkeypatch):
    contract, parity = tmp_path / "contract", tmp_path / "parity"
    contract.mkdir()
    parity.mkdir()
    manifest = b'{"fixture":true}'
    proof = json.dumps({"impressions_replayed": 2232748, "candidate_exposures_replayed": 83507374}).encode()
    parity_bytes = b'{"fixture_parity":true}'
    (contract / "manifest.json").write_bytes(manifest)
    (contract / "verification.json").write_bytes(proof)
    (parity / "manifest.json").write_bytes(parity_bytes)
    monkeypatch.setattr(subject, "CONTRACT_MANIFEST", hashlib.sha256(manifest).hexdigest())
    monkeypatch.setattr(subject, "CONTRACT_PROOF", hashlib.sha256(proof).hexdigest())
    monkeypatch.setattr(sys, "argv", [
        "run_jev_large_training", "--contract", str(contract), "--parity-cache", str(parity),
        "--parity-sha256", hashlib.sha256(parity_bytes).hexdigest(),
        "--source-revision", "a" * 40, "--output-root", "fixture_not_submitted",
    ])
    jobs = []

    def capture(job):
        jobs.append(job)
        return SimpleNamespace(name="fixture_only_no_cloud_allocation")

    monkeypatch.setattr(subject.ml_client.jobs, "create_or_update", capture)
    subject.main()
    assert len(jobs) == 1 and len(jobs[0].jobs) == 1
    job = jobs[0]
    assert job.name == subject.RUN_NAME
    node = next(iter(job.jobs.values()))
    assert job.tags["jev_experiment"] == "large_semantic_training"
    assert node.resources.instance_count == 1
    assert node.resources.instance_type == "Singularity.ND12am_A100_v4"
    assert node.limits.timeout == 21600
    inputs = node._to_dict()["inputs"]
    assert inputs["source_revision"] == "a" * 40
    assert inputs["contract"]["mode"] == inputs["parity_cache"]["mode"] == "download"
    assert not any("dev" in name or "test" in name for name in inputs)
