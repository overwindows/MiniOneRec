import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from data import MINDPointwiseSFTDataset
from evaluate_mind_jev import build_jev_state
from mind_utils import load_news
from uniform_jev_features import canonical, check_parity, load_parity, missing_rows, training_states


def row(index, label=0):
    return {"impression_id": str(index), "candidate_id": "N" + str(index), "label": label}


def test_missing_shards_partition_exact_order_and_labels():
    rows = [row(i) for i in range(10)]
    old = rows[:2]
    chosen = []
    for stage in range(4):
        pending, shard = missing_rows(rows, old, stage, shard_rows=2)
        assert pending == rows[2:]
        chosen.extend(shard)
    assert chosen == rows[2:]
    with pytest.raises(ValueError, match="unique"):
        missing_rows(rows + rows[:1], old, 0, shard_rows=2)
    with pytest.raises(ValueError, match="bounded"):
        missing_rows(rows, old, 4)
    with pytest.raises(ValueError, match="empty"):
        missing_rows(rows[:2], old, 0)


def test_history_and_abstract_prompt_exactly_match_dataset(tmp_path):
    news = tmp_path / "news.tsv"
    news.write_text(
        "".join(f"N{i}\tc\ts\tTitle{i}\t{'Abstract' if i % 2 else ''}\tu\t[]\t[]\n"
                for i in range(35)), encoding="utf-8",
    )
    behaviors = tmp_path / "behaviors.tsv"
    behaviors.write_text(
        "1\tU1\t11/14/2019 1:00:00 PM\t"
        + " ".join(["N0", "missing"] + [f"N{i}" for i in range(1, 34)])
        + "\tN34-1 N0-0\n", encoding="utf-8",
    )
    dataset = MINDPointwiseSFTDataset(
        str(behaviors), str(news), None, max_history=30, use_abstract=True,
        negative_sampling="uniform",
    )
    metadata = [{name: sample[name] for name in ("impression_id", "candidate_id", "label")}
                for sample in dataset.samples]
    states = training_states(metadata, behaviors, load_news(str(news), use_abstract=True))
    for state, sample in zip(states, dataset.samples, strict=True):
        assert state["state"] == build_jev_state(sample["history"], sample["candidate"])
    with pytest.raises(ValueError, match="histories"):
        training_states([row(99)], behaviors, {})


def test_parity_requires_exact_manifest_and_contents(tmp_path):
    rows = [row(i) for i in range(16)]
    (tmp_path / "rows.json").write_bytes(canonical(rows))
    x = np.arange(16 * 2560, dtype=np.float32).reshape(16, 2560)
    base = np.arange(16, dtype=np.float32)
    np.savez(tmp_path / "features.npz", features=x, margins=base)
    manifest = {"sha256": {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("rows.json", "features.npz")
    }}
    manifest_bytes = canonical(manifest)
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    sha = hashlib.sha256(manifest_bytes).hexdigest()
    actual_rows, actual_x, actual_base = load_parity(tmp_path, sha)
    assert actual_rows == rows
    np.testing.assert_array_equal(actual_x, x)
    np.testing.assert_array_equal(actual_base, base)
    with pytest.raises(ValueError, match="manifest"):
        load_parity(tmp_path, "0" * 64)
    (tmp_path / "rows.json").write_bytes(b"[]\n")
    with pytest.raises(ValueError, match="artifact"):
        load_parity(tmp_path, sha)


def test_component_and_executor_keep_download_input_absolute():
    root = Path(__file__).resolve().parents[1]
    component = (root / "pipeline" / "components" / "mind_jev_eval" / "uniform_features.yaml").read_text()
    config = json.loads((root / "pipeline" / "components" / "mind_jev_eval"
                         / "config_jev_uniform_features.json").read_text())
    executor = (root / "pipeline" / "components" / "mind_jev_eval" / "pipeline_executor.py").read_text()
    assert "mode: download" in component
    assert "--uniform-parity-cache ${{inputs.parity_cache}}" in component
    assert "'uniform_parity_cache': args.uniform_parity_cache" in executor
    command = config["commands"][-1]["command"]
    assert "--parity-cache {uniform_parity_cache}" in command
    assert "--parity-sha256 {uniform_parity_sha256}" in command
    assert config["error_handling"]["retry_failed_commands"] is False


def test_parity_checks_bits_including_signed_zero():
    x = np.zeros((16, 2560), dtype=np.float32)
    base = np.zeros(16, dtype=np.float32)
    check_parity(x.copy(), base.copy(), x, base)
    changed = base.copy()
    changed[0] = -0.
    with pytest.raises(AssertionError):
        check_parity(x, changed, x, base)
    with pytest.raises(ValueError, match="dtype"):
        check_parity(x.astype(np.float64), base, x, base)
