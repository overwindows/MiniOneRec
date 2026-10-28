from datetime import datetime, timedelta, timezone
import json

import pytest

import uniform_jev_readout as training


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_incomplete_feature_campaign_cannot_fit_or_open_dev(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign.json"
    write(campaign, {
        "stage": "serial_frozen_feature_extraction",
        "campaign_deadline": (datetime.now(timezone.utc) + timedelta(hours=34)).isoformat(),
    })
    features = tmp_path / "features"
    features.mkdir()
    write(features / "result.json", {})
    calls = []
    monkeypatch.setattr(training, "fit_streamed", lambda *args, **kwargs: calls.append(args))
    output = tmp_path / "fit"
    with pytest.raises(ValueError, match="incomplete"):
        training.train(features, campaign, tmp_path, tmp_path, tmp_path / "head", output, "a" * 40)
    state = json.loads((output / "state.json").read_bytes())
    assert state["stage"] == "failed" and not state["opens_dev_gate"]
    assert state["gpu_submissions"] == state["official_dev_evaluations"] == 0
    assert not calls and not (output / "selected-full.head.npz").exists()


def test_expired_aggregate_budget_cannot_be_reset(tmp_path):
    campaign = tmp_path / "campaign.json"
    write(campaign, {
        "campaign_deadline": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    })
    output = tmp_path / "fit"
    with pytest.raises(TimeoutError, match="deadline"):
        training.train(tmp_path, campaign, tmp_path, tmp_path, tmp_path / "head", output, "a" * 40)
    assert not output.exists()


def test_existing_fit_is_never_overwritten(tmp_path):
    campaign = tmp_path / "campaign.json"
    write(campaign, {
        "campaign_deadline": (datetime.now(timezone.utc) + timedelta(hours=34)).isoformat(),
    })
    output = tmp_path / "fit"
    output.mkdir()
    sentinel = output / "state.json"
    sentinel.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="never"):
        training.train(tmp_path, campaign, tmp_path, tmp_path, tmp_path / "head", output, "a" * 40)
    assert sentinel.read_text() == "original"
