import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


SPEC = importlib.util.spec_from_file_location(
    "large_training_entry", Path(__file__).parent / "components" / "mind_jev_large_training" / "entry.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def invoke(monkeypatch, tmp_path, failure=None):
    revision = "a" * 40
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "entry.py", "--mount", str(tmp_path), "--contract", str(tmp_path),
        "--parity-cache", str(tmp_path), "--source-revision", revision,
        "--output-root", "fixture", "--parity-sha256", "b" * 64,
    ])
    calls = []

    def run(command, **kwargs):
        assert kwargs["check"] is True
        calls.append(command)
        if failure == "dependencies" and command[-3:] == ["-m", "pip", "check"]:
            raise subprocess.CalledProcessError(1, command)
        if failure == "import" and "-c" in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(entry.subprocess, "run", run)
    monkeypatch.setattr(entry.subprocess, "check_output", lambda *args, **kwargs: revision)
    return calls


def test_dependency_and_real_model_import_checks_precede_training(monkeypatch, tmp_path):
    calls = invoke(monkeypatch, tmp_path)
    entry.main()
    dependency = next(i for i, call in enumerate(calls) if call[-3:] == ["-m", "pip", "check"])
    probe = next(i for i, call in enumerate(calls) if "-c" in call)
    training = next(i for i, call in enumerate(calls) if any("full_impression_training.py" in arg for arg in call))
    assert dependency < probe < training
    assert "from transformers import Qwen3_5ForConditionalGeneration" in calls[probe][-1]
    assert "'fla-core', 'triton'" in calls[probe][-1]
    compile(calls[probe][-1], "<runtime-preflight>", "exec")


@pytest.mark.parametrize("failure", ["dependencies", "import"])
def test_failed_preflight_cannot_start_training(monkeypatch, tmp_path, failure):
    calls = invoke(monkeypatch, tmp_path, failure)
    with pytest.raises(subprocess.CalledProcessError):
        entry.main()
    assert not any("full_impression_training.py" in arg for call in calls for arg in call)
