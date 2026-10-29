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
        if failure == "import" and command[1:3] == ["-u", "-c"]:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(entry.subprocess, "run", run)
    monkeypatch.setattr(entry.subprocess, "check_output", lambda *args, **kwargs: revision)
    return calls


def test_dependency_and_real_model_import_checks_precede_training(monkeypatch, tmp_path):
    calls = invoke(monkeypatch, tmp_path)
    entry.main()
    dependency = next(i for i, call in enumerate(calls) if call[-3:] == ["-m", "pip", "check"])
    probe = next(i for i, call in enumerate(calls) if call[1:3] == ["-u", "-c"])
    training = next(i for i, call in enumerate(calls) if any("full_impression_training.py" in arg for arg in call))
    assert dependency < probe < training
    assert "from transformers import Qwen3_5ForConditionalGeneration" in calls[probe][-1]
    assert "'fla-core', 'triton'" in calls[probe][-1]
    compile(calls[probe][-1], "<runtime-preflight>", "exec")


def test_qualified_managed_python_is_selected(monkeypatch, tmp_path):
    calls = invoke(monkeypatch, tmp_path)
    entry.main()
    creation = next(call for call in calls if "--managed-python" in call)
    assert creation[creation.index("--python") + 1] == "3.11.9"
    assert any("uv==0.6.14" in call for call in calls)
    probe = next(call for call in calls if call[1:3] == ["-u", "-c"])
    assert "platform.python_version() != '3.11.9'" in probe[-1]
    install = next(call for call in calls if "-r" in call)
    assert install[install.index("-c") + 1].endswith("requirements.lock.txt")


def test_reconstructed_dependencies_cannot_drift_to_other_torch_or_transformers_builds():
    root = Path(__file__).parent / "components" / "mind_jev_large_training"
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    locked = (root / "requirements.lock.txt").read_text(encoding="utf-8")
    assert "--index-url https://pypi.org/simple" in requirements
    assert "--extra-index-url" not in requirements
    for package in ("torch", "torchvision"):
        expected = "2.6.0" if package == "torch" else "0.21.0"
        assert f"{package}==={expected}\n" in requirements
        assert f"{package}=={expected}\n" in locked
        assert f"{package}=={expected}+" not in locked
    for package, expected in (("transformers", "5.21.0"), ("tokenizers", "0.23.3")):
        assert f"{package}=={expected}\n" in requirements
        assert f"{package}=={expected}\n" in locked


@pytest.mark.parametrize("failure", ["dependencies", "import"])
def test_failed_preflight_cannot_start_training(monkeypatch, tmp_path, failure):
    calls = invoke(monkeypatch, tmp_path, failure)
    with pytest.raises(subprocess.CalledProcessError):
        entry.main()
    assert not any("full_impression_training.py" in arg for call in calls for arg in call)
