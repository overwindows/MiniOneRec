import os
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(os.name == "nt", reason="Bash workflow fixtures run in qualified WSL")
REPO = Path(__file__).resolve().parents[1]
FAKE_EVALUATOR = """
import argparse, os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--behaviors_path', required=True)
parser.add_argument('--output_file', required=True)
parser.add_argument('--output_scores_file')
parser.add_argument('--max_impressions', type=int, default=0)
args, _ = parser.parse_known_args()
gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
if gpu == os.environ.get('FAIL_GPU'):
    raise SystemExit(9)
rows = Path(args.behaviors_path).read_text().splitlines()
if args.max_impressions:
    rows = rows[:args.max_impressions]
predictions, scores = [], []
for row in rows:
    fields = row.split('\\t')
    predictions.append(fields[0] + ' ' + fields[4] + '\\n')
    scores.append(fields[0] + '\\t' + '\\t'.join('1' for _ in fields[4].split()) + '\\n')
Path(args.output_file).write_text(''.join(predictions))
if args.output_scores_file and gpu != os.environ.get('MISSING_SCORES_GPU'):
    Path(args.output_scores_file).write_text(''.join(scores))
"""


@pytest.fixture
def workspace(tmp_path):
    for directory in ("scripts", "src", "data/test", "outputs/scores"):
        (tmp_path / directory).mkdir(parents=True)
    for name in ("eval_mind_pointwise.sh", "eval_mind_pointwise_ensemble.sh"):
        (tmp_path / "scripts" / name).write_text((REPO / "scripts" / name).read_text())
    for name in ("split_mind.py", "merge_mind.py"):
        shutil.copyfile(REPO / "src" / name, tmp_path / "src" / name)
    (tmp_path / "src/evaluate_mind_pointwise.py").write_text(FAKE_EVALUATOR)
    (tmp_path / "data/test/behaviors.tsv").write_text(
        "".join(f"{i}\tU\tt\t\tN1 N2\n" for i in range(1, 5)))
    (tmp_path / "data/test/news.tsv").write_text("fixture news\n")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1",
                   "PYTHON": sys.executable, "MIND_ROOT": str(tmp_path / "data"),
                   "OUTPUT_FILE": str(tmp_path / "outputs/predictions.txt"),
                   "OUTPUT_SCORES_FILE": str(tmp_path / "outputs/scores.txt"),
                   "KEEP_SCORES_DIR": str(tmp_path / "outputs/scores"),
                   "MIND_SIZE": "large", "SPLIT": "test", "MAX_IMPRESSIONS": "0",
                   "CF_ALPHA": "0", "USE_ABSTRACT": "0", "USE_SUBCATEGORY": "0",
                   "USE_CHAT_TEMPLATE": "0"}
    return tmp_path, environment


def run(workspace, script="eval_mind_pointwise.sh", models=("fixture_model",), max_impressions=None):
    root, environment = workspace
    arguments = list(models) + (["test"] if script == "eval_mind_pointwise.sh" else [])
    if max_impressions is not None:
        arguments.append(str(max_impressions))
    return subprocess.run(["bash", "scripts/" + script, *arguments], cwd=root, env=environment,
                          capture_output=True, text=True, timeout=60)


def test_complete_shards_merge_scores_and_predictions_without_gpu(workspace):
    root, _ = workspace
    result = run(workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "outputs/predictions.txt").read_text() == "".join(f"{i} N1 N2\n" for i in range(1, 5))
    assert (root / "outputs/scores.txt").read_text() == "".join(f"{i}\t1\t1\n" for i in range(1, 5))
    assert not list((root / "outputs").glob(".test-shards.*"))


def test_failed_gpu_retains_successful_shards_and_never_publishes_partial_output(workspace):
    root, environment = workspace
    environment["FAIL_GPU"] = "1"
    result = run(workspace)
    assert result.returncode != 0
    assert not (root / "outputs/predictions.txt").exists()
    assert not (root / "outputs/scores.txt").exists()
    pending = list((root / "outputs").glob(".test-shards.*"))
    assert len(pending) == 1 and (pending[0] / "0.txt").is_file()
    assert (pending[0] / "0_scores.txt").is_file()


def test_missing_scores_never_publish_predicted_ranks_as_complete(workspace):
    root, environment = workspace
    environment["MISSING_SCORES_GPU"] = "1"
    result = run(workspace)
    assert result.returncode != 0
    assert not (root / "outputs/predictions.txt").exists()
    assert not (root / "outputs/scores.txt").exists()
    assert list((root / "outputs").glob(".test-shards.*"))


def test_ensemble_failure_retains_per_model_outputs_without_overwriting_final_result(workspace):
    root, environment = workspace
    environment["FAIL_GPU"] = "1"
    final = root / "outputs/predictions.txt"
    final.write_text("prior accepted result\n")
    result = run(workspace, "eval_mind_pointwise_ensemble.sh", ("model_a", "model_b"))
    assert result.returncode != 0
    assert final.read_text() == "prior accepted result\n"
    pending = list((root / "outputs/scores").glob(".pending-wave.*"))
    assert len(pending) == 1 and (pending[0] / "model_1_scores.txt").is_file()
    assert (pending[0] / "model_1_predictions.txt").is_file()


def test_insufficient_gpu_allocation_fails_before_starting_any_model(workspace):
    root, environment = workspace
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    result = run(workspace, "eval_mind_pointwise_ensemble.sh", ("model_a", "model_b"))
    assert result.returncode != 0 and "require at least 2 visible GPUs" in result.stderr
    assert not list((root / "outputs/scores").rglob("model_*.log"))


@pytest.mark.parametrize("script", ("eval_mind_pointwise.sh", "eval_mind_pointwise_ensemble.sh"))
def test_explicitly_disabled_gpus_are_not_reenabled(workspace, script):
    root, environment = workspace
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = run(workspace, script)
    assert result.returncode != 0 and "disables GPU evaluation" in result.stderr
    assert not list((root / "outputs/scores").rglob("model_*.log"))

def test_parallel_screening_validates_the_actual_per_gpu_limit(workspace):
    root, _ = workspace
    (root / "data/test/behaviors.tsv").write_text(
        "".join(f"{i}\tU\tt\t\tN1 N2\n" for i in range(1, 9)))
    result = run(workspace, max_impressions=4)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "outputs/predictions.txt").read_text() == "".join(
        f"{i} N1 N2\n" for i in (1, 2, 5, 6))
    assert (root / "outputs/scores.txt").read_text() == "".join(
        f"{i}\t1\t1\n" for i in (1, 2, 5, 6))


def test_parallel_screening_never_turns_a_small_limit_into_full_inference(workspace):
    root, _ = workspace
    result = run(workspace, max_impressions=1)
    assert result.returncode != 0 and "limit must be at least the GPU count" in result.stderr
    assert not (root / "outputs/predictions.txt").exists()
    assert not list((root / "outputs").glob(".test-shards.*"))


@pytest.mark.parametrize("visible_count", (0, 1))
def test_unset_gpu_mask_discovers_visible_devices_without_loading_a_gpu(workspace, visible_count):
    root, environment = workspace
    environment.pop("CUDA_VISIBLE_DEVICES")
    (root / "torch.py").write_text(
        f"class cuda:\n    @staticmethod\n    def device_count():\n        return {visible_count}\n")
    result = run(workspace, "eval_mind_pointwise_ensemble.sh", ("model_a", "model_b"))
    assert result.returncode != 0
    assert ("No visible GPUs" if visible_count == 0 else "got 1") in result.stderr
    assert not list((root / "outputs/scores").rglob("model_*.log"))


def test_component_respects_worker_gpu_allocation_in_both_waves():
    configuration = json.loads(
        (REPO / "pipeline/components/mind_ensemble_eval/config_mind_ensemble_eval.json").read_text())
    waves = [item["command"] for item in configuration["commands"]
             if "bash scripts/eval_mind_pointwise_ensemble.sh" in item["command"]]
    assert len(waves) == 2 and all("CUDA_VISIBLE_DEVICES=" not in command for command in waves)
