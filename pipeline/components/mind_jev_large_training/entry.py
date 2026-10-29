"""Use a private pinned environment, without multi-node setup or telemetry login."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mount", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--parity-cache", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--parity-sha256", required=True)
    parser.add_argument("--max-seconds", type=int, default=21600)
    parser.add_argument("--task", choices=("training", "official-dev"), default="training")
    parser.add_argument("--training-proof", type=Path)
    parser.add_argument("--training-proof-sha256")
    parser.add_argument("--training-result-sha256")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.source_revision):
        parser.error("An immutable source commit is required")
    if not 0 < args.max_seconds <= 21600:
        parser.error("The worker cannot extend the six-hour maximum")
    if args.task == "official-dev" and (
            args.training_proof is None or any(not re.fullmatch("[0-9a-f]{64}", value or "")
                                               for value in (args.training_proof_sha256,
                                                             args.training_result_sha256))):
        parser.error("Frozen independent training proof and result hashes are required for official dev")
    root = Path.cwd()
    source = root / "MiniOneRec"
    subprocess.run(["git", "clone", "--quiet", "https://github.com/overwindows/MiniOneRec.git", str(source)], check=True)
    subprocess.run(["git", "fetch", str(root / "source.bundle"), "HEAD"], cwd=source, check=True)
    subprocess.run(["git", "checkout", "--detach", args.source_revision], cwd=source, check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != args.source_revision:
        raise ValueError("Worker source commit differs")
    print(actual, flush=True)
    env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1", WANDB_DISABLED="true", WANDB_MODE="disabled")
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE="1", UV_PYTHON_INSTALL_DIR=str(root / "jev-managed-python"))
    bootstrap = root / "jev-bootstrap-env"
    subprocess.run([sys.executable, "-m", "venv", str(bootstrap)], check=True, env=env)
    bootstrap_python = bootstrap / "bin" / "python"
    subprocess.run([str(bootstrap_python), "-m", "pip", "install", "--quiet", "uv==0.6.14"],
                   check=True, env=env)
    environment = root / "jev-large-env"
    subprocess.run([str(bootstrap_python), "-m", "uv", "venv", "--managed-python", "--python",
                    "3.11.9", "--seed", str(environment)], check=True, env=env)
    python = environment / "bin" / "python"
    dependencies = source / "pipeline" / "components" / "mind_jev_large_training"
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", "-r",
                    str(dependencies / "requirements.txt"), "-c", str(dependencies / "requirements.lock.txt")],
                   check=True, env=env)
    env.update(CUDA_VISIBLE_DEVICES="0", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if args.task == "official-dev":
        env["NPY_DISABLE_CPU_FEATURES"] = "AVX512F,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL"
    subprocess.run([str(python), "-m", "pip", "check"], check=True, env=env)
    subprocess.run([
        str(python), "-u", "-c",
        "import json, sys, platform; from importlib.metadata import version\n"
        "if platform.python_version() != '3.11.9':\n"
        "    raise RuntimeError('Qualified managed Python is required')\n"
        "print(json.dumps({'stage': 'runtime_import_preflight', 'python': sys.version, "
        "'versions': {name: version(name) for name in "
        "('torch', 'torchvision', 'transformers', 'tokenizers', 'flash-linear-attention', "
        "'fla-core', 'triton')}}), flush=True); "
        "from transformers import Qwen3_5ForConditionalGeneration; "
        "print('Qwen3_5ForConditionalGeneration import verified', flush=True)",
    ], check=True, env=env)
    script = "full_impression_evaluation.py" if args.task == "official-dev" else "full_impression_training.py"
    command = [
        str(python), "-u", str(source / "src" / script),
        "--contract", str(args.contract), "--parity-cache", str(args.parity_cache),
        "--parity-sha256", args.parity_sha256, "--source-revision", actual,
        "--model-path", str(args.mount / "shares/users/wuc/models/APUS-OpenJev-v1-4B"),
        "--small-data", str(args.mount / "shares/users/wuc/data/MIND_small"),
        "--output", str(args.mount / args.output_root),
        "--max-seconds", str(args.max_seconds),
    ]
    if args.task == "official-dev":
        command.extend([
            "--training-artifacts",
            str(args.mount / "shares/users/wuc/output_dir/jev_readout_recovery_20261028/large_semantic_04"),
            "--training-proof", str(args.training_proof),
            "--training-proof-sha256", args.training_proof_sha256,
            "--training-result-sha256", args.training_result_sha256,
            "--dev-data", str(args.mount / "shares/users/wuc/data/MIND_large/dev"),
        ])
    subprocess.run(command, check=True, cwd=source, env=env)


if __name__ == "__main__":
    main()
