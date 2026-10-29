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
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.source_revision):
        parser.error("An immutable source commit is required")
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
    environment = root / "jev-large-env"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True, env=env)
    python = environment / "bin" / "python"
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", "-r",
                    str(source / "pipeline" / "components" / "mind_jev_large_training" / "requirements.txt")],
                   check=True, env=env)
    env.update(CUDA_VISIBLE_DEVICES="0", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run([str(python), "-m", "pip", "check"], check=True, env=env)
    subprocess.run([
        str(python), "-u", "-c",
        "import json, sys; from importlib.metadata import version; "
        "print(json.dumps({'stage': 'runtime_import_preflight', 'python': sys.version, "
        "'versions': {name: version(name) for name in "
        "('torch', 'torchvision', 'transformers', 'tokenizers', 'flash-linear-attention', "
        "'fla-core', 'triton')}}), flush=True); "
        "from transformers import Qwen3_5ForConditionalGeneration; "
        "print('Qwen3_5ForConditionalGeneration import verified', flush=True)",
    ], check=True, env=env)
    subprocess.run([
        str(python), "-u", str(source / "src" / "full_impression_training.py"),
        "--contract", str(args.contract), "--parity-cache", str(args.parity_cache),
        "--parity-sha256", args.parity_sha256, "--source-revision", actual,
        "--model-path", str(args.mount / "shares/users/wuc/models/APUS-OpenJev-v1-4B"),
        "--small-data", str(args.mount / "shares/users/wuc/data/MIND_small"),
        "--output", str(args.mount / args.output_root),
    ], check=True, cwd=source, env=env)


if __name__ == "__main__":
    main()
