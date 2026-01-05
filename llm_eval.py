import argparse
import json
import os
import sys
import time
import logging
import numpy as np

# Set environment variables BEFORE importing torch to prevent distributed initialization
# These must be set before any torch imports
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    # Use a random port to avoid conflicts
    import random
    os.environ["MASTER_PORT"] = str(random.randint(20000, 65000))

import torch

"""
DECOUPLED DATASET DOWNLOAD AND EVALUATION
==========================================

This module supports a decoupled workflow to avoid FUSE conflicts:

PROBLEM:
--------
When datasets are downloaded during evaluation, multiple GPU processes may
attempt concurrent file access through FUSE, causing:
  - Filesystem errors and conflicts
  - Unpredictable performance degradation
  - Slow network I/O interfering with GPU computation

SOLUTION:
---------
Use pre-downloaded datasets in offline mode:

  Stage 1 (Pre-download, run once):
    python download_eval_datasets.py \\
        --tasks "mmlu,hellaswag,arc_challenge" \\
        --cache_dir ./eval_cache \\
        --verbose

  Stage 2 (Evaluation with offline mode):
    python llm_eval.py \\
        --model_path Qwen/Qwen3-1.7B \\
        --tasks "mmlu,hellaswag,arc_challenge" \\
        --cache_dir ./eval_cache \\
        --offline_mode \\
        --batch_size 32

PARAMETERS:
-----------
  --cache_dir: Path to pre-downloaded datasets (sets HF_DATASETS_CACHE)
  --offline_mode: Enable offline mode (requires pre-downloaded datasets)
                  Sets: HF_DATASETS_OFFLINE=1, TRANSFORMERS_OFFLINE=1

ENVIRONMENT VARIABLES:
----------------------
  HF_DATASETS_CACHE: Directory containing cached datasets
  HF_DATASETS_OFFLINE: Set to "1" to prevent new downloads
  TRANSFORMERS_OFFLINE: Set to "1" to prevent model downloads

WORKFLOW:
---------
  1. Download phase (runs once):
     - Uses lm_eval.tasks to identify required datasets
     - Downloads via datasets.load_dataset() to HF_DATASETS_CACHE
     - Takes 5-30 minutes depending on datasets

  2. Evaluation phase (can run multiple times on same cache):
     - Sets offline environment variables
     - Uses cached datasets from HF_DATASETS_CACHE
     - No network I/O, pure GPU computation
     - Avoids FUSE conflicts entirely

This design separates concerns: data preparation vs. model evaluation.
"""


def _build_model_args(model_path, tokenizer_path, dtype, trust_remote_code, max_length, use_accelerate=False, num_gpus=None):
    args = [f"pretrained={model_path}"]
    if tokenizer_path:
        args.append(f"tokenizer={tokenizer_path}")
    if dtype:
        args.append(f"dtype={dtype}")
    if trust_remote_code:
        args.append("trust_remote_code=true")
    if max_length is not None:
        args.append(f"max_length={max_length}")

    # Multi-GPU support using device_map
    if use_accelerate:
        # Use device_map="auto" for multi-GPU distribution
        args.append("device_map=auto")
        # Set parallelism flag
        args.append("parallelize=true")

    return ",".join(args)


def _make_json_serializable(obj):
    """Convert non-serializable objects to JSON-compatible types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_json_serializable(v) for v in obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'dtype'):  # Handle numpy dtypes and similar
        return str(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def main():
    parser = argparse.ArgumentParser(description="Run LLM capability evals via lm-eval-harness.")
    parser.add_argument("--model_path", required=True, help="HF model path or checkpoint directory.")
    parser.add_argument("--tokenizer_path", default="", help="Optional tokenizer path.")
    parser.add_argument(
        "--tasks",
        default="mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval",
        help="Comma-separated lm-eval task names.",
    )
    parser.add_argument("--output_dir", default="llm_eval", help="Where to write results JSON.")
    parser.add_argument("--batch_size", default="auto", help="Batch size (or 'auto').")
    parser.add_argument("--device", default="cuda", help="Device for lm-eval (e.g., cuda, cpu).")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype (e.g., bfloat16, float16).")
    parser.add_argument("--use_accelerate", action="store_true", help="Use Accelerate for multi-GPU inference.")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all available).")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Few-shot examples.")
    parser.add_argument("--limit", type=float, default=None, help="Optional eval limit (0.0-1.0 for fraction, or N for first N examples).")
    parser.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code.")
    parser.add_argument("--max_length", type=int, default=None, help="Optional max length.")
    parser.add_argument("--cache_dir", default=None, help="HF datasets cache directory (for pre-downloaded datasets).")
    parser.add_argument(
        "--offline_mode",
        action="store_true",
        help="Force offline mode (default unless --online_mode is set).",
    )
    parser.add_argument(
        "--online_mode",
        action="store_true",
        help="Allow dataset/model downloads during evaluation (not recommended with FUSE).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    # Disable torch distributed for single GPU to avoid port conflicts
    if not args.use_accelerate:
        # Use GPU 0 by default unless CUDA_VISIBLE_DEVICES is already set
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Prevent accelerate from trying to use distributed mode
        os.environ["ACCELERATE_USE_FSDP"] = "false"
        os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
        # Disable torch distributed initialization
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"

    if args.offline_mode and args.online_mode:
        raise SystemExit("Choose only one of --offline_mode or --online_mode.")

    # Configure dataset cache directory if provided
    if args.cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir
        logging.info(f"Using datasets cache directory: {args.cache_dir}")

    # Default to offline mode unless explicitly overridden
    offline = args.offline_mode or not args.online_mode
    if offline:
        if args.cache_dir and not os.path.isdir(args.cache_dir):
            raise SystemExit(
                f"Cache directory not found: {args.cache_dir}. "
                "Run download_eval_datasets.py first to pre-download datasets."
            )
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logging.info("Running in offline mode (requires pre-downloaded datasets)")
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    try:
        from lm_eval import evaluator
    except Exception as exc:
        raise SystemExit(
            "lm-eval-harness is not installed. Install with: pip install -r requirements-eval.txt"
        ) from exc

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise SystemExit("No tasks provided. Use --tasks with comma-separated task names.")

    # Determine number of GPUs to use
    available_gpus = torch.cuda.device_count()
    if args.use_accelerate:
        num_gpus = args.num_gpus if args.num_gpus else available_gpus
        logging.info(f"Multi-GPU mode enabled: Using {num_gpus}/{available_gpus} GPUs with Accelerate")
    else:
        num_gpus = None
        logging.info(f"Single-GPU mode: Using GPU 0 (total available: {available_gpus})")

    logging.info(f"Starting evaluation for model: {args.model_path}")
    logging.info(f"Tasks to evaluate: {', '.join(tasks)}")
    if args.limit:
        logging.info(f"Evaluation limit: {args.limit}")

    model_args = _build_model_args(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path or None,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_length=args.max_length,
        use_accelerate=args.use_accelerate,
        num_gpus=num_gpus,
    )

    logging.info(f"Model arguments: {model_args}")
    logging.info("=" * 60)
    logging.info("Starting evaluation process...")
    logging.info("=" * 60)
    logging.info("Step 1/3: Resolving tasks (this may take 30-60 minutes for MMLU)...")
    logging.info("Step 2/3: Loading model into memory...")
    logging.info("Step 3/3: Running evaluation on all tasks...")
    logging.info("")
    logging.info("NOTE: Task resolution is SLOW for MMLU (expands 57 subtasks)")
    logging.info("      You will see model loading progress after task resolution completes")
    logging.info("")

    start_time = time.time()
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=args.batch_size,
        device=args.device,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )
    elapsed_time = time.time() - start_time

    logging.info(f"Evaluation completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(os.path.abspath(args.model_path))
    out_path = os.path.join(args.output_dir, f"{model_name}_{stamp}.json")

    logging.info(f"Writing results to: {out_path}")

    # Convert results to JSON-serializable format
    serializable_results = _make_json_serializable(results)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✓ Evaluation completed successfully!")
    print(f"{'='*60}")
    print(f"Results saved to: {out_path}")
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
