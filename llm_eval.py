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
    logging.info("Loading model and starting evaluation... (this may take a while)")

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
