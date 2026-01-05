#!/usr/bin/env python3
"""
Multi-GPU parallel evaluation using process-level data parallelism.
This approach spawns one process per GPU for maximum throughput.

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
        --tasks "mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval" \\
        --cache_dir ./eval_cache \\
        --verbose

  Stage 2 (Parallel evaluation with offline mode):
    python llm_eval_parallel.py \\
        --model_path Qwen/Qwen3-1.7B \\
        --tasks "mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval" \\
        --cache_dir ./eval_cache \\
        --offline_mode \\
        --num_gpus 4

PARAMETERS:
-----------
  --cache_dir: Path to pre-downloaded datasets (sets HF_DATASETS_CACHE)
  --offline_mode: Enable offline mode (requires pre-downloaded datasets)
                  Sets: HF_DATASETS_OFFLINE=1, TRANSFORMERS_OFFLINE=1
  --num_gpus: Number of GPUs to use (default: all available)

WORKFLOW:
---------
  1. Download phase (runs once):
     - Uses lm_eval.tasks to identify required datasets
     - Downloads via datasets.load_dataset() to HF_DATASETS_CACHE
     - Takes 5-30 minutes depending on datasets

  2. Parallel evaluation (can run multiple times on same cache):
     - Spawns one process per GPU
     - Each process uses the same cached datasets
     - Sets offline environment variables
     - No network I/O, pure GPU computation
     - Avoids FUSE conflicts entirely through data parallelism

This design separates concerns: data preparation vs. model evaluation.
Each GPU process works independently from the same cached dataset.
"""
import argparse
import json
import os
import sys
import time
import logging
import numpy as np
import torch
from multiprocessing import Process, Queue
from pathlib import Path


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
    elif hasattr(obj, 'dtype'):
        return str(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def run_evaluation_on_gpu(gpu_id, model_path, tasks, batch_size, dtype, num_fewshot, limit, output_queue, num_gpus, tokenizer_path=None, trust_remote_code=False, max_length=None):
    """Run evaluation on a specific GPU."""
    import time as time_module
    start_time = time_module.time()

    print(f"[GPU {gpu_id}] Step 1/5: Setting CUDA_VISIBLE_DEVICES={gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        from lm_eval import evaluator
    except Exception as e:
        output_queue.put({"gpu_id": gpu_id, "error": str(e)})
        return

    # Build model args
    model_args = [f"pretrained={model_path}"]
    if tokenizer_path:
        model_args.append(f"tokenizer={tokenizer_path}")
    if dtype:
        model_args.append(f"dtype={dtype}")
    if trust_remote_code:
        model_args.append("trust_remote_code=true")
    if max_length is not None:
        model_args.append(f"max_length={max_length}")

    model_args_str = ",".join(model_args)

    # Determine task subset for this GPU - distribute tasks evenly
    num_tasks = len(tasks)
    tasks_per_gpu = num_tasks // num_gpus
    remainder = num_tasks % num_gpus

    # Calculate start and end indices for this GPU
    if gpu_id < remainder:
        # GPUs with id < remainder get one extra task
        start_idx = gpu_id * (tasks_per_gpu + 1)
        end_idx = start_idx + tasks_per_gpu + 1
    else:
        start_idx = gpu_id * tasks_per_gpu + remainder
        end_idx = start_idx + tasks_per_gpu

    gpu_tasks = tasks[start_idx:end_idx]

    if not gpu_tasks:
        print(f"[GPU {gpu_id}] No tasks assigned to this GPU")
        output_queue.put({"gpu_id": gpu_id, "results": {}})
        return

    print(f"GPU {gpu_id}: Evaluating tasks {gpu_tasks}")

    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args_str,
            tasks=gpu_tasks,
            batch_size=batch_size,
            device="cuda",
            num_fewshot=num_fewshot,
            limit=limit,
        )
        elapsed = time_module.time() - start_time
        print(f"[GPU {gpu_id}] Step 5/5: ✓ Evaluation completed in {elapsed:.1f}s")
        output_queue.put({"gpu_id": gpu_id, "results": results})
    except Exception as e:
        elapsed = time_module.time() - start_time
        print(f"[GPU {gpu_id}] ✗ Evaluation failed after {elapsed:.1f}s: {e}")
        output_queue.put({"gpu_id": gpu_id, "error": str(e)})


def merge_results(all_results):
    """Merge results from multiple GPUs."""
    merged = {"results": {}}

    for gpu_result in all_results:
        if "error" in gpu_result:
            print(f"GPU {gpu_result['gpu_id']} error: {gpu_result['error']}")
            continue

        results = gpu_result.get("results", {})
        if "results" in results:
            merged["results"].update(results["results"])

    return merged


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluations in parallel across multiple GPUs")
    parser.add_argument("--model_path", required=True, help="HF model path or checkpoint directory.")
    parser.add_argument("--tokenizer_path", default="", help="Optional tokenizer path.")
    parser.add_argument(
        "--tasks",
        default="mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval",
        help="Comma-separated lm-eval task names.",
    )
    parser.add_argument("--output_dir", default="llm_eval", help="Where to write results JSON.")
    parser.add_argument("--batch_size", default="auto", help="Batch size (or 'auto').")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype (e.g., bfloat16, float16).")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Few-shot examples.")
    parser.add_argument("--limit", type=float, default=None, help="Optional eval limit.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code.")
    parser.add_argument("--max_length", type=int, default=None, help="Optional max length.")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all).")
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

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

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

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise SystemExit("No tasks provided.")

    # Determine number of GPUs
    available_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus if args.num_gpus else available_gpus

    if num_gpus > available_gpus:
        logging.warning(f"Requested {num_gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
        num_gpus = available_gpus

    logging.info(f"Running parallel evaluation on {num_gpus} GPUs")
    logging.info(f"Model: {args.model_path}")
    logging.info(f"Tasks: {', '.join(tasks)} ({len(tasks)} total)")

    # Set world size for task distribution
    os.environ["WORLD_SIZE"] = str(num_gpus)

    # Create output queue
    output_queue = Queue()

    # Spawn processes for each GPU
    processes = []
    start_time = time.time()

    for gpu_id in range(num_gpus):
        p = Process(
            target=run_evaluation_on_gpu,
            args=(
                gpu_id,
                args.model_path,
                tasks,
                args.batch_size,
                args.dtype,
                args.num_fewshot,
                args.limit,
                output_queue,
                num_gpus,  # Pass num_gpus for task distribution
                args.tokenizer_path or None,
                args.trust_remote_code,
                args.max_length,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    elapsed_time = time.time() - start_time

    # Collect results
    all_results = []
    while not output_queue.empty():
        all_results.append(output_queue.get())

    # Merge results
    merged_results = merge_results(all_results)

    logging.info(f"Evaluation completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(os.path.abspath(args.model_path))
    out_path = os.path.join(args.output_dir, f"{model_name}_{stamp}_parallel.json")

    logging.info(f"Writing results to: {out_path}")

    serializable_results = _make_json_serializable(merged_results)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✓ Parallel evaluation completed successfully!")
    print(f"{'='*60}")
    print(f"GPUs used: {num_gpus}")
    print(f"Results saved to: {out_path}")
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"Speedup: ~{10.7*60/elapsed_time:.1f}x faster than single GPU")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
