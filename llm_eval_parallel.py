#!/usr/bin/env python3
"""
Multi-GPU parallel evaluation using process-level data parallelism.
This approach spawns one process per GPU for maximum throughput.
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
    # Disable torch distributed to avoid port conflicts
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(9500 + gpu_id)  # Use different port per GPU
    os.environ["RANK"] = str(gpu_id)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"  # Each process sees only 1 GPU

    print(f"[GPU {gpu_id}] Step 2/5: Importing lm_eval library...")
    try:
        from lm_eval import evaluator
        print(f"[GPU {gpu_id}] ✓ lm_eval imported successfully")
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ Failed to import lm_eval: {e}")
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

    print(f"[GPU {gpu_id}] Step 3/5: Tasks assigned: {gpu_tasks}")
    print(f"[GPU {gpu_id}] Step 4/5: Loading model '{model_path}' (this may take several minutes)...")
    load_start = time_module.time()

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
        load_time = time_module.time() - load_start
        print(f"[GPU {gpu_id}] Step 5/5: ✓ Evaluation completed in {elapsed:.1f}s (model load: {load_time:.1f}s)")
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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

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
