#!/usr/bin/env python3
"""
Pre-download evaluation datasets for lm-eval-harness tasks.

This script intentionally separates dataset download from model evaluation to
avoid FUSE conflicts during multi-GPU runs.
"""
import argparse
import logging
import os
import sys
from typing import Dict, Iterable


def _resolve_tasks(task_names: Iterable[str]) -> Dict[str, object]:
    logging.info("Step 1/2: Resolving tasks from lm-eval-harness...")
    logging.info("  Requested tasks: %s", ", ".join(task_names))

    try:
        from lm_eval import tasks as lm_tasks
        logging.info("  ✓ lm_eval.tasks module imported successfully")
    except Exception as exc:
        raise SystemExit(
            "lm-eval-harness is not installed. Install with: pip install -r requirements-eval.txt"
        ) from exc

    # Try legacy get_task_dict() first - it's much faster than TaskManager initialization
    if hasattr(lm_tasks, "get_task_dict"):
        logging.info("  Using fast get_task_dict() interface (legacy)")
        task_dict = lm_tasks.get_task_dict(task_names)
        logging.info("  DEBUG: Got %d items from get_task_dict()", len(task_dict))

        # Filter out ConfigurableGroup objects - they're just containers
        # The individual tasks (like mmlu_anatomy, etc.) are already in task_dict
        filtered_dict = {}
        for key, value in task_dict.items():
            type_name = type(value).__name__
            value_str = str(value)
            logging.info("  DEBUG: Task '%s' has type '%s'", key, type_name)

            # Skip ConfigurableGroup containers - keep only downloadable tasks
            if type_name == 'ConfigurableGroup' or 'ConfigurableGroup' in value_str:
                logging.info("    → Skipping group container '%s' (individual tasks already included)", key)
                continue

            filtered_dict[key] = value

        logging.info("  ✓ Resolved %d downloadable task(s) after filtering", len(filtered_dict))
        return filtered_dict

    # Fall back to TaskManager if get_task_dict not available
    if hasattr(lm_tasks, "TaskManager"):
        logging.info("  Using TaskManager interface (slower - scans all tasks)")
        manager = lm_tasks.TaskManager()
        if hasattr(manager, "get_task_dict"):
            logging.info("  Resolving via TaskManager.get_task_dict()")
            task_dict = manager.get_task_dict(task_names)
            # Filter out ConfigurableGroup objects
            filtered_dict = {}
            for key, value in task_dict.items():
                type_name = type(value).__name__
                value_str = str(value)
                # Skip ConfigurableGroup containers
                if type_name == 'ConfigurableGroup' or 'ConfigurableGroup' in value_str:
                    logging.info("    → Skipping group container '%s' (individual tasks already included)", key)
                    continue
                filtered_dict[key] = value
            logging.info("  ✓ Resolved %d downloadable task(s) after filtering", len(filtered_dict))
            return filtered_dict
        if hasattr(manager, "load_task_or_group"):
            logging.info("  Resolving via TaskManager.load_task_or_group()")
            task_dict = {}
            for name in task_names:
                logging.info("    Loading task/group: %s", name)
                task_dict.update(manager.load_task_or_group(name))
            # Filter out ConfigurableGroup objects
            filtered_dict = {}
            for key, value in task_dict.items():
                type_name = type(value).__name__
                value_str = str(value)
                # Skip ConfigurableGroup containers
                if type_name == 'ConfigurableGroup' or 'ConfigurableGroup' in value_str:
                    logging.info("    → Skipping group container '%s' (individual tasks already included)", key)
                    continue
                filtered_dict[key] = value
            logging.info("  ✓ Resolved %d downloadable task(s) after filtering", len(filtered_dict))
            return filtered_dict

    raise SystemExit("Unsupported lm-eval-harness version; cannot resolve tasks.")


def _force_download(task_name: str, task: object) -> None:
    logging.info("    Downloading dataset for task: %s", task_name)
    
    if hasattr(task, "download"):
        logging.info("      Method: task.download()")
        task.download()
        logging.info("      ✓ Download completed")
        return

    for getter_name in ("training_docs", "validation_docs", "test_docs", "fewshot_docs"):
        getter = getattr(task, getter_name, None)
        if getter is None:
            continue
        logging.info("      Method: %s()", getter_name)
        iterator = getter()
        try:
            next(iter(iterator))
            logging.info("      ✓ Dataset accessed successfully")
        except StopIteration:
            logging.info("      ✓ Empty dataset (no download needed)")
        return

    if hasattr(task, "dataset"):
        logging.info("      Method: task.dataset property")
        _ = task.dataset
        logging.info("      ✓ Dataset loaded")
        return

    raise RuntimeError(f"Task {task_name} does not expose a downloadable dataset interface.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download lm-eval datasets.")
    parser.add_argument(
        "--tasks",
        default="mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval",
        help="Comma-separated lm-eval task names.",
    )
    parser.add_argument("--cache_dir", default=None, help="HF datasets cache directory.")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed logging.")
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Ensure downloads are allowed for the pre-download stage.
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    if args.cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir
        logging.info("Using datasets cache directory: %s", args.cache_dir)

    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not task_names:
        raise SystemExit("No tasks provided. Use --tasks with comma-separated task names.")

    logging.info("="*60)
    logging.info("Starting dataset pre-download")
    logging.info("="*60)
    
    task_dict = _resolve_tasks(task_names)
    if not task_dict:
        raise SystemExit(f"No tasks resolved from: {', '.join(task_names)}")

    logging.info("")
    logging.info("Step 2/2: Downloading datasets...")
    logging.info("  Total tasks to download: %d", len(task_dict))
    
    failures = []
    success_count = 0
    for idx, (name, task) in enumerate(task_dict.items(), 1):
        # Convert name to string in case it's not already
        name_str = str(name)
        logging.info("  [%d/%d] Processing task: %s", idx, len(task_dict), name_str)
        try:
            _force_download(name_str, task)
            success_count += 1
            logging.info("    ✓ Success")
        except Exception as exc:
            logging.error("    ✗ Failed: %s", exc)
            failures.append(name_str)

    logging.info("")
    logging.info("="*60)
    logging.info("Download Summary:")
    logging.info("  Successful: %d/%d", success_count, len(task_dict))
    if failures:
        logging.error("  Failed: %d/%d", len(failures), len(task_dict))
        # Convert all failure items to strings to ensure join works
        failure_names = [str(f) for f in failures]
        logging.error("  Failed tasks: %s", ", ".join(failure_names))
        logging.info("="*60)
        raise SystemExit(f"Failed to download datasets for: {', '.join(failure_names)}")
    
    logging.info("  ✓ All datasets downloaded successfully!")
    logging.info("="*60)
    print("\n✓ Dataset pre-download completed successfully.")


if __name__ == "__main__":
    main()
