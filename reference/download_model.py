#!/usr/bin/env python3
"""
Pre-download model and datasets to avoid I/O bottlenecks during evaluation.
"""
import argparse
import sys
from pathlib import Path


def download_model(model_name: str, cache_dir: str = None):
    """Download model from HuggingFace Hub to local cache."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Install required packages: pip install transformers huggingface_hub")
        sys.exit(1)

    print(f"🔽 Downloading model: {model_name}")
    print(f"📁 Cache directory: {cache_dir or 'default (~/.cache/huggingface)'}")
    print()

    # Download model files
    print("Downloading model files...")
    try:
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
        )
        print(f"✅ Model files downloaded")
    except Exception as e:
        print(f"⚠️  Model download issue: {e}")

    # Load tokenizer (lightweight, ensures config files are present)
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"✅ Tokenizer loaded ({len(tokenizer)} tokens)")
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"✅ Model '{model_name}' is ready for evaluation!")
    print(f"{'='*60}\n")


def download_datasets(task_names: list[str]):
    """Pre-download evaluation datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("⚠️  datasets library not installed, skipping dataset download")
        return

    # Map of lm-eval tasks to HuggingFace dataset names
    dataset_map = {
        "mmlu": ("cais/mmlu", "all"),
        "hellaswag": ("Rowan/hellaswag", None),
        "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge"),
        "winogrande": ("allenai/winogrande", "winogrande_xl"),
        "gsm8k": ("openai/gsm8k", "main"),
    }

    print(f"\n🔽 Pre-downloading datasets for tasks: {', '.join(task_names)}")

    for task in task_names:
        if task in dataset_map:
            dataset_name, config = dataset_map[task]
            try:
                print(f"  Downloading {task} ({dataset_name})...")
                if config:
                    load_dataset(dataset_name, config)
                else:
                    load_dataset(dataset_name)
                print(f"  ✅ {task} downloaded")
            except Exception as e:
                print(f"  ⚠️  {task} download failed: {e}")
        else:
            print(f"  ⚠️  Unknown dataset mapping for: {task}")

    print(f"\n✅ Dataset download complete\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-download model and datasets for LLM evaluation")
    parser.add_argument("model_name", help="Model name from HuggingFace (e.g., Qwen/Qwen3-4B-Instruct-2507)")
    parser.add_argument("--tasks", default="mmlu,hellaswag,arc_challenge,winogrande,gsm8k",
                       help="Comma-separated task names")
    parser.add_argument("--cache-dir", default=None, help="Cache directory for downloads")
    parser.add_argument("--skip-datasets", action="store_true", help="Skip dataset download")
    args = parser.parse_args()

    # Download model
    download_model(args.model_name, args.cache_dir)

    # Download datasets
    if not args.skip_datasets:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
        download_datasets(task_list)

    print("\n🚀 Ready to run evaluation!")


if __name__ == "__main__":
    main()
