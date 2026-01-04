#!/usr/bin/env python3
"""
Extract LLM evaluation results from JSON files and create a markdown table.
"""
import json
import os
import sys
from pathlib import Path

def extract_results(json_file):
    """Extract key metrics from evaluation results."""
    try:
        with open(json_file, 'r') as f:
            # Read first part to check structure
            first_line = f.readline()
            f.seek(0)

            # Try to load full JSON
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # If file is too large or malformed, try to find results section
                print(f"Warning: Could not parse full JSON for {json_file}", file=sys.stderr)
                return None

        results = data.get('results', {})

        # Define primary metrics for each task
        primary_metrics = {
            'mmlu': 'acc,none',
            'hellaswag': 'acc_norm,none',
            'arc_challenge': 'acc_norm,none',
            'winogrande': 'acc,none',
            'gsm8k': 'exact_match,flexible-extract',
            'ifeval': 'prompt_level_strict_acc,none'
        }

        extracted = {}
        for task_name, metric_key in primary_metrics.items():
            if task_name in results and metric_key in results[task_name]:
                extracted[task_name] = results[task_name][metric_key]

        return extracted

    except Exception as e:
        print(f"Error reading {json_file}: {e}", file=sys.stderr)
        return None


def main():
    llm_eval_dir = Path("/home/aiscuser/wuc/MiniOneRec/llm_eval")

    if not llm_eval_dir.exists():
        print(f"Directory {llm_eval_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Find all JSON files
    json_files = sorted(llm_eval_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {llm_eval_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_files)} result files")
    print()

    # Process each file
    all_results = {}
    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        results = extract_results(json_file)
        if results:
            # Use filename as key, extract model name
            model_name = json_file.stem.rsplit('_', 1)[0]  # Remove timestamp
            if model_name not in all_results:
                all_results[model_name] = results
            print(f"  ✓ Extracted {len(results)} metrics")
        else:
            print(f"  ✗ Failed to extract results")

    print()
    print("="*80)
    print("MARKDOWN TABLE")
    print("="*80)
    print()

    # Create markdown table
    if all_results:
        # Headers
        tasks = ['mmlu', 'hellaswag', 'arc_challenge', 'winogrande', 'gsm8k', 'ifeval']

        print("| Model | MMLU | HellaSwag | ARC-C | Winogrande | GSM8K | IFEval |")
        print("|-------|------|-----------|-------|------------|-------|--------|")

        for model_name, results in all_results.items():
            row = [model_name]
            for task in tasks:
                if task in results:
                    score = results[task]
                    # Format as percentage
                    if score <= 1.0:
                        row.append(f"{score*100:.2f}")
                    else:
                        row.append(f"{score:.2f}")
                else:
                    row.append("-")

            print(f"| {' | '.join(row)} |")

    else:
        print("No results extracted")

    print()


if __name__ == "__main__":
    main()
