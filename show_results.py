#!/usr/bin/env python3
"""
Display evaluation results from lm-eval-harness JSON output.
Usage: python show_results.py <path_to_results.json>
"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Display LLM evaluation results")
    parser.add_argument("json_file", help="Path to the evaluation results JSON file")
    parser.add_argument("--detailed", action="store_true", help="Show all metrics, not just primary ones")
    args = parser.parse_args()

    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.json_file}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{args.json_file}'", file=sys.stderr)
        sys.exit(1)

    results = data.get('results', {})

    if not results:
        print("No results found in JSON file", file=sys.stderr)
        sys.exit(1)

    print('='*70)
    print('EVALUATION RESULTS')
    print('='*70)
    print()

    # Define primary metrics for each task
    primary_metrics = {
        'mmlu': 'acc,none',
        'hellaswag': 'acc_norm,none',
        'arc_challenge': 'acc_norm,none',
        'winogrande': 'acc,none',
        'gsm8k': 'exact_match,flexible-extract',
        'ifeval': 'prompt_level_strict_acc,none'
    }

    # Print summary
    print("SUMMARY:")
    print("-" * 70)

    for task_name in sorted(results.keys()):
        metrics = results[task_name]

        if not isinstance(metrics, dict):
            continue

        # Get primary metric
        primary_metric = primary_metrics.get(task_name)

        if primary_metric and primary_metric in metrics:
            score = metrics[primary_metric]
            if score <= 1.0:
                print(f"  {task_name.upper():30s}: {score*100:6.2f}%")
            else:
                print(f"  {task_name.upper():30s}: {score:6.2f}")
        elif args.detailed:
            # Show all metrics if detailed mode
            print(f"\n{task_name.upper()}:")
            for metric_name, value in sorted(metrics.items()):
                if isinstance(value, (int, float)) and not metric_name.endswith('_stderr'):
                    if value <= 1.0:
                        print(f"    {metric_name:35s}: {value*100:6.2f}%")
                    else:
                        print(f"    {metric_name:35s}: {value:6.2f}")

    # Show detailed metrics if requested
    if args.detailed:
        print()
        print("="*70)
        print("DETAILED METRICS:")
        print("="*70)
        print()

        for task_name in sorted(results.keys()):
            metrics = results[task_name]

            if not isinstance(metrics, dict):
                continue

            print(f"\n{task_name.upper()}:")
            print("-" * 50)
            for metric_name, value in sorted(metrics.items()):
                if isinstance(value, (int, float)):
                    if value <= 1.0:
                        print(f"  {metric_name:40s}: {value*100:6.2f}%")
                    else:
                        print(f"  {metric_name:40s}: {value:6.2f}")

    print()
    print('='*70)
    print()
    print(f"✓ Results loaded from: {args.json_file}")
    print()


if __name__ == "__main__":
    main()
