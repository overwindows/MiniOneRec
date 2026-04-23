"""
Merge MIND prediction files from multiple GPUs and calculate final metrics.

Usage:
    python merge_mind.py --input_path ./temp --output_path predictions.txt --cuda_list "0,1,2,3"
"""

import fire
import os
from tqdm import tqdm


def merge(input_path, output_path=None, cuda_list=None, calculate_metrics=True):
    """
    Merge MIND prediction files from multiple GPU processes.

    Args:
        input_path: Directory containing prediction files (GPU_ID.txt format)
        output_path: Path to save merged predictions (optional)
        cuda_list: Comma-separated list of GPU IDs or list of ints (optional, auto-detect if not provided)
        calculate_metrics: If True, only calculate and print metrics without saving merged file
    """
    # Parse cuda_list
    if cuda_list is None:
        # Auto-detect: find all .txt files whose stem is a plain GPU ID (integer)
        txt_files = [f for f in os.listdir(input_path)
                     if f.endswith('.txt') and f[:-4].isdigit()]
        cuda_list = [int(f[:-4]) for f in txt_files]
        print(f"Auto-detected GPU files: {sorted(cuda_list)}")
    elif isinstance(cuda_list, str):
        cuda_list = [int(x.strip()) for x in cuda_list.split(',') if x.strip() and x.strip().isdigit()]

    predictions = []
    failed_files = []

    print(f"Merging predictions from {len(cuda_list)} GPU(s)...")

    # Read prediction files from each GPU
    for gpu_id in tqdm(sorted(cuda_list), desc="Reading prediction files"):
        pred_file = os.path.join(input_path, f'{gpu_id}.txt')

        if not os.path.exists(pred_file):
            print(f"WARNING: File not found: {pred_file}")
            failed_files.append(pred_file)
            continue

        try:
            with open(pred_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    predictions.append(line)
        except Exception as e:
            print(f"ERROR: Failed to read {pred_file}: {e}")
            failed_files.append(pred_file)

    if failed_files:
        print(f"\nWARNING: {len(failed_files)} file(s) failed to load:")
        for f in failed_files:
            print(f"  - {f}")

    if not predictions:
        raise RuntimeError("Merge failed: no predictions found!")

    print(f"\n✓ Successfully merged {len(predictions)} impressions")

    # Save merged predictions if output_path is provided
    if output_path:
        print(f"Writing merged predictions to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(pred + '\n')
        print(f"✓ Saved {len(predictions)} predictions to {output_path}")

    # If only calculating metrics (for dev/test evaluation without labels)
    if not calculate_metrics:
        return

    print("\nNote: Metrics calculation requires labels (dev split only).")
    print("For test split predictions, submit to MIND leaderboard.")


if __name__ == "__main__":
    fire.Fire(merge)
