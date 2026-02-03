"""
Split MIND behaviors.tsv file across multiple GPUs for parallel evaluation.

Usage:
    python split_mind.py --input_path behaviors.tsv --output_path ./temp --cuda_list "0,1,2,3"
"""

import fire
import os
from tqdm import tqdm


def split(input_path, output_path, cuda_list):
    """
    Split MIND behaviors.tsv file into multiple parts for parallel GPU processing.

    Args:
        input_path: Path to behaviors.tsv file
        output_path: Directory to save split files
        cuda_list: Comma-separated list of GPU IDs (e.g., "0,1,2,3") or list of ints
    """
    # Parse cuda_list
    if isinstance(cuda_list, str):
        cuda_list = [int(x.strip()) for x in cuda_list.split(',') if x.strip()]

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Read input file and count lines
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    num_gpus = len(cuda_list)

    print(f"Total impressions: {total_lines}")
    print(f"Splitting across {num_gpus} GPUs: {cuda_list}")

    # Split data across GPUs
    lines_per_gpu = total_lines // num_gpus

    for i, gpu_id in enumerate(cuda_list):
        start_idx = i * lines_per_gpu
        # Last GPU gets remaining lines
        end_idx = (i + 1) * lines_per_gpu if i < num_gpus - 1 else total_lines

        output_file = os.path.join(output_path, f'{gpu_id}.tsv')

        # Write split file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(lines[start_idx:end_idx])

        num_lines = end_idx - start_idx
        print(f"GPU {gpu_id}: {num_lines} impressions → {output_file}")

    print(f"✓ Split completed: {num_gpus} files created in {output_path}")


if __name__ == "__main__":
    fire.Fire(split)
