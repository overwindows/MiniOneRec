"""
Split MIND behaviors.tsv file across multiple GPUs for parallel evaluation.

Usage:
    python split_mind.py --input_path behaviors.tsv --output_path ./temp --cuda_list "0,1,2,3"
"""

import fire
import os
from tqdm import tqdm


def parse_cuda_list(cuda_list):
    if isinstance(cuda_list, int) and not isinstance(cuda_list, bool):
        cuda_list = [cuda_list]
    elif isinstance(cuda_list, str):
        fields = [field.strip() for field in cuda_list.split(",")]
        if not all(field.isdecimal() for field in fields):
            raise ValueError("GPU IDs must be nonnegative integers")
        cuda_list = [int(field) for field in fields]
    if (not isinstance(cuda_list, (list, tuple)) or not cuda_list
            or any(not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0 for gpu in cuda_list)
            or len(set(cuda_list)) != len(cuda_list)):
        raise ValueError("GPU IDs must be a nonempty, unique list of nonnegative integers")
    return list(cuda_list)


def split(input_path, output_path, cuda_list):
    """
    Split MIND behaviors.tsv file into multiple parts for parallel GPU processing.

    Args:
        input_path: Path to behaviors.tsv file
        output_path: Directory to save split files
        cuda_list: A GPU ID, comma-separated IDs, or a list/tuple of ints
    """
    cuda_list = parse_cuda_list(cuda_list)

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
