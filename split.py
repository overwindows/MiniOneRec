import fire
import os
import pandas as pd

def split(input_path, output_path, cuda_list):
    """
    Split a CSV file into multiple parts for parallel GPU processing.

    Args:
        input_path: Path to input CSV file
        output_path: Directory to save split files
        cuda_list: List of GPU IDs (can be string "4,5,6,7" or list [4,5,6,7] or int 4)
    """
    # Handle different input types for cuda_list
    if isinstance(cuda_list, str):
        # Parse comma-separated string: "4,5,6,7" -> [4, 5, 6, 7]
        cuda_list = [int(x.strip()) for x in cuda_list.split(',') if x.strip()]
    elif isinstance(cuda_list, int):
        cuda_list = [cuda_list]
    else:
        cuda_list = list(cuda_list)

    # Read input data
    df = pd.read_csv(input_path)
    # df = df.sample(frac=1).reset_index(drop=True)

    # Create output directory if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Split data evenly across GPUs
    df_len = len(df)
    cuda_num = len(cuda_list)

    print(f"Splitting {df_len} rows across {cuda_num} GPUs: {cuda_list}")

    for i in range(cuda_num):
        start = i * df_len // cuda_num
        end = (i+1) * df_len // cuda_num
        output_file = f'{output_path}/{cuda_list[i]}.csv'
        # CRITICAL FIX: index=False to avoid corrupting row alignment during merge
        df[start:end].to_csv(output_file, index=False)
        print(f"  Created {output_file}: rows {start}-{end-1} ({end-start} rows)")
        
if __name__ == '__main__':
    fire.Fire(split)
