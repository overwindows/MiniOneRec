import fire
import pandas as pd
import json
import os
from tqdm import tqdm

def merge(input_path, output_path, cuda_list):
    """
    Merge JSON result files from multiple GPU processes.

    Args:
        input_path: Directory containing split result files
        output_path: Path to save merged results
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

    data = []
    failed_files = []
    success_count = 0

    print(f"Merging results from {len(cuda_list)} GPU(s): {cuda_list}")

    for i in tqdm(cuda_list, desc="Merging JSON files"):
        json_file = f'{input_path}/{i}.json'

        # Check if file exists
        if not os.path.exists(json_file):
            print(f"\n  WARNING: Missing file {json_file} - GPU {i} may have failed")
            failed_files.append(json_file)
            continue

        # Try to load JSON with error handling
        try:
            with open(json_file, 'r') as f:
                chunk_data = json.load(f)
                data.extend(chunk_data)
                success_count += 1
                print(f"\n  ✓ Loaded {json_file}: {len(chunk_data)} records")
        except json.JSONDecodeError as e:
            print(f"\n  ERROR: Corrupted JSON in {json_file}: {e}")
            failed_files.append(json_file)
            continue
        except Exception as e:
            print(f"\n  ERROR: Failed to read {json_file}: {e}")
            failed_files.append(json_file)
            continue

    # Report merge summary
    print(f"\n{'='*60}")
    print(f"Merge Summary:")
    print(f"  Successfully merged: {success_count}/{len(cuda_list)} files")
    print(f"  Total records: {len(data)}")

    if failed_files:
        print(f"\n  ⚠️  Failed files ({len(failed_files)}):")
        for f in failed_files:
            print(f"    - {f}")
        print(f"\n  WARNING: Results are incomplete!")

    # Save merged results
    if data:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n  ✓ Saved merged results to: {output_path}")
    else:
        print(f"\n  ERROR: No data to save - all GPU processes may have failed!")
        raise RuntimeError("Merge failed: no valid data found")

    print(f"{'='*60}\n")

    return len(data)

if __name__ == '__main__':
    fire.Fire(merge)
