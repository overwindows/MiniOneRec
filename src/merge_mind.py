"""
Merge and validate MIND prediction or score shards from GPU evaluators.

Usage:
    python merge_mind.py --input_path ./temp --output_path predictions.txt --cuda_list "0,1,2,3"
"""

import fire
import os
from collections import Counter
import math
from tqdm import tqdm

from split_mind import parse_cuda_list


def merge(input_path, output_path=None, cuda_list=None, calculate_metrics=True,
          behaviors_path=None, max_impressions=0, score_files=False):
    """
    Merge MIND prediction files from multiple GPU processes.

    Args:
        input_path: Directory containing prediction files (GPU_ID.txt format)
        output_path: Path to save merged predictions (optional)
        cuda_list: A GPU ID, comma-separated IDs, or a list/tuple of ints (auto-detect if omitted)
        calculate_metrics: If True, print the reminder that labeled metrics require a separate evaluator
        behaviors_path: Optional source for exact impression/candidate coverage validation
        max_impressions: Per-shard limit used by the evaluator,0 means complete source
        score_files: Read GPU_ID_scores.txt instead of ranked GPU_ID.txt predictions
    """
    if (not isinstance(max_impressions, int) or isinstance(max_impressions, bool)
            or max_impressions < 0):
        raise ValueError("The per-shard impression limit must be a nonnegative integer")
    suffix = "_scores.txt" if score_files else ".txt"
    if cuda_list is None:
        txt_files = [f for f in os.listdir(input_path)
                     if f.endswith(suffix) and f[:-len(suffix)].isdigit()]
        cuda_list = sorted(int(f[:-len(suffix)]) for f in txt_files)
        print(f"Auto-detected GPU files: {sorted(cuda_list)}")
    cuda_list = parse_cuda_list(cuda_list)

    predictions = []
    identifiers = set()

    print(f"Merging predictions from {len(cuda_list)} GPU(s)...")

    for gpu_id in tqdm(sorted(cuda_list), desc="Reading prediction files"):
        pred_file = os.path.join(input_path, f'{gpu_id}{suffix}')
        rows = 0
        with open(pred_file, 'r', encoding='utf-8') as f:
            for number, line in enumerate(f, 1):
                line = line.strip()
                fields = line.split()
                if len(fields) < 2 or fields[0] in identifiers:
                    raise ValueError(f"Malformed or duplicate impression in {pred_file}:{number}")
                if score_files and not all(math.isfinite(float(value)) for value in fields[1:]):
                    raise ValueError(f"Nonfinite scores in {pred_file}:{number}")
                identifiers.add(fields[0])
                predictions.append(line)
                rows += 1
        if not rows:
            raise ValueError(f"Empty prediction shard: {pred_file}")

    if not predictions:
        raise RuntimeError("Merge failed: no predictions found!")

    if behaviors_path:
        expected = []
        with open(behaviors_path, encoding="utf-8") as handle:
            seen = set()
            for number, line in enumerate(handle, 1):
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 5 or not parts[0] or parts[0] in seen or not parts[4].split():
                    raise ValueError(f"Malformed or duplicate source impression at line {number}")
                seen.add(parts[0])
                candidates = [token.rsplit("-", 1)[0] if token.endswith(("-0", "-1")) else token
                              for token in parts[4].split()]
                expected.append((parts[0], candidates))
        if max_impressions:
            size = len(expected) // len(cuda_list)
            limited = []
            for index in range(len(cuda_list)):
                start = index * size
                end = start + size if index < len(cuda_list) - 1 else len(expected)
                limited.extend(expected[start:min(start + max_impressions, end)])
            expected = limited
        if identifiers != {identifier for identifier, _ in expected}:
            raise ValueError("Prediction impressions do not exactly cover the requested source")
        records = {line.split()[0]: line for line in predictions}
        for identifier, candidates in expected:
            fields = records[identifier].split()[1:]
            if (len(fields) != len(candidates)
                    or (not score_files and Counter(fields) != Counter(candidates))):
                raise ValueError(f"Candidate coverage differs for impression {identifier}")
        predictions = [records[identifier] for identifier, _ in expected]

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
