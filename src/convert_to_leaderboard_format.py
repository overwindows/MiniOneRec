"""
Convert internal prediction format to MIND leaderboard submission format.

Internal format (one line per impression):
    impression_id N12345 N67890 N11111   <- news IDs in ranked order

MIND leaderboard format:
    impression_id [3,1,2]   <- rank of each candidate in original order

Usage:
    python src/convert_to_leaderboard_format.py \
        --predictions path/to/test_predictions.txt \
        --behaviors path/to/test/behaviors.tsv \
        --output path/to/submission.txt
"""

import argparse
from tqdm import tqdm


def convert(predictions_path: str, behaviors_path: str, output_path: str):
    # Load original candidate order from behaviors.tsv
    print(f"Loading candidate order from: {behaviors_path}")
    original_order = {}
    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading behaviors"):
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            impression_id = parts[0]
            candidates = [imp.split("-")[0] for imp in parts[4].split() if imp]
            original_order[impression_id] = candidates
    print(f"Loaded {len(original_order)} impressions")

    # Load predictions
    print(f"\nLoading predictions from: {predictions_path}")
    predictions = {}
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            impression_id = parts[0]
            ranked_news_ids = parts[1:]
            predictions[impression_id] = ranked_news_ids
    print(f"Loaded {len(predictions)} predictions")

    # Convert and write
    print(f"\nConverting to leaderboard format...")
    missing = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for impression_id, candidates in tqdm(original_order.items(), desc="Converting"):
            if impression_id not in predictions:
                missing += 1
                continue

            ranked = predictions[impression_id]
            # Map each news ID to its predicted rank (1-indexed, lower = better)
            rank_map = {nid: rank + 1 for rank, nid in enumerate(ranked)}

            # Assign rank for each candidate in original order
            # Candidates not in predictions get the worst rank
            worst_rank = len(ranked) + 1
            ranks = [rank_map.get(nid, worst_rank) for nid in candidates]

            ranks_str = ",".join(str(r) for r in ranks)
            f.write(f"{impression_id} [{ranks_str}]\n")

    print(f"\n✓ Saved submission to: {output_path}")
    if missing > 0:
        print(f"WARNING: {missing} impressions missing from predictions")


def main():
    parser = argparse.ArgumentParser(description="Convert predictions to MIND leaderboard format")
    parser.add_argument("--predictions", required=True, help="Internal predictions file (impression_id + ranked news IDs)")
    parser.add_argument("--behaviors", required=True, help="behaviors.tsv for the split being submitted")
    parser.add_argument("--output", required=True, help="Output submission file")
    args = parser.parse_args()

    convert(args.predictions, args.behaviors, args.output)


if __name__ == "__main__":
    main()
