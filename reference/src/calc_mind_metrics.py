"""
Calculate MIND evaluation metrics from prediction file.

This script reads predictions (ranked news IDs) and compares them against
ground truth labels from the behaviors file to compute AUC, MRR, and nDCG metrics.

Usage:
    python calc_mind_metrics.py --predictions ./results_mind/dev_predictions.txt --behaviors ../data/MIND/dev/behaviors.tsv
"""

import argparse
import numpy as np
from tqdm import tqdm

from mind_utils import auc_score, mrr_score, ndcg_score


def load_ground_truth(behaviors_path: str):
    """Load ground truth labels from behaviors file."""
    ground_truth = {}

    print(f"Loading ground truth from: {behaviors_path}")
    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading behaviors"):
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue

            impression_id = parts[0]
            impressions = parts[4].split()

            labels_dict = {}
            for imp in impressions:
                if "-" not in imp:
                    continue
                nid, label = imp.rsplit("-", 1)
                labels_dict[nid] = int(label)

            ground_truth[impression_id] = labels_dict

    print(f"✓ Loaded {len(ground_truth)} impressions with ground truth labels")
    return ground_truth


def calculate_metrics(predictions_path: str, behaviors_path: str):
    """Calculate metrics from predictions file."""

    # Load ground truth
    ground_truth = load_ground_truth(behaviors_path)

    # Load predictions
    print(f"\nLoading predictions from: {predictions_path}")
    predictions = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            impression_id = parts[0]
            ranked_news_ids = parts[1:]
            predictions.append((impression_id, ranked_news_ids))

    print(f"✓ Loaded {len(predictions)} predictions")

    # Calculate metrics
    aucs = []
    mrrs = []
    ndcg5 = []
    ndcg10 = []

    matched = 0
    missing_gt = 0

    print("\nCalculating metrics...")
    for impression_id, ranked_news_ids in tqdm(predictions, desc="Processing impressions"):
        if impression_id not in ground_truth:
            missing_gt += 1
            continue

        gt_labels = ground_truth[impression_id]

        # Build labels and scores based on prediction ranking
        labels = []
        scores = []
        for rank, nid in enumerate(ranked_news_ids):
            if nid in gt_labels:
                labels.append(gt_labels[nid])
                # Higher rank = higher score (inverted rank as score)
                scores.append(len(ranked_news_ids) - rank)

        # Skip if no valid candidates
        if not labels or sum(labels) == 0:
            continue

        # Calculate metrics for this impression
        aucs.append(auc_score(labels, scores))
        mrrs.append(mrr_score(labels, scores))
        ndcg5.append(ndcg_score(labels, scores, 5))
        ndcg10.append(ndcg_score(labels, scores, 10))
        matched += 1

    # Print results
    print("\n" + "="*50)
    print("MIND Evaluation Metrics")
    print("="*50)
    print(f"Total predictions: {len(predictions)}")
    print(f"Matched with ground truth: {matched}")
    if missing_gt > 0:
        print(f"Missing ground truth: {missing_gt}")
    print(f"\nResults:")
    print(f"  AUC:      {np.mean(aucs):.4f}")
    print(f"  MRR:      {np.mean(mrrs):.4f}")
    print(f"  nDCG@5:   {np.mean(ndcg5):.4f}")
    print(f"  nDCG@10:  {np.mean(ndcg10):.4f}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Calculate MIND evaluation metrics from predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions file (e.g., ./results_mind/dev_predictions.txt)")
    parser.add_argument("--behaviors", required=True, help="Path to behaviors file with ground truth (e.g., ../data/MIND/dev/behaviors.tsv)")
    args = parser.parse_args()

    calculate_metrics(args.predictions, args.behaviors)


if __name__ == "__main__":
    main()
