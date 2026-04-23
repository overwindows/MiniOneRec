"""
Merge per-GPU DOCA evaluation score files into one.

Each GPU writes a score file with lines: label1\\tlabel2...\\t|\\tscore1\\tscore2...
This script concatenates them and computes aggregate metrics.

Usage:
    python src/merge_doca.py \
        --input_path temp_doca/ \
        --output_path results_doca/dev_scores.txt \
        --cuda_list 0,1,2,3
"""

import argparse
import os
import numpy as np


def auc_score(labels, scores):
    pairs = list(zip(labels, scores))
    n_pos = sum(l for l, _ in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    correct = 0
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            if pairs[i][0] > pairs[j][0]:
                if pairs[i][1] > pairs[j][1]:
                    correct += 1
                elif pairs[i][1] == pairs[j][1]:
                    correct += 0.5
    return correct / (n_pos * n_neg)


def mrr_score(labels, scores):
    ranked = sorted(zip(labels, scores), key=lambda x: -x[1])
    for i, (label, _) in enumerate(ranked, 1):
        if label == 1:
            return 1.0 / i
    return 0.0


def dcg_score(labels, scores, k):
    ranked = sorted(zip(labels, scores), key=lambda x: -x[1])[:k]
    return sum(l / np.log2(i + 2) for i, (l, _) in enumerate(ranked))


def ndcg_score(labels, scores, k):
    dcg = dcg_score(labels, scores, k)
    ideal = dcg_score(labels, labels, k)
    return dcg / ideal if ideal > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Directory with per-GPU score files")
    parser.add_argument("--output_path", required=True, help="Merged output file")
    parser.add_argument("--cuda_list", required=True, help="Comma-separated GPU IDs")
    args = parser.parse_args()

    gpu_ids = [g.strip() for g in args.cuda_list.split(",") if g.strip()]

    all_lines = []
    for gpu_id in gpu_ids:
        score_file = os.path.join(args.input_path, f"{gpu_id}_scores.txt")
        if not os.path.exists(score_file):
            print(f"WARNING: {score_file} not found, skipping GPU {gpu_id}")
            continue
        with open(score_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"GPU {gpu_id}: {len(lines)} feeds")
        all_lines.extend(lines)

    print(f"\nTotal merged: {len(all_lines)} feeds")

    # Write merged file
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")
    print(f"Merged scores saved to: {args.output_path}")

    # Compute metrics from merged scores
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    for line in all_lines:
        parts = line.split("\t|\t")
        if len(parts) != 2:
            continue
        labels = [int(x) for x in parts[0].split("\t")]
        scores = [float(x) for x in parts[1].split("\t")]
        if sum(labels) == 0:
            continue
        a = auc_score(labels, scores)
        if a is not None:
            aucs.append(a)
        mrrs.append(mrr_score(labels, scores))
        ndcg5s.append(ndcg_score(labels, scores, 5))
        ndcg10s.append(ndcg_score(labels, scores, 10))

    if aucs:
        print(f"\n=========================================")
        print(f"DOCA Evaluation Results (merged)")
        print(f"=========================================")
        print(f"Feeds evaluated: {len(aucs)}")
        print(f"AUC:     {np.mean(aucs):.4f}")
        print(f"MRR:     {np.mean(mrrs):.4f}")
        print(f"nDCG@5:  {np.mean(ndcg5s):.4f}")
        print(f"nDCG@10: {np.mean(ndcg10s):.4f}")
    else:
        print("No metrics computed (no feeds with clicks)")


if __name__ == "__main__":
    main()
