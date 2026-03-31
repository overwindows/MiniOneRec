"""
Sweep temperature values on dev predictions to find the best calibration for ensemble.

Temperature scaling divides logits by T before softmax, which scales the score
log P(Yes) - log P(No) by ~1/T. For a single model, rankings within impressions
are unchanged. For ensembles, calibrating each model's score range before averaging
can improve combined performance.

Usage:
    # Sweep temperatures for a single score file on dev
    python src/sweep_temperature.py \
        --score_file path/to/model_scores.txt \
        --behaviors_path data/MIND_large/dev/behaviors.tsv \
        --temperatures 0.5 0.8 1.0 1.2 1.5 2.0

    # Sweep for ensemble: find best per-model temperatures
    python src/sweep_temperature.py \
        --score_files path/to/model_1_scores.txt path/to/model_2_scores.txt \
        --behaviors_path data/MIND_large/dev/behaviors.tsv
"""

import argparse
import math
import numpy as np
from tqdm import tqdm


# ── Metric helpers ────────────────────────────────────────────────────────────

def _rankdata(scores):
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(sorted_idx):
        j = i
        while j + 1 < len(sorted_idx) and scores[sorted_idx[j]] == scores[sorted_idx[j + 1]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg
        i = j + 1
    return ranks


def auc_score(labels, scores):
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    ranks = _rankdata(scores)
    pos_rank_sum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (pos_rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def ndcg_score(labels, scores, k):
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    dcg = sum(1.0 / math.log2(r + 2) for r, idx in enumerate(sorted_idx[:k]) if labels[idx] == 1)
    ideal = sum(1.0 / math.log2(r + 2) for r in range(min(sum(labels), k)))
    return dcg / ideal if ideal > 0 else 0.0


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_scores(path):
    """Load score file: impression_id → list of float scores."""
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            imp_id = parts[0]
            scores[imp_id] = [float(x) for x in parts[1:]]
    return scores


def load_labels(behaviors_path):
    """Load ground truth labels: impression_id → list of int labels."""
    labels = {}
    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading behaviors"):
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            imp_id = parts[0]
            imp_labels = []
            for imp in parts[4].split():
                if "-" in imp:
                    _, lbl = imp.rsplit("-", 1)
                    imp_labels.append(int(lbl))
                else:
                    imp_labels.append(0)
            if any(l == 1 for l in imp_labels):
                labels[imp_id] = imp_labels
    return labels


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(scores_dict, labels_dict, temperature=1.0):
    """Evaluate a score file at a given temperature. Returns (auc, ndcg5, ndcg10)."""
    aucs, n5s, n10s = [], [], []
    for imp_id, lbls in labels_dict.items():
        if imp_id not in scores_dict:
            continue
        raw = scores_dict[imp_id]
        if len(raw) != len(lbls):
            continue
        scaled = [s / temperature for s in raw]
        aucs.append(auc_score(lbls, scaled))
        n5s.append(ndcg_score(lbls, scaled, 5))
        n10s.append(ndcg_score(lbls, scaled, 10))
    if not aucs:
        return 0.0, 0.0, 0.0
    return float(np.mean(aucs)), float(np.mean(n5s)), float(np.mean(n10s))


def evaluate_ensemble(score_files, labels_dict, temperatures):
    """Average scores across models (each scaled by its temperature) and evaluate."""
    all_scores = [load_scores(f) for f in score_files]
    aucs, n5s, n10s = [], [], []
    for imp_id, lbls in labels_dict.items():
        candidate_scores = None
        for i, scores_dict in enumerate(all_scores):
            if imp_id not in scores_dict:
                continue
            raw = scores_dict[imp_id]
            if len(raw) != len(lbls):
                continue
            scaled = [s / temperatures[i] for s in raw]
            if candidate_scores is None:
                candidate_scores = scaled
            else:
                candidate_scores = [a + b for a, b in zip(candidate_scores, scaled)]
        if candidate_scores is None:
            continue
        aucs.append(auc_score(lbls, candidate_scores))
        n5s.append(ndcg_score(lbls, candidate_scores, 5))
        n10s.append(ndcg_score(lbls, candidate_scores, 10))
    if not aucs:
        return 0.0, 0.0, 0.0
    return float(np.mean(aucs)), float(np.mean(n5s)), float(np.mean(n10s))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sweep temperature for score calibration")
    parser.add_argument("--score_file", help="Single score file to sweep")
    parser.add_argument("--score_files", nargs="+", help="Multiple score files for ensemble sweep")
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--temperatures", nargs="+", type=float,
                        default=[0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0],
                        help="Temperatures to sweep")
    args = parser.parse_args()

    print("Loading ground truth labels...")
    labels = load_labels(args.behaviors_path)
    print(f"Loaded {len(labels)} impressions with positive labels")

    if args.score_file:
        # Single model sweep
        print(f"\nLoading scores from: {args.score_file}")
        scores = load_scores(args.score_file)
        print(f"Loaded {len(scores)} score entries")

        print(f"\n{'T':>6}  {'AUC':>8}  {'nDCG@5':>8}  {'nDCG@10':>8}")
        print("-" * 40)
        best = None
        for T in args.temperatures:
            auc, n5, n10 = evaluate(scores, labels, T)
            marker = " ← best" if best is None or auc > best[0] else ""
            if best is None or auc > best[0]:
                best = (auc, n5, n10, T)
            print(f"{T:>6.2f}  {auc:>8.4f}  {n5:>8.4f}  {n10:>8.4f}{marker}")

        print(f"\nBest temperature: T={best[3]:.2f}  AUC={best[0]:.4f}  nDCG@5={best[1]:.4f}")

    elif args.score_files:
        # Ensemble sweep: try all equal-T first, then report
        n = len(args.score_files)
        print(f"\nEnsemble sweep over {n} models")
        print(f"\n{'T (all)':>8}  {'AUC':>8}  {'nDCG@5':>8}  {'nDCG@10':>8}")
        print("-" * 44)
        best = None
        for T in args.temperatures:
            temps = [T] * n
            auc, n5, n10 = evaluate_ensemble(args.score_files, labels, temps)
            marker = " ← best" if best is None or auc > best[0] else ""
            if best is None or auc > best[0]:
                best = (auc, n5, n10, T)
            print(f"{T:>8.2f}  {auc:>8.4f}  {n5:>8.4f}  {n10:>8.4f}{marker}")

        print(f"\nBest uniform temperature: T={best[3]:.2f}  AUC={best[0]:.4f}  nDCG@5={best[1]:.4f}")
        print(f"\nTip: re-run with individual per-model T values to fine-tune further.")

    else:
        parser.error("Provide --score_file or --score_files")


if __name__ == "__main__":
    main()
