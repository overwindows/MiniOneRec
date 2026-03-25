"""
Ensemble pre-computed pointwise scores from multiple models.

Each score file is produced by evaluate_mind_pointwise.py --output_scores_file.
Format: impression_id<TAB>score1<TAB>score2<TAB>...  (one line per impression)

This is much faster than running models in-process because scoring is already done
by the fast parallel eval; this script just averages numbers.

Usage:
    python src/ensemble_from_scores.py \
        --score_files scores_model1.txt scores_model2.txt \
        --behaviors_path data/MIND_large/dev/behaviors.tsv \
        --output_file ensemble_predictions.txt \
        --weights 1.0 0.8
"""

import argparse
import math
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def load_scores(path: str) -> Dict[str, List[float]]:
    """Load score file → {impression_id: [score1, score2, ...]}"""
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            imp_id = parts[0]
            scores[imp_id] = [float(s) for s in parts[1:]]
    return scores


def auc_score(labels, scores):
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    pairs = sorted(zip(scores, labels), reverse=True)
    pos_above = 0
    auc = 0.0
    for s, l in pairs:
        if l == 1:
            pos_above += 1
        else:
            auc += pos_above
    return auc / (pos * neg)


def mrr_score(labels, scores):
    for rank, idx in enumerate(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True), 1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


def ndcg_score(labels, scores, k):
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    dcg = sum(labels[i] / math.log2(r + 2) for r, i in enumerate(top_k))
    ideal = sum(1.0 / math.log2(r + 2) for r in range(min(sum(labels), k)))
    return dcg / ideal if ideal > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_files", nargs="+", required=True,
                        help="Raw score files from evaluate_mind_pointwise.py --output_scores_file")
    parser.add_argument("--behaviors_path", required=True,
                        help="behaviors.tsv — needed for labels and candidate order")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Per-file weights (default: equal)")
    parser.add_argument("--output_file", default=None,
                        help="Save ranked predictions (optional)")
    args = parser.parse_args()

    n = len(args.score_files)
    weights = args.weights or [1.0] * n
    if len(weights) != n:
        raise ValueError(f"--weights length ({len(weights)}) must match --score_files ({n})")
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    print(f"Loading {n} score files...")
    all_scores = []
    for i, path in enumerate(args.score_files):
        s = load_scores(path)
        all_scores.append(s)
        print(f"  [{i+1}] {len(s)} impressions — {path}")

    common_ids = set(all_scores[0].keys())
    for s in all_scores[1:]:
        common_ids &= set(s.keys())
    print(f"  {len(common_ids)} impressions in common across all files\n")

    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    predictions = []

    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Ensembling", unit="imp"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            impression_id = parts[0]
            if impression_id not in common_ids:
                continue

            imp_list = parts[4].split()
            labels, candidate_ids = [], []
            for item in imp_list:
                if "-" in item:
                    nid, lbl = item.rsplit("-", 1)
                    labels.append(int(lbl))
                else:
                    nid = item
                    labels.append(0)
                candidate_ids.append(nid)

            n_cands = len(labels)

            # Weighted average of scores across models
            ensemble = [0.0] * n_cands
            for w, model_scores in zip(weights, all_scores):
                m = model_scores[impression_id]
                for j in range(min(n_cands, len(m))):
                    ensemble[j] += w * m[j]

            if sum(labels) > 0:
                aucs.append(auc_score(labels, ensemble))
                mrrs.append(mrr_score(labels, ensemble))
                ndcg5s.append(ndcg_score(labels, ensemble, 5))
                ndcg10s.append(ndcg_score(labels, ensemble, 10))

            if args.output_file:
                ranked = sorted(range(len(ensemble)), key=lambda i: ensemble[i], reverse=True)
                predictions.append((impression_id, [candidate_ids[i] for i in ranked]))

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    print(f"\n{'='*55}")
    print(f"MIND Ensemble Results ({n} models)")
    print(f"{'='*55}")
    for i, (path, w) in enumerate(zip(args.score_files, weights)):
        print(f"  [{i+1}] w={w:.3f}  {path}")
    print(f"Impressions evaluated: {len(aucs)}")
    print(f"AUC:     {_avg(aucs):.4f}")
    print(f"MRR:     {_avg(mrrs):.4f}")
    print(f"nDCG@5:  {_avg(ndcg5s):.4f}")
    print(f"nDCG@10: {_avg(ndcg10s):.4f}")
    print(f"{'='*55}")

    if args.output_file and predictions:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for imp_id, ranked_ids in predictions:
                f.write(f"{imp_id} {' '.join(ranked_ids)}\n")
        print(f"Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
