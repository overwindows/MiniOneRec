#!/usr/bin/env python3
"""
Compute item-based collaborative filtering scores for MIND news recommendation.

For each (user, candidate) pair in dev/test behaviors:
  CF_score(u, c) = mean over history h of:
      |users_who_clicked(c) ∩ users_who_clicked(h)|
      / sqrt(|users_who_clicked(c)| * |users_who_clicked(h)|)

This is cosine similarity in the "users who clicked this item" space,
averaged over the user's reading history. Pure collaborative signal —
no text content used.

Usage:
    python src/compute_cf_scores.py \
        --train_behaviors data/MIND_large/train/behaviors.tsv \
        --eval_behaviors  data/MIND_large/dev/behaviors.tsv \
        --output          data/MIND_large/dev/cf_scores.tsv \
        --max_history 30
"""

import argparse
import math
import os
from collections import defaultdict
from typing import Dict, Set

from tqdm import tqdm


def build_inverted_index(train_behaviors_path: str, max_history: int) -> Dict[str, Set[str]]:
    """
    Build news_id → set of user_ids who have that item in their history.
    Also adds clicked items from training impressions.
    """
    news_to_users: Dict[str, Set[str]] = defaultdict(set)

    with open(train_behaviors_path, 'r', encoding='utf-8') as f:
        total = sum(1 for _ in f)

    with open(train_behaviors_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc="Building inverted index"):
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            user_id = parts[1]
            history_ids = parts[3].split()
            if max_history > 0:
                history_ids = history_ids[-max_history:]

            for news_id in history_ids:
                news_to_users[news_id].add(user_id)

            # Also add clicked impressions as positive signal
            for imp in parts[4].split():
                if '-' in imp:
                    news_id, label = imp.rsplit('-', 1)
                    if label == '1':
                        news_to_users[news_id].add(user_id)

    return news_to_users


def compute_cf_score(
    candidate: str,
    history: list,
    news_to_users: Dict[str, Set[str]],
) -> float:
    """
    Cosine CF score: mean over history of sim(candidate, h) where
    sim uses user-overlap (co-click cosine similarity).
    """
    users_c = news_to_users.get(candidate)
    if not users_c or not history:
        return 0.0

    pop_c = len(users_c)
    total = 0.0
    count = 0
    for h in history:
        users_h = news_to_users.get(h)
        if not users_h:
            continue
        overlap = len(users_c & users_h)
        if overlap > 0:
            total += overlap / math.sqrt(pop_c * len(users_h))
        count += 1

    # Divide by len(history) (not count of known items) so scores are comparable
    # across users with different cold-start rates in the training index.
    return total / len(history) if history else 0.0


def compute_and_save_scores(
    train_behaviors_path: str,
    eval_behaviors_path: str,
    output_path: str,
    max_history: int,
) -> None:
    print("=" * 60)
    print("MIND Item-Based CF Score Computation")
    print("=" * 60)
    print(f"Train behaviors: {train_behaviors_path}")
    print(f"Eval behaviors:  {eval_behaviors_path}")
    print(f"Output:          {output_path}")
    print(f"Max history:     {max_history}")
    print()

    # Step 1: build inverted index from training data
    news_to_users = build_inverted_index(train_behaviors_path, max_history)
    print(f"Inverted index built: {len(news_to_users):,} news items")
    print()

    # Step 2: score dev/test impressions
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(eval_behaviors_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        f_out.write("impression_id\tnews_id\tcf_score\n")

        total = sum(1 for _ in open(eval_behaviors_path, encoding='utf-8'))
        f_in.seek(0)

        scored = 0
        for line in tqdm(f_in, total=total, desc="Scoring impressions"):
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            impression_id = parts[0]
            history_ids = parts[3].split()
            if max_history > 0:
                history_ids = history_ids[-max_history:]

            for imp in parts[4].split():
                if '-' not in imp:
                    continue
                news_id = imp.rsplit('-', 1)[0]
                score = compute_cf_score(news_id, history_ids, news_to_users)
                f_out.write(f"{impression_id}\t{news_id}\t{score:.6f}\n")
                scored += 1

    print(f"\nScored {scored:,} (impression, candidate) pairs → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute item-based CF scores for MIND")
    parser.add_argument("--train_behaviors", required=True)
    parser.add_argument("--eval_behaviors", required=True)
    parser.add_argument("--output", required=True, help="Output TSV: impression_id, news_id, cf_score")
    parser.add_argument("--max_history", type=int, default=30)
    args = parser.parse_args()

    compute_and_save_scores(
        train_behaviors_path=args.train_behaviors,
        eval_behaviors_path=args.eval_behaviors,
        output_path=args.output,
        max_history=args.max_history,
    )


if __name__ == "__main__":
    main()
