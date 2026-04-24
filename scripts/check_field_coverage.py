"""
Analyze field coverage in DOCA v8 JSONL data.

Checks how many feeds have non-empty interactions, conversations, interests, etc.
Helps diagnose whether new v8 signals are sparse/noisy.

Usage:
    python scripts/check_field_coverage.py --jsonl data/doca_v8/train.jsonl
    python scripts/check_field_coverage.py --jsonl data/doca_v8/train.jsonl --max_feeds 10000
"""

import argparse
import json
import numpy as np
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Check field coverage in DOCA JSONL")
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--max_feeds", type=int, default=-1, help="Limit feeds to check (-1=all)")
    args = parser.parse_args()

    total = 0
    has_interests = 0
    has_neg_interests = 0
    has_conversation = 0
    has_interactions = 0
    has_clicks = 0
    has_thumbs_up = 0
    has_thumbs_down = 0
    has_shown_10d = 0
    has_candidates = 0
    has_any_click = 0

    interest_counts = []
    neg_interest_counts = []
    conversation_counts = []       # number of conversation groups
    conversation_msg_counts = []   # total messages across groups
    click_counts = []
    thumbs_up_counts = []
    thumbs_down_counts = []
    shown_counts = []
    candidate_counts = []
    pos_candidate_counts = []

    with open(args.jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if args.max_feeds > 0 and total >= args.max_feeds:
                break

            feed = json.loads(line)
            total += 1

            # Interests
            interests = feed.get('interests', [])
            interest_counts.append(len(interests))
            if interests:
                has_interests += 1

            # Negative interests
            neg = feed.get('negative_interests', [])
            neg_interest_counts.append(len(neg))
            if neg:
                has_neg_interests += 1

            # Conversation (v8: grouped by conversation_id)
            conv = feed.get('conversation', [])
            conversation_counts.append(len(conv))
            msg_total = 0
            for group in conv:
                msgs = group.get('messages', []) if isinstance(group, dict) else []
                msg_total += len(msgs)
            conversation_msg_counts.append(msg_total)
            if conv:
                has_conversation += 1

            # Interactions (v8 new field)
            interactions = feed.get('interactions', {})
            clicks = interactions.get('clicks', [])
            tu = interactions.get('thumbsUp', [])
            td = interactions.get('thumbsDown', [])
            click_counts.append(len(clicks))
            thumbs_up_counts.append(len(tu))
            thumbs_down_counts.append(len(td))
            if clicks or tu or td:
                has_interactions += 1
            if clicks:
                has_clicks += 1
            if tu:
                has_thumbs_up += 1
            if td:
                has_thumbs_down += 1

            # Shown 10d
            shown = feed.get('shown_10d', [])
            shown_counts.append(len(shown))
            if shown:
                has_shown_10d += 1

            # Candidates
            candidates = feed.get('candidates', [])
            candidate_counts.append(len(candidates))
            pos = [c for c in candidates if c.get('is_clicked')]
            pos_candidate_counts.append(len(pos))
            if candidates:
                has_candidates += 1
            if pos:
                has_any_click += 1

    def pct(x):
        return f"{x}/{total} ({100*x/total:.1f}%)"

    def stats(arr, name):
        a = np.array(arr)
        if len(a) == 0:
            return
        nz = np.count_nonzero(a)
        print(f"  {name}:")
        print(f"    non-zero: {nz}/{total} ({100*nz/total:.1f}%)")
        print(f"    min={a.min()}, median={int(np.median(a))}, mean={a.mean():.1f}, "
              f"p90={int(np.percentile(a, 90))}, p99={int(np.percentile(a, 99))}, max={a.max()}")
        if nz < total:
            a_nz = a[a > 0]
            if len(a_nz) > 0:
                print(f"    (non-zero only) median={int(np.median(a_nz))}, mean={a_nz.mean():.1f}, max={a_nz.max()}")

    print(f"{'='*60}")
    print(f"FIELD COVERAGE REPORT")
    print(f"{'='*60}")
    print(f"File: {args.jsonl}")
    print(f"Total feeds: {total}\n")

    print(f"--- Feed-level field presence ---")
    print(f"  interests:          {pct(has_interests)}")
    print(f"  negative_interests: {pct(has_neg_interests)}")
    print(f"  conversation:       {pct(has_conversation)}")
    print(f"  interactions (any): {pct(has_interactions)}")
    print(f"    clicks:           {pct(has_clicks)}")
    print(f"    thumbsUp:         {pct(has_thumbs_up)}")
    print(f"    thumbsDown:       {pct(has_thumbs_down)}")
    print(f"  shown_10d:          {pct(has_shown_10d)}")
    print(f"  candidates:         {pct(has_candidates)}")
    print(f"  has_click (>=1 pos):{pct(has_any_click)}")

    print(f"\n--- Field size distributions ---")
    stats(interest_counts, "interests count")
    stats(neg_interest_counts, "negative_interests count")
    stats(conversation_counts, "conversation groups")
    stats(conversation_msg_counts, "conversation total messages")
    stats(click_counts, "interactions.clicks count")
    stats(thumbs_up_counts, "interactions.thumbsUp count")
    stats(thumbs_down_counts, "interactions.thumbsDown count")
    stats(shown_counts, "shown_10d count")
    stats(candidate_counts, "candidates count")
    stats(pos_candidate_counts, "clicked candidates count")

    # v8 新信号的交叉分析
    print(f"\n--- v8 new signals vs click rate ---")
    feeds_with_interactions = []
    feeds_without_interactions = []
    with open(args.jsonl, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if args.max_feeds > 0 and count >= args.max_feeds:
                break
            feed = json.loads(line)
            count += 1
            interactions = feed.get('interactions', {})
            has_any = bool(interactions.get('clicks') or interactions.get('thumbsUp') or interactions.get('thumbsDown'))
            candidates = feed.get('candidates', [])
            n_pos = sum(1 for c in candidates if c.get('is_clicked'))
            n_total = len(candidates)
            if n_total == 0:
                continue
            ctr = n_pos / n_total
            if has_any:
                feeds_with_interactions.append(ctr)
            else:
                feeds_without_interactions.append(ctr)

    if feeds_with_interactions:
        a = np.array(feeds_with_interactions)
        print(f"  Feeds WITH interactions:    n={len(a)}, mean CTR={a.mean():.4f}")
    if feeds_without_interactions:
        a = np.array(feeds_without_interactions)
        print(f"  Feeds WITHOUT interactions: n={len(a)}, mean CTR={a.mean():.4f}")


if __name__ == "__main__":
    main()
