"""
Download and prepare DOCA feed data from Databricks for pointwise SFT training.

Queries `mai_ws_discover.analytics.ods_doca_feed_grounded_v7_partitioned`,
extracts relevant fields, and saves as JSONL files (train/dev split).

Usage:
    python src/prepare_doca.py --output_dir data/doca --train_days 12 --dev_days 2

    # Custom date range
    python src/prepare_doca.py --output_dir data/doca --start_date 20260407 --end_date 20260420
"""

import argparse
import json
import os
import random
import subprocess
import shutil
from datetime import datetime


def get_databricks_token():
    """Get AAD token for Databricks via az CLI."""
    az_path = shutil.which('az')
    if az_path is None:
        for p in [
            r'C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd',
            r'C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin\az.cmd',
            os.path.expandvars(r'%LOCALAPPDATA%\Programs\Azure CLI\wbin\az.cmd'),
        ]:
            if os.path.exists(p):
                az_path = p
                break
    if az_path is None:
        raise RuntimeError("az CLI not found. Run 'az login' first.")

    token = subprocess.check_output(
        [az_path, 'account', 'get-access-token',
         '--resource', '2ff814a6-3304-4ab8-85cb-cd0e6f879c1d',
         '--query', 'accessToken', '-o', 'tsv'],
        text=True, shell=True
    ).strip()
    return token


def connect_databricks(token):
    from databricks import sql as dbsql
    return dbsql.connect(
        server_hostname='adb-3355567219430035.15.azuredatabricks.net',
        http_path='/sql/1.0/warehouses/9c266703e61b038d',
        access_token=token,
    )


def parse_json_field(raw):
    """Safely parse a JSON string field, return empty list on failure."""
    if raw is None:
        return []
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def process_candidates(cards_raw):
    """Extract (title, summary, is_clicked) from candidate_cards JSON."""
    cards = parse_json_field(cards_raw)
    results = []
    for card in cards:
        results.append({
            'itemid': card.get('itemid', ''),
            'title': card.get('title', ''),
            'summary': card.get('summary', ''),
            'is_clicked': bool(card.get('is_clicked', False)),
        })
    return results


def process_interests(interests_raw):
    """Extract structured interests."""
    interests = parse_json_field(interests_raw)
    results = []
    for i in interests:
        results.append({
            'name': i.get('name', ''),
            'keywords': i.get('keywords', []),
            'strength': i.get('strength', 0.0),
            'domain': i.get('domain', ''),
        })
    # Sort by strength descending
    results.sort(key=lambda x: x['strength'], reverse=True)
    return results


def process_conversation(conv_raw, max_messages=30):
    """Extract human messages from conversation history (most recent first)."""
    convs = parse_json_field(conv_raw)
    # Only keep human messages for user signal
    human_msgs = [
        {
            'text': m.get('text', ''),
            'createdAt': m.get('createdAt', ''),
        }
        for m in convs if m.get('author') == 'human'
    ]
    # Sort by time descending (most recent first), take last N
    human_msgs.sort(key=lambda x: x['createdAt'], reverse=True)
    return human_msgs[:max_messages]


def process_interactions(interactions_raw):
    """Extract interaction events (thumbsUp/thumbsDown)."""
    interactions = parse_json_field(interactions_raw)
    results = []
    for act in interactions:
        results.append({
            'event_time': act.get('event_time', ''),
            'type': act.get('clickScenario', ''),
        })
    return results


def process_shown(shown_raw, max_items=20):
    """Extract recently shown article titles."""
    shown = parse_json_field(shown_raw)
    results = []
    for s in shown:
        title = s.get('cardTitle', '')
        if title:
            results.append(title)
    return results[:max_items]


def process_row(row, cols):
    """Process a single DB row into a clean dict."""
    d = dict(zip(cols, row))
    return {
        'feed_id': d['feedId'],
        'user_id': d['user_id'],
        'ref_ts': str(d['ref_ts']),
        'bizdate': d['bizdate'],
        'candidates': process_candidates(d['candidate_cards']),
        'interests': process_interests(d['interests']),
        'negative_interests': process_interests(d['negative_interests']),
        'conversation': process_conversation(d['conversation']),
        'interactions_90d': process_interactions(d['interactions_90d']),
        'shown_10d': process_shown(d['shown_10d']),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare DOCA feed data for pointwise SFT")
    parser.add_argument("--output_dir", type=str, default="data/doca")
    parser.add_argument("--train_days", type=int, default=12,
                        help="Number of most recent days for training (rest for dev)")
    parser.add_argument("--dev_days", type=int, default=2,
                        help="Number of most recent days for dev set")
    parser.add_argument("--start_date", type=str, default=None,
                        help="Start bizdate (e.g. 20260407). If not set, auto-detect.")
    parser.add_argument("--end_date", type=str, default=None,
                        help="End bizdate (e.g. 20260420). If not set, auto-detect.")
    parser.add_argument("--max_messages", type=int, default=30,
                        help="Max human messages to keep from conversation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Connecting to Databricks...")
    token = get_databricks_token()
    conn = connect_databricks(token)
    cursor = conn.cursor()

    TABLE = "mai_ws_discover.analytics.ods_doca_feed_grounded_v7_partitioned"

    # Get available dates
    print("Fetching available dates...")
    cursor.execute(f"SELECT DISTINCT bizdate FROM {TABLE} ORDER BY bizdate")
    all_dates = [row[0] for row in cursor.fetchall()]
    print(f"Available dates: {all_dates}")

    # Filter date range
    if args.start_date:
        all_dates = [d for d in all_dates if d >= args.start_date]
    if args.end_date:
        all_dates = [d for d in all_dates if d <= args.end_date]

    if not all_dates:
        print("ERROR: No dates available in the specified range.")
        return

    # Split into train/dev by date
    # Dev = last N days, Train = rest
    dev_dates = set(all_dates[-args.dev_days:])
    train_dates = set(all_dates[:-args.dev_days])

    print(f"Train dates ({len(train_dates)}): {sorted(train_dates)}")
    print(f"Dev dates ({len(dev_dates)}): {sorted(dev_dates)}")

    # Download and process data
    fields = (
        "feedId, user_id, ref_ts, bizdate, "
        "candidate_cards, interests, negative_interests, "
        "conversation, interactions_90d, shown_10d"
    )

    train_path = os.path.join(args.output_dir, "train.jsonl")
    dev_path = os.path.join(args.output_dir, "dev.jsonl")

    train_count = 0
    dev_count = 0
    train_candidates = 0
    dev_candidates = 0
    train_clicks = 0
    dev_clicks = 0

    # Process each date
    for date in sorted(all_dates):
        is_dev = date in dev_dates
        split_name = "dev" if is_dev else "train"
        out_path = dev_path if is_dev else train_path

        print(f"\nProcessing {date} -> {split_name}...")
        cursor.execute(f"SELECT {fields} FROM {TABLE} WHERE bizdate='{date}'")
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        print(f"  Fetched {len(rows)} feeds")

        mode = 'a' if os.path.exists(out_path) and date != sorted(all_dates)[0] else 'w'
        # For dev, check if it's the first dev date
        if is_dev and date == sorted(dev_dates)[0]:
            mode = 'w'
        # For train, check if it's the first train date
        if not is_dev and date == sorted(train_dates)[0]:
            mode = 'w'

        with open(out_path, mode, encoding='utf-8') as f:
            for row in rows:
                record = process_row(row, cols)
                n_cands = len(record['candidates'])
                n_clicks = sum(1 for c in record['candidates'] if c['is_clicked'])

                # Skip feeds with no candidates
                if n_cands == 0:
                    continue

                f.write(json.dumps(record, ensure_ascii=False) + '\n')

                if is_dev:
                    dev_count += 1
                    dev_candidates += n_cands
                    dev_clicks += n_clicks
                else:
                    train_count += 1
                    train_candidates += n_cands
                    train_clicks += n_clicks

    cursor.close()
    conn.close()

    # Summary
    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print("=" * 50)
    print(f"\nTrain: {train_path}")
    print(f"  Feeds: {train_count}")
    print(f"  Total candidates: {train_candidates}")
    print(f"  Total clicks: {train_clicks}")
    print(f"  CTR: {train_clicks / max(train_candidates, 1):.4f}")

    print(f"\nDev: {dev_path}")
    print(f"  Feeds: {dev_count}")
    print(f"  Total candidates: {dev_candidates}")
    print(f"  Total clicks: {dev_clicks}")
    print(f"  CTR: {dev_clicks / max(dev_candidates, 1):.4f}")

    # Save metadata
    meta = {
        'train_dates': sorted(train_dates),
        'dev_dates': sorted(dev_dates),
        'train_feeds': train_count,
        'dev_feeds': dev_count,
        'train_candidates': train_candidates,
        'dev_candidates': dev_candidates,
        'train_clicks': train_clicks,
        'dev_clicks': dev_clicks,
        'max_messages': args.max_messages,
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
