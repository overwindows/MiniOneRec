#!/usr/bin/env python3
"""
Prepare MIND dataset for VERL RL training with POINTWISE format.

Converts MIND behaviors.tsv and news.tsv into VERL parquet format with:
- prompt: User history + single candidate with Yes/No question
- ground_truth: "Yes" or "No"
- extra_info: Dict containing label, impression_id, etc.

Key difference from prepare_mind_rl.py (listwise):
- Creates N samples per impression (one per candidate)
- Each sample asks "Will user read this article? Yes/No"
- Matches the format used in sft_mind_pointwise_ds.py

Usage:
    python prepare_mind_rl_pointwise.py \
        --behaviors_path ../data/MIND/train/behaviors.tsv \
        --news_path ../data/MIND/train/news.tsv \
        --output_parquet ../data/MIND/train/rl_pointwise_train.parquet \
        --max_history 30 \
        --neg_ratio 1.0

Author: MiniOneRec
"""

import argparse
import random
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from mind_utils import load_news, build_pointwise_prompt as _build_prompt


def build_pointwise_prompt(history_items: List[Dict[str, str]], candidate: Dict[str, str]) -> List[Dict[str, str]]:
    """Build pointwise prompt in VERL chat format."""
    prompt_text = _build_prompt(history_items, candidate)
    return [{"role": "user", "content": prompt_text}]


def prepare_mind_pointwise_for_rl(
    behaviors_path: str,
    news_path: str,
    output_parquet: str,
    max_history: int = 30,
    neg_ratio: float = 1.0,
    use_abstract: bool = False,
    max_samples: int = 0,
    seed: int = 42,
) -> None:
    """
    Convert MIND behaviors and news to VERL parquet format for pointwise RL.

    Args:
        behaviors_path: Path to behaviors.tsv
        news_path: Path to news.tsv
        output_parquet: Output parquet file path
        max_history: Maximum number of history items to include
        neg_ratio: Ratio of negatives to positives per impression
        use_abstract: Whether to include news abstracts
        max_samples: Maximum number of samples (0 = all)
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("MIND Pointwise RL Data Preparation")
    print("=" * 60)
    print(f"Behaviors: {behaviors_path}")
    print(f"News: {news_path}")
    print(f"Output: {output_parquet}")
    print(f"Max history: {max_history}")
    print(f"Neg ratio: {neg_ratio}")
    print(f"Use abstracts: {use_abstract}")
    print(f"Seed: {seed}")
    print("=" * 60)
    print()

    random.seed(seed)

    # Load news
    print("Loading news articles...")
    news = load_news(news_path, use_abstract)
    print(f"Loaded {len(news)} news articles")
    print()

    # Process behaviors
    print("Processing behaviors...")
    data = []
    skipped_impressions = 0
    total_positives = 0
    total_negatives = 0

    with open(behaviors_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(behaviors_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(tqdm(f, total=total_lines, desc="Processing impressions")):
            if max_samples > 0 and len(data) >= max_samples:
                break

            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            impression_id = parts[0]
            user_id = parts[1]
            timestamp = parts[2]
            history_ids = parts[3].split()[-max_history:]  # Take last N history items
            impressions = parts[4].split()

            # Parse candidates and labels
            positives = []
            negatives = []

            for imp in impressions:
                if '-' not in imp:
                    continue
                news_id, label = imp.rsplit('-', 1)

                if news_id not in news:
                    continue

                if int(label) == 1:
                    positives.append(news_id)
                else:
                    negatives.append(news_id)

            # Skip if no positives
            if not positives:
                skipped_impressions += 1
                continue

            # Build history
            history_items = [news[nid] for nid in history_ids if nid in news]

            # Sample negatives based on neg_ratio
            num_neg_to_sample = int(len(positives) * neg_ratio)
            if num_neg_to_sample > 0 and negatives:
                # Hard negative sampling (50% same category, 50% different)
                pos_categories = set(news[pid].get('category', '') for pid in positives)

                hard_negs = [nid for nid in negatives if news[nid].get('category', '') in pos_categories]
                easy_negs = [nid for nid in negatives if news[nid].get('category', '') not in pos_categories]

                num_hard = num_neg_to_sample // 2
                num_easy = num_neg_to_sample - num_hard

                sampled_negs = []
                if hard_negs:
                    sampled_negs.extend(random.sample(hard_negs, min(num_hard, len(hard_negs))))
                if len(sampled_negs) < num_neg_to_sample and easy_negs:
                    remaining = num_neg_to_sample - len(sampled_negs)
                    sampled_negs.extend(random.sample(easy_negs, min(remaining, len(easy_negs))))

                negatives = sampled_negs

            # Create samples for all positives
            for pos_id in positives:
                prompt = build_pointwise_prompt(history_items, news[pos_id])
                data.append({
                    'prompt': prompt,
                    'data_source': 'mind_pointwise',
                    'reward_model': {
                        'ground_truth': 'Yes'
                    },
                    'extra_info': {
                        'label': 1,
                        'news_id': pos_id,
                        'impression_id': impression_id,
                        'user_id': user_id,
                        'timestamp': timestamp,
                        'num_history': len(history_items),
                        'candidate_text': news[pos_id]['text'],
                        'candidate_category': news[pos_id].get('category', ''),
                    }
                })
                total_positives += 1

            # Create samples for sampled negatives
            for neg_id in negatives:
                prompt = build_pointwise_prompt(history_items, news[neg_id])
                data.append({
                    'prompt': prompt,
                    'data_source': 'mind_pointwise',
                    'reward_model': {
                        'ground_truth': 'No'
                    },
                    'extra_info': {
                        'label': 0,
                        'news_id': neg_id,
                        'impression_id': impression_id,
                        'user_id': user_id,
                        'timestamp': timestamp,
                        'num_history': len(history_items),
                        'candidate_text': news[neg_id]['text'],
                        'candidate_category': news[neg_id].get('category', ''),
                    }
                })
                total_negatives += 1

    # Shuffle all samples
    random.shuffle(data)

    print()
    print("=" * 60)
    print("Processing Summary:")
    print("=" * 60)
    print(f"Total impressions processed: {total_lines}")
    print(f"Skipped impressions (no positives): {skipped_impressions}")
    print(f"Total samples created: {len(data)}")
    print(f"  - Positive samples (Yes): {total_positives}")
    print(f"  - Negative samples (No): {total_negatives}")
    print(f"  - Ratio: {total_negatives / max(total_positives, 1):.2f}")
    print("=" * 60)
    print()

    if not data:
        print("ERROR: No valid samples created!")
        return

    # Convert to DataFrame and save
    print(f"Saving to parquet: {output_parquet}")
    df = pd.DataFrame(data)

    # Print statistics
    print()
    print("Dataset Statistics:")
    print(f"  Total samples: {len(df)}")

    def _prompt_len(prompt_value):
        if isinstance(prompt_value, list) and prompt_value:
            content = prompt_value[0].get("content", "")
            return len(content)
        return len(str(prompt_value))

    print(f"  Avg prompt length: {df['prompt'].apply(_prompt_len).mean():.1f} chars")
    print()

    # Save parquet
    df.to_parquet(output_parquet, index=False, engine='pyarrow')
    print(f"Saved {len(df)} samples to {output_parquet}")
    print()

    # Print sample
    print("Sample data:")
    print("-" * 60)
    sample = df.iloc[0]
    extra = sample['extra_info']
    reward_model = sample['reward_model']
    sample_prompt = sample['prompt']
    if isinstance(sample_prompt, list) and sample_prompt:
        sample_prompt = sample_prompt[0].get("content", "")
    print(f"Prompt (truncated): {str(sample_prompt)[:300]}...")
    print(f"Ground truth: {reward_model['ground_truth']}")
    print(f"Label: {extra['label']}")
    print(f"Candidate: {extra['candidate_text'][:50]}...")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIND dataset for VERL RL training (pointwise format)"
    )
    parser.add_argument(
        '--behaviors_path',
        required=True,
        help='Path to behaviors.tsv file'
    )
    parser.add_argument(
        '--news_path',
        required=True,
        help='Path to news.tsv file'
    )
    parser.add_argument(
        '--output_parquet',
        required=True,
        help='Output parquet file path'
    )
    parser.add_argument(
        '--max_history',
        type=int,
        default=30,
        help='Maximum number of history items (default: 30)'
    )
    parser.add_argument(
        '--neg_ratio',
        type=float,
        default=1.0,
        help='Ratio of negatives to positives per impression (default: 1.0)'
    )
    parser.add_argument(
        '--use_abstract',
        action='store_true',
        help='Include news abstracts in text'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=0,
        help='Maximum number of samples to process (0 = all, default: 0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    prepare_mind_pointwise_for_rl(
        behaviors_path=args.behaviors_path,
        news_path=args.news_path,
        output_parquet=args.output_parquet,
        max_history=args.max_history,
        neg_ratio=args.neg_ratio,
        use_abstract=args.use_abstract,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
