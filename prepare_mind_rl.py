#!/usr/bin/env python3
"""
Prepare MIND dataset for VERL RL training.

Converts MIND behaviors.tsv and news.tsv into VERL parquet format with:
- prompt: User history formatted as input
- ground_truth: Clicked news title
- extra_info: Dict containing candidates, labels, and other metadata

Usage:
    python prepare_mind_rl.py \
        --behaviors_path ../data/MIND/train/behaviors.tsv \
        --news_path ../data/MIND/train/news.tsv \
        --output_parquet ../data/MIND/train/rl_train.parquet \
        --max_history 50 \
        --use_abstract False

Author: MiniOneRec
"""

import argparse
import json
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm


def load_news(news_path: str, use_abstract: bool) -> Dict[str, Dict[str, str]]:
    """
    Load news articles from news.tsv.

    Args:
        news_path: Path to news.tsv file
        use_abstract: Whether to include abstracts in news text

    Returns:
        Dictionary mapping news_id -> dict(title, text, category)
    """
    news = {}
    with open(news_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            news_id = parts[0]
            category = parts[1] if len(parts) > 1 else ""
            title = parts[3]
            abstract = parts[4] if len(parts) > 4 else ""

            if use_abstract and abstract:
                text = f"{title} {abstract}"
            else:
                text = title
            news[news_id] = {
                "title": title,
                "text": text,
                "category": category,
            }

    return news


_OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_prompt(history_items: List[Dict[str, str]], candidates: List[Dict[str, str]], use_numeric: bool = True) -> List[Dict[str, str]]:
    """
    Build multiple-choice ranking prompt.

    Args:
        history_items: List of news dicts in user's reading history
        candidates: List of candidate news dicts
        use_numeric: If True, use numeric indices (1, 2, 3...) instead of letters (A, B, C...)
                     This allows for more than 26 candidates.

    Returns:
        Chat-style prompt list for VERL
    """
    prompt = "Role: You are a news recommendation assistant.\n"
    prompt += "Task: Select the most relevant news article for the user based on their reading history.\n\n"
    prompt += "User History:\n"
    if history_items:
        for i, item in enumerate(history_items, 1):
            category = f" ({item['category']})" if item.get("category") else ""
            prompt += f"{i}. [Title] {item['text']}{category}\n"
    else:
        prompt += "(No reading history)\n"

    prompt += "\n"
    prompt += "Candidate News Articles:\n"

    if use_numeric:
        for i, cand in enumerate(candidates, 1):
            category = f" ({cand['category']})" if cand.get("category") else ""
            prompt += f"{i}. [Title] {cand['text']}{category}\n"
        prompt += "\n"
        prompt += "Please analyze the user's interests and select the best article from the candidates above.\n"
        prompt += "Output only the candidate number.\n\n"
        prompt += "Answer:"
    else:
        # Legacy letter-based format (limited to 26 candidates)
        for i, cand in enumerate(candidates):
            letter = _OPTION_LETTERS[i] if i < len(_OPTION_LETTERS) else str(i + 1)
            category = f" ({cand['category']})" if cand.get("category") else ""
            prompt += f"{letter}. [Title] {cand['text']}{category}\n"
        prompt += "\n"
        prompt += "Please analyze the user's interests and select the best article from the candidates above.\n"
        prompt += "Output only the option letter.\n\n"
        prompt += "Answer:"

    return [{"role": "user", "content": prompt}]


def prepare_mind_for_rl(
    behaviors_path: str,
    news_path: str,
    output_parquet: str,
    max_history: int = 50,
    use_abstract: bool = False,
    max_samples: int = 0,
    min_candidates: int = 2,
    max_candidates: int = 100,
    use_numeric: bool = True
) -> None:
    """
    Convert MIND behaviors and news to VERL parquet format.

    Args:
        behaviors_path: Path to behaviors.tsv
        news_path: Path to news.tsv
        output_parquet: Output parquet file path
        max_history: Maximum number of history items to include
        use_abstract: Whether to include news abstracts
        max_samples: Maximum number of samples (0 = all)
        min_candidates: Minimum number of candidates per impression
        max_candidates: Maximum number of candidates per impression
        use_numeric: Use numeric indices (1,2,3...) instead of letters (A,B,C...)
                     This allows more than 26 candidates per impression.
    """
    print("=" * 60)
    print("MIND RL Data Preparation")
    print("=" * 60)
    print(f"Behaviors: {behaviors_path}")
    print(f"News: {news_path}")
    print(f"Output: {output_parquet}")
    print(f"Max history: {max_history}")
    print(f"Use abstracts: {use_abstract}")
    print(f"Candidate range: [{min_candidates}, {max_candidates}]")
    print("=" * 60)
    print()

    # Load news
    print("Loading news articles...")
    news = load_news(news_path, use_abstract)
    print(f"✓ Loaded {len(news)} news articles")
    print()

    # Process behaviors
    print("Processing behaviors...")
    data = []
    skipped = 0
    skipped_reasons = {
        'no_clicks': 0,
        'too_few_candidates': 0,
        'too_many_candidates': 0,
        'missing_news': 0
    }

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
            candidates = []
            labels = []
            clicked_titles = []

            for imp in impressions:
                if '-' not in imp:
                    continue
                news_id, label = imp.rsplit('-', 1)

                if news_id not in news:
                    skipped_reasons['missing_news'] += 1
                    continue

                candidates.append(news[news_id])
                label_int = int(label)
                labels.append(label_int)

                if label_int == 1:
                    clicked_titles.append(news[news_id]["text"])

            # Validation checks
            if not clicked_titles:
                skipped += 1
                skipped_reasons['no_clicks'] += 1
                continue

            if len(candidates) < min_candidates:
                skipped += 1
                skipped_reasons['too_few_candidates'] += 1
                continue

            if len(candidates) > max_candidates:
                # Truncate candidates (keep all clicked + random sample of non-clicked)
                clicked_indices = [i for i, l in enumerate(labels) if l == 1]
                non_clicked_indices = [i for i, l in enumerate(labels) if l == 0]

                # Keep all clicked
                keep_indices = clicked_indices.copy()

                # Sample non-clicked to fill up to max_candidates
                remaining_slots = max_candidates - len(clicked_indices)
                if remaining_slots > 0 and non_clicked_indices:
                    import random
                    random.seed(line_idx)  # Deterministic sampling
                    sampled_non_clicked = random.sample(
                        non_clicked_indices,
                        min(remaining_slots, len(non_clicked_indices))
                    )
                    keep_indices.extend(sampled_non_clicked)

                # Shuffle to avoid position bias
                random.shuffle(keep_indices)

                candidates = [candidates[i] for i in keep_indices]
                labels = [labels[i] for i in keep_indices]

            # Only apply 26-letter limitation if NOT using numeric format
            if not use_numeric and len(candidates) > len(_OPTION_LETTERS):
                # Limit to match multiple-choice letters.
                clicked_indices = [i for i, l in enumerate(labels) if l == 1]
                non_clicked_indices = [i for i, l in enumerate(labels) if l == 0]
                if len(clicked_indices) >= len(_OPTION_LETTERS):
                    import random
                    random.seed(line_idx + 9973)
                    keep_indices = random.sample(clicked_indices, len(_OPTION_LETTERS))
                else:
                    keep_indices = clicked_indices.copy()
                    remaining_slots = len(_OPTION_LETTERS) - len(clicked_indices)
                    if remaining_slots > 0 and non_clicked_indices:
                        import random
                        random.seed(line_idx + 9973)
                        sampled_non_clicked = random.sample(
                            non_clicked_indices,
                            min(remaining_slots, len(non_clicked_indices))
                        )
                        keep_indices.extend(sampled_non_clicked)
                random.shuffle(keep_indices)
                candidates = [candidates[i] for i in keep_indices]
                labels = [labels[i] for i in keep_indices]

            # Build prompt from history
            history_items = [news[nid] for nid in history_ids if nid in news]
            prompt = build_prompt(history_items, candidates, use_numeric=use_numeric)

            # Use first clicked title as ground truth
            ground_truth = clicked_titles[0]

            # Build extra_info dict
            # For numeric format, use string numbers; for letter format, use letters
            if use_numeric:
                option_letters = [str(i) for i in range(1, len(candidates) + 1)]
            else:
                option_letters = list(_OPTION_LETTERS[:len(candidates)])

            extra_info = {
                'candidates': [c["text"] for c in candidates],
                'labels': labels,
                'option_letters': option_letters,
                'use_numeric': use_numeric,
                'impression_id': impression_id,
                'user_id': user_id,
                'timestamp': timestamp,
                'num_history': len(history_items),
                'num_candidates': len(candidates),
                'num_clicked': sum(labels)
            }

            # VERL expects reward_model dict containing ground_truth
            data.append({
                'prompt': prompt,
                'data_source': 'mind',  # Required by VERL reward loop
                'reward_model': {
                    'ground_truth': ground_truth
                },
                'extra_info': extra_info  # Keep as dict for VERL compatibility
            })

    print()
    print("=" * 60)
    print("Processing Summary:")
    print("=" * 60)
    print(f"Total impressions processed: {total_lines}")
    print(f"Valid samples created: {len(data)}")
    print(f"Skipped samples: {skipped}")
    print(f"  - No clicks: {skipped_reasons['no_clicks']}")
    print(f"  - Too few candidates: {skipped_reasons['too_few_candidates']}")
    print(f"  - Missing news: {skipped_reasons['missing_news']}")
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
    # ground_truth is now inside reward_model dict
    avg_gt_len = df['reward_model'].apply(lambda x: len(x['ground_truth'])).mean()
    print(f"  Avg ground truth length: {avg_gt_len:.1f} chars")
    print()

    # Save parquet
    df.to_parquet(output_parquet, index=False, engine='pyarrow')
    print(f"✓ Saved {len(df)} samples to {output_parquet}")
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
    print(f"Prompt (truncated): {str(sample_prompt)[:200]}...")
    print(f"Ground truth: {reward_model['ground_truth']}")
    print(f"Num candidates: {extra['num_candidates']}")
    print(f"Num clicked: {extra['num_clicked']}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIND dataset for VERL RL training"
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
        default=50,
        help='Maximum number of history items (default: 50)'
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
        '--min_candidates',
        type=int,
        default=2,
        help='Minimum number of candidates per impression (default: 2)'
    )
    parser.add_argument(
        '--max_candidates',
        type=int,
        default=100,
        help='Maximum number of candidates per impression (default: 100)'
    )
    parser.add_argument(
        '--use_numeric',
        action='store_true',
        default=True,
        help='Use numeric indices (1,2,3...) instead of letters (A,B,C...). Enables >26 candidates. (default: True)'
    )
    parser.add_argument(
        '--use_letters',
        action='store_true',
        help='Use letter indices (A,B,C...) instead of numbers. Limits to 26 candidates max.'
    )

    args = parser.parse_args()

    # use_letters overrides use_numeric (for backward compatibility)
    use_numeric = not args.use_letters

    prepare_mind_for_rl(
        behaviors_path=args.behaviors_path,
        news_path=args.news_path,
        output_parquet=args.output_parquet,
        max_history=args.max_history,
        use_abstract=args.use_abstract,
        max_samples=args.max_samples,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        use_numeric=use_numeric
    )


if __name__ == '__main__':
    main()
