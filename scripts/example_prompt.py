"""Show example prompts for the DOCA pointwise SFT pipeline.

Usage:
    python scripts/example_prompt.py
    python scripts/example_prompt.py --from_jsonl data/doca/train.jsonl --index 0
"""

import json
import sys
import os
import argparse


# ---------------------------------------------------------------------------
# Inline prompt builder (mirrors DOCAPointwiseSFTDataset._build_prompt)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a content recommendation assistant. "
    "Based on a user's interest profile, conversation history, interaction history, "
    "and previously shown articles, predict whether they will click on a given article.\n\n"
    "Output format: Answer ONLY \"Yes\" or \"No\". Do not explain.\n\n"
    "Ranking guidelines (highest to lowest priority):\n"
    "1. Source signal priority: Inline Curation (user explicitly selected, strongest signal) "
    "> User Interaction (clicks/likes) > Chat History (inferred from messages).\n"
    "2. Interest strength: High (0.9-1.0) > Medium (0.8-0.9) > Exploratory (<0.8).\n"
    "3. Long-term interest relevance: How well does the article align with established interests?\n"
    "4. Short-term task relevance: How relevant is it to the user's recent activities and needs?\n"
    "5. USER_INTERACTION affinity: topical overlap with thumbs-up, clicked, or "
    "thumbs-down card titles in user interactions.\n"
    "6. Negative-interest match: whether the candidate matches a topic the user has "
    "shown disinterest in (disliked interests or thumbs-down cards).\n"
    "7. Freshness: Prefer up-to-date content; consider if information might be outdated.\n"
    "8. Importance: How significant is this content for the user?\n"
    "9. Novelty: Prefer content the user hasn't seen recently (check shown articles).\n\n"
    "Also consider:\n"
    "- Articles matching disliked interests should NOT be clicked.\n"
    "- [CURATED] messages in conversations indicate the strongest user intent.\n"
    "- Interest rationale explains WHY something is an interest — use it to judge relevance."
)


def build_prompt(user_context, candidate, label):
    """Build pointwise prompt — same logic as DOCAPointwiseSFTDataset._build_prompt."""
    parts = []

    # 1. User interests
    interests = user_context.get('interests', [])
    if interests:
        parts.append("User interests:")
        for i, intr in enumerate(interests, 1):
            name = intr.get('name', '')
            strength = intr.get('strength', 0)
            domain = intr.get('domain', '')
            sources = ', '.join(intr.get('sources', []))
            intent = intr.get('intent', '')
            classification = intr.get('classification', '')
            status = intr.get('status', '')
            keywords = ', '.join(intr.get('keywords', [])[:5])
            line = f"{i}. {name} (strength={strength:.2f}"
            if domain:
                line += f", {domain}"
            if sources:
                line += f", source: {sources}"
            line += ")"
            meta = []
            if classification:
                meta.append(classification)
            if intent:
                meta.append(f"intent: {intent}")
            if status:
                meta.append(status)
            if meta:
                line += f" [{', '.join(meta)}]"
            line += f" - {keywords}"
            parts.append(line)
            rationale = intr.get('rationale', '')
            if rationale:
                if len(rationale) > 200:
                    rationale = rationale[:200] + "..."
                parts.append(f"   Reason: {rationale}")

    # 2. Negative interests
    neg_interests = user_context.get('negative_interests', [])
    if neg_interests:
        parts.append("\nDislikes:")
        for i, intr in enumerate(neg_interests, 1):
            name = intr.get('name', '')
            keywords = ', '.join(intr.get('keywords', [])[:5])
            sources = ', '.join(intr.get('sources', []))
            line = f"{i}. {name}"
            if sources:
                line += f" (source: {sources})"
            line += f" - {keywords}"
            parts.append(line)
            rationale = intr.get('rationale', '')
            if rationale:
                if len(rationale) > 200:
                    rationale = rationale[:200] + "..."
                parts.append(f"   Reason: {rationale}")

    # 3. Conversation history (grouped by conversation_id, user+assistant)
    conversation = user_context.get('conversation', [])
    if conversation:
        parts.append("\nRecent conversations:")
        for gi, group in enumerate(conversation, 1):
            started_at = (group.get('started_at') or '')[:16].replace('T', ' ')
            head = f"  Conversation {gi}"
            if started_at:
                head += f" ({started_at})"
            head += ":"
            parts.append(head)
            for msg in group.get('messages', []):
                text = (msg.get('text') or '').strip()
                if not text:
                    continue
                if len(text) > 150:
                    text = text[:150] + "..."
                author = msg.get('author', '?')
                role = 'user' if author in ('human', 'user') else 'assistant'
                if msg.get('is_inline_curation'):
                    parts.append(f"    [{role}] [CURATED] {text}")
                else:
                    parts.append(f"    [{role}] {text}")

    # 4. User interactions
    interactions = user_context.get('interactions', {})
    thumbs_up = interactions.get('thumbsUp', [])
    thumbs_down = interactions.get('thumbsDown', [])
    clicks = interactions.get('clicks', [])
    if thumbs_up or thumbs_down or clicks:
        if thumbs_up:
            parts.append("\nUser interactions (positive signals, thumbs-up):")
            for t in thumbs_up:
                parts.append(f"- {t}")
        if thumbs_down:
            parts.append("\nUser interactions (negative signals, thumbs-down):")
            for t in thumbs_down:
                parts.append(f"- {t}")
        if clicks:
            parts.append("\nUser interactions (click signals):")
            for t in clicks:
                parts.append(f"- {t}")

    # 5. Shown 10d
    shown = user_context.get('shown_10d', [])
    if shown:
        parts.append("\nRecently shown articles:")
        for item in shown:
            if isinstance(item, dict):
                title = item.get('title', '')
                date = item.get('event_time', '')[:10]
                parts.append(f'- "{title}" ({date})')
            else:
                parts.append(f'- "{item}"')

    # 6. Candidate
    parts.append("\nCandidate article:")
    parts.append(f"Title: {candidate.get('title', '')}")
    summary = candidate.get('summary', '')
    if summary:
        parts.append(f"Summary: {summary}")

    parts.append("\nWill this user click on this article? Answer:")

    prompt = '\n'.join(parts)
    target = " Yes" if label == 1 else " No"
    return prompt, target


def make_mock_sample():
    """Build a realistic mock sample for demonstration."""
    return {
        'user_context': {
            'interests': [
                {
                    'name': 'Artificial Intelligence',
                    'strength': 0.95,
                    'domain': 'Technology',
                    'sources': ['chat_history', 'user_interactions'],
                    'intent': 'stay informed',
                    'classification': 'topic',
                    'status': 'stable',
                    'keywords': ['LLM', 'GPT', 'machine learning', 'neural networks'],
                    'rationale': 'User frequently discusses AI models and LLM capabilities in chat',
                },
                {
                    'name': 'Cryptocurrency',
                    'strength': 0.82,
                    'domain': 'Finance',
                    'sources': ['user_interactions'],
                    'intent': 'explore',
                    'classification': 'topic',
                    'status': 'emerging',
                    'keywords': ['Bitcoin', 'Ethereum', 'DeFi', 'crypto regulation'],
                },
                {
                    'name': 'Space Exploration',
                    'strength': 0.70,
                    'domain': 'Science',
                    'sources': ['chat_history'],
                    'intent': 'discover',
                    'classification': 'topic',
                    'status': 'ephemeral',
                    'keywords': ['NASA', 'SpaceX', 'Mars', 'rocket'],
                },
            ],
            'negative_interests': [
                {
                    'name': 'Celebrity Gossip',
                    'keywords': ['paparazzi', 'scandal', 'breakup'],
                    'sources': ['user_interactions'],
                    'domain': 'Entertainment',
                    'rationale': 'User thumbs-downed multiple celebrity articles',
                },
            ],
            'conversation': [
                {
                    'id': 'conv-abc123',
                    'started_at': '2026-04-20T14:30:00Z',
                    'messages': [
                        {'author': 'human', 'text': "What's the latest on GPT-5 benchmarks?", 'createdAt': '2026-04-20T14:30:00Z'},
                        {'author': 'bot', 'text': 'GPT-5 has shown significant improvements in reasoning tasks, with a 15% gain on MMLU and 22% on HumanEval compared to GPT-4o.', 'createdAt': '2026-04-20T14:30:15Z'},
                        {'author': 'human', 'text': 'How does it compare to Claude?', 'createdAt': '2026-04-20T14:31:00Z'},
                    ],
                },
                {
                    'id': 'conv-def456',
                    'started_at': '2026-04-19T09:15:00Z',
                    'messages': [
                        {'author': 'human', 'text': 'Tell me about Bitcoin ETF approval status', 'createdAt': '2026-04-19T09:15:00Z', 'is_inline_curation': True},
                        {'author': 'bot', 'text': 'The SEC approved three new spot Bitcoin ETFs last week, bringing the total to 15 approved funds.', 'createdAt': '2026-04-19T09:15:20Z'},
                    ],
                },
            ],
            'interactions': {
                'clicks': ['OpenAI Releases GPT-5 API', 'Ethereum Staking Rewards Hit New High'],
                'thumbsUp': ['AI Safety Summit 2026 Outcomes'],
                'thumbsDown': ['Top 10 Celebrity Breakups This Month'],
            },
            'shown_10d': [
                {'title': 'OpenAI Releases GPT-5 API', 'event_time': '2026-04-19T10:00:00Z'},
                {'title': 'Bitcoin Surges Past $100K', 'event_time': '2026-04-18T08:30:00Z'},
                {'title': 'NASA Artemis III Update', 'event_time': '2026-04-17T12:00:00Z'},
                {'title': 'AI Safety Summit 2026 Outcomes', 'event_time': '2026-04-16T09:45:00Z'},
            ],
        },
        'candidate': {
            'title': 'New AI Regulation Framework Proposed by EU',
            'summary': 'The European Union has unveiled a comprehensive regulatory framework for artificial intelligence, covering model transparency, safety benchmarks, and deployment guidelines for high-risk applications.',
        },
        'label': 1,
        'user_flight_ids': 'discover-rk-ura,discover-convstarter,discover-picks-ntfy',
    }


def show_prompt(sample, label_name="example"):
    """Print the full prompt for a sample."""
    prompt, target = build_prompt(sample['user_context'], sample['candidate'], sample['label'])

    print("=" * 80)
    print(f"[{label_name}] label={sample['label']} target='{target}'")
    if sample.get('user_flight_ids'):
        print(f"[{label_name}] flights={sample['user_flight_ids']}")
    print("=" * 80)

    print("\n--- SYSTEM PROMPT ---")
    print(SYSTEM_PROMPT)

    print("\n--- USER PROMPT ---")
    print(prompt)

    print(f"\n--- TARGET ---")
    print(f"'{target}'")
    print("=" * 80)


def show_from_jsonl(jsonl_path, index=0):
    """Load a real sample from JSONL and show its prompt."""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == index:
                feed = json.loads(line.strip())
                break
        else:
            print(f"ERROR: index {index} out of range")
            return

    # Expand first candidate
    candidates = feed.get('candidates', [])
    if not candidates:
        print("ERROR: no candidates in this feed")
        return

    user_context = {
        'interests': feed.get('interests', []),
        'negative_interests': feed.get('negative_interests', []),
        'conversation': feed.get('conversation', []),
        'interactions': feed.get('interactions', {}),
        'shown_10d': feed.get('shown_10d', []),
    }

    print(f"Feed: {feed.get('feed_id', '?')}  user: {feed.get('user_id', '?')}  "
          f"bizdate: {feed.get('bizdate', '?')}  candidates: {len(candidates)}")
    print(f"Flights: {feed.get('user_flight_ids', '')[:100]}...")
    print()

    # Show first clicked candidate, or first candidate if no clicks
    clicked = [c for c in candidates if c.get('is_clicked')]
    cand = clicked[0] if clicked else candidates[0]
    label = 1 if cand.get('is_clicked') else 0

    sample = {
        'user_context': user_context,
        'candidate': cand,
        'label': label,
        'user_flight_ids': feed.get('user_flight_ids', ''),
    }
    show_prompt(sample, label_name=f"jsonl[{index}]")


def main():
    parser = argparse.ArgumentParser(description="Show example DOCA prompts")
    parser.add_argument("--from_jsonl", type=str, default="",
                        help="Path to JSONL file to load a real sample from")
    parser.add_argument("--index", type=int, default=0,
                        help="Index of the feed in the JSONL file")
    args = parser.parse_args()

    if args.from_jsonl:
        show_from_jsonl(args.from_jsonl, index=args.index)
    else:
        print("Using mock sample (pass --from_jsonl to use real data)\n")
        sample = make_mock_sample()
        show_prompt(sample)


if __name__ == "__main__":
    main()
