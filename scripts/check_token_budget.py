"""
Analyze per-section token usage in DOCA v8 prompts.

Shows where tokens are being spent: system prompt, interests, dislikes,
conversations, interactions, shown articles, candidate.

Usage:
    python scripts/check_token_budget.py --jsonl data/doca_v8/train.jsonl
    python scripts/check_token_budget.py --jsonl data/doca_v8/train.jsonl --max_feeds 5000
"""

import argparse
import json
import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from data import DOCAPointwiseSFTDataset


def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_sections(ds, sample):
    """Build prompt section by section to measure each part's token cost."""
    uc = sample['user_context']
    cand = sample['candidate']
    label = sample['label']

    sections = {}

    # System prompt
    sections['system_prompt'] = ds.SYSTEM_PROMPT + "\n\n"

    # 1. Interests
    interests = uc.get('interests', [])
    parts = []
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
    sections['interests'] = '\n'.join(parts) + '\n' if parts else ''

    # 1b. Interest rationale only (subset for analysis)
    rationale_parts = []
    for intr in interests:
        r = intr.get('rationale', '')
        if r:
            if len(r) > 200:
                r = r[:200] + "..."
            rationale_parts.append(f"   Reason: {r}")
    sections['interests_rationale_only'] = '\n'.join(rationale_parts) + '\n' if rationale_parts else ''

    # 2. Negative interests
    neg = uc.get('negative_interests', [])
    parts = []
    if neg:
        parts.append("\nDislikes:")
        for i, intr in enumerate(neg, 1):
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
    sections['dislikes'] = '\n'.join(parts) + '\n' if parts else ''

    # 3. Conversations
    conv = uc.get('conversation', [])
    parts = []
    if conv:
        parts.append("\nRecent conversations:")
        for gi, group in enumerate(conv, 1):
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
    sections['conversations'] = '\n'.join(parts) + '\n' if parts else ''

    # 4. Interactions
    interactions = uc.get('interactions', {})
    tu = interactions.get('thumbsUp', [])
    td = interactions.get('thumbsDown', [])
    clicks = interactions.get('clicks', [])
    parts = []
    if tu or td or clicks:
        if tu:
            parts.append("\nUser interactions (positive signals, thumbs-up):")
            for t in tu:
                parts.append(f"- {t}")
        if td:
            parts.append("\nUser interactions (negative signals, thumbs-down):")
            for t in td:
                parts.append(f"- {t}")
        if clicks:
            parts.append("\nUser interactions (click signals):")
            for t in clicks:
                parts.append(f"- {t}")
    sections['interactions'] = '\n'.join(parts) + '\n' if parts else ''

    # 5. Shown
    shown = uc.get('shown_10d', [])
    parts = []
    if shown:
        parts.append("\nRecently shown articles:")
        for item in shown:
            if isinstance(item, dict):
                title = item.get('title', '')
                date = item.get('event_time', '')[:10]
                parts.append(f'- "{title}" ({date})')
            else:
                parts.append(f'- "{item}"')
    sections['shown_10d'] = '\n'.join(parts) + '\n' if parts else ''

    # 6. Candidate
    parts = []
    parts.append("\nCandidate article:")
    parts.append(f"Title: {cand.get('title', '')}")
    summary = cand.get('summary', '')
    if summary:
        parts.append(f"Summary: {summary}")
    parts.append("\nWill this user click on this article? Answer:")
    sections['candidate'] = '\n'.join(parts)

    return sections


def main():
    parser = argparse.ArgumentParser(description="Analyze per-section token budget")
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max_feeds", type=int, default=5000)
    parser.add_argument("--neg_ratio", type=float, default=2.0)
    parser.add_argument("--max_interests", type=int, default=0)
    parser.add_argument("--max_conversation_msgs", type=int, default=15)
    parser.add_argument("--max_shown", type=int, default=10)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ds = DOCAPointwiseSFTDataset(
        jsonl_path=args.jsonl,
        tokenizer=tokenizer,
        max_len=8192,  # large enough to not truncate
        neg_ratio=args.neg_ratio,
        max_interests=args.max_interests,
        max_conversation_msgs=args.max_conversation_msgs,
        max_shown=args.max_shown,
        sample=args.max_feeds if args.max_feeds > 0 else -1,
    )

    section_names = [
        'system_prompt', 'interests', 'interests_rationale_only',
        'dislikes', 'conversations', 'interactions', 'shown_10d', 'candidate'
    ]
    section_tokens = {name: [] for name in section_names}
    total_tokens = []

    n = len(ds)
    print(f"Analyzing {n} samples...\n")

    for i in range(n):
        sample = ds.samples[i]
        sections = build_sections(ds, sample)

        t = 0
        for name in section_names:
            text = sections.get(name, '')
            tok = count_tokens(tokenizer, text) if text else 0
            section_tokens[name].append(tok)
            # Don't double-count rationale (it's a subset of interests)
            if name != 'interests_rationale_only':
                t += tok
        total_tokens.append(t)

    print(f"{'='*70}")
    print(f"PER-SECTION TOKEN BUDGET (n={n} samples)")
    print(f"{'='*70}")
    print(f"{'Section':<30} {'Mean':>6} {'Median':>7} {'P90':>6} {'P99':>6} {'Max':>6} {'% of total':>10}")
    print(f"{'-'*70}")

    total_mean = np.mean(total_tokens)
    for name in section_names:
        arr = np.array(section_tokens[name])
        nz = np.count_nonzero(arr)
        mean = arr.mean()
        pct = 100 * mean / total_mean if total_mean > 0 else 0
        label = name
        if name == 'interests_rationale_only':
            label = '  (rationale subset)'
        fill = f" [{nz}/{n} filled]" if nz < n else ""
        print(f"{label:<30} {mean:>6.0f} {int(np.median(arr)):>7} {int(np.percentile(arr, 90)):>6} "
              f"{int(np.percentile(arr, 99)):>6} {int(arr.max()):>6} {pct:>9.1f}%{fill}")

    print(f"{'-'*70}")
    arr = np.array(total_tokens)
    print(f"{'TOTAL':<30} {arr.mean():>6.0f} {int(np.median(arr)):>7} {int(np.percentile(arr, 90)):>6} "
          f"{int(np.percentile(arr, 99)):>6} {int(arr.max()):>6} {'100.0%':>10}")

    print(f"\n--- What-if analysis (token savings) ---")
    # Simulate dropping rationale
    savings_no_rationale = np.mean(section_tokens['interests_rationale_only'])
    print(f"Drop interest rationale:     save ~{savings_no_rationale:.0f} tok/sample ({100*savings_no_rationale/total_mean:.1f}%)")

    # Simulate capping interactions
    int_arr = np.array(section_tokens['interactions'])
    if int_arr.max() > 0:
        savings_no_interactions = int_arr.mean()
        print(f"Drop interactions entirely:  save ~{savings_no_interactions:.0f} tok/sample ({100*savings_no_interactions/total_mean:.1f}%)")

    # Simulate shorter conversations
    conv_arr = np.array(section_tokens['conversations'])
    if conv_arr.max() > 0:
        savings_no_conv = conv_arr.mean()
        print(f"Drop conversations entirely: save ~{savings_no_conv:.0f} tok/sample ({100*savings_no_conv/total_mean:.1f}%)")

    # Simulate no shown
    shown_arr = np.array(section_tokens['shown_10d'])
    if shown_arr.max() > 0:
        savings_no_shown = shown_arr.mean()
        print(f"Drop shown_10d entirely:     save ~{savings_no_shown:.0f} tok/sample ({100*savings_no_shown/total_mean:.1f}%)")

    sys_arr = np.array(section_tokens['system_prompt'])
    if sys_arr.max() > 0:
        savings_sys = sys_arr.mean()
        print(f"Drop system prompt:          save ~{savings_sys:.0f} tok/sample ({100*savings_sys/total_mean:.1f}%)")


if __name__ == "__main__":
    main()
