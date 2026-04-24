"""
Evaluate DOCA pointwise click prediction using Azure OpenAI API.

Calls GPT model to predict Yes/No for each candidate, extracts
logprobs for scoring, and computes AUC/MRR/nDCG metrics.

Usage:
    python src/evaluate_doca_openai.py --eval_jsonl data/doca_v8/dev.jsonl --max_feeds 50

    # Custom endpoint/model
    python src/evaluate_doca_openai.py --eval_jsonl data/doca_v8/dev.jsonl \
        --endpoint https://... --deployment gpt-5.1 --api_key KEY

    # Multi-worker parallel (faster)
    python src/evaluate_doca_openai.py --eval_jsonl data/doca_v8/dev.jsonl \
        --max_feeds 200 --max_workers 8
"""

import argparse
import json
import math
import os
import sys
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- Metrics ----

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


def ndcg_score(labels, scores, k):
    ranked = sorted(zip(labels, scores), key=lambda x: -x[1])[:k]
    dcg = sum(l / np.log2(i + 2) for i, (l, _) in enumerate(ranked))
    ideal = sorted(labels, reverse=True)
    idcg = sum(label / np.log2(i + 2) for i, label in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


# ---- Prompt building (same as evaluate_doca_pointwise.py) ----

def build_doca_prompt(user_context, candidate, max_interests=0,
                      max_conversation_msgs=15, max_shown=10):
    parts = []

    interests = user_context.get('interests', [])
    if max_interests > 0:
        interests = interests[:max_interests]
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

    conversation = user_context.get('conversation', [])[:max_conversation_msgs]
    if conversation:
        parts.append("\nRecent conversations:")
        for msg in conversation:
            text = msg.get('text', '').strip()
            if text:
                if len(text) > 150:
                    text = text[:150] + "..."
                if msg.get('is_inline_curation'):
                    parts.append(f'- [CURATED] "{text}"')
                else:
                    parts.append(f'- "{text}"')

    shown = user_context.get('shown_10d', [])[:max_shown]
    if shown:
        parts.append("\nRecently shown articles:")
        for item in shown:
            if isinstance(item, dict):
                title = item.get('title', '')
                date = item.get('event_time', '')[:10]
                parts.append(f'- "{title}" ({date})')
            else:
                parts.append(f'- "{item}"')

    parts.append("\nCandidate article:")
    parts.append(f"Title: {candidate.get('title', '')}")
    summary = candidate.get('summary', '')
    if summary:
        parts.append(f"Summary: {summary}")

    parts.append("\nWill this user click on this article? Answer:")

    return '\n'.join(parts)


SYSTEM_PROMPT = (
    "You are a content recommendation assistant. "
    "Based on a user's interest profile, conversation history, and previously shown articles, "
    "predict whether they will click on a given article.\n\n"
    "Output format: Answer \"Yes\" or \"No\" followed by a confidence score 0-100. "
    "Example: 'Yes 85' or 'No 20'. Do not explain.\n\n"
    "Ranking guidelines (highest to lowest priority):\n"
    "1. Source signal priority: Inline Curation (user explicitly selected, strongest signal) "
    "> User Interaction (clicks/likes) > Chat History (inferred from messages).\n"
    "2. Interest strength: High (0.9-1.0) > Medium (0.8-0.9) > Exploratory (<0.8).\n"
    "3. Long-term interest relevance: How well does the article align with established interests?\n"
    "4. Short-term task relevance: How relevant is it to the user's recent activities and needs?\n"
    "5. Freshness: Prefer up-to-date content; consider if information might be outdated.\n"
    "6. Importance: How significant is this content for the user?\n"
    "7. Novelty: Prefer content the user hasn't seen recently (check shown articles).\n\n"
    "Also consider:\n"
    "- Articles matching disliked interests should NOT be clicked.\n"
    "- [CURATED] messages in conversations indicate the strongest user intent.\n"
    "- Interest rationale explains WHY something is an interest — use it to judge relevance."
)


def _parse_score(reply):
    """Parse 'Yes 85' or 'No 20' into a numeric score in [-1, 1]."""
    import re
    reply = reply.strip().lower()
    # Try to extract confidence number
    m = re.search(r'(yes|no)\s*(\d+)', reply)
    if m:
        answer = m.group(1)
        conf = min(int(m.group(2)), 100) / 100.0  # normalize to [0, 1]
        return conf if answer == 'yes' else -conf
    # Fallback: binary
    if reply.startswith('yes'):
        return 1.0
    elif reply.startswith('no'):
        return -1.0
    return 0.0


def score_candidate_openai(client, deployment, user_prompt, max_retries=3):
    """Call OpenAI API and extract Yes/No + confidence score."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_completion_tokens=200,
            )

            choice = response.choices[0]
            reply = choice.message.content.strip() if choice.message.content else ""
            return _parse_score(reply)

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                print(f"  API error after {max_retries} retries: {e}")
                return 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate DOCA with Azure OpenAI API")
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--endpoint", default="https://msncompanioneu2.cognitiveservices.azure.com/")
    parser.add_argument("--deployment", default="gpt-5.1")
    parser.add_argument("--api_key", default=None, help="Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var)")
    parser.add_argument("--api_version", default="2024-12-01-preview")
    parser.add_argument("--max_feeds", type=int, default=50, help="Max feeds with clicks to evaluate")
    parser.add_argument("--max_workers", type=int, default=4, help="Parallel API calls per feed")
    parser.add_argument("--max_interests", type=int, default=0)
    parser.add_argument("--max_conversation_msgs", type=int, default=15)
    parser.add_argument("--max_shown", type=int, default=10)
    parser.add_argument("--output_scores_file", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    from openai import AzureOpenAI

    api_key = args.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Provide --api_key or set AZURE_OPENAI_API_KEY env var")
        sys.exit(1)

    client = AzureOpenAI(
        api_version=args.api_version,
        azure_endpoint=args.endpoint,
        api_key=api_key,
    )

    # Quick connectivity test
    print("Testing API connectivity...")
    try:
        test_resp = client.chat.completions.create(
            model=args.deployment,
            messages=[{"role": "user", "content": "Say OK."}],
            max_completion_tokens=5,
        )
        print(f"API OK: model={test_resp.model}, reply={test_resp.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"API connection FAILED: {e}")
        sys.exit(1)

    print(f"\nEvaluating DOCA with Azure OpenAI ({args.deployment})...")
    print(f"  Eval file: {args.eval_jsonl}")
    print(f"  Max feeds: {args.max_feeds}")
    print(f"  Max workers: {args.max_workers}")

    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    raw_scores = []
    count = 0
    skipped = 0
    total_api_calls = 0

    with open(args.eval_jsonl, 'r', encoding='utf-8') as f:
        pbar = tqdm(total=args.max_feeds, desc="Evaluating", unit="feed")
        for line in f:
            line = line.strip()
            if not line:
                continue

            feed = json.loads(line)
            candidates = feed.get('candidates', [])
            if not candidates:
                continue

            labels = [1 if c.get('is_clicked') else 0 for c in candidates]
            if sum(labels) == 0:
                skipped += 1
                continue

            user_context = {
                'interests': feed.get('interests', []),
                'negative_interests': feed.get('negative_interests', []),
                'conversation': feed.get('conversation', []),
                'shown_10d': feed.get('shown_10d', []),
            }

            # Build prompts for all candidates
            prompts = [
                build_doca_prompt(user_context, cand,
                                  max_interests=args.max_interests,
                                  max_conversation_msgs=args.max_conversation_msgs,
                                  max_shown=args.max_shown)
                for cand in candidates
            ]

            # Score candidates in parallel
            scores = [0.0] * len(prompts)
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {
                    executor.submit(score_candidate_openai, client, args.deployment, p): i
                    for i, p in enumerate(prompts)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    scores[idx] = future.result()
                    total_api_calls += 1

            auc_val = auc_score(labels, scores)
            if auc_val is not None:
                aucs.append(auc_val)
            mrrs.append(mrr_score(labels, scores))
            ndcg5s.append(ndcg_score(labels, scores, 5))
            ndcg10s.append(ndcg_score(labels, scores, 10))

            if args.output_scores_file:
                scores_str = "\t".join(f"{s:.6f}" for s in scores)
                labels_str = "\t".join(str(l) for l in labels)
                raw_scores.append(f"{labels_str}\t|\t{scores_str}")

            count += 1
            pbar.update(1)
            pbar.set_postfix({
                'AUC': f'{np.mean(aucs):.4f}' if aucs else 'N/A',
                'MRR': f'{np.mean(mrrs):.4f}' if mrrs else 'N/A',
                'calls': total_api_calls,
            })

            if count >= args.max_feeds:
                break

        pbar.close()

    print(f"\n{'=' * 50}")
    print(f"DOCA Evaluation (Azure OpenAI: {args.deployment})")
    print(f"{'=' * 50}")
    print(f"Feeds evaluated: {count}")
    print(f"Feeds skipped (no clicks): {skipped}")
    print(f"Total API calls: {total_api_calls}")

    if aucs:
        print(f"\nAUC:     {np.mean(aucs):.4f}")
        print(f"MRR:     {np.mean(mrrs):.4f}")
        print(f"nDCG@5:  {np.mean(ndcg5s):.4f}")
        print(f"nDCG@10: {np.mean(ndcg10s):.4f}")
    else:
        print("No metrics computed")

    if args.output_scores_file and raw_scores:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_scores_file)), exist_ok=True)
        with open(args.output_scores_file, "w", encoding="utf-8") as f:
            for line in raw_scores:
                f.write(line + "\n")
        print(f"\nScores saved to: {args.output_scores_file}")


if __name__ == "__main__":
    main()
