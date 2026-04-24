"""
Evaluate DOCA models trained with point-wise SFT (Yes/No classification).

Scores each candidate independently by computing log P("Yes") - log P("No")
and uses those scores to rank candidates within each feed impression.

Metrics: AUC, MRR, nDCG@5, nDCG@10 (per-feed, then averaged).

Usage:
    python src/evaluate_doca_pointwise.py \\
        --model_path output_dir/sft_doca_pointwise_*/final_checkpoint \\
        --eval_jsonl data/doca_v8/dev.jsonl \\
        --flash_attn \\
        --max_feeds 1000

    # Quick mode (500 feeds)
    python src/evaluate_doca_pointwise.py --model_path ... --eval_jsonl ... --quick
"""

import argparse
import json
import os
import sys
import random
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DOCAPointwiseSFTDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ---- Metrics (same as MIND) ----

def auc_score(labels, scores):
    """AUC for a single impression."""
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
    """MRR for a single impression."""
    ranked = sorted(zip(labels, scores), key=lambda x: -x[1])
    for i, (label, _) in enumerate(ranked):
        if label == 1:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_score(labels, scores, k):
    """nDCG@k for a single impression."""
    ranked = sorted(zip(labels, scores), key=lambda x: -x[1])
    dcg = sum(
        label / np.log2(i + 2)
        for i, (label, _) in enumerate(ranked[:k])
    )
    ideal = sorted(labels, reverse=True)
    idcg = sum(
        label / np.log2(i + 2)
        for i, label in enumerate(ideal[:k])
    )
    if idcg == 0:
        return 0.0
    return dcg / idcg


def build_doca_prompt(user_context, candidate, max_interests=0, max_conversation_msgs=15,
                      max_shown=10):
    """Build prompt matching DOCAPointwiseSFTDataset._build_prompt format."""
    parts = []

    # 1. User interests (with sources, intent, classification, status, rationale)
    interests = user_context.get('interests', [])[:max_interests] if max_interests > 0 else user_context.get('interests', [])
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

    # 2. Negative interests (with sources and rationale)
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
    conversation = user_context.get('conversation', [])[:max_conversation_msgs]
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

    # 4. User interactions (clicks, thumbsUp, thumbsDown from interactions_90d)
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

    # 6. Candidate
    parts.append("\nCandidate article:")
    parts.append(f"Title: {candidate.get('title', '')}")
    summary = candidate.get('summary', '')
    if summary:
        parts.append(f"Summary: {summary}")

    parts.append("\nWill this user click on this article? Answer:")

    return '\n'.join(parts)


def batch_score_candidates(
    model, tokenizer, user_context, candidates,
    device, yes_token_id, no_token_id,
    batch_size=8, use_chat_template=False,
    max_interests=0, max_conversation_msgs=15,
    max_shown=10,
    temperature=1.0,
) -> List[float]:
    """Score multiple candidates in batches."""

    prompts = []
    for cand in candidates:
        prompt = build_doca_prompt(
            user_context, cand,
            max_interests=max_interests,
            max_conversation_msgs=max_conversation_msgs,
            max_shown=max_shown,
        )
        if use_chat_template:
            messages = [
                {"role": "system", "content": DOCAPointwiseSFTDataset.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompts.append(prompt)

    all_prompt_ids = [
        tokenizer.encode(p, add_special_tokens=(not use_chat_template))
        for p in prompts
    ]

    scores = []
    for batch_start in range(0, len(all_prompt_ids), batch_size):
        batch_ids = all_prompt_ids[batch_start:batch_start + batch_size]
        max_len = max(len(ids) for ids in batch_ids)
        padded_ids = []
        attention_masks = []

        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            yes_log_probs = log_probs[:, yes_token_id]
            no_log_probs = log_probs[:, no_token_id]
            batch_scores = (yes_log_probs - no_log_probs).cpu().tolist()
            scores.extend(batch_scores)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_jsonl", required=True, help="Path to dev.jsonl from prepare_doca.py")
    parser.add_argument("--max_feeds", type=int, default=0, help="Max feeds to evaluate (0=all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for scoring candidates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention 2")
    parser.add_argument("--use_chat_template", action="store_true", help="Use chat template for instruct models")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 500 feeds")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_interests", type=int, default=0, help="Max interests to include (0=all)")
    parser.add_argument("--max_conversation_msgs", type=int, default=15)
    parser.add_argument("--max_shown", type=int, default=10)
    parser.add_argument("--output_scores_file", help="Output raw scores file")
    args = parser.parse_args()

    if args.quick and args.max_feeds == 0:
        args.max_feeds = 500

    set_seed(args.seed)

    print(f"Loading model from: {args.model_path}")
    from pathlib import Path
    local_only = os.path.isabs(args.model_path) or os.path.isdir(args.model_path)
    model_path_arg = Path(args.model_path) if local_only else args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path_arg, trust_remote_code=True, local_files_only=local_only)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if local_only:
        model_kwargs["local_files_only"] = True
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")

    model = AutoModelForCausalLM.from_pretrained(model_path_arg, **model_kwargs)
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    # Get Yes/No token IDs
    yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode(" No", add_special_tokens=False)
    if len(yes_tokens) != 1:
        print(f"WARNING: ' Yes' tokenized into {len(yes_tokens)} tokens {yes_tokens}; using last token.")
    if len(no_tokens) != 1:
        print(f"WARNING: ' No' tokenized into {len(no_tokens)} tokens {no_tokens}; using last token.")
    yes_token_id = yes_tokens[0] if len(yes_tokens) == 1 else yes_tokens[-1]
    no_token_id = no_tokens[0] if len(no_tokens) == 1 else no_tokens[-1]
    print(f"Yes token ID: {yes_token_id} ('{tokenizer.decode([yes_token_id])}')")
    print(f"No token ID: {no_token_id} ('{tokenizer.decode([no_token_id])}')")

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    aucs = []
    mrrs = []
    ndcg5 = []
    ndcg10 = []
    raw_scores = []
    count = 0
    skipped_no_pos = 0

    # Count total feeds
    total_feeds = 0
    with open(args.eval_jsonl, 'r', encoding='utf-8') as f:
        for _ in f:
            total_feeds += 1
    total_to_process = min(total_feeds, args.max_feeds) if args.max_feeds > 0 else total_feeds

    print(f"\nEvaluating DOCA point-wise (Yes/No)...")
    print(f"  Eval file: {args.eval_jsonl}")
    print(f"  Total feeds: {total_feeds}")
    print(f"  Processing: {total_to_process}")
    print(f"  Batch size: {args.batch_size}")
    if args.quick:
        print(f"  Quick mode: {args.max_feeds} feeds")
    print()

    with open(args.eval_jsonl, 'r', encoding='utf-8') as f:
        pbar = tqdm(total=total_to_process, desc="Evaluating feeds", unit="feed")
        for line in f:
            line = line.strip()
            if not line:
                continue

            feed = json.loads(line)
            candidates = feed.get('candidates', [])
            if not candidates:
                continue

            # Build user context
            user_context = {
                'interests': feed.get('interests', []),
                'negative_interests': feed.get('negative_interests', []),
                'conversation': feed.get('conversation', []),
                'shown_10d': feed.get('shown_10d', []),
            }

            labels = [1 if c.get('is_clicked') else 0 for c in candidates]

            # Skip feeds with no positives (can't compute AUC/MRR)
            if sum(labels) == 0:
                skipped_no_pos += 1
                pbar.update(1)
                continue

            # Score all candidates
            scores = batch_score_candidates(
                model, tokenizer, user_context, candidates,
                device, yes_token_id, no_token_id,
                batch_size=args.batch_size,
                use_chat_template=args.use_chat_template,
                max_interests=args.max_interests,
                max_conversation_msgs=args.max_conversation_msgs,
                max_shown=args.max_shown,
                temperature=args.temperature,
            )

            auc_val = auc_score(labels, scores)
            if auc_val is not None:
                aucs.append(auc_val)
            mrrs.append(mrr_score(labels, scores))
            ndcg5.append(ndcg_score(labels, scores, 5))
            ndcg10.append(ndcg_score(labels, scores, 10))

            if args.output_scores_file:
                scores_str = "\t".join(f"{s:.6f}" for s in scores)
                labels_str = "\t".join(str(l) for l in labels)
                raw_scores.append(f"{labels_str}\t|\t{scores_str}")

            count += 1
            pbar.update(1)
            pbar.set_postfix({
                'AUC': f'{_avg(aucs):.4f}',
                'MRR': f'{_avg(mrrs):.4f}',
                'nDCG@5': f'{_avg(ndcg5):.4f}',
                'nDCG@10': f'{_avg(ndcg10):.4f}',
            })

            if args.max_feeds and count >= args.max_feeds:
                break

        pbar.close()

    print(f"\nDOCA Evaluation (Point-wise, Yes/No)")
    print(f"Feeds processed: {count}")
    print(f"Feeds with clicks (evaluated): {len(aucs)}")
    print(f"Feeds skipped (no clicks): {skipped_no_pos}")

    if aucs:
        print(f"\nAUC:     {_avg(aucs):.4f}")
        print(f"MRR:     {_avg(mrrs):.4f}")
        print(f"nDCG@5:  {_avg(ndcg5):.4f}")
        print(f"nDCG@10: {_avg(ndcg10):.4f}")
    else:
        print("No metrics computed (no feeds with clicks)")

    if args.output_scores_file and raw_scores:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_scores_file)), exist_ok=True)
        with open(args.output_scores_file, "w", encoding="utf-8") as f:
            for line in raw_scores:
                f.write(line + "\n")
        print(f"\nWrote {len(raw_scores)} score lines to: {args.output_scores_file}")


if __name__ == "__main__":
    main()
