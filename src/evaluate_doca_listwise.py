"""
Evaluate DOCA models trained with list-wise SFT.

The model generates text output (e.g., "1, 3") indicating which candidates
it predicts the user will click. We parse these indices and compute metrics.

For scoring/ranking, we also compute a soft score per candidate by checking
how early and how often the model mentions each candidate index.

Metrics: AUC (per-feed avg), MRR, nDCG@5, nDCG@10.

Usage:
    python src/evaluate_doca_listwise.py \\
        --model_path output_dir/sft_doca_listwise_*/final_checkpoint \\
        --eval_jsonl data/doca/dev.jsonl \\
        --max_feeds 1000

    # Quick mode (500 feeds)
    python src/evaluate_doca_listwise.py --model_path ... --eval_jsonl ... --quick
"""

import argparse
import json
import os
import sys
import re
import random
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DOCAListwiseSFTDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
    for i, (label, _) in enumerate(ranked):
        if label == 1:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_score(labels, scores, k):
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


def parse_predicted_indices(text: str, num_candidates: int) -> List[int]:
    """Parse model output like '1, 3' into list of 1-based indices."""
    text = text.strip().lower()
    if text == "none" or not text:
        return []
    # Extract all integers
    numbers = re.findall(r'\d+', text)
    indices = []
    for n in numbers:
        idx = int(n)
        if 1 <= idx <= num_candidates:
            indices.append(idx)
    return list(dict.fromkeys(indices))  # deduplicate, preserve order


def indices_to_scores(predicted_indices: List[int], num_candidates: int) -> List[float]:
    """
    Convert predicted click indices to per-candidate scores for ranking metrics.

    Candidates mentioned first get higher scores. Unmentioned candidates get 0.
    """
    scores = [0.0] * num_candidates
    if not predicted_indices:
        return scores
    for rank, idx in enumerate(predicted_indices):
        # Higher score for earlier mentions; 1-based index
        scores[idx - 1] = len(predicted_indices) - rank
    return scores


def logprob_scores_for_candidates(
    model, tokenizer, prompt, num_candidates, device,
    use_chat_template=False,
) -> List[float]:
    """
    Compute log P(candidate_index_token) at the first generation position.

    For each candidate 1..N, we look at the log probability the model assigns
    to that number token right after the prompt. This gives a continuous score
    per candidate, making AUC/MRR/nDCG meaningful.

    Returns a list of N floats (log-probs), one per candidate.
    """
    if use_chat_template:
        messages = [
            {"role": "system", "content": DOCAListwiseSFTDataset.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    tokens = tokenizer.encode(prompt, add_special_tokens=(not use_chat_template))
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        # Logits at the last position = next token prediction
        logits = outputs.logits[0, -1, :]  # (vocab_size,)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Get log-prob for each candidate number token: " 1", " 2", ..., " N"
    scores = []
    for i in range(1, num_candidates + 1):
        # Try " {i}" (with leading space, as trained)
        token_ids = tokenizer.encode(f" {i}", add_special_tokens=False)
        # Use the first token (the number token)
        tid = token_ids[0]
        scores.append(log_probs[tid].item())

    return scores


def build_listwise_prompt(user_context, candidates, max_interests=0,
                          max_conversation_msgs=15, max_shown=10):
    """Build prompt matching DOCAListwiseSFTDataset._build_prompt format (without target)."""
    parts = []

    # 1. User interests
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

    # 3. Conversation history
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

    # 4. Shown 10d
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

    # 5. Candidate list (numbered)
    parts.append("\nCandidate articles:")
    for j, cand in enumerate(candidates, 1):
        title = cand.get('title', '')
        summary = cand.get('summary', '')
        if summary:
            parts.append(f"{j}. {title} — {summary}")
        else:
            parts.append(f"{j}. {title}")

    parts.append("\nWhich articles will this user click? Output the article numbers:")

    return '\n'.join(parts)


def generate_prediction(model, tokenizer, prompt, device, max_new_tokens=32,
                        use_chat_template=False):
    """Generate model prediction for a listwise prompt."""
    if use_chat_template:
        messages = [
            {"role": "system", "content": DOCAListwiseSFTDataset.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    tokens = tokenizer.encode(prompt, add_special_tokens=(not use_chat_template))
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated_ids = outputs[0][len(tokens):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--max_feeds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 500 feeds")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_interests", type=int, default=0)
    parser.add_argument("--max_conversation_msgs", type=int, default=15)
    parser.add_argument("--max_shown", type=int, default=10)
    parser.add_argument("--max_candidates", type=int, default=10, help="Max candidates per feed (match training)")
    parser.add_argument("--output_scores_file", help="Output raw scores file")
    args = parser.parse_args()

    if args.quick and args.max_feeds == 0:
        args.max_feeds = 500

    set_seed(args.seed)
    rng = random.Random(args.seed)

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

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    aucs = []
    mrrs = []
    ndcg5 = []
    ndcg10 = []
    raw_scores = []
    count = 0
    skipped_no_pos = 0
    exact_match = 0
    total_predicted = 0
    total_actual = 0
    total_correct = 0

    # Count total feeds
    total_feeds = 0
    with open(args.eval_jsonl, 'r', encoding='utf-8') as f:
        for _ in f:
            total_feeds += 1
    total_to_process = min(total_feeds, args.max_feeds) if args.max_feeds > 0 else total_feeds

    print(f"\nEvaluating DOCA list-wise...")
    print(f"  Eval file: {args.eval_jsonl}")
    print(f"  Total feeds: {total_feeds}")
    print(f"  Processing: {total_to_process}")
    print(f"  Max new tokens: {args.max_new_tokens}")
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
            if len(candidates) < 2:
                pbar.update(1)
                continue

            # Shuffle and limit candidates to match training distribution
            indexed_candidates = list(enumerate(candidates))
            rng.shuffle(indexed_candidates)

            if args.max_candidates > 0 and len(indexed_candidates) > args.max_candidates:
                # Ensure at least one positive is included
                positives = [(i, c) for i, c in indexed_candidates if c.get('is_clicked')]
                negatives = [(i, c) for i, c in indexed_candidates if not c.get('is_clicked')]
                if positives:
                    keep_pos = positives[:args.max_candidates]
                    remaining = args.max_candidates - len(keep_pos)
                    keep_neg = negatives[:remaining]
                    indexed_candidates = keep_pos + keep_neg
                    rng.shuffle(indexed_candidates)
                else:
                    indexed_candidates = indexed_candidates[:args.max_candidates]

            candidates = [c for _, c in indexed_candidates]

            user_context = {
                'interests': feed.get('interests', []),
                'negative_interests': feed.get('negative_interests', []),
                'conversation': feed.get('conversation', []),
                'shown_10d': feed.get('shown_10d', []),
            }

            labels = [1 if c.get('is_clicked') else 0 for c in candidates]
            actual_clicked = set(i + 1 for i, l in enumerate(labels) if l == 1)

            if sum(labels) == 0:
                skipped_no_pos += 1
                pbar.update(1)
                continue

            # Build prompt
            prompt = build_listwise_prompt(
                user_context, candidates,
                max_interests=args.max_interests,
                max_conversation_msgs=args.max_conversation_msgs,
                max_shown=args.max_shown,
            )

            # Logprob-based scoring: continuous score per candidate
            logprob_sc = logprob_scores_for_candidates(
                model, tokenizer, prompt, len(candidates), device,
                use_chat_template=args.use_chat_template,
            )

            # Also generate text for classification metrics
            prediction = generate_prediction(
                model, tokenizer, prompt, device,
                max_new_tokens=args.max_new_tokens,
                use_chat_template=args.use_chat_template,
            )

            # Parse predicted indices for classification metrics
            predicted_indices = parse_predicted_indices(prediction, len(candidates))
            predicted_set = set(predicted_indices)

            # Classification metrics (from generated text)
            correct = predicted_set & actual_clicked
            total_predicted += len(predicted_set)
            total_actual += len(actual_clicked)
            total_correct += len(correct)
            if predicted_set == actual_clicked:
                exact_match += 1

            # Ranking metrics (from logprob scores — continuous!)
            auc_val = auc_score(labels, logprob_sc)
            if auc_val is not None:
                aucs.append(auc_val)
            mrrs.append(mrr_score(labels, logprob_sc))
            ndcg5.append(ndcg_score(labels, logprob_sc, 5))
            ndcg10.append(ndcg_score(labels, logprob_sc, 10))

            if args.output_scores_file:
                scores_str = "\t".join(f"{s:.4f}" for s in logprob_sc)
                labels_str = "\t".join(str(l) for l in labels)
                raw_scores.append(f"{labels_str}\t|\t{scores_str}\t|\t{prediction.strip()}")

            count += 1
            pbar.update(1)
            pbar.set_postfix({
                'AUC': f'{_avg(aucs):.4f}',
                'MRR': f'{_avg(mrrs):.4f}',
                'nDCG@10': f'{_avg(ndcg10):.4f}',
            })

            if args.max_feeds and count >= args.max_feeds:
                break

        pbar.close()

    print(f"\nDOCA Evaluation (List-wise)")
    print(f"Feeds processed: {count}")
    print(f"Feeds with clicks (evaluated): {len(aucs)}")
    print(f"Feeds skipped (no clicks): {skipped_no_pos}")

    if aucs:
        print(f"\n--- Ranking Metrics ---")
        print(f"AUC:     {_avg(aucs):.4f}")
        print(f"MRR:     {_avg(mrrs):.4f}")
        print(f"nDCG@5:  {_avg(ndcg5):.4f}")
        print(f"nDCG@10: {_avg(ndcg10):.4f}")

        print(f"\n--- Classification Metrics ---")
        precision = total_correct / max(total_predicted, 1)
        recall = total_correct / max(total_actual, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        print(f"Precision: {precision:.4f} ({total_correct}/{total_predicted})")
        print(f"Recall:    {recall:.4f} ({total_correct}/{total_actual})")
        print(f"F1:        {f1:.4f}")
        print(f"Exact Match: {exact_match}/{count} ({100*exact_match/max(count,1):.1f}%)")
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
