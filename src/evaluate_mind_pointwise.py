"""
Evaluate MIND models trained with point-wise SFT (Yes/No classification).

This evaluation scores each candidate independently by computing P("Yes" | prompt)
and uses those scores to rank candidates within each impression.

Key Features:
- Point-wise scoring: Each candidate evaluated independently
- Scores P(" Yes") vs P(" No") for each (history, candidate) pair
- Ranks candidates by Yes probability
- Uses official MIND metrics (AUC, MRR, nDCG@5, nDCG@10)
- Supports Flash Attention 2 for faster inference

Usage:
    python evaluate_mind_pointwise.py \
        --model_path output_dir/sft_mind_pointwise_*/final_checkpoint \
        --behaviors_path ../data/MIND/dev/behaviors.tsv \
        --news_path ../data/MIND/dev/news.tsv \
        --flash_attn \
        --max_impressions 1000  # Optional: for quick testing

    # Quick mode (500 impressions for fast iteration)
    python evaluate_mind_pointwise.py --model_path ... --quick
"""

import argparse
import random
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from mind_utils import (
    load_news,
    build_pointwise_prompt,
    build_pointwise_prompt_subcategory,
    parse_behaviors_line,
    auc_score,
    mrr_score,
    ndcg_score,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def batch_score_candidates_pointwise(
    model,
    tokenizer,
    history: List[dict],
    candidates: List[dict],
    device,
    yes_token_id: int,
    no_token_id: int,
    batch_size: int = 8,
    use_chat_template: bool = False,
    use_subcategory: bool = False,
    temperature: float = 1.0,
    use_recency: bool = False,
    use_profile_summary: bool = False,
    impression_timestamp: str = None,
) -> List[float]:
    """
    Score multiple candidates in batches for efficiency.

    Returns:
        List of scores for each candidate
    """
    # Build all prompts
    if use_subcategory:
        prompts = [
            build_pointwise_prompt_subcategory(history, cand, tokenizer=tokenizer, use_chat_template=use_chat_template)
            for cand in candidates
        ]
    else:
        prompts = [
            build_pointwise_prompt(
                history, cand, tokenizer=tokenizer, use_chat_template=use_chat_template,
                use_recency=use_recency, use_profile_summary=use_profile_summary,
                impression_timestamp=impression_timestamp,
            )
            for cand in candidates
        ]

    # Tokenize all prompts
    # For chat templates, add_special_tokens is already handled
    all_prompt_ids = [
        tokenizer.encode(p, add_special_tokens=(not use_chat_template))
        for p in prompts
    ]

    scores = []

    # Process in batches
    for batch_start in range(0, len(all_prompt_ids), batch_size):
        batch_ids = all_prompt_ids[batch_start:batch_start + batch_size]

        # Pad to same length
        max_len = max(len(ids) for ids in batch_ids)
        padded_ids = []
        attention_masks = []

        for ids in batch_ids:
            pad_len = max_len - len(ids)
            # Left padding
            padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Get logits at last position (after "Answer:")
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
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--use_subcategory", action="store_true", help="Use [category/subcategory] format in prompts")
    parser.add_argument("--max_history", type=int, default=0, help="Max history items (0=unlimited)")
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for scoring candidates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", help="Output prediction file for MIND leaderboard")
    parser.add_argument("--output_scores_file", help="Output raw scores file for ensemble (impression_id score1 score2 ...)")
    parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention 2")
    parser.add_argument("--use_chat_template", action="store_true", help="Use chat template (for instruct models)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: evaluate 500 impressions")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for logit scaling (default=1.0, no effect); mainly useful for ensemble calibration")
    parser.add_argument("--use_recency", action="store_true", help="Mark 5 most recent history items with '(recent)' tag")
    parser.add_argument("--use_profile_summary", action="store_true", help="Prepend top-3 category interest summary to prompt")
    parser.add_argument("--cf_scores_file", default="", help="TSV file with CF scores (impression_id, news_id, cf_score) to blend with LLM scores")
    parser.add_argument("--cf_alpha", type=float, default=0.3, help="Blend weight for CF: final = (1-alpha)*LLM + alpha*CF (default 0.3)")
    args = parser.parse_args()

    # Quick mode overrides max_impressions
    if args.quick and args.max_impressions == 0:
        args.max_impressions = 500

    set_seed(args.seed)

    # Load CF scores if provided
    cf_scores: dict = {}
    if args.cf_scores_file:
        print(f"Loading CF scores from: {args.cf_scores_file}")
        with open(args.cf_scores_file, 'r', encoding='utf-8') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    imp_id, news_id, score = parts
                    cf_scores[(imp_id, news_id)] = float(score)
        print(f"Loaded {len(cf_scores):,} CF scores (alpha={args.cf_alpha})")

    print(f"Loading news from: {args.news_path}")
    news = load_news(args.news_path, args.use_abstract)
    print(f"Loaded {len(news)} news articles")

    print(f"Loading model from: {args.model_path}")
    import os as _os
    from pathlib import Path as _Path
    # Treat any absolute path or existing directory as local to avoid
    # huggingface_hub validate_repo_id rejecting absolute paths
    local_only = _os.path.isabs(args.model_path) or _os.path.isdir(args.model_path)
    model_path_arg = _Path(args.model_path) if local_only else args.model_path
    # Checkpoint subdirs may not contain tokenizer files; fall back to parent dir
    tokenizer_path = model_path_arg
    if local_only and not (_Path(model_path_arg) / "tokenizer_config.json").exists():
        parent = _Path(model_path_arg).parent
        if (parent / "tokenizer_config.json").exists():
            print(f"Tokenizer files not found in checkpoint dir, loading from parent: {parent}")
            tokenizer_path = parent
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=local_only)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load model with optional Flash Attention 2
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if local_only:
        model_kwargs["local_files_only"] = True
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")

    model = AutoModelForCausalLM.from_pretrained(
        model_path_arg, **model_kwargs
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    # Get Yes/No token IDs
    yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode(" No", add_special_tokens=False)

    # Handle tokenizers that might split " Yes"/" No" into multiple tokens.
    # Use [-1] (the content token e.g. "Yes") rather than [0] (which could be a generic space).
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
    predictions = []
    raw_scores = []

    # Count total lines for progress bar
    total_lines = None
    if not args.max_impressions:
        with open(args.behaviors_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

    total_to_process = total_lines or args.max_impressions or None

    count = 0
    skipped_malformed = 0

    print(f"\nEvaluating with point-wise format (Yes/No)...")
    print(f"Use abstract: {args.use_abstract}")
    print(f"Max history: {'unlimited' if args.max_history == 0 else args.max_history}")
    print(f"Batch size: {args.batch_size}")
    if args.quick:
        print(f"Quick mode: {args.max_impressions} impressions")
    print()

    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=total_to_process, desc="Evaluating impressions", unit="impression")
        for line in f:
            parsed = parse_behaviors_line(line)
            if parsed is None:
                skipped_malformed += 1
                continue

            impression_id, _, impression_ts, history_ids, imp_list = parsed

            if args.max_history > 0:
                history_ids = history_ids[-args.max_history:]

            labels = []
            candidate_objs = []
            candidate_ids = []

            for nid, label in imp_list:
                if nid not in news:
                    candidate_objs.append({'text': '[MISSING_NEWS]', 'category': ''})
                else:
                    candidate_objs.append(news[nid])

                candidate_ids.append(nid)
                labels.append(label)

            # Skip if no candidates
            if not candidate_objs:
                continue

            # Build history
            history_objs = [news[nid] for nid in history_ids if nid in news]

            # Score all candidates using batched point-wise scoring
            scores = batch_score_candidates_pointwise(
                model, tokenizer, history_objs, candidate_objs, device,
                yes_token_id, no_token_id, args.batch_size, args.use_chat_template,
                args.use_subcategory,
                temperature=args.temperature,
                use_recency=args.use_recency,
                use_profile_summary=args.use_profile_summary,
                impression_timestamp=impression_ts if (args.use_recency or args.use_profile_summary or impression_ts) else None,
            )

            # Blend with CF scores if available
            if cf_scores:
                import numpy as _np
                lm_arr = _np.array(scores)
                cf_arr = _np.array([cf_scores.get((impression_id, nid), 0.0) for nid in candidate_ids])
                # Min-max normalise each signal to [0,1] within the impression
                def _norm(x):
                    mn, mx = x.min(), x.max()
                    return (x - mn) / (mx - mn + 1e-9)
                scores = ((1 - args.cf_alpha) * _norm(lm_arr) + args.cf_alpha * _norm(cf_arr)).tolist()

            # Compute metrics (only if we have positive labels)
            if sum(labels) > 0:
                aucs.append(auc_score(labels, scores))
                mrrs.append(mrr_score(labels, scores))
                ndcg5.append(ndcg_score(labels, scores, 5))
                ndcg10.append(ndcg_score(labels, scores, 10))

            # Generate ranked predictions
            if args.output_file:
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                ranked_news_ids = [candidate_ids[i] for i in ranked_indices]
                predictions.append((impression_id, ranked_news_ids))

            # Save raw scores for ensemble
            if args.output_scores_file:
                scores_str = "\t".join(f"{s:.6f}" for s in scores)
                raw_scores.append(f"{impression_id}\t{scores_str}")

            count += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'AUC': f'{_avg(aucs):.4f}',
                'MRR': f'{_avg(mrrs):.4f}',
                'nDCG@5': f'{_avg(ndcg5):.4f}',
                'nDCG@10': f'{_avg(ndcg10):.4f}'
            })

            if args.max_impressions and count >= args.max_impressions:
                break

        pbar.close()

    print("\nMIND Evaluation (Point-wise, Yes/No)")
    print(f"Impressions processed: {count}")
    if skipped_malformed > 0:
        print(f"Skipped malformed lines: {skipped_malformed}")

    if aucs:
        print(f"AUC:     {_avg(aucs):.4f}")
        print(f"MRR:     {_avg(mrrs):.4f}")
        print(f"nDCG@5:  {_avg(ndcg5):.4f}")
        print(f"nDCG@10: {_avg(ndcg10):.4f}")
    else:
        print("No metrics computed (test set has no labels)")

    # Write predictions
    if args.output_file and predictions:
        print(f"\nWriting predictions to: {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for impression_id, ranked_news_ids in predictions:
                f.write(f"{impression_id} {' '.join(ranked_news_ids)}\n")
        print(f"Wrote {len(predictions)} predictions")

    # Write raw scores for ensemble
    if args.output_scores_file and raw_scores:
        import os as _os2
        _os2.makedirs(_os2.path.dirname(_os2.path.abspath(args.output_scores_file)), exist_ok=True)
        print(f"\nWriting raw scores to: {args.output_scores_file}")
        with open(args.output_scores_file, "w", encoding="utf-8") as f:
            for line in raw_scores:
                f.write(line + "\n")
        print(f"Wrote {len(raw_scores)} score lines")


if __name__ == "__main__":
    main()
