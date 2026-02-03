import argparse
import json
import math
import random
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_news(news_path: str, use_abstract: bool) -> dict:
    news = {}
    with open(news_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            news_id = parts[0]
            title = parts[3]
            abstract = parts[4] if len(parts) > 4 else ""
            if use_abstract and abstract:
                news[news_id] = f"{title} {abstract}"
            else:
                news[news_id] = title
    return news


def build_prompt(history_titles: List[str]) -> str:
    history_text = ", ".join([f"\"{t}\"" for t in history_titles if t])
    return (
        "### User Input: \n"
        f"The user has read the following news before: {history_text}\n\n"
        "### Response:\n"
    )


def score_candidates(
    model,
    tokenizer,
    prompt_ids: List[int],
    candidates: List[str],
    device,
    batch_size: int = 4,
) -> List[float]:
    """Score candidates in batches to avoid OOM on large vocabularies."""
    cand_ids = [tokenizer.encode(c, add_special_tokens=False) + [tokenizer.eos_token_id] for c in candidates]
    input_ids = [prompt_ids + ids for ids in cand_ids]

    all_scores = []

    # Process candidates in batches
    for batch_start in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[batch_start:batch_start + batch_size]
        max_len = max(len(seq) for seq in batch_input_ids)

        padded = []
        attention = []
        labels = []
        for seq in batch_input_ids:
            pad_len = max_len - len(seq)
            padded.append([tokenizer.pad_token_id] * pad_len + seq)
            attention.append([0] * pad_len + [1] * len(seq))
            label = [-100] * (pad_len + len(prompt_ids)) + seq[len(prompt_ids) :]
            labels.append(label)

        input_ids_t = torch.tensor(padded, dtype=torch.long, device=device)
        attention_t = torch.tensor(attention, dtype=torch.long, device=device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids_t, attention_mask=attention_t)
            logits = outputs.logits[:, :-1, :]
            target = labels_t[:, 1:]
            mask = target != -100

            # Compute log_softmax and gather in one go to reduce memory
            vocab_size = logits.size(-1)
            target_clamped = torch.clamp(target, 0, vocab_size - 1)

            # Use cross-entropy computation directly to avoid storing full log_probs
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_logp = torch.gather(log_probs, -1, target_clamped.unsqueeze(-1)).squeeze(-1)
            token_logp = token_logp * mask
            scores = token_logp.sum(dim=-1)

            all_scores.extend(scores.detach().cpu().tolist())

            # Clear CUDA cache after each batch
            del logits, log_probs, token_logp, outputs
            torch.cuda.empty_cache()

    return all_scores


def auc_score(labels: List[int], scores: List[float]) -> float:
    """
    Compute AUC score using sklearn's roc_auc_score.
    This matches the official MIND evaluation script.
    """
    pos = sum(labels)
    if pos == 0 or pos == len(labels):
        return 0.5
    return roc_auc_score(labels, scores)


def mrr_score(labels: List[int], scores: List[float]) -> float:
    """
    Compute MRR score (Mean Reciprocal Rank).
    Matches official MIND evaluation: averages RR over all clicked items.
    """
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rr_scores = []
    for rank, idx in enumerate(sorted_idx, start=1):
        if labels[idx] == 1:
            rr_scores.append(1.0 / rank)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def ndcg_score(labels: List[int], scores: List[float], k: int) -> float:
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    dcg = 0.0
    for rank, idx in enumerate(sorted_idx[:k], start=1):
        if labels[idx] == 1:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(sum(labels), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--max_history", type=int, default=50)
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for scoring candidates (reduce if OOM)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", help="Output prediction file for MIND leaderboard (ImpressionID and ranked news IDs)")
    args = parser.parse_args()

    set_seed(args.seed)

    news = load_news(args.news_path, args.use_abstract)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, fix_mistral_regex=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    # Get device from first parameter (handles multi-GPU correctly)
    device = next(model.parameters()).device

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    aucs = []
    mrrs = []
    ndcg5 = []
    ndcg10 = []
    predictions = []  # For MIND leaderboard format

    # Count total lines efficiently using wc -l (fast for large files)
    # Skip counting if max_impressions is set (for quick tests)
    import subprocess
    total_lines = None
    if not args.max_impressions:
        try:
            total_lines = int(subprocess.check_output(['wc', '-l', args.behaviors_path]).split()[0])
        except:
            pass

    # Determine total impressions to process
    if total_lines:
        total_to_process = total_lines
    elif args.max_impressions > 0:
        total_to_process = args.max_impressions
    else:
        total_to_process = None  # No total, just show count

    count = 0
    skipped_malformed = 0
    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=total_to_process, desc="Evaluating impressions", unit="impression")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                skipped_malformed += 1
                continue
            impression_id = parts[0]  # ImpressionID for leaderboard submission
            history = parts[3].split()[-args.max_history :]
            impressions = parts[4].split()
            labels = []
            candidates = []
            candidate_ids = []  # Store news IDs for prediction output
            missing_count = 0
            for imp in impressions:
                if "-" not in imp:
                    continue
                nid, label = imp.rsplit("-", 1)
                # Use placeholder for missing news instead of skipping
                if nid not in news:
                    candidates.append("[MISSING_NEWS]")
                    missing_count += 1
                else:
                    candidates.append(news[nid])
                candidate_ids.append(nid)
                labels.append(int(label))

            # Skip only if no candidates at all (should rarely happen)
            if not candidates:
                continue

            history_titles = [news.get(nid, "") for nid in history if nid in news]
            prompt = build_prompt(history_titles)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

            scores = score_candidates(model, tokenizer, prompt_ids, candidates, device, args.batch_size)

            # NOTE: Length normalization removed - raw log probabilities work better for ranking
            # The model already learns appropriate length distributions during training
            # Length normalization was creating bias toward short, vague titles

            # Compute metrics (only if we have positive labels for dev/train evaluation)
            if sum(labels) > 0:
                aucs.append(auc_score(labels, scores))
                mrrs.append(mrr_score(labels, scores))
                ndcg5.append(ndcg_score(labels, scores, 5))
                ndcg10.append(ndcg_score(labels, scores, 10))

            # Generate ranked predictions (news IDs sorted by score, descending)
            # This is needed for all impressions (including test set where labels are unknown)
            if args.output_file:
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                ranked_news_ids = [candidate_ids[i] for i in ranked_indices]
                predictions.append((impression_id, ranked_news_ids))

            count += 1

            # Update progress bar with current metrics
            pbar.update(1)
            pbar.set_postfix({
                'AUC': f'{_avg(aucs):.4f}',
                'MRR': f'{_avg(mrrs):.4f}',
                'nDCG@5': f'{_avg(ndcg5):.4f}',
                'nDCG@10': f'{_avg(ndcg10):.4f}'
            })

            # Print periodic status updates for parallel execution (every 100 impressions)
            if count % 100 == 0:
                import sys
                print(f"\rProcessed {count} impressions | AUC: {_avg(aucs):.4f} | MRR: {_avg(mrrs):.4f}",
                      file=sys.stderr, flush=True)

            if args.max_impressions and count >= args.max_impressions:
                break

        pbar.close()

    print("\nMIND Evaluation")
    print(f"Impressions processed: {count}")
    if skipped_malformed > 0:
        print(f"⚠️  Skipped malformed lines: {skipped_malformed}")
    if aucs:  # Only print metrics if we have labels (dev/train set)
        print(f"AUC:  {_avg(aucs):.4f}")
        print(f"MRR:  {_avg(mrrs):.4f}")
        print(f"nDCG@5:  {_avg(ndcg5):.4f}")
        print(f"nDCG@10: {_avg(ndcg10):.4f}")
    else:
        print("No metrics computed (test set has no labels)")

    # Write predictions to file for MIND leaderboard submission
    if args.output_file and predictions:
        print(f"\nWriting predictions to: {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for impression_id, ranked_news_ids in predictions:
                # MIND leaderboard format: ImpressionID [ranked news IDs separated by space]
                f.write(f"{impression_id} {' '.join(ranked_news_ids)}\n")
        print(f"✓ Wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main()
