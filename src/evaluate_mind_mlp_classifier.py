"""
Evaluate MIND models trained with the frozen LLM + MLP classifier approach.

Loads a frozen Qwen3-1.7B backbone + saved MLP head checkpoint, then scores
each candidate by: sigmoid(MLP(last_token_hidden_state_of_backbone)).

Usage:
    python src/evaluate_mind_mlp_classifier.py \\
        --model_path   Qwen/Qwen3-1.7B \\
        --mlp_checkpoint output_dir/mlp_classifier/best_mlp.pt \\
        --behaviors_path data/MIND/dev/behaviors.tsv \\
        --news_path      data/MIND/dev/news.tsv \\
        --flash_attn

    # Quick mode (500 impressions)
    python src/evaluate_mind_mlp_classifier.py ... --quick
"""

import argparse
import os
import random
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mind_utils import (
    load_news,
    build_pointwise_prompt,
    build_pointwise_prompt_subcategory,
    parse_behaviors_line,
    auc_score,
    mrr_score,
    ndcg_score,
)


# ---------------------------------------------------------------------------
# MLP head (must match training definition)
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def batch_score_mlp(
    backbone,
    mlp_head: MLPHead,
    tokenizer,
    history: List[dict],
    candidates: List[dict],
    device,
    batch_size: int = 8,
    use_chat_template: bool = False,
    use_subcategory: bool = False,
    max_len: int = 2048,
) -> List[float]:
    """
    Score each candidate independently using frozen backbone + MLP head.

    Returns a list of float scores (higher = more relevant).
    """
    # Build prompts (same format as training)
    if use_subcategory:
        prompts = [
            build_pointwise_prompt_subcategory(history, cand, tokenizer=tokenizer,
                                               use_chat_template=use_chat_template)
            for cand in candidates
        ]
    else:
        prompts = [
            build_pointwise_prompt(history, cand, tokenizer=tokenizer,
                                   use_chat_template=use_chat_template)
            for cand in candidates
        ]

    # Tokenize (no padding yet — will left-pad per batch)
    all_ids = [
        tokenizer.encode(p, add_special_tokens=(not use_chat_template),
                         truncation=True, max_length=max_len)
        for p in prompts
    ]

    scores = []

    for batch_start in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[batch_start:batch_start + batch_size]

        # Left-pad to uniform length in the batch
        max_batch_len = max(len(ids) for ids in batch_ids)
        padded, masks = [], []
        for ids in batch_ids:
            pad_len = max_batch_len - len(ids)
            padded.append([tokenizer.pad_token_id] * pad_len + ids)
            masks.append([0] * pad_len + [1] * len(ids))

        input_ids      = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks,  dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Last token of last transformer layer → (batch, hidden_size)
            last_hidden = outputs.hidden_states[-1][:, -1, :].float()
            logits = mlp_head(last_hidden)                    # (batch,)
            batch_scores = torch.sigmoid(logits).cpu().tolist()

        scores.extend(batch_scores)

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MIND with frozen LLM + MLP classifier"
    )
    parser.add_argument("--model_path",       required=True,
                        help="Path to (or HF id of) Qwen3-1.7B backbone")
    parser.add_argument("--mlp_checkpoint",   required=True,
                        help="Path to best_mlp.pt saved by train_mind_mlp_classifier.py")
    parser.add_argument("--behaviors_path",   required=True)
    parser.add_argument("--news_path",        required=True)
    parser.add_argument("--use_abstract",     action="store_true")
    parser.add_argument("--use_subcategory",  action="store_true")
    parser.add_argument("--use_chat_template",action="store_true")
    parser.add_argument("--max_history",      type=int, default=0,
                        help="Max history items (0=unlimited)")
    parser.add_argument("--max_impressions",  type=int, default=0)
    parser.add_argument("--batch_size",       type=int, default=8)
    parser.add_argument("--max_len",          type=int, default=2048)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--flash_attn",       action="store_true",
                        help="Use Flash Attention 2 for faster inference")
    parser.add_argument("--quick",            action="store_true",
                        help="Quick mode: evaluate 500 impressions")
    parser.add_argument("--output_file",      default="",
                        help="Write ranked predictions for MIND leaderboard")
    parser.add_argument("--output_scores_file", default="",
                        help="Write raw scores TSV for ensemble")
    args = parser.parse_args()

    if args.quick and args.max_impressions == 0:
        args.max_impressions = 500

    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Load backbone
    # ------------------------------------------------------------------
    from pathlib import Path as _Path
    local_only = os.path.isabs(args.model_path) or os.path.isdir(args.model_path)
    model_path_arg = _Path(args.model_path) if local_only else args.model_path
    local_kwargs   = {"local_files_only": True} if local_only else {}

    # Tokenizer fallback: if checkpoint dir has no tokenizer, try parent
    tokenizer_path = model_path_arg
    if local_only and not (_Path(model_path_arg) / "tokenizer_config.json").exists():
        parent = _Path(model_path_arg).parent
        if (parent / "tokenizer_config.json").exists():
            print(f"Tokenizer not in checkpoint dir; loading from: {parent}")
            tokenizer_path = parent

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, **local_kwargs
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Loading backbone from: {args.model_path}")
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto",
                    "output_hidden_states": True}
    if local_only:
        model_kwargs["local_files_only"] = True
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")

    backbone = AutoModelForCausalLM.from_pretrained(model_path_arg, **model_kwargs)
    backbone.eval()
    device = next(backbone.parameters()).device
    print(f"Backbone on device: {device}")

    # ------------------------------------------------------------------
    # Load MLP head
    # ------------------------------------------------------------------
    print(f"Loading MLP checkpoint: {args.mlp_checkpoint}")
    ckpt = torch.load(args.mlp_checkpoint, map_location=device)
    hidden_size    = ckpt.get("hidden_size",    backbone.config.hidden_size)
    mlp_hidden_dim = ckpt.get("mlp_hidden_dim", 256)
    saved_auc      = ckpt.get("best_auc",       None)
    saved_epoch    = ckpt.get("epoch",          "?")

    mlp_head = MLPHead(hidden_size, mlp_hidden_dim).to(device)
    mlp_head.load_state_dict(ckpt["mlp_state_dict"])
    mlp_head.eval()

    print(f"MLP loaded  hidden_size={hidden_size}  mlp_hidden_dim={mlp_hidden_dim}")
    if saved_auc:
        print(f"  Checkpoint from epoch {saved_epoch}, val AUC={saved_auc:.4f}")

    # ------------------------------------------------------------------
    # Load news
    # ------------------------------------------------------------------
    print(f"Loading news from: {args.news_path}")
    news = load_news(args.news_path, args.use_abstract)
    print(f"Loaded {len(news):,} articles")

    # ------------------------------------------------------------------
    # Evaluation loop (mirrors evaluate_mind_pointwise.py)
    # ------------------------------------------------------------------
    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    predictions, raw_scores = [], []

    total_lines = None
    if not args.max_impressions:
        with open(args.behaviors_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
    total_to_process = total_lines or args.max_impressions or None

    count = 0
    skipped = 0

    print(f"\nEvaluating with frozen LLM + MLP classifier...")
    print(f"  Use abstract:     {args.use_abstract}")
    print(f"  Max history:      {'unlimited' if args.max_history == 0 else args.max_history}")
    print(f"  Batch size:       {args.batch_size}")
    if args.quick:
        print(f"  Quick mode:       {args.max_impressions} impressions")
    print()

    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=total_to_process, desc="Evaluating", unit="impression")

        for line in f:
            parsed = parse_behaviors_line(line)
            if parsed is None:
                skipped += 1
                continue

            impression_id, _, impression_ts, history_ids, imp_list = parsed

            if args.max_history > 0:
                history_ids = history_ids[-args.max_history:]

            labels, candidate_objs, candidate_ids = [], [], []
            for nid, label in imp_list:
                candidate_objs.append(news.get(nid, {'text': '[MISSING]', 'category': ''}))
                candidate_ids.append(nid)
                labels.append(label)

            if not candidate_objs:
                continue

            history_objs = [news[nid] for nid in history_ids if nid in news]

            scores = batch_score_mlp(
                backbone, mlp_head, tokenizer,
                history_objs, candidate_objs, device,
                batch_size=args.batch_size,
                use_chat_template=args.use_chat_template,
                use_subcategory=args.use_subcategory,
                max_len=args.max_len,
            )

            if sum(labels) > 0:
                aucs.append(auc_score(labels, scores))
                mrrs.append(mrr_score(labels, scores))
                ndcg5s.append(ndcg_score(labels, scores, 5))
                ndcg10s.append(ndcg_score(labels, scores, 10))

            if args.output_file:
                ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                predictions.append((impression_id, [candidate_ids[i] for i in ranked]))

            if args.output_scores_file:
                raw_scores.append(
                    impression_id + "\t" + "\t".join(f"{s:.6f}" for s in scores)
                )

            count += 1
            pbar.update(1)
            pbar.set_postfix({
                'AUC':     f'{_avg(aucs):.4f}',
                'MRR':     f'{_avg(mrrs):.4f}',
                'nDCG@5':  f'{_avg(ndcg5s):.4f}',
                'nDCG@10': f'{_avg(ndcg10s):.4f}',
            })

            if args.max_impressions and count >= args.max_impressions:
                break

        pbar.close()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print("\nMIND Evaluation (Frozen LLM + MLP Classifier)")
    print(f"Impressions processed: {count:,}")
    if skipped:
        print(f"Skipped malformed lines: {skipped}")

    if aucs:
        print(f"AUC:     {_avg(aucs):.4f}")
        print(f"MRR:     {_avg(mrrs):.4f}")
        print(f"nDCG@5:  {_avg(ndcg5s):.4f}")
        print(f"nDCG@10: {_avg(ndcg10s):.4f}")
    else:
        print("No metrics (test set has no labels)")

    if args.output_file and predictions:
        print(f"\nWriting predictions to: {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for imp_id, ranked_ids in predictions:
                f.write(f"{imp_id} {' '.join(ranked_ids)}\n")
        print(f"Wrote {len(predictions):,} predictions")

    if args.output_scores_file and raw_scores:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_scores_file)), exist_ok=True)
        print(f"\nWriting raw scores to: {args.output_scores_file}")
        with open(args.output_scores_file, "w", encoding="utf-8") as f:
            f.write("\n".join(raw_scores) + "\n")
        print(f"Wrote {len(raw_scores):,} score lines")


if __name__ == "__main__":
    main()
