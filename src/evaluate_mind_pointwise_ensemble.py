"""
Evaluate MIND using an ensemble of multiple point-wise models.

Each model scores candidates independently (Yes/No log probs).
Models are loaded sequentially to keep VRAM manageable.
Final score = weighted average of per-model scores.

Usage:
    # Equal-weight ensemble of 2 models
    python src/evaluate_mind_pointwise_ensemble.py \
        --model_paths path/to/model1/final_checkpoint path/to/model2/final_checkpoint \
        --behaviors_path data/MIND_large/dev/behaviors.tsv \
        --news_path data/MIND_large/dev/news.tsv

    # 3-model ensemble with custom weights
    python src/evaluate_mind_pointwise_ensemble.py \
        --model_paths model1 model2 model3 \
        --weights 1.0 0.8 0.8 \
        --behaviors_path ... --news_path ...
"""

import argparse
import gc
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from mind_utils import (
    load_news,
    build_pointwise_prompt,
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
        torch.cuda.manual_seed_all(seed)


def batch_score_pointwise(
    model,
    tokenizer,
    history: List[dict],
    candidates: List[dict],
    device,
    yes_token_id: int,
    no_token_id: int,
    batch_size: int = 8,
    use_chat_template: bool = False,
) -> List[float]:
    prompts = [
        build_pointwise_prompt(history, cand, tokenizer=tokenizer, use_chat_template=use_chat_template)
        for cand in candidates
    ]
    all_prompt_ids = [
        tokenizer.encode(p, add_special_tokens=(not use_chat_template))
        for p in prompts
    ]

    scores = []
    for batch_start in range(0, len(all_prompt_ids), batch_size):
        batch_ids = all_prompt_ids[batch_start:batch_start + batch_size]
        max_len = max(len(ids) for ids in batch_ids)
        padded_ids, attention_masks = [], []
        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            batch_scores = (log_probs[:, yes_token_id] - log_probs[:, no_token_id]).cpu().tolist()
            scores.extend(batch_scores)

    return scores


def score_all_impressions(
    model_path: str,
    behaviors_path: str,
    news: dict,
    max_history: int,
    max_impressions: int,
    batch_size: int,
    flash_attn: bool,
    use_chat_template: bool,
    use_abstract: bool,
) -> Dict[str, List[float]]:
    """Load one model, score all impressions, return {impression_id: [scores]}."""

    print(f"\n  Loading tokenizer...")
    import os
    local_only = os.path.isdir(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=local_only
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if local_only:
        model_kwargs["local_files_only"] = True
    if flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model loaded on: {device}")

    yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode(" No", add_special_tokens=False)
    yes_token_id = yes_tokens[0] if len(yes_tokens) == 1 else yes_tokens[-1]
    no_token_id = no_tokens[0] if len(no_tokens) == 1 else no_tokens[-1]

    all_scores: Dict[str, List[float]] = {}
    count = 0

    with open(behaviors_path, "r", encoding="utf-8") as f:
        total_lines = None
        if not max_impressions:
            total_lines = sum(1 for _ in f)
            f.seek(0)

        pbar = tqdm(total=total_lines or max_impressions, desc="  Scoring", unit="imp")
        for line in f:
            parsed = parse_behaviors_line(line)
            if parsed is None:
                continue

            impression_id, _, _, history_ids, imp_list = parsed

            if max_history > 0:
                history_ids = history_ids[-max_history:]

            candidate_objs = [news.get(nid, {'text': '[MISSING]', 'category': ''}) for nid, _ in imp_list]
            history_objs = [news[nid] for nid in history_ids if nid in news]

            if not candidate_objs:
                continue

            scores = batch_score_pointwise(
                model, tokenizer, history_objs, candidate_objs,
                device, yes_token_id, no_token_id, batch_size, use_chat_template
            )
            all_scores[impression_id] = scores

            count += 1
            pbar.update(1)
            if max_impressions and count >= max_impressions:
                break

        pbar.close()

    print(f"  Scored {count} impressions")

    # Unload model to free VRAM
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Model unloaded")

    return all_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="Paths to pointwise model checkpoints (2 or more)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Per-model weights for score averaging (default: equal)")
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--max_history", type=int, default=0)
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if len(args.model_paths) < 2:
        raise ValueError("Ensemble requires at least 2 model paths")

    # Normalize weights
    weights = args.weights or [1.0] * len(args.model_paths)
    if len(weights) != len(args.model_paths):
        raise ValueError(f"--weights length ({len(weights)}) must match --model_paths ({len(args.model_paths)})")
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    set_seed(args.seed)

    print(f"Loading news from: {args.news_path}")
    news = load_news(args.news_path, args.use_abstract)
    print(f"Loaded {len(news)} articles")

    print(f"\n{'='*50}")
    print(f"Ensemble of {len(args.model_paths)} models")
    for i, (mp, w) in enumerate(zip(args.model_paths, weights)):
        print(f"  [{i+1}] weight={w:.3f}  {mp}")
    print(f"{'='*50}")

    # Score impressions with each model sequentially
    all_model_scores: List[Dict[str, List[float]]] = []
    for i, model_path in enumerate(args.model_paths):
        print(f"\n[Model {i+1}/{len(args.model_paths)}] {model_path}")
        model_scores = score_all_impressions(
            model_path=model_path,
            behaviors_path=args.behaviors_path,
            news=news,
            max_history=args.max_history,
            max_impressions=args.max_impressions,
            batch_size=args.batch_size,
            flash_attn=args.flash_attn,
            use_chat_template=args.use_chat_template,
            use_abstract=args.use_abstract,
        )
        all_model_scores.append(model_scores)

    # Collect impression IDs present in all models
    common_ids = set(all_model_scores[0].keys())
    for ms in all_model_scores[1:]:
        common_ids &= set(ms.keys())
    print(f"\n{len(common_ids)} impressions scored by all models")

    # Compute ensemble metrics
    print("\nComputing ensemble metrics...")
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    predictions = []

    # Re-read behaviors to get labels and candidate IDs
    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_behaviors_line(line)
            if parsed is None:
                continue
            impression_id, _, _, _, imp_list = parsed

            if impression_id not in common_ids:
                continue

            labels = [label for _, label in imp_list]
            candidate_ids = [nid for nid, _ in imp_list]

            # Weighted average of scores across models
            n_cands = len(labels)
            ensemble_scores = [0.0] * n_cands
            for w, model_scores in zip(weights, all_model_scores):
                m_scores = model_scores[impression_id]
                for j in range(min(n_cands, len(m_scores))):
                    ensemble_scores[j] += w * m_scores[j]

            if sum(labels) > 0:
                aucs.append(auc_score(labels, ensemble_scores))
                mrrs.append(mrr_score(labels, ensemble_scores))
                ndcg5s.append(ndcg_score(labels, ensemble_scores, 5))
                ndcg10s.append(ndcg_score(labels, ensemble_scores, 10))

            if args.output_file:
                ranked_indices = sorted(range(len(ensemble_scores)), key=lambda i: ensemble_scores[i], reverse=True)
                predictions.append((impression_id, [candidate_ids[i] for i in ranked_indices]))

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    print(f"\n{'='*50}")
    print(f"MIND Ensemble Evaluation ({len(args.model_paths)} models)")
    print(f"{'='*50}")
    print(f"Impressions evaluated: {len(aucs)}")
    print(f"Weights: {[f'{w:.3f}' for w in weights]}")
    print(f"AUC:     {_avg(aucs):.4f}")
    print(f"MRR:     {_avg(mrrs):.4f}")
    print(f"nDCG@5:  {_avg(ndcg5s):.4f}")
    print(f"nDCG@10: {_avg(ndcg10s):.4f}")
    print(f"{'='*50}")

    if args.output_file and predictions:
        import os
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for impression_id, ranked_news_ids in predictions:
                f.write(f"{impression_id} {' '.join(ranked_news_ids)}\n")
        print(f"Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
