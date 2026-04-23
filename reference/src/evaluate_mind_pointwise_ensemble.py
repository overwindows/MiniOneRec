"""
Evaluate MIND using an ensemble of multiple point-wise models.

Each model scores candidates independently (Yes/No log probs).
Models are loaded sequentially to keep VRAM manageable.
Final score = weighted average of per-model scores.

Per-model use_chat_template and use_abstract are supported so models
trained with different settings can be correctly combined.
If not specified, flags are auto-detected from the checkpoint path
(contains "_chat" → chat_template=1, contains "_abstract" → abstract=1).

Usage:
    # 2-model ensemble — auto-detect flags from path
    python src/evaluate_mind_pointwise_ensemble.py \
        --model_paths model_abstract_chat/final_checkpoint model_base/final_checkpoint \
        --behaviors_path data/MIND_large/dev/behaviors.tsv \
        --news_path data/MIND_large/dev/news.tsv

    # Explicit per-model flags
    python src/evaluate_mind_pointwise_ensemble.py \
        --model_paths model1 model2 model3 \
        --use_chat_template 1 1 0 \
        --use_abstract 1 0 0 \
        --weights 1.0 0.8 0.8 \
        --behaviors_path ... --news_path ...
"""

import argparse
import gc
import os
import random
from typing import Dict, List

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


def autodetect_flags(model_path: str):
    """Infer use_chat_template and use_abstract from checkpoint directory name."""
    name = os.path.basename(os.path.normpath(model_path))
    # walk up one level if path ends with final_checkpoint / checkpoint-XXXXX
    if name in ("final_checkpoint",) or name.startswith("checkpoint-"):
        name = os.path.basename(os.path.dirname(os.path.normpath(model_path)))
    use_chat_template = 1 if "_chat" in name else 0
    use_abstract = 1 if "_abstract" in name else 0
    return use_chat_template, use_abstract


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
    news_with_abstract: dict,
    news_without_abstract: dict,
    use_abstract: bool,
    use_chat_template: bool,
    max_history: int,
    max_impressions: int,
    batch_size: int,
    flash_attn: bool,
) -> Dict[str, List[float]]:
    """Load one model, score all impressions, unload. Returns {impression_id: [scores]}."""

    news = news_with_abstract if use_abstract else news_without_abstract

    print(f"  Loading tokenizer...")
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
    print(f"  Model on: {device}")

    yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode(" No", add_special_tokens=False)
    yes_token_id = yes_tokens[0] if len(yes_tokens) == 1 else yes_tokens[-1]
    no_token_id = no_tokens[0] if len(no_tokens) == 1 else no_tokens[-1]

    all_scores: Dict[str, List[float]] = {}
    count = 0

    with open(behaviors_path, "r", encoding="utf-8") as f:
        if not max_impressions:
            total_lines = sum(1 for _ in f)
            f.seek(0)
        else:
            total_lines = max_impressions

        pbar = tqdm(total=total_lines, desc="  Scoring", unit="imp")
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
                        help="Paths to pointwise checkpoints (2 or more)")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Per-model weights (default: equal). Length must match --model_paths.")
    parser.add_argument("--use_chat_template", nargs="+", type=int, default=None,
                        help="Per-model chat template flag (0/1). Auto-detected from path if omitted.")
    parser.add_argument("--use_abstract", nargs="+", type=int, default=None,
                        help="Per-model abstract flag (0/1). Auto-detected from path if omitted.")
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--max_history", type=int, default=0)
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n = len(args.model_paths)
    if n < 2:
        raise ValueError("Ensemble requires at least 2 model paths")

    # Resolve per-model flags (auto-detect if not provided)
    chat_flags = []
    abstract_flags = []
    for i, mp in enumerate(args.model_paths):
        auto_chat, auto_abstract = autodetect_flags(mp)
        c = args.use_chat_template[i] if args.use_chat_template and i < len(args.use_chat_template) else auto_chat
        a = args.use_abstract[i] if args.use_abstract and i < len(args.use_abstract) else auto_abstract
        chat_flags.append(bool(c))
        abstract_flags.append(bool(a))

    # Normalize weights
    weights = args.weights or [1.0] * n
    if len(weights) != n:
        raise ValueError(f"--weights length ({len(weights)}) must match --model_paths ({n})")
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    set_seed(args.seed)

    # Load news — load both versions if models differ in use_abstract
    need_abstract = any(abstract_flags)
    need_plain = any(not f for f in abstract_flags)

    print(f"Loading news from: {args.news_path}")
    news_with_abstract = load_news(args.news_path, use_abstract=True) if need_abstract else {}
    news_without_abstract = load_news(args.news_path, use_abstract=False) if need_plain else {}
    total_news = len(news_with_abstract) or len(news_without_abstract)
    print(f"Loaded {total_news} articles")

    print(f"\n{'='*55}")
    print(f"Ensemble of {n} models")
    print(f"{'='*55}")
    for i, mp in enumerate(args.model_paths):
        print(f"  [{i+1}] w={weights[i]:.3f}  chat={int(chat_flags[i])}  abstract={int(abstract_flags[i])}")
        print(f"       {mp}")
    print(f"{'='*55}")

    # Score each model sequentially
    all_model_scores: List[Dict[str, List[float]]] = []
    for i, model_path in enumerate(args.model_paths):
        print(f"\n[Model {i+1}/{n}] chat={int(chat_flags[i])} abstract={int(abstract_flags[i])}")
        print(f"  Path: {model_path}")
        model_scores = score_all_impressions(
            model_path=model_path,
            behaviors_path=args.behaviors_path,
            news_with_abstract=news_with_abstract,
            news_without_abstract=news_without_abstract,
            use_abstract=abstract_flags[i],
            use_chat_template=chat_flags[i],
            max_history=args.max_history,
            max_impressions=args.max_impressions,
            batch_size=args.batch_size,
            flash_attn=args.flash_attn,
        )
        all_model_scores.append(model_scores)

    # Only evaluate impressions present in all models
    common_ids = set(all_model_scores[0].keys())
    for ms in all_model_scores[1:]:
        common_ids &= set(ms.keys())
    print(f"\n{len(common_ids)} impressions scored by all {n} models")

    # Compute ensemble metrics
    print("Computing ensemble metrics...")
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    predictions = []

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
            n_cands = len(labels)

            # Weighted average across models
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
                ranked = sorted(range(len(ensemble_scores)), key=lambda i: ensemble_scores[i], reverse=True)
                predictions.append((impression_id, [candidate_ids[i] for i in ranked]))

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    print(f"\n{'='*55}")
    print(f"MIND Ensemble Evaluation ({n} models)")
    print(f"{'='*55}")
    for i, mp in enumerate(args.model_paths):
        print(f"  [{i+1}] w={weights[i]:.3f}  {os.path.basename(os.path.dirname(mp))}")
    print(f"Impressions evaluated: {len(aucs)}")
    print(f"AUC:     {_avg(aucs):.4f}")
    print(f"MRR:     {_avg(mrrs):.4f}")
    print(f"nDCG@5:  {_avg(ndcg5s):.4f}")
    print(f"nDCG@10: {_avg(ndcg10s):.4f}")
    print(f"{'='*55}")

    if args.output_file and predictions:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for impression_id, ranked_news_ids in predictions:
                f.write(f"{impression_id} {' '.join(ranked_news_ids)}\n")
        print(f"Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
