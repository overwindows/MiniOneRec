"""Bounded frozen-JEV residual-readout experiment; never updates the backbone."""

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_mind_jev import JevScorer, build_jev_state, parse_behaviors_line
from jev_decision import candidate_decision_loss
from jev_noul_data import JEVNoulPointwiseDataset
from mind_utils import auc_score, mrr_score, ndcg_score, load_news


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pair_indices(labels, groups):
    buckets = defaultdict(list)
    for index, group in enumerate(groups):
        buckets[group].append(index)
    left, right, weights = [], [], []
    for indices in buckets.values():
        positive = [i for i in indices if labels[i] == 1]
        negative = [i for i in indices if labels[i] == 0]
        if not positive or not negative:
            raise ValueError("Each training impression must have both classes")
        for p in positive:
            for n in negative:
                left.append(p)
                right.append(n)
                weights.append(1 / (len(positive) * len(negative)))
    return (
        torch.tensor(left), torch.tensor(right),
        torch.tensor(weights, dtype=torch.float64) / len(buckets),
    )


def fit_head(features, margins, labels, groups, regularization, objective, max_iter=100):
    if objective not in ("binary", "pairwise") or regularization <= 0:
        raise ValueError("Expected binary/pairwise objective and positive regularization")
    features = np.asarray(features, dtype=np.float64)
    margins = np.asarray(margins, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if (features.ndim != 2 or margins.shape != labels.shape
            or margins.shape != (len(features),) or len(groups) != len(features)
            or not len(features) or set(np.unique(labels)) != {0, 1}
            or not all(np.isfinite(a).all() for a in (features, margins, labels))):
        raise ValueError("Expected finite aligned features, margins and both binary classes")
    mean = features.mean(axis=0)
    scale = features.std(axis=0).clip(min=1e-6)
    x = torch.from_numpy((features - mean) / scale)
    base = torch.from_numpy(margins)
    targets = torch.from_numpy(labels)
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    pairs = pair_indices(labels, groups) if objective == "pairwise" else None
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=max_iter, line_search_fn="strong_wolfe",
        tolerance_grad=1e-7, tolerance_change=1e-10,
    )

    def closure():
        optimizer.zero_grad()
        prediction = base + x @ weight + bias
        if pairs is None:
            loss = F.binary_cross_entropy_with_logits(prediction, targets)
        else:
            left, right, pair_weight = pairs
            loss = (F.softplus(prediction[right] - prediction[left]) * pair_weight).sum()
        loss = loss + 0.5 * regularization * weight.square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Residual-head objective is nonfinite")
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = float(closure().detach())
    result = {
        "mean": mean, "scale": scale, "weight": weight.detach().numpy(),
        "bias": np.asarray(float(bias.detach())),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Residual-head parameters are nonfinite")
    return result, loss


def predict(head, features, margins):
    if head is None:
        return np.asarray(margins, dtype=np.float64)
    return margins + ((features - head["mean"]) / head["scale"]) @ head["weight"] + head["bias"]


def metrics(labels, scores, groups):
    buckets = defaultdict(list)
    for index, group in enumerate(groups):
        buckets[group].append(index)
    values = []
    for indices in buckets.values():
        y, s = np.asarray(labels)[indices], np.asarray(scores)[indices]
        if not np.isfinite(s).all() or not (0 < y.sum() < len(y)):
            raise ValueError("Ranking metrics require finite scores and both classes")
        values.append([auc_score(y, s), mrr_score(y, s), ndcg_score(y, s, 5), ndcg_score(y, s, 10)])
    return dict(zip(("auc", "mrr", "ndcg5", "ndcg10"), np.mean(values, axis=0).tolist()))


def extract(scorer, rows, output, name, chunk_size=16, report_progress=True):
    features, margins = [], []

    def capture(_module, inputs, logits):
        if inputs[0].ndim != 3 or inputs[0].shape[1] != 1:
            raise RuntimeError("Expected exactly one normalized decision representation")
        features.append(inputs[0][:, 0].detach().float().cpu().numpy())
        selected = logits[:, 0, scorer.cand_tokens].detach().float()
        margins.append((selected[:, 0] - selected[:, 1]).cpu().numpy())

    handle = scorer.model.lm_head.register_forward_hook(capture)
    try:
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start + chunk_size]
            probabilities = scorer.score_yes_batch([r["state"] for r in chunk], chunk_size)
            if any(not 0 <= probability <= 1 for probability in probabilities):
                raise RuntimeError("Feature extraction must not abstain on any candidate")
            if report_progress and start % 256 == 0:
                print(json.dumps({"stage": name, "processed": start + len(chunk), "total": len(rows)}), flush=True)
    finally:
        handle.remove()
    x, base = np.concatenate(features), np.concatenate(margins)
    if len(x) != len(rows) or not np.isfinite(x).all() or not np.isfinite(base).all():
        raise RuntimeError("Feature extraction coverage or finiteness failed")
    if output is not None:
        np.savez(output / (name + ".features.npz"), features=x, margins=base)
        save_json(output / (name + ".rows.json"), [
            {key: row[key] for key in ("impression_id", "candidate_id", "label")}
            for row in rows
        ])
    return x, base


def behavior_rows(path, news, start=0, count=None):
    records = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            if count is not None and index >= start + count:
                break
            parsed = parse_behaviors_line(line)
            if parsed is None:
                raise ValueError("Malformed behavior row")
            impression, history, candidates = parsed
            records.append((impression, [
                {
                    "impression_id": impression, "candidate_id": nid, "label": label,
                    "state": build_jev_state([news[h] for h in history[-30:]], news[nid]),
                }
                for nid, label in candidates
            ]))
    if count is not None and len(records) != count:
        raise ValueError("Requested behavior window is incomplete")
    return records


def loader_parity(scorer, rows, output, model_path):
    from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq

    subset = rows[:4]
    _, frozen = extract(scorer, subset, None, "conditional_loader_parity", 2)
    causal = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(scorer.device).eval()
    proxy = copy.copy(scorer)
    proxy.model = causal
    _, alternative = extract(proxy, subset, None, "causal_loader_parity", 2)
    samples = []
    for row in subset:
        ids = scorer._encode(row["state"])
        target = scorer.cand_tokens[0 if row["label"] else 1]
        samples.append({
            "input_ids": ids + [target], "labels": [-100] * len(ids) + [target],
            "attention_mask": [1] * (len(ids) + 1),
        })
    tokenizer = scorer.tokenizer
    previous_pad, previous_side = tokenizer.pad_token, tokenizer.padding_side
    try:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        collated = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)(samples)
        collated = {key: value.to(scorer.device) for key, value in collated.items()}
        causal.train()
        with torch.no_grad():
            loss, result = candidate_decision_loss(causal, collated, scorer.cand_tokens)
        logits = result["logits"]
        training = (logits[:, 0] - logits[:, 1]).cpu().numpy()
    finally:
        tokenizer.pad_token, tokenizer.padding_side = previous_pad, previous_side
    evidence = {
        "conditional_margins": frozen.tolist(), "causal_eval_margins": alternative.tolist(),
        "causal_training_margins": training.tolist(), "training_loss": float(loss),
        "loader_max_margin_difference": float(np.max(np.abs(frozen - alternative))),
        "training_max_margin_difference": float(np.max(np.abs(frozen - training))),
        "head_rows_identical": bool(torch.equal(
            scorer.model.lm_head.weight[scorer.cand_tokens],
            causal.lm_head.weight[scorer.cand_tokens],
        )),
    }
    save_json(output / "loader-parity.json", evidence)
    del proxy, causal, result, collated
    torch.cuda.empty_cache()
    if (not evidence["head_rows_identical"]
            or evidence["loader_max_margin_difference"] > 0.25
            or evidence["training_max_margin_difference"] > 0.25):
        raise RuntimeError("Zero-step loader/training parity failed; inspect loader-parity.json")
    return evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--train-pairs", type=int, default=16384)
    ap.add_argument("--validation-impressions", type=int, default=256)
    ap.add_argument("--dev-start", type=int, default=5000)
    ap.add_argument("--dev-impressions", type=int, default=1000)
    args = ap.parse_args()
    if min(args.train_pairs, args.validation_impressions, args.dev_impressions) < 1 or args.dev_start < 0:
        ap.error("Positive sample limits and nonnegative dev start are required")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Refusing to overwrite a completed probe")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    train_dir, dev_dir = args.data_root / "train", args.data_root / "dev"
    with (train_dir / "behaviors.tsv").open(encoding="utf-8") as handle:
        train_impressions = sum(1 for _ in handle)
    if train_impressions <= args.validation_impressions:
        raise ValueError("Train holdout must leave training impressions available")
    validation_records = behavior_rows(
        train_dir / "behaviors.tsv", load_news(str(train_dir / "news.tsv"), True),
        train_impressions - args.validation_impressions, args.validation_impressions,
    )
    validation_ids = {group for group, _ in validation_records}
    scorer = JevScorer(args.model_path, prompt_contract="legacy", max_prompt_tokens=4095)
    for parameter in scorer.model.parameters():
        parameter.requires_grad_(False)
    dataset = JEVNoulPointwiseDataset(
        str(train_dir / "behaviors.tsv"), str(train_dir / "news.tsv"), scorer.tokenizer,
        max_len=4096, sample=-1, seed=42, max_history=30, neg_ratio=1.0,
        hard_neg_ratio=0.5, use_abstract=True, use_chat_template=False,
    )
    grouped = defaultdict(list)
    for sample in dataset.samples:
        if sample["impression_id"] not in validation_ids:
            grouped[sample["impression_id"]].append(sample)
    training_rows = []
    for samples in grouped.values():
        if {sample["label"] for sample in samples} != {0, 1}:
            continue
        if len(training_rows) + len(samples) > args.train_pairs:
            break
        training_rows.extend({
            "impression_id": s["impression_id"], "candidate_id": s["candidate_id"],
            "label": s["label"], "state": build_jev_state(s["history"], s["candidate"]),
        } for s in samples)
    if not training_rows:
        raise ValueError("No complete mixed-label training groups fit the sample budget")
    train_ids = {row["impression_id"] for row in training_rows}
    if train_ids & validation_ids:
        raise RuntimeError("Training/selection impression leakage")
    del dataset, grouped
    parity = loader_parity(scorer, training_rows, output, args.model_path)
    train_x, train_base = extract(scorer, training_rows, output, "train")
    validation_rows = [row for _, rows in validation_records for row in rows]
    validation_x, validation_base = extract(scorer, validation_rows, output, "train_holdout")
    labels = np.asarray([r["label"] for r in training_rows])
    groups = [r["impression_id"] for r in training_rows]
    val_labels = np.asarray([r["label"] for r in validation_rows])
    val_groups = [r["impression_id"] for r in validation_rows]
    candidates = [{
        "name": "frozen", "head": None,
        "validation": metrics(val_labels, validation_base, val_groups),
    }]
    for objective in ("binary", "pairwise"):
        for regularization in (100.0, 10.0, 1.0, 0.1, 0.01):
            head, loss = fit_head(train_x, train_base, labels, groups, regularization, objective)
            validation = metrics(val_labels, predict(head, validation_x, validation_base), val_groups)
            name = f"{objective}_l2_{regularization:g}"
            np.savez(output / (name + ".head.npz"), **head)
            candidates.append({
                "name": name, "head": head, "validation": validation, "training_loss": loss,
            })
            print(json.dumps({"stage": "head_selection", "name": name, "validation": validation}), flush=True)
    selected = max(candidates, key=lambda candidate: candidate["validation"]["auc"])
    # Model selection is finished before any new dev features or labels are read.
    save_json(output / "train-selection.json", {
        "selected": selected["name"], "candidates": [
            {key: value for key, value in c.items() if key != "head"} for c in candidates
        ],
    })
    dev_records = behavior_rows(
        dev_dir / "behaviors.tsv", load_news(str(dev_dir / "news.tsv"), True),
        args.dev_start, args.dev_impressions,
    )
    dev_rows = [row for _, rows in dev_records for row in rows]
    dev_x, dev_base = extract(scorer, dev_rows, output, "dev")
    dev_labels = np.asarray([r["label"] for r in dev_rows])
    dev_groups = [r["impression_id"] for r in dev_rows]
    scores = predict(selected["head"], dev_x, dev_base)
    frozen_metrics = metrics(dev_labels, dev_base, dev_groups)
    selected_metrics = metrics(dev_labels, scores, dev_groups)
    with (output / "dev.scores.jsonl").open("w", encoding="utf-8") as handle:
        position = 0
        for impression, rows in dev_records:
            end = position + len(rows)
            handle.write(json.dumps({
                "impression_id": impression,
                "candidate_ids": [row["candidate_id"] for row in rows],
                "labels": [row["label"] for row in rows],
                "frozen_margins": dev_base[position:end].tolist(),
                "selected_margins": scores[position:end].tolist(),
            }, allow_nan=False) + "\n")
            position = end
    evidence = {
        "source_revision": args.source_revision,
        "backbone": "frozen; only CPU residual readouts fitted", "loader_parity": parity,
        "training_pairs": len(training_rows), "training_impressions": len(train_ids),
        "train_holdout_impressions": len(validation_ids),
        "selection": "maximum train-holdout AUC; frozen candidate wins ties; no dev tuning",
        "selected": selected["name"], "dev_start_zero_based": args.dev_start,
        "dev_impressions": len(dev_records), "dev_candidates": len(dev_rows),
        "frozen": frozen_metrics, "selected_metrics": selected_metrics,
        "delta": {key: selected_metrics[key] - frozen_metrics[key] for key in frozen_metrics},
        "data_sha256": {
            f"{split}/{name}": hashlib.sha256((args.data_root / split / name).read_bytes()).hexdigest()
            for split in ("train", "dev") for name in ("news.tsv", "behaviors.tsv")
        },
        "limitations": [
            "Frozen-backbone residual readout, not native JEV two-exit training or full-model SFT.",
            "Unseen dev window 5001-6000 by default, not the locked first5000 comparison.",
            "One deterministic training subset; no SOTA or repeated-seed claim.",
        ],
    }
    save_json(output / "result.json", evidence)
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
