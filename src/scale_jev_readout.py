"""Scale only the frozen readout's training data; never evaluate dev here."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_jev_readout import load_head
from evaluate_mind_jev import JevScorer
from jev_noul_data import JEVNoulPointwiseDataset
from mind_utils import load_news
from probe_jev_readout import (
    behavior_rows, extract, fit_head, metrics, predict, sample_training_rows, save_json,
)


TRAIN_PAIR_LIMIT = 131072
REFERENCE_RESULT_SHA256 = "67ceef2c76caf9068a1f16af8eac2dfd0a006aef4d0661ecc753dbe72f1620a6"
REFERENCE_HEAD_SHA256 = "d7796a02baeed17a72acc42b23ae70f494654560dd3c2f39528cc54febe48f38"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cached_features(root, name, rows, dimension):
    identities = [
        {key: row[key] for key in ("impression_id", "candidate_id", "label")}
        for row in rows
    ]
    if json.loads((root / (name + ".rows.json")).read_bytes()) != identities:
        raise ValueError("Cached feature rows differ from the exact deterministic training sample")
    with np.load(root / (name + ".features.npz"), allow_pickle=False) as stored:
        features, margins = stored["features"], stored["margins"]
    if (features.shape != (len(rows), dimension) or margins.shape != (len(rows),)
            or not np.isfinite(features).all() or not np.isfinite(margins).all()):
        raise ValueError("Cached feature dimensions or numeric values are invalid")
    return features, margins


def check_cached_model(scorer, rows, features, margins, name):
    actual_x, actual_base = extract(scorer, rows[:16], None, name, report_progress=False)
    if (not np.array_equal(actual_x, features[:16])
            or not np.array_equal(actual_base, margins[:16])):
        raise ValueError("Cached features differ from a fresh identical-batch frozen-model replay")


def write_holdout_scores(output, records, margins, reference, scaled):
    position = 0
    with (output / "train-holdout.scores.jsonl").open("w", encoding="utf-8") as handle:
        for impression, rows in records:
            end = position + len(rows)
            handle.write(json.dumps({
                "impression_id": impression,
                "candidate_ids": [row["candidate_id"] for row in rows],
                "labels": [row["label"] for row in rows],
                "frozen_margins": margins[position:end].tolist(),
                "reference_margins": reference[position:end].tolist(),
                "scaled_margins": scaled[position:end].tolist(),
            }, allow_nan=False) + "\n")
            position = end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--reference-head", type=Path, required=True)
    ap.add_argument("--reference-head-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source-revision", required=True)
    args = ap.parse_args()
    if args.reference_head_sha256 != REFERENCE_HEAD_SHA256:
        ap.error("This data-only experiment requires the confirmed original head")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Training-only scale output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    reference_head, head_bytes = load_head(args.reference_head, args.reference_head_sha256)
    reference_root = args.reference_head.parent
    if sha256_file(reference_root / "result.json") != REFERENCE_RESULT_SHA256:
        raise ValueError("Reference pilot provenance checksum differs")
    original = json.loads((reference_root / "result.json").read_bytes())
    selection = json.loads((reference_root / "train-selection.json").read_bytes())
    if (original["selected"] != "binary_l2_0.01" or selection["selected"] != original["selected"]
            or original["training_pairs"] != 16383 or original["train_holdout_impressions"] != 256):
        raise ValueError("Reference training protocol differs")
    train_dir = args.data_root / "train"
    data_hashes = {}
    for filename in ("news.tsv", "behaviors.tsv"):
        key = "train/" + filename
        data_hashes[key] = sha256_file(train_dir / filename)
        if data_hashes[key] != original["data_sha256"][key]:
            raise ValueError("Training inputs differ from the original head's inputs")
    with (train_dir / "behaviors.tsv").open(encoding="utf-8") as handle:
        train_count = sum(1 for _ in handle)
    records = behavior_rows(
        train_dir / "behaviors.tsv", load_news(str(train_dir / "news.tsv"), True),
        train_count - 256, 256,
    )
    validation_ids = {impression for impression, _ in records}
    validation_rows = [row for _, rows in records for row in rows]
    scorer = JevScorer(args.model_path, prompt_contract="legacy", max_prompt_tokens=4095)
    for parameter in scorer.model.parameters():
        parameter.requires_grad_(False)
    dataset = JEVNoulPointwiseDataset(
        str(train_dir / "behaviors.tsv"), str(train_dir / "news.tsv"), scorer.tokenizer,
        max_len=4096, sample=-1, seed=42, max_history=30, neg_ratio=1.0,
        hard_neg_ratio=0.5, use_abstract=True, use_chat_template=False,
    )
    rows = sample_training_rows(dataset.samples, validation_ids, TRAIN_PAIR_LIMIT)
    del dataset
    cached_count = original["training_pairs"]
    if len(rows) <= cached_count:
        raise ValueError("Scaling must add training pairs")
    dimension = len(reference_head["weight"])
    cached_x, cached_base = load_cached_features(
        reference_root, "train", rows[:cached_count], dimension,
    )
    validation_x, validation_base = load_cached_features(
        reference_root, "train_holdout", validation_rows, dimension,
    )
    check_cached_model(scorer, rows, cached_x, cached_base, "train_cache_replay")
    check_cached_model(scorer, validation_rows, validation_x, validation_base, "holdout_cache_replay")
    labels = np.asarray([row["label"] for row in validation_rows])
    groups = [row["impression_id"] for row in validation_rows]
    reference_scores = predict(reference_head, validation_x, validation_base)
    reference_metrics = metrics(labels, reference_scores, groups)
    expected_metrics = next(c["validation"] for c in selection["candidates"]
                            if c["name"] == original["selected"])
    if any(abs(reference_metrics[key] - expected_metrics[key]) > 1e-12
           for key in reference_metrics):
        raise ValueError("Reference holdout metrics do not reproduce the original selection")
    cache_hashes = {
        name: sha256_file(reference_root / name)
        for name in ("train.features.npz", "train.rows.json",
                     "train_holdout.features.npz", "train_holdout.rows.json")
    }
    save_json(args.output / "protocol.json", {
        "source_revision": args.source_revision, "training_only": True,
        "training_pair_limit": TRAIN_PAIR_LIMIT, "training_pairs": len(rows),
        "cached_training_pairs": cached_count,
        "reference_head_sha256": args.reference_head_sha256,
        "reference_result_sha256": REFERENCE_RESULT_SHA256,
        "cache_sha256": cache_hashes, "data_sha256": data_hashes,
        "objective": "binary", "regularization": 0.01, "seed": 42,
        "train_holdout_impressions": 256, "fresh_cache_replay_rows_per_split": 16,
        "dev_evaluation": "none; selection uses training holdout only",
    })
    new_x, new_base = extract(scorer, rows[cached_count:], None, "new_training_features")
    train_x = np.concatenate([cached_x, new_x])
    train_base = np.concatenate([cached_base, new_base])
    del new_x, new_base, cached_x, cached_base
    np.savez(args.output / "train.features.npz", features=train_x, margins=train_base)
    save_json(args.output / "train.rows.json", [
        {key: row[key] for key in ("impression_id", "candidate_id", "label")}
        for row in rows
    ])
    head, loss = fit_head(
        train_x, train_base, np.asarray([row["label"] for row in rows]),
        [row["impression_id"] for row in rows], 0.01, "binary",
    )
    scaled_scores = predict(head, validation_x, validation_base)
    scaled_metrics = metrics(labels, scaled_scores, groups)
    candidate_name = "binary_l2_0.01_scaled"
    np.savez(args.output / (candidate_name + ".head.npz"), **head)
    (args.output / "reference_readout.head.npz").write_bytes(head_bytes)
    gate = scaled_metrics["auc"] > reference_metrics["auc"]
    selected = candidate_name if gate else "reference_readout"
    save_json(args.output / "train-selection.json", {
        "selected": selected,
        "candidates": [
            {"name": "reference_readout", "validation": reference_metrics},
            {"name": candidate_name, "validation": scaled_metrics, "training_loss": loss},
        ],
        "rule": "Strictly larger train-holdout AUC; confirmed reference wins exact ties",
    })
    (args.output / "loader-parity.json").write_bytes(
        (reference_root / "loader-parity.json").read_bytes()
    )
    write_holdout_scores(
        args.output, records, validation_base, reference_scores, scaled_scores,
    )
    indices = np.sort(np.random.default_rng(42).choice(
        len(validation_rows), min(128, len(validation_rows)), replace=False,
    ))
    np.savez(
        args.output / "train-holdout-spotcheck.npz", indices=indices,
        features=validation_x[indices], margins=validation_base[indices],
        reference=reference_scores[indices], scaled=scaled_scores[indices],
    )
    save_json(args.output / "result.json", {
        "source_revision": args.source_revision, "training_only": True,
        "training_pairs": len(rows), "training_impressions": len({r["impression_id"] for r in rows}),
        "cached_training_pairs": cached_count, "new_training_pairs": len(rows) - cached_count,
        "train_holdout_impressions": 256, "holdout_candidates": len(validation_rows),
        "selected": selected, "scaled_head_opens_dev_gate": gate,
        "reference_head_sha256": args.reference_head_sha256,
        "scaled_head_sha256": sha256_file(args.output / (candidate_name + ".head.npz")),
        "reference": reference_metrics, "scaled": scaled_metrics,
        "frozen": metrics(labels, validation_base, groups),
        "data_sha256": data_hashes, "feature_cache_sha256": cache_hashes,
        "train_features_sha256": sha256_file(args.output / "train.features.npz"),
        "limitations": [
            "Training-only selection evidence, not a new dev or SOTA result.",
            "Same256-impression training holdout as original selection; not independent robustness.",
            "Only data quantity changes: binary L2 0.01,seed42,unchanged frozen backbone.",
        ],
    })
    print(json.dumps({
        "stage": "training_scale_completed", "selected": selected,
        "reference": reference_metrics, "scaled": scaled_metrics,
        "scaled_head_opens_dev_gate": gate,
    }), flush=True)


if __name__ == "__main__":
    main()
