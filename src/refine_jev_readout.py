"""Refine a frozen readout on verified training caches, without reading dev."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_jev_readout import load_head
from probe_jev_readout import fit_head, metrics, predict, save_json
from scale_jev_readout import load_cached_features, sha256_file


CANDIDATES = (
    ("binary", 0.1), ("binary", 0.001),
    ("pairwise", 0.1), ("pairwise", 0.01), ("pairwise", 0.001),
)
REFERENCE_NAME = "scaled_reference"


def load_inputs(cache, provenance_path, provenance_sha256, head_path, head_sha256, expected_rows):
    if sha256_file(provenance_path) != provenance_sha256:
        raise ValueError("Training provenance checksum differs")
    provenance = json.loads(provenance_path.read_bytes())
    if (not provenance["training_only"] or not provenance["scaled_head_opens_dev_gate"]
            or provenance["scaled_head_sha256"] != head_sha256):
        raise ValueError("Reference is not the pinned, selected training-scale head")
    head, head_bytes = load_head(head_path, head_sha256)
    required = {
        "train.features.npz": provenance["train_features_sha256"],
        "train_holdout.features.npz": provenance["feature_cache_sha256"]["train_holdout.features.npz"],
        "train_holdout.rows.json": provenance["feature_cache_sha256"]["train_holdout.rows.json"],
    }
    for name, expected in required.items():
        if sha256_file(cache / name) != expected:
            raise ValueError("Training cache checksum differs: " + name)
    rows = json.loads((cache / "train.rows.json").read_bytes())
    holdout = json.loads((cache / "train_holdout.rows.json").read_bytes())
    if rows != json.loads(expected_rows.read_bytes()):
        raise ValueError("Training IDs, order or labels differ from the independently audited sample")
    if (len(rows) != provenance["training_pairs"]
            or len({r["impression_id"] for r in rows}) != provenance["training_impressions"]
            or len(holdout) != provenance["holdout_candidates"]
            or len({r["impression_id"] for r in holdout}) != provenance["train_holdout_impressions"]):
        raise ValueError("Training or holdout coverage differs")
    if {r["impression_id"] for r in rows} & {r["impression_id"] for r in holdout}:
        raise ValueError("Training/holdout impression leakage")
    for split in (rows, holdout):
        groups = defaultdict(set)
        for row in split:
            groups[row["impression_id"]].add(row["label"])
        if any(labels != {0, 1} for labels in groups.values()):
            raise ValueError("Every impression must have both binary classes")
    dimension = len(head["weight"])
    x, base = load_cached_features(cache, "train", rows, dimension)
    holdout_x, holdout_base = load_cached_features(cache, "train_holdout", holdout, dimension)
    scores = predict(head, holdout_x, holdout_base)
    actual = metrics(
        [r["label"] for r in holdout], scores, [r["impression_id"] for r in holdout],
    )
    if any(abs(actual[key] - provenance["scaled"][key]) > 1e-12 for key in actual):
        raise ValueError("Pinned reference does not reproduce its verified training-holdout metrics")
    return provenance, head_bytes, rows, holdout, x, base, holdout_x, holdout_base, scores


def refine(cache, provenance_path, provenance_sha256, head_path, head_sha256,
           expected_rows, parity_path, output, source_revision):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refinement output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    parity = json.loads(parity_path.read_bytes())
    if (not parity["head_rows_identical"] or parity["loader_max_margin_difference"] > 0.25
            or parity["training_max_margin_difference"] > 0.25):
        raise ValueError("Original frozen-loader parity evidence is invalid")
    provenance, head_bytes, rows, holdout, x, base, hx, hb, reference = load_inputs(
        cache, provenance_path, provenance_sha256, head_path, head_sha256, expected_rows,
    )
    protocol = {
        "source_revision": source_revision, "training_only": True,
        "training_pairs": len(rows), "train_holdout_impressions": provenance["train_holdout_impressions"],
        "reference_head_sha256": head_sha256, "training_provenance_sha256": provenance_sha256,
        "cache_sha256": {p.name: sha256_file(p) for p in (
            cache / "train.features.npz", cache / "train.rows.json",
            cache / "train_holdout.features.npz", cache / "train_holdout.rows.json",
        )},
        "candidates": [{"objective": obj, "regularization": reg} for obj, reg in CANDIDATES],
        "optimizer": "Existing CPU float64 LBFGS,100 iterations maximum,zero initialization",
        "selection": "Maximum reused training-holdout AUC; scaled reference wins ties",
        "dev_evaluation": "none; no dev files or dev feature caches are read",
        "minimum_dev_start": 16000,
    }
    save_json(output / "protocol.json", protocol)
    (output / (REFERENCE_NAME + ".head.npz")).write_bytes(head_bytes)
    (output / "loader-parity.json").write_bytes(parity_path.read_bytes())
    labels = np.asarray([r["label"] for r in rows])
    groups = [r["impression_id"] for r in rows]
    hy = np.asarray([r["label"] for r in holdout])
    hg = [r["impression_id"] for r in holdout]
    candidates = [{"name": REFERENCE_NAME, "validation": provenance["scaled"]}]
    all_scores = {REFERENCE_NAME: reference}
    for objective, regularization in CANDIDATES:
        name = f"{objective}_l2_{regularization:g}_refined"
        print(json.dumps({"stage": "fitting", "candidate": name}), flush=True)
        head, loss = fit_head(x, base, labels, groups, regularization, objective)
        scores = predict(head, hx, hb)
        values = metrics(hy, scores, hg)
        np.savez(output / (name + ".head.npz"), **head)
        candidates.append({"name": name, "validation": values, "training_loss": loss})
        all_scores[name] = scores
        print(json.dumps({"stage": "candidate_completed", "candidate": name, "validation": values}), flush=True)
        del head
    selected = max(candidates, key=lambda candidate: candidate["validation"]["auc"])
    save_json(output / "train-selection.json", {
        "selected": selected["name"], "candidates": candidates, "rule": protocol["selection"],
        "minimum_dev_start": protocol["minimum_dev_start"],
    })
    grouped = defaultdict(list)
    for index, row in enumerate(holdout):
        grouped[row["impression_id"]].append(index)
    with (output / "train-holdout.scores.jsonl").open("w", encoding="utf-8") as handle:
        for impression, indices in grouped.items():
            handle.write(json.dumps({
                "impression_id": impression,
                "candidate_ids": [holdout[i]["candidate_id"] for i in indices],
                "labels": hy[indices].tolist(), "frozen_margins": hb[indices].tolist(),
                "candidate_margins": {name: scores[indices].tolist() for name, scores in all_scores.items()},
            }, allow_nan=False) + "\n")
    indices = np.sort(np.random.default_rng(42).choice(len(holdout), min(128, len(holdout)), replace=False))
    np.savez(
        output / "train-holdout-spotcheck.npz", indices=indices,
        features=hx[indices], margins=hb[indices], selected=all_scores[selected["name"]][indices],
    )
    result = {
        "source_revision": source_revision, "training_only": True,
        "training_pairs": len(rows), "training_impressions": provenance["training_impressions"],
        "train_holdout_impressions": provenance["train_holdout_impressions"],
        "holdout_candidates": len(holdout), "selected": selected["name"],
        "reference_head_sha256": head_sha256, "reference": provenance["scaled"],
        "selected_metrics": selected["validation"],
        "selected_head_sha256": sha256_file(output / (selected["name"] + ".head.npz")),
        "refined_head_opens_dev_gate": selected["name"] != REFERENCE_NAME,
        "data_sha256": provenance["data_sha256"],
        "training_provenance_sha256": provenance_sha256,
        "cache_sha256": protocol["cache_sha256"],
        "limitations": [
            "Reused training-holdout selection,not independent dev or SOTA evidence.",
            "Same training sample/backbone; objective and regularization search only.",
            "One sample; no repeated-seed robustness claim.",
        ],
    }
    save_json(output / "result.json", result)
    print(json.dumps({"stage": "refinement_completed", **result}), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--training-result", required=True, type=Path)
    ap.add_argument("--training-result-sha256", required=True)
    ap.add_argument("--reference-head", required=True, type=Path)
    ap.add_argument("--reference-head-sha256", required=True)
    ap.add_argument("--expected-training-rows", required=True, type=Path)
    ap.add_argument("--loader-parity", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--source-revision", required=True)
    args = ap.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    refine(
        args.cache, args.training_result, args.training_result_sha256,
        args.reference_head, args.reference_head_sha256, args.expected_training_rows,
        args.loader_parity, args.output, args.source_revision,
    )


if __name__ == "__main__":
    main()
