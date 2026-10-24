"""Train a small nonlinear correction using pinned training features only."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from evaluate_jev_readout import load_head
from probe_jev_readout import metrics, predict, save_json
from refine_jev_readout import load_inputs, REFERENCE_NAME
from scale_jev_readout import sha256_file


CHECKPOINTS = (4, 8, 16)
WIDTH = 16
BATCH_SIZE = 2048
MINIMUM_AUC_GAIN = 0.005
REFERENCE_FIELDS = ("mean", "scale", "weight", "bias")
FIELDS = set(REFERENCE_FIELDS) | {"hidden_weight", "hidden_bias", "output_weight", "output_bias"}


def validate_head(head):
    if set(head) != FIELDS:
        raise ValueError("Unexpected nonlinear-readout fields")
    if any(value.dtype.kind != "f" or not np.isfinite(value).all() for value in head.values()):
        raise ValueError("Nonlinear-readout parameters must be finite floating-point arrays")
    shape = head["weight"].shape
    hidden = head["hidden_bias"].shape
    if (len(shape) != 1 or not shape[0] or len(hidden) != 1 or not hidden[0]
            or head["mean"].shape != shape or head["scale"].shape != shape
            or not (head["scale"] > 0).all() or head["bias"].shape != ()
            or head["hidden_weight"].shape != (shape[0], hidden[0])
            or head["output_weight"].shape != hidden or head["output_bias"].shape != ()):
        raise ValueError("Invalid nonlinear-readout dimensions or scale")


def load_nonlinear_head(path, expected_sha256):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Nonlinear-readout checksum mismatch")
    with np.load(path, allow_pickle=False) as stored:
        head = dict(stored)
    validate_head(head)
    return head, data


def predict_nonlinear(head, features, margins):
    validate_head(head)
    if (features.ndim != 2 or features.shape[1:] != head["weight"].shape
            or margins.shape != (len(features),)
            or not np.isfinite(features).all() or not np.isfinite(margins).all()):
        raise ValueError("Expected finite aligned nonlinear-readout features and margins")
    scores = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), BATCH_SIZE):
        end = start + BATCH_SIZE
        x = ((features[start:end] - head["mean"]) / head["scale"]).astype(np.float32)
        hidden = np.maximum(x @ head["hidden_weight"] + head["hidden_bias"], 0)
        scores[start:end] = (
            predict(head, features[start:end], margins[start:end])
            + hidden @ head["output_weight"] + head["output_bias"]
        )
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonlinear-readout predictions are nonfinite")
    return scores


def fit_checkpoints(features, margins, labels, reference, checkpoints=CHECKPOINTS, deadline=None):
    if (features.ndim != 2 or features.shape[1:] != reference["weight"].shape
            or margins.shape != labels.shape or labels.shape != (len(features),)
            or not len(features) or set(np.unique(labels)) != {0, 1}
            or not all(np.isfinite(value).all() for value in (features, margins, labels))
            or not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints
            or checkpoints[0] < 1):
        raise ValueError("Expected aligned mixed-label training data and increasing positive checkpoints")
    base = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), BATCH_SIZE):
        end = start + BATCH_SIZE
        base[start:end] = predict(reference, features[start:end], margins[start:end])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        network = nn.Sequential(
            nn.Linear(features.shape[1], WIDTH), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(WIDTH, 1),
        )
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.01)
        generator = torch.Generator().manual_seed(42)
        for epoch in range(1, checkpoints[-1] + 1):
            network.train()
            order = torch.randperm(len(features), generator=generator).numpy()
            total_loss = 0.
            for start in range(0, len(features), BATCH_SIZE):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Nonlinear training exceeded its fixed CPU deadline")
                indices = order[start:start + BATCH_SIZE]
                x = ((features[indices] - reference["mean"]) / reference["scale"]).astype(np.float32)
                optimizer.zero_grad()
                logits = torch.from_numpy(base[indices]) + network(torch.from_numpy(x)).squeeze(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.from_numpy(labels[indices].astype(np.float64)),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonlinear training loss is nonfinite")
                loss.backward()
                if any(not torch.isfinite(parameter.grad).all() for parameter in network.parameters()):
                    raise FloatingPointError("Nonlinear training gradients are nonfinite")
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
            print(json.dumps({
                "stage": "nonlinear_epoch", "epoch": epoch,
                "training_loss": total_loss / len(features),
            }), flush=True)
            if epoch in checkpoints:
                head = {key: reference[key].copy() for key in REFERENCE_FIELDS}
                head.update(
                    hidden_weight=network[0].weight.detach().numpy().T.copy(),
                    hidden_bias=network[0].bias.detach().numpy().copy(),
                    output_weight=network[-1].weight.detach().numpy()[0].copy(),
                    output_bias=np.asarray(float(network[-1].bias.detach()[0]), dtype=np.float32),
                )
                validate_head(head)
                yield epoch, head, total_loss / len(features)


def select_candidate(candidates):
    reference = candidates[0]
    if reference["name"] != REFERENCE_NAME:
        raise ValueError("The pinned reference must be the first selection candidate")
    best = max(candidates, key=lambda candidate: candidate["validation"]["auc"])
    opens = (
        best["validation"]["auc"] - reference["validation"]["auc"] >= MINIMUM_AUC_GAIN
        and all(best["validation"][key] >= reference["validation"][key]
                for key in ("mrr", "ndcg5", "ndcg10"))
    )
    return (best if opens else reference), best


def train(cache, provenance_path, provenance_sha256, head_path, head_sha256,
          expected_rows, parity_path, output, source_revision):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonlinear training output must be empty")
    deadline = time.monotonic() + 1800
    provenance, head_bytes, rows, holdout, x, base, hx, hb, reference_scores = load_inputs(
        cache, provenance_path, provenance_sha256, head_path, head_sha256, expected_rows,
    )
    parity = json.loads(parity_path.read_bytes())
    differences = np.asarray([
        parity["loader_max_margin_difference"], parity["training_max_margin_difference"],
    ], dtype=float)
    if (not parity["head_rows_identical"] or not np.isfinite(differences).all()
            or (differences < 0).any() or (differences > 0.25).any()):
        raise ValueError("Original frozen-loader parity evidence is invalid")
    reference_head, _ = load_head(head_path, head_sha256)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "source_revision": source_revision, "training_only": True,
        "training_pairs": len(rows), "training_provenance_sha256": provenance_sha256,
        "reference_head_sha256": head_sha256,
        "architecture": f"Retained linear margin + Linear({len(reference_head['weight'])},16)/ReLU/Dropout(0.1)/Linear(16,1)",
        "initialization": "seed42;zero output layer preserves reference initially",
        "optimizer": "CPU AdamW,lr0.001,weight_decay0.01,batch2048,seed42",
        "checkpoints": list(CHECKPOINTS), "cpu_threads_maximum": 4,
        "timeout_seconds": 1800, "minimum_dev_start": 26000,
        "selection": "Best reused-training-holdout AUC;reference wins ties",
        "dev_gate": "Best candidate needs AUC gain>=0.005 and no MRR/nDCG5/nDCG10 regression;otherwise retain reference",
        "dev_evaluation": "none;no dev data or scores read;no automatic GPU submission",
        "cache_sha256": {p.name: sha256_file(p) for p in (
            cache / "train.features.npz", cache / "train.rows.json",
            cache / "train_holdout.features.npz", cache / "train_holdout.rows.json",
        )},
    }
    save_json(output / "protocol.json", protocol)
    (output / (REFERENCE_NAME + ".head.npz")).write_bytes(head_bytes)
    (output / "loader-parity.json").write_bytes(parity_path.read_bytes())
    labels = np.asarray([row["label"] for row in rows])
    hy = np.asarray([row["label"] for row in holdout])
    hg = [row["impression_id"] for row in holdout]
    candidates = [{"name": REFERENCE_NAME, "validation": provenance["scaled"]}]
    all_scores = {REFERENCE_NAME: reference_scores}
    for epoch, head, loss in fit_checkpoints(x, base, labels, reference_head, deadline=deadline):
        name = f"mlp16_epoch{epoch}"
        scores = predict_nonlinear(head, hx, hb)
        values = metrics(hy, scores, hg)
        np.savez(output / (name + ".head.npz"), **head)
        candidates.append({"name": name, "validation": values, "training_loss": loss})
        all_scores[name] = scores
        print(json.dumps({"stage": "nonlinear_candidate", "name": name, "validation": values}), flush=True)
    selected, best = select_candidate(candidates)
    save_json(output / "train-selection.json", {
        "selected": selected["name"], "best_candidate": best["name"], "candidates": candidates,
        "rule": protocol["selection"], "dev_gate": protocol["dev_gate"], "minimum_dev_start": 26000,
    })
    grouped = defaultdict(list)
    for index, row in enumerate(holdout):
        grouped[row["impression_id"]].append(index)
    with (output / "train-holdout.scores.jsonl").open("w", encoding="utf-8") as handle:
        for impression, indices in grouped.items():
            handle.write(json.dumps({
                "impression_id": impression, "candidate_ids": [holdout[i]["candidate_id"] for i in indices],
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
        "holdout_candidates": len(holdout), "selected": selected["name"], "best_candidate": best["name"],
        "reference_head_sha256": head_sha256, "reference": provenance["scaled"],
        "selected_metrics": selected["validation"], "best_candidate_metrics": best["validation"],
        "selected_head_sha256": sha256_file(output / (selected["name"] + ".head.npz")),
        "nonlinear_head_opens_dev_gate": selected["name"] != REFERENCE_NAME,
        "training_provenance_sha256": provenance_sha256,
        "candidate_head_sha256": {
            candidate["name"]: sha256_file(output / (candidate["name"] + ".head.npz"))
            for candidate in candidates
        },
        "data_sha256": provenance["data_sha256"], "cache_sha256": protocol["cache_sha256"],
        "limitations": [
            "Reused training holdout,not independent dev/SOTA or repeated-seed evidence.",
            "Three fixed checkpoints on one seed;selection may overfit the holdout.",
            "Only cached training features;the backbone and retained linear head stay frozen.",
        ],
    }
    save_json(output / "result.json", result)
    print(json.dumps({"stage": "nonlinear_training_completed", **result}), flush=True)
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
    train(
        args.cache, args.training_result, args.training_result_sha256,
        args.reference_head, args.reference_head_sha256, args.expected_training_rows,
        args.loader_parity, args.output, args.source_revision,
    )


if __name__ == "__main__":
    main()
