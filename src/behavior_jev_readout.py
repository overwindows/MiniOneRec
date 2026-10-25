"""Fit cross-fitted item-popularity corrections without reading dev data."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_jev_readout import load_head
from nonlinear_jev_readout import select_candidate
from probe_jev_readout import fit_head, metrics, predict, save_json
from refine_jev_readout import load_inputs, REFERENCE_NAME
from scale_jev_readout import sha256_file


FOLDS = 5
FEATURE_NAMES = ("log_exposure_rate", "log_click_rate", "smoothed_ctr_log_odds", "known_item")


def impression_fold(impression):
    return int.from_bytes(hashlib.sha256(("42:" + impression).encode()).digest()[:8], "big") % FOLDS


def build_counts(path, excluded, expected_rows=()):
    expected = defaultdict(list)
    for row in expected_rows:
        expected[row["impression_id"]].append((row["candidate_id"], row["label"]))
    counts, seen = {}, set()
    totals = np.zeros((FOLDS + 1, 2), dtype=np.int64)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 5 or not fields[0] or not fields[4]:
                raise ValueError(f"Malformed training behavior at line {number}")
            impression = fields[0]
            if impression in seen:
                raise ValueError("Duplicate training impression: " + impression)
            seen.add(impression)
            items = []
            for token in fields[4].split():
                news, label = token.rsplit("-", 1)
                if not news or label not in ("0", "1"):
                    raise ValueError(f"Invalid training candidate at line {number}")
                items.append((news, int(label)))
            if len(dict(items)) != len(items):
                raise ValueError("Duplicate candidate in training impression: " + impression)
            if impression in expected:
                actual = dict(items)
                if any(actual.get(news) != label for news, label in expected[impression]):
                    raise ValueError("Cached training candidate or label differs from source behaviors")
            if impression in excluded:
                continue
            fold = impression_fold(impression) + 1
            totals[0] += (len(items), sum(label for _, label in items))
            totals[fold] += (len(items), sum(label for _, label in items))
            for news, label in items:
                if news not in counts:
                    counts[news] = np.zeros_like(totals)
                counts[news][0] += (1, label)
                counts[news][fold] += (1, label)
    if not set(expected) <= seen or not set(excluded) <= seen:
        raise ValueError("Source behaviors do not cover every expected/excluded impression")
    if not counts or not 0 < totals[0, 1] < totals[0, 0]:
        raise ValueError("Popularity statistics require both training classes")
    return counts, totals


def behavior_features(rows, counts, totals, cross_fitted):
    values = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    empty = np.zeros((FOLDS + 1, 2), dtype=np.int64)
    for index, row in enumerate(rows):
        fold = impression_fold(row["impression_id"]) + 1 if cross_fitted else 0
        available = totals[0] - totals[fold] if cross_fitted else totals[0]
        item = counts.get(row["candidate_id"], empty)
        exposure, clicks = item[0] - item[fold] if cross_fitted else item[0]
        n, positive = available
        if (not 0 < positive < n or not 0 <= clicks <= exposure <= n):
            raise ValueError("Invalid fold-excluded item counts or background class coverage")
        prior = positive / n
        ctr = (clicks + 20. * prior) / (exposure + 20.)
        values[index] = (
            np.log1p(1e6 * exposure / n), np.log1p(1e6 * clicks / positive),
            np.log(ctr / (1. - ctr)) - np.log(prior / (1. - prior)), float(exposure > 0),
        )
    if not np.isfinite(values).all():
        raise FloatingPointError("Behavioral features are nonfinite")
    return values


def save_counts(path, counts, totals):
    names = sorted(counts)
    np.savez(path, item_ids=np.asarray(names), counts=np.stack([counts[name] for name in names]), totals=totals)


def load_counts(path, expected_sha256):
    if sha256_file(path) != expected_sha256:
        raise ValueError("Popularity-statistics checksum differs")
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != {"item_ids", "counts", "totals"}:
            raise ValueError("Unexpected popularity-statistics fields")
        names, counts, totals = stored["item_ids"], stored["counts"], stored["totals"]
    if (names.ndim != 1 or names.dtype.kind != "U" or not len(names)
            or len(set(names.tolist())) != len(names) or not all(names)
            or counts.shape != (len(names), FOLDS + 1, 2) or totals.shape != (FOLDS + 1, 2)
            or counts.dtype.kind not in "iu" or totals.dtype.kind not in "iu"
            or (counts < 0).any() or (totals < 0).any()
            or (counts[:, :, 1] > counts[:, :, 0]).any()
            or not np.array_equal(counts.sum(axis=0), totals)
            or not np.array_equal(counts[:, 1:].sum(axis=1), counts[:, 0])
            or not np.array_equal(totals[1:].sum(axis=0), totals[0])):
        raise ValueError("Invalid popularity-statistics dimensions or count identities")
    return dict(zip(names.tolist(), counts)), totals


def train(cache, provenance_path, provenance_sha256, head_path, head_sha256,
          expected_rows, train_behaviors, output, source_revision):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Behavioral-readout output must be empty")
    provenance, reference_bytes, rows, holdout, x, base, hx, hb, reference_scores = load_inputs(
        cache, provenance_path, provenance_sha256, head_path, head_sha256, expected_rows,
    )
    if sha256_file(train_behaviors) != provenance["data_sha256"]["train/behaviors.tsv"]:
        raise ValueError("Training behaviors checksum differs from frozen-feature provenance")
    reference_head, _ = load_head(head_path, head_sha256)
    training_scores = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(rows), 2048):
        end = start + 2048
        training_scores[start:end] = predict(reference_head, x[start:end], base[start:end])
    del x, hx
    excluded = {row["impression_id"] for row in holdout}
    counts, totals = build_counts(train_behaviors, excluded, rows + holdout)
    train_x = behavior_features(rows, counts, totals, cross_fitted=True)
    holdout_x = behavior_features(holdout, counts, totals, cross_fitted=False)
    output.mkdir(parents=True, exist_ok=True)
    save_counts(output / "population.npz", counts, totals)
    protocol = {
        "source_revision": source_revision, "training_only": True,
        "reference_head_sha256": head_sha256, "training_provenance_sha256": provenance_sha256,
        "train_behaviors_sha256": sha256_file(train_behaviors),
        "feature_names": list(FEATURE_NAMES), "folds": FOLDS,
        "fold_rule": "SHA256('42:'+impression_id),first8bytes big-endian modulo5",
        "fold_exclusion": "Fit rows exclude their entire impression fold;all256 holdout impressions excluded from every statistic",
        "smoothing_strength": 20, "population_sha256": sha256_file(output / "population.npz"),
        "candidates": ["binary_l2_0.01", "pairwise_l2_0.01"],
        "optimizer": "CPU float64 LBFGS,100 iterations maximum,4threads",
        "dev_gate": "Best training-holdout AUC gain>=0.005 AND all secondary metrics nondecreasing;otherwise retain exact reference",
        "minimum_dev_start": 26000, "dev_evaluation": "none;no dev inputs or scores read",
    }
    save_json(output / "protocol.json", protocol)
    (output / (REFERENCE_NAME + ".head.npz")).write_bytes(reference_bytes)
    labels = np.asarray([row["label"] for row in rows])
    groups = [row["impression_id"] for row in rows]
    hy = np.asarray([row["label"] for row in holdout])
    hg = [row["impression_id"] for row in holdout]
    candidates = [{"name": REFERENCE_NAME, "validation": provenance["scaled"]}]
    all_scores = {REFERENCE_NAME: reference_scores}
    for objective in ("binary", "pairwise"):
        name = f"behavior_{objective}_l2_0.01"
        head, loss = fit_head(train_x, training_scores, labels, groups, .01, objective)
        scores = predict(head, holdout_x, reference_scores)
        values = metrics(hy, scores, hg)
        np.savez(output / (name + ".head.npz"), **head)
        candidates.append({"name": name, "validation": values, "training_loss": loss})
        all_scores[name] = scores
        print(json.dumps({"stage": "behavior_candidate", "name": name, "validation": values}), flush=True)
    selected, best = select_candidate(candidates)
    save_json(output / "train-selection.json", {
        "selected": selected["name"], "best_candidate": best["name"], "candidates": candidates,
        "minimum_dev_start": 26000, "dev_gate": protocol["dev_gate"],
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
    np.savez(output / "behavior-features.npz", train=train_x, holdout=holdout_x)
    result = {
        "source_revision": source_revision, "training_only": True,
        "training_pairs": len(rows), "training_impressions": provenance["training_impressions"],
        "train_holdout_impressions": len(excluded), "holdout_candidates": len(holdout),
        "selected": selected["name"], "best_candidate": best["name"],
        "selected_metrics": selected["validation"], "best_candidate_metrics": best["validation"],
        "reference": provenance["scaled"], "reference_head_sha256": head_sha256,
        "training_provenance_sha256": provenance_sha256, "data_sha256": provenance["data_sha256"],
        "population_sha256": protocol["population_sha256"],
        "selected_head_sha256": sha256_file(output / (selected["name"] + ".head.npz")),
        "candidate_head_sha256": {
            candidate["name"]: sha256_file(output / (candidate["name"] + ".head.npz"))
            for candidate in candidates
        },
        "behavior_head_opens_dev_gate": selected["name"] != REFERENCE_NAME,
        "minimum_dev_start": 26000,
        "limitations": [
            "Reused training holdout;not independent dev or SOTA evidence.",
            "Popularity uses all non-holdout official training impressions,not only cached fitting rows.",
            "Five-fold impression exclusion prevents direct own-impression target leakage,not a temporal validation claim.",
        ],
    }
    save_json(output / "result.json", result)
    print(json.dumps({"stage": "behavior_training_completed", **result}), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--training-result", required=True, type=Path)
    ap.add_argument("--training-result-sha256", required=True)
    ap.add_argument("--reference-head", required=True, type=Path)
    ap.add_argument("--reference-head-sha256", required=True)
    ap.add_argument("--expected-training-rows", required=True, type=Path)
    ap.add_argument("--train-behaviors", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--source-revision", required=True)
    args = ap.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    train(
        args.cache, args.training_result, args.training_result_sha256,
        args.reference_head, args.reference_head_sha256, args.expected_training_rows,
        args.train_behaviors, args.output, args.source_revision,
    )


if __name__ == "__main__":
    main()
