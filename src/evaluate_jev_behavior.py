"""Apply one frozen behavioral correction to checksum-pinned JEV dev scores."""

import argparse
import json
from pathlib import Path

import numpy as np

from behavior_jev_readout import behavior_features, load_counts
from evaluate_jev_readout import load_head
from evaluate_mind_jev import parse_behaviors_line
from probe_jev_readout import metrics, predict, save_json
from scale_jev_readout import sha256_file


def evaluate(training_root, training_sha256, base_result_path, base_result_sha256,
             base_scores_path, base_scores_sha256, data_root, output,
             source_revision, dev_start=26000, dev_impressions=5000):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Behavioral evaluation output must be empty")
    for path, expected in (
        (training_root / "result.json", training_sha256),
        (base_result_path, base_result_sha256), (base_scores_path, base_scores_sha256),
    ):
        if sha256_file(path) != expected:
            raise ValueError("Behavioral evaluation input checksum differs: " + path.name)
    trained = json.loads((training_root / "result.json").read_bytes())
    selection = json.loads((training_root / "train-selection.json").read_bytes())
    baseline = json.loads(base_result_path.read_bytes())
    if (not trained["training_only"] or not trained["behavior_head_opens_dev_gate"]
            or trained["selected"] != selection["selected"]
            or dev_start < max(26000, trained["minimum_dev_start"], selection["minimum_dev_start"])
            or dev_impressions < 1
            or baseline["dev_start_zero_based"] != dev_start
            or baseline["dev_impressions"] != dev_impressions
            or baseline["head_sha256"] != trained["reference_head_sha256"]):
        raise ValueError("Frozen behavior selection,reference or untouched dev window differs")
    candidates = selection["candidates"]
    reference = next(c for c in candidates if c["name"] == "scaled_reference")
    best = max(candidates, key=lambda c: c["validation"]["auc"])
    if (best["name"] != trained["selected"]
            or best["validation"]["auc"] - reference["validation"]["auc"] < .005
            or any(best["validation"][key] < reference["validation"][key]
                   for key in ("mrr", "ndcg5", "ndcg10"))):
        raise ValueError("Selected behavioral head fails its original training-only compute gate")
    head_path = training_root / (trained["selected"] + ".head.npz")
    head, head_bytes = load_head(head_path, trained["selected_head_sha256"])
    if head["weight"].shape != (4,):
        raise ValueError("Behavioral head must have exactly four features")
    counts, totals = load_counts(training_root / "population.npz", trained["population_sha256"])
    dev = data_root / "dev"
    for name in ("news.tsv", "behaviors.tsv"):
        if sha256_file(dev / name) != baseline["data_sha256"]["dev/" + name]:
            raise ValueError("Behavioral dev data differs from pinned base evaluation")
    expected = []
    with (dev / "behaviors.tsv").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if dev_start <= index < dev_start + dev_impressions:
                parsed = parse_behaviors_line(line)
                if parsed is None or not parsed[2]:
                    raise ValueError("Malformed behavior in registered dev window")
                expected.append(parsed)
            if index >= dev_start + dev_impressions:
                break
    records = [json.loads(line) for line in base_scores_path.read_text(encoding="utf-8").splitlines()]
    if len(expected) != dev_impressions or len(records) != dev_impressions:
        raise ValueError("Incomplete registered behavioral dev window")
    rows, frozen_parts, reference_parts = [], [], []
    for (impression, _, items), record in zip(expected, records, strict=True):
        ids = [item[0] for item in items]
        labels = [item[1] for item in items]
        if (record["impression_id"] != impression or record["candidate_ids"] != ids
                or record["labels"] != labels
                or len(record["selected_margins"]) != len(items)
                or len(record["frozen_margins"]) != len(items)):
            raise ValueError("Base dev IDs,order,labels or score coverage differ")
        rows.extend({"impression_id": impression, "candidate_id": nid, "label": label} for nid, label in items)
        frozen_parts.extend(record["frozen_margins"])
        reference_parts.extend(record["selected_margins"])
    if len(rows) != baseline["dev_candidates"]:
        raise ValueError("Base candidate count differs from exact dev coverage")
    if len({record["impression_id"] for record in records}) != len(records):
        raise ValueError("Duplicate behavioral dev impression")
    frozen = np.asarray(frozen_parts, dtype=np.float64)
    reference_scores = np.asarray(reference_parts, dtype=np.float64)
    if not np.isfinite(frozen).all() or not np.isfinite(reference_scores).all():
        raise ValueError("Base behavioral evaluation scores must be finite")
    features = behavior_features(rows, counts, totals, cross_fitted=False)
    scores = predict(head, features, reference_scores)
    labels = [row["label"] for row in rows]
    groups = [row["impression_id"] for row in rows]
    values = metrics(labels, scores, groups)
    reference_values = metrics(labels, reference_scores, groups)
    frozen_values = metrics(labels, frozen, groups)
    if any(abs(reference_values[key] - baseline["selected_metrics"][key]) > 1e-12 for key in values):
        raise ValueError("Base retained-head metrics do not reproduce their pinned result")
    if any(abs(frozen_values[key] - baseline["frozen"][key]) > 1e-12 for key in values):
        raise ValueError("Base frozen-control metrics do not reproduce their pinned result")
    output.mkdir(parents=True, exist_ok=True)
    (output / "selected.head.npz").write_bytes(head_bytes)
    for source, target in (
        (training_root / "result.json", "head-training-result.json"),
        (training_root / "train-selection.json", "train-selection.json"),
        (training_root / "population.npz", "population.npz"),
        (base_result_path, "base-result.json"),
    ):
        (output / target).write_bytes(source.read_bytes())
    with (output / "dev.scores.jsonl").open("w", encoding="utf-8") as handle:
        position = 0
        for record in records:
            end = position + len(record["candidate_ids"])
            handle.write(json.dumps({
                "impression_id": record["impression_id"], "candidate_ids": record["candidate_ids"],
                "labels": record["labels"], "frozen_margins": frozen[position:end].tolist(),
                "reference_margins": reference_scores[position:end].tolist(),
                "selected_margins": scores[position:end].tolist(),
            }, allow_nan=False) + "\n")
            position = end
    np.savez(output / "behavior-features.npz", features=features)
    result = {
        "source_revision": source_revision, "selected": trained["selected"],
        "head_sha256": trained["selected_head_sha256"],
        "reference_head_sha256": trained["reference_head_sha256"],
        "population_sha256": trained["population_sha256"],
        "training_result_sha256": training_sha256,
        "base_source_revision": baseline["source_revision"],
        "base_result_sha256": base_result_sha256, "base_scores_sha256": base_scores_sha256,
        "dev_start_zero_based": dev_start, "dev_impressions": dev_impressions,
        "dev_candidates": len(rows), "selected_metrics": values,
        "reference_metrics": reference_values, "frozen": frozen_values,
        "primary_comparison": "selected_minus_reference",
        "data_sha256": baseline["data_sha256"],
        "limitations": [
            "Frozen training-selected behavioral correction;no dev fitting or reselection.",
            "Partial MIND-small dev,one training sample/seed;not MIND-large/test or SOTA.",
            "Training-holdout selection was reused;this window must remain separate from all earlier dev windows.",
        ],
    }
    save_json(output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-root", required=True, type=Path)
    ap.add_argument("--training-result-sha256", required=True)
    ap.add_argument("--base-result", required=True, type=Path)
    ap.add_argument("--base-result-sha256", required=True)
    ap.add_argument("--base-scores", required=True, type=Path)
    ap.add_argument("--base-scores-sha256", required=True)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--dev-start", type=int, default=26000)
    ap.add_argument("--dev-impressions", type=int, default=5000)
    args = ap.parse_args()
    evaluate(
        args.training_root, args.training_result_sha256,
        args.base_result, args.base_result_sha256, args.base_scores, args.base_scores_sha256,
        args.data_root, args.output, args.source_revision, args.dev_start, args.dev_impressions,
    )


if __name__ == "__main__":
    main()
