"""Apply two frozen behavioral corrections to a fresh official-dev feature pass."""

import argparse
import json
from pathlib import Path

import numpy as np

from behavior_jev_readout import behavior_features, load_counts
from evaluate_jev_readout import load_head
from evaluate_mind_jev import parse_behaviors_line
from probe_jev_readout import metrics, predict, save_json
from scale_jev_readout import sha256_file


def evaluate(full_root, full_sha256, retained_root, retained_sha256,
             base_result_path, base_result_sha256, base_scores_path, base_scores_sha256,
             data_root, output, source_revision, dev_start=31000, dev_impressions=5000):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Full-training dev output must be empty")
    for path, digest in (
        (full_root / "result.json", full_sha256), (retained_root / "result.json", retained_sha256),
        (base_result_path, base_result_sha256), (base_scores_path, base_scores_sha256),
    ):
        if sha256_file(path) != digest:
            raise ValueError("Full-training dev input checksum differs: " + path.name)
    full = json.loads((full_root / "result.json").read_bytes())
    retained = json.loads((retained_root / "result.json").read_bytes())
    base = json.loads(base_result_path.read_bytes())
    if (not full["full_training"] or not full["training_only"]
            or full["training_pairs"] != 469264 or full["training_impressions"] != 156965
            or full["train_holdout_impressions"] != 0 or full["artificial_pair_cap"] is not None
            or full["quality_result"] is not None or dev_start < max(31000, full["minimum_dev_start"])
            or dev_impressions < 1 or base["dev_start_zero_based"] != dev_start
            or base["dev_impressions"] != dev_impressions
            or base["head_sha256"] != full["text_head_sha256"]
            or base["reference_head_sha256"] != retained["reference_head_sha256"]):
        raise ValueError("Full-training coverage,models or fresh official-dev window differs")
    full_head, _ = load_head(
        full_root / (full["behavior_selected"] + ".head.npz"), full["behavior_head_sha256"],
    )
    retained_head, _ = load_head(
        retained_root / (retained["selected"] + ".head.npz"), retained["selected_head_sha256"],
    )
    if full_head["weight"].shape != (4,) or retained_head["weight"].shape != (4,):
        raise ValueError("Both frozen behavioral corrections must have four features")
    full_counts, full_totals = load_counts(full_root / "population.npz", full["population_sha256"])
    old_counts, old_totals = load_counts(retained_root / "population.npz", retained["population_sha256"])
    dev = data_root / "dev"
    for name in ("news.tsv", "behaviors.tsv"):
        if sha256_file(dev / name) != base["data_sha256"]["dev/" + name]:
            raise ValueError("Full-training official-dev inputs differ from the baseline pass")
    expected = []
    with (dev / "behaviors.tsv").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if dev_start <= index < dev_start + dev_impressions:
                parsed = parse_behaviors_line(line)
                if parsed is None or not parsed[2]:
                    raise ValueError("Malformed behavior in the registered full-training dev window")
                expected.append(parsed)
            if index >= dev_start + dev_impressions:
                break
    records = [json.loads(line) for line in base_scores_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != dev_impressions or len(expected) != dev_impressions:
        raise ValueError("Incomplete full-training dev coverage")
    rows, full_parts, retained_parts, frozen_parts = [], [], [], []
    for (impression, _, items), record in zip(expected, records, strict=True):
        if (record["impression_id"] != impression
                or record["candidate_ids"] != [item[0] for item in items]
                or record["labels"] != [item[1] for item in items]
                or any(len(record[name]) != len(items) for name in
                       ("selected_margins", "reference_margins", "frozen_margins"))):
            raise ValueError("Full-training baseline IDs,order,labels or margins differ")
        rows.extend({"impression_id": impression, "candidate_id": nid, "label": label} for nid, label in items)
        full_parts.extend(record["selected_margins"])
        retained_parts.extend(record["reference_margins"])
        frozen_parts.extend(record["frozen_margins"])
    if (len(rows) != base["dev_candidates"]
            or len({record["impression_id"] for record in records}) != dev_impressions):
        raise ValueError("Full-training dev candidate count or unique impression coverage differs")
    full_text, retained_text, frozen = [np.asarray(part, dtype=np.float64) for part in
                                       (full_parts, retained_parts, frozen_parts)]
    if not all(np.isfinite(part).all() for part in (full_text, retained_text, frozen)):
        raise ValueError("Full-training baseline scores must be finite")
    full_features = behavior_features(rows, full_counts, full_totals, cross_fitted=False)
    old_features = behavior_features(rows, old_counts, old_totals, cross_fitted=False)
    selected = predict(full_head, full_features, full_text)
    reference = predict(retained_head, old_features, retained_text)
    labels = np.asarray([row["label"] for row in rows])
    groups = [row["impression_id"] for row in rows]
    for scores, expected_values in (
        (full_text, base["selected_metrics"]), (retained_text, base["reference_metrics"]),
        (frozen, base["frozen"]),
    ):
        actual = metrics(labels, scores, groups)
        if any(abs(actual[name] - expected_values[name]) > 1e-12 for name in actual):
            raise ValueError("Full-training baseline metrics do not reproduce their pinned result")
    result = {
        "source_revision": source_revision, "full_training_result_sha256": full_sha256,
        "retained_training_result_sha256": retained_sha256,
        "base_result_sha256": base_result_sha256, "base_scores_sha256": base_scores_sha256,
        "base_source_revision": base["source_revision"],
        "selected": "full_training_hybrid", "reference": "retained_subset_hybrid",
        "text_head_sha256": full["text_head_sha256"],
        "behavior_head_sha256": full["behavior_head_sha256"],
        "population_sha256": full["population_sha256"],
        "reference_text_head_sha256": retained["reference_head_sha256"],
        "reference_behavior_head_sha256": retained["selected_head_sha256"],
        "reference_population_sha256": retained["population_sha256"],
        "dev_start_zero_based": dev_start, "dev_impressions": dev_impressions,
        "dev_candidates": len(rows), "selected_metrics": metrics(labels, selected, groups),
        "reference_metrics": metrics(labels, reference, groups), "frozen": base["frozen"],
        "data_sha256": base["data_sha256"], "primary_comparison": "selected_minus_reference",
        "limitations": [
            "All official MIND-small training impressions with existing negative sampling,not every unclicked candidate.",
            "Fixed full-training recipe;no dev fitting,reselection or pooling earlier windows.",
            "One training seed and partial official dev,not full-dev/test,MIND-large or SOTA.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "behavior-features.npz", selected=full_features, reference=old_features)
    with (output / "dev.scores.jsonl").open("w", encoding="utf-8") as handle:
        position = 0
        for record in records:
            end = position + len(record["labels"])
            handle.write(json.dumps({
                "impression_id": record["impression_id"], "candidate_ids": record["candidate_ids"],
                "labels": record["labels"], "selected_margins": selected[position:end].tolist(),
                "reference_margins": reference[position:end].tolist(),
                "frozen_margins": frozen[position:end].tolist(),
            }, allow_nan=False) + "\n")
            position = end
    save_json(output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full-root", "retained-root", "base-result", "base-scores", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("full-sha256", "retained-sha256", "base-result-sha256", "base-scores-sha256", "source-revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dev-start", type=int, default=31000)
    parser.add_argument("--dev-impressions", type=int, default=5000)
    args = parser.parse_args()
    evaluate(args.full_root, args.full_sha256, args.retained_root, args.retained_sha256,
             args.base_result, args.base_result_sha256, args.base_scores, args.base_scores_sha256,
             args.data_root, args.output, args.source_revision, args.dev_start, args.dev_impressions)
