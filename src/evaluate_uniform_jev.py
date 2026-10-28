"""Replay frozen uniform and retained models on all cached fresh-dev candidates."""

import argparse
from itertools import islice
import json
from pathlib import Path

import numpy as np

from behavior_jev_readout import behavior_features, load_counts
from evaluate_jev_content import paired_intervals, METRICS
from evaluate_jev_readout import load_head
from jev_content_ranker import behavior_records, digest, ranking_metrics, write_json
from probe_jev_readout import predict
from streamed_jev_readout import predict_streamed, select_joint
from uniform_jev_features import DATA_SHA256, DEV_SHA256
from uniform_jev_readout import FEATURE_REVISION, REFERENCE_SHA, ROWS_SHA


RETAINED_RESULT_SHA = "1670421674adeb8fb41f858d065edd5b276b97e3fcddb7fa0bdde23d761e63fe"
RETAINED_CORRECTION_SHA = "32e8af31786d059d90b47c32ecd118cfe0633f9c7c8d4922ed52deb703d2ff68"
RETAINED_POPULATION_SHA = "ba4d7855aabe6894c6eecd9c9a0e69f1d326200891bc174dae048f9841ecc62c"


def checked_json(path, sha):
    if digest(path) != sha:
        raise ValueError("Frozen evaluation input checksum differs: " + path.name)
    return json.loads(path.read_bytes())


def evaluate(training, training_sha, dev_cache, data_root, retained_training, retained_text, output):
    if output.exists():
        raise FileExistsError("Uniform dev evaluation must preserve previous results")
    trained = checked_json(training / "result.json", training_sha)
    if (trained["stage"] != "full_training_fitted" or not trained["opens_dev_gate"]
            or not trained["training_only"] or trained["training_rows"] != 472626
            or trained["training_impressions"] != 156965
            or trained["final_refit_holdout_impressions"] != 0
            or trained["feature_source_revision"] != FEATURE_REVISION
            or trained["evaluator_sha256"] != digest(Path(__file__))):
        raise ValueError("Frozen full-training model/evaluator/gate differs")
    selection = checked_json(training / "train-selection.json", trained["selection_sha256"])
    best, opens = select_joint(selection["candidates"], selection["controls"])
    if not opens or best["name"] != selection["selected"] or not selection["opens_dev_gate"]:
        raise ValueError("Registered training-only selection gate is not open")
    retained = checked_json(retained_training / "result.json", RETAINED_RESULT_SHA)
    if (retained["selected_head_sha256"] != RETAINED_CORRECTION_SHA
            or retained["population_sha256"] != RETAINED_POPULATION_SHA):
        raise ValueError("Retained model/population differs")
    head, _ = load_head(training / "selected-full.head.npz", trained["head_sha256"])
    reference, _ = load_head(retained_text, REFERENCE_SHA)
    correction, _ = load_head(
        retained_training / (retained["selected"] + ".head.npz"), RETAINED_CORRECTION_SHA,
    )
    if len(head["weight"]) != 2564 or len(reference["weight"]) != 2560 or len(correction["weight"]) != 4:
        raise ValueError("Unexpected frozen model feature dimensions")
    counts, totals = load_counts(training / "population.npz", trained["population_sha256"])
    old_counts, old_totals = load_counts(retained_training / "population.npz", RETAINED_POPULATION_SHA)
    for relative, sha in DEV_SHA256.items():
        if digest(data_root / relative) != sha:
            raise ValueError("Official fresh-dev source differs")
    records = list(islice(behavior_records(data_root / "dev" / "behaviors.tsv"), 41000, 46000))
    if len(records) != 5000:
        raise ValueError("Incomplete registered fresh-dev impression window")
    expected = [
        {"impression_id": record["impression_id"], "candidate_id": name, "label": label}
        for record in records for name, label in record["items"]
    ]
    manifest = json.loads((dev_cache / "manifest.json").read_bytes())
    if (manifest["stage"] != 4 or manifest["source_revision"] != FEATURE_REVISION
            or manifest["rows"] != len(expected) or manifest["dimension"] != 2560
            or manifest["training_only"] or manifest["backbone_fitting"]
            or manifest["parity"] != "exact_original_contiguous_16_row_batch"
            or manifest["parity_manifest_sha256"] != trained["parity_manifest_sha256"]
            or manifest["data_sha256"] != {**DATA_SHA256, **DEV_SHA256}):
        raise ValueError("Fresh-dev frozen-feature provenance differs")
    actual = checked_json(dev_cache / "rows.json", manifest["rows_sha256"])
    if actual != expected:
        raise ValueError("Fresh-dev candidate identities/order/labels/coverage differ")
    if digest(dev_cache / "features.npz") != manifest["features_sha256"]:
        raise ValueError("Full fresh-dev feature archive checksum differs")
    with np.load(dev_cache / "features.npz", allow_pickle=False) as stored:
        if set(stored.files) != {"features", "margins"}:
            raise ValueError("Unexpected fresh-dev feature archive fields")
        x, frozen = stored["features"], stored["margins"]
    if (x.shape != (len(expected), 2560) or frozen.shape != (len(expected),)
            or x.dtype != np.float32 or frozen.dtype != np.float32):
        raise ValueError("Fresh-dev feature dimensions or dtype differ")
    empty = np.empty((len(x), 0))
    text = predict_streamed(reference, x, empty, frozen)
    scores = {
        "frozen": frozen, "retained_text": text,
        "retained": predict(correction, behavior_features(expected, old_counts, old_totals, False), text),
        "uniform": predict_streamed(head, x, behavior_features(expected, counts, totals, False), frozen),
    }
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise FloatingPointError("Nonfinite frozen fresh-dev model predictions")
    output.mkdir()
    all_values = {name: [] for name in scores}
    position = 0
    with (output / "dev.scores.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            end = position + len(record["items"])
            subset = expected[position:end]
            row_scores = {}
            for name, values in scores.items():
                part = values[position:end]
                metric = ranking_metrics(subset, part)
                if metric["scored_impressions"] != 1 or metric["single_class_impressions"]:
                    raise ValueError("Every official-dev impression must contribute all metrics")
                all_values[name].append([metric["metrics"][key] for key in METRICS])
                row_scores[name] = part.tolist()
            handle.write(json.dumps({
                "impression_id": record["impression_id"],
                "candidate_ids": [name for name, _ in record["items"]],
                "labels": [label for _, label in record["items"]], **row_scores,
            }, allow_nan=False) + "\n")
            position = end
    if position != len(expected):
        raise ValueError("Incomplete fresh-dev score coverage")
    result = {
        "source_revision": trained["source_revision"], "feature_source_revision": FEATURE_REVISION,
        "dataset": "official MIND-small dev", "dev_start_zero_based": 41000,
        "dev_impressions": 5000, "dev_candidates": len(expected), "all_candidates": True,
        "metrics": {name: dict(zip(METRICS, np.mean(values, axis=0).tolist()))
                    for name, values in all_values.items()},
        "paired_uniform_minus_retained": paired_intervals(all_values["uniform"], all_values["retained"]),
        "training_result_sha256": training_sha, "head_sha256": trained["head_sha256"],
        "uniform_training_rows_sha256": ROWS_SHA, "reference_head_sha256": REFERENCE_SHA,
        "reference_correction_sha256": RETAINED_CORRECTION_SHA,
        "reference_population_sha256": RETAINED_POPULATION_SHA,
        "dev_manifest_sha256": digest(dev_cache / "manifest.json"),
        "dev_scores_sha256": digest(output / "dev.scores.jsonl"),
        "independent_verification": "required_before_promotion",
        "competition_test_goal_achieved": False,
    }
    write_json(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "dev-cache", "data-root", "retained-training", "retained-text", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--training-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(
        args.training, args.training_sha256, args.dev_cache, args.data_root,
        args.retained_training, args.retained_text, args.output,
    )), flush=True)
