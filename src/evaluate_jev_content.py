"""Evaluate one frozen train-selected content ranker on a registered dev window."""

import argparse
from itertools import islice
import json
from pathlib import Path

import joblib
import numpy as np

from behavior_jev_readout import behavior_features, load_counts
from evaluate_jev_readout import load_head
from jev_content_ranker import (
    Content, FEATURE_NAMES, assembled_features, behavior_records, content_features,
    digest, ranking_metrics, read_news, write_json,
)
from probe_jev_readout import predict


DEV_START = 36000
DEV_IMPRESSIONS = 5000
DEV_NEWS_SHA = "e5d144667558c449d16084fe6bfa01940c2f5d5785ea9f9110292bc3b94eb822"
DEV_BEHAVIORS_SHA = "b6c460e33b1a8693252ded6e626da7d3ccf78920eea2ec11889020bb7d8443ef"
FULL_TEXT_SHA = "5392ad5738d09a9e0df9a502754ceb833f807f9f4c8b9fbc6a7ceded1adf13bc"
RETAINED_TEXT_SHA = "279ef6c47f09e7fe49b803810159cac6fba7ee23d612ea33318af6ec37f86457"
RETAINED_BEHAVIOR_SHA = "32e8af31786d059d90b47c32ecd118cfe0633f9c7c8d4922ed52deb703d2ff68"
CONTROL_SOURCE = "e44b2e1147b5b7d637dc37d354ede943ba84ec57"
METRICS = ("auc", "mrr", "ndcg5", "ndcg10")


def paired_intervals(selected, reference):
    delta = np.asarray(selected, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 4 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Bootstrap requires finite paired impression metrics")
    generator = np.random.default_rng(42)
    resamples = []
    for _ in range(32):
        indices = generator.integers(len(delta), size=(64, len(delta)))
        resamples.append(delta[indices].mean(axis=1))
    intervals = np.quantile(np.concatenate(resamples), [.025, .975], axis=0).T
    return {
        key: {"delta": float(delta[:, column].mean()), "ci95": intervals[column].tolist()}
        for column, key in enumerate(METRICS)
    }


def checked_json(path, expected):
    if digest(path) != expected:
        raise ValueError("Evaluation input checksum differs: " + path.name)
    return json.loads(path.read_bytes())


def evaluate(training, training_sha, reference_training, reference_sha,
             base_result, base_result_sha, base_scores, base_scores_sha,
             dev_news, dev_behaviors, output):
    if output.exists():
        raise FileExistsError("Content evaluation must not overwrite an existing output")
    trained = checked_json(training / "result.json", training_sha)
    reference = checked_json(reference_training / "result.json", reference_sha)
    base = checked_json(base_result, base_result_sha)
    selection = checked_json(training / "train-selection.json", trained["train_selection_sha256"])
    if (trained["stage"] != "full_training_fitted" or not trained["training_only"]
            or not trained["opens_dev_gate"] or trained["selected"] != selection["selected"]
            or trained["training_impressions"] != 156965 or trained["training_rows"] != 469264
            or trained["final_refit_holdout_impressions"] != 0
            or base["dev_start_zero_based"] != DEV_START
            or base["dev_impressions"] != DEV_IMPRESSIONS
            or base["source_revision"] != CONTROL_SOURCE
            or base["head_sha256"] != FULL_TEXT_SHA
            or base["reference_head_sha256"] != RETAINED_TEXT_SHA
            or reference["selected_head_sha256"] != RETAINED_BEHAVIOR_SHA):
        raise ValueError("Frozen model, training coverage, control or registered dev window differs")
    best = max(selection["candidates"][1:], key=lambda value: value["metrics"]["auc"])
    if (best["name"] != trained["selected"]
            or best["metrics"]["auc"] - selection["candidates"][0]["metrics"]["auc"] < .005):
        raise ValueError("Selected content model fails its original train-only gate")
    for path, expected in (
        (training / "model.joblib", trained["model_sha256"]),
        (base_scores, base_scores_sha), (dev_news, DEV_NEWS_SHA),
        (dev_behaviors, DEV_BEHAVIORS_SHA),
    ):
        if digest(path) != expected:
            raise ValueError("Evaluation artifact/data checksum differs: " + path.name)
    if base["data_sha256"] != {"dev/news.tsv": DEV_NEWS_SHA, "dev/behaviors.tsv": DEV_BEHAVIORS_SHA}:
        raise ValueError("GPU control used different official dev data")
    payload = joblib.load(training / "model.joblib")
    if (payload["recipe"]["name"] != trained["selected"]
            or tuple(payload["recipe"]["columns"]) != tuple(range(len(FEATURE_NAMES)))):
        raise ValueError("Frozen model uses an unexpected content feature contract")
    content = Content(**payload["content"])
    news = {**payload["news"], **read_news(dev_news)}
    table = content.table(news)
    counts, totals = load_counts(training / "population.npz", trained["population_sha256"])
    reference_counts, reference_totals = load_counts(
        reference_training / "population.npz", reference["population_sha256"],
    )
    reference_head, _ = load_head(
        reference_training / (reference["selected"] + ".head.npz"), RETAINED_BEHAVIOR_SHA,
    )
    records = list(islice(behavior_records(dev_behaviors), DEV_START, DEV_START + DEV_IMPRESSIONS))
    if len(records) != DEV_IMPRESSIONS:
        raise ValueError("Incomplete registered dev source coverage")
    raw = [json.loads(line) for line in base_scores.read_text(encoding="utf-8").splitlines()]
    if len(raw) != DEV_IMPRESSIONS:
        raise ValueError("Incomplete GPU dev score coverage")
    for record, scores in zip(records, raw, strict=True):
        if (record["impression_id"] != scores["impression_id"]
                or [name for name, _ in record["items"]] != scores["candidate_ids"]
                or [label for _, label in record["items"]] != scores["labels"]):
            raise ValueError("GPU candidate identities, order or labels differ from official dev")
    output.mkdir()
    all_values = {name: [] for name in ("selected", "retained", "frozen", "full_text", "retained_text")}
    candidate_count = 0
    with (output / "dev.scores.jsonl").open("w", encoding="utf-8") as handle:
        for start in range(0, len(records), 64):
            batch, original = records[start:start + 64], raw[start:start + 64]
            rows, content_x = content_features(batch, table)
            frozen = np.concatenate([record["frozen_margins"] for record in original])
            retained_text = np.concatenate([record["reference_margins"] for record in original])
            full_text = np.concatenate([record["selected_margins"] for record in original])
            features = assembled_features(rows, content_x, frozen, counts, totals, False)
            selected = payload["model"].decision_function(features)
            retained = predict(
                reference_head, behavior_features(rows, reference_counts, reference_totals, False),
                retained_text,
            )
            parts = dict(selected=selected, retained=retained, frozen=frozen,
                         full_text=full_text, retained_text=retained_text)
            position = 0
            for record in batch:
                end = position + len(record["items"])
                subset = rows[position:end]
                score_parts = {}
                for name, values in parts.items():
                    score = values[position:end]
                    result = ranking_metrics(subset, score)
                    if result["single_class_impressions"] or result["scored_impressions"] != 1:
                        raise ValueError("Official dev metrics must not exclude any impression")
                    all_values[name].append([result["metrics"][key] for key in METRICS])
                    score_parts[name] = score.tolist()
                handle.write(json.dumps({
                    "impression_id": record["impression_id"],
                    "candidate_ids": [name for name, _ in record["items"]],
                    "labels": [label for _, label in record["items"]], "scores": score_parts,
                }, allow_nan=False) + "\n")
                position = end
            if position != len(rows):
                raise ValueError("Scored batch coverage differs")
            candidate_count += len(rows)
    values = {name: dict(zip(METRICS, np.mean(arm, axis=0).tolist())) for name, arm in all_values.items()}
    if candidate_count != base["dev_candidates"]:
        raise ValueError("Scored candidate count differs from the GPU control")
    for name, base_field in (("frozen", "frozen"), ("full_text", "selected_metrics"),
                             ("retained_text", "reference_metrics")):
        if any(abs(values[name][key] - base[base_field][key]) > 1e-12 for key in METRICS):
            raise ValueError("GPU control metrics do not independently reproduce")
    contrast = paired_intervals(all_values["selected"], all_values["retained"])
    result = {
        "stage": "official_dev_evaluated", "selected": trained["selected"],
        "training_result_sha256": training_sha, "reference_training_result_sha256": reference_sha,
        "base_result_sha256": base_result_sha, "base_scores_sha256": base_scores_sha,
        "model_sha256": trained["model_sha256"], "dev_start_zero_based": DEV_START,
        "dev_impressions": DEV_IMPRESSIONS, "dev_candidates": candidate_count,
        "metrics": values, "primary_comparison": "selected_minus_retained", "paired": contrast,
        "bootstrap": {"seed": 42, "resamples": 2048, "unit": "impression", "adjustment": "none"},
        "positive_primary_ci": contrast["auc"]["ci95"][0] > 0,
        "project_auc_target": .8, "project_target_achieved": False,
        "raw_scores_sha256": digest(output / "dev.scores.jsonl"),
        "data_sha256": base["data_sha256"],
        "limitations": [
            "One frozen train-selected model;no dev fitting, tuning or reselection.",
            "Partial official MIND-small dev,not full dev,MIND-large or competition test.",
            "Internal sampled-training AUC was only an allocation gate,not benchmark quality.",
            "Even partial-dev AUC>=0.80 would not establish competition-test acceptance.",
        ],
    }
    write_json(output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("training", "reference-training", "base-result", "base-scores",
                  "dev-news", "dev-behaviors", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("training-sha", "reference-sha", "base-result-sha", "base-scores-sha"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args()
    evaluate(args.training, args.training_sha, args.reference_training, args.reference_sha,
             args.base_result, args.base_result_sha, args.base_scores, args.base_scores_sha,
             args.dev_news, args.dev_behaviors, args.output)


if __name__ == "__main__":
    main()
