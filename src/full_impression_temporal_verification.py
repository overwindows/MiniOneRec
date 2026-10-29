"""Complete source/feature/design/metric replay and independent numeric-tree scoring."""

import argparse
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np

from full_impression_contract import check_deadline
from full_impression_evaluation import TRAINING_RESULT_SHA, load_gate
from full_impression_temporal_training import NAMES, PRIOR_PROOF, feature_batch
from full_impression_training import MANIFEST_SHA, PROOF_SHA, batches, load_contract
from full_impression_verification import metrics
from jev_content_ranker import digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric_tree_scores(model, features):
    require(model._preprocessor is None and np.array_equal(model.classes_, [0, 1])
            and model.n_iter_ == len(model._predictors), "Only the frozen numeric binary booster")
    require(features.ndim == 2 and np.isfinite(features).all(), "Finite full-context matrix required")
    result = np.full(len(features), float(model._baseline_prediction[0, 0]), dtype=np.float64)
    for trees in model._predictors:
        require(len(trees) == 1, "One binary tree per fixed round")
        nodes = trees[0].nodes
        splits = nodes["is_leaf"] == 0
        require(not nodes["is_categorical"].any() and np.isfinite(nodes["value"]).all()
                and np.isfinite(nodes["num_threshold"][splits]).all()
                and np.all(nodes["feature_idx"][splits] < features.shape[1])
                and np.all(nodes["left"][splits] > np.flatnonzero(splits))
                and np.all(nodes["right"][splits] > np.flatnonzero(splits))
                and np.all(nodes["left"][splits] < len(nodes))
                and np.all(nodes["right"][splits] < len(nodes)), "Finite acyclic numeric tree required")
        positions = np.zeros(len(features), dtype=np.int32)
        pending = np.arange(len(features))
        while len(pending):
            current = positions[pending]
            leaf = nodes["is_leaf"][current].astype(bool)
            result[pending[leaf]] += nodes["value"][current[leaf]]
            pending, current = pending[~leaf], current[~leaf]
            if len(pending):
                left = features[pending, nodes["feature_idx"][current]] <= nodes["num_threshold"][current]
                positions[pending] = np.where(left, nodes["left"][current], nodes["right"][current])
    return result


def independent_population(arrays, news, include_last):
    lengths = np.diff(arrays["candidate_offsets.npy"]).astype(np.int64)
    days = arrays["training_days.npy"]
    dates = np.unique(days if include_last else days[days < days.max()])
    exposure_days = np.repeat(days, lengths)
    included = np.ones(len(exposure_days), dtype=bool) if include_last else exposure_days < days.max()
    encoded = np.searchsorted(dates, exposure_days[included]) * (news + 1)
    encoded += arrays["candidate_news.npy"][included]
    exposure = np.bincount(encoded, minlength=len(dates) * (news + 1))
    labels = arrays["candidate_labels.npy"][included]
    clicks = np.bincount(encoded[labels == 1], minlength=len(dates) * (news + 1))
    return dates, np.stack((exposure, clicks), axis=1).reshape(len(dates), news + 1, 2)


def verify_design(directory, indices, arrays, features, expected, deadline):
    x = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(directory / "labels.npy", mmap_mode="r", allow_pickle=False)
    weight = np.load(directory / "weights.npy", mmap_mode="r", allow_pickle=False)
    require(x.shape == (expected[1], len(NAMES)) and x.dtype == np.float64
            and y.shape == weight.shape == (expected[1],) and y.dtype == np.uint8
            and weight.dtype == np.float64, "Exact complete training-design inventory")
    offsets, position, positives = arrays["candidate_offsets.npy"], 0, 0
    for batch in batches(indices, arrays, candidate_limit=1000000, impression_limit=32768):
        check_deadline(deadline)
        source = np.concatenate([np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in batch])
        end = position + len(source)
        lengths = (offsets[batch + 1] - offsets[batch]).astype(np.int64)
        require(np.array_equal(x[position:end], features[source])
                and np.array_equal(y[position:end], arrays["candidate_labels.npy"][source])
                and np.array_equal(weight[position:end], np.repeat(1. / lengths, lengths)),
                "Every source candidate/label/feature/unit-impression weight must replay")
        positives += int(y[position:end].sum())
        position = end
    require((len(indices), position, positives) == expected, "No missing fitting impression/exposure/positive")


def run(args):
    deadline = time.monotonic() + args.max_seconds
    require(0 < args.max_seconds <= 3600 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and np.__version__ == "1.26.4", "Bounded pinned CPU-only verification")
    dispatch = {key: bool(np.core._multiarray_umath.__cpu_features__[key])
                for key in ("AVX2", "AVX512F", "AVX512_SKX", "AVX512_ICL")}
    require(dispatch == {"AVX2": True, "AVX512F": False, "AVX512_SKX": False, "AVX512_ICL": False},
            "Exact original tied-score numeric runtime")
    require(not args.proof.exists(), "Never overwrite independent evidence")
    prior = load_gate(args.training_artifacts, args.training_proof, PRIOR_PROOF, TRAINING_RESULT_SHA)
    _, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    result_path = args.artifacts / "result.json"
    result_sha = digest(result_path)
    result = json.loads(result_path.read_bytes())
    require(result["source_revision"] == args.source_revision
            and result["strictly_earlier_day_statistics"]
            and result["label_dependent_selection_counts_exclude_last_training_day"]
            and not result["dev_or_test_contents_read"] and result["gpu_allocations"] == 0
            and result["feature_names"] == list(NAMES), "Only the preregistered temporal experiment")
    for name, key in (("content.joblib", "content_sha256"), ("content.table.npz", "content_table_sha256"),
                      ("selection.population.npz", "selection_population_sha256"),
                      ("source.features.npy", "source_features_sha256")):
        require(digest(args.artifacts / name) == result[key], "Complete frozen artifact changed")
    margin_path = args.training_artifacts / "article.margins.npy"
    require(digest(margin_path) == result["frozen_margins_sha256"]
            == prior["feature_extraction"]["margins_sha256"], "Original exact-parity frozen margin changed")
    margins = np.load(margin_path, mmap_mode="r", allow_pickle=False)
    dates, counts = independent_population(arrays, 101527, False)
    with np.load(args.artifacts / "selection.population.npz", allow_pickle=False) as stored:
        require(set(stored.files) == {"days", "counts"}
                and np.array_equal(stored["days"], dates) and np.array_equal(stored["counts"], counts)
                and int(counts[:, :, 0].sum()) == 66107268
                and int(counts[:, :, 1].sum()) == 2712612,
                "Independent all-source counts must EXCLUDE every validation target")
    ids = json.loads((args.contract / "news.ids.json").read_bytes())
    news = {}
    with (args.contract / "news.metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            news[row["news_id"]] = {"category": row["category"], "subcategory": row["subcategory"],
                                   "text": row["title"] + " " + row["abstract"]}
    require(list(news) == ids, "Complete metadata identity order")
    content = joblib.load(args.artifacts / "content.joblib")
    table = content.table(news)
    with np.load(args.artifacts / "content.table.npz", allow_pickle=False) as cached:
        require(all(np.array_equal(cached[name], expected) for name, expected in
                    zip(("vectors", "categories", "subcategories", "lengths"), table[1:])),
                "Every frozen lexical article row must replay from source metadata")
    features = np.load(args.artifacts / "source.features.npy", mmap_mode="r", allow_pickle=False)
    require(features.shape == (83507374, len(NAMES)) and features.dtype == np.float64,
            "Every official-training candidate context required")
    position = seen = 0
    for batch in batches(np.arange(len(arrays["training_days.npy"])), arrays):
        check_deadline(deadline)
        expected = feature_batch(batch, arrays, ids, table, margins, dates, counts)
        require(np.array_equal(features[position:position + len(expected)], expected),
                "Complete target-free source context replay differs")
        position += len(expected)
        seen += len(batch)
        if seen % 100000 < len(batch):
            print(json.dumps({"stage": "independent_temporal_features", "impressions": seen}), flush=True)
    require((seen, position) == (2232748, 83507374), "Complete feature source coverage")
    days = arrays["training_days.npy"]
    fitting, validation = np.flatnonzero(days < days.max()), np.flatnonzero(days == days.max())
    verify_design(args.artifacts / "selection.design", fitting, arrays, features,
                  (1801231, 66107268, 2712612), deadline)
    selected = result["selection"]
    require(tuple(selected["fitting"][key] for key in ("impressions", "exposures", "positives")) ==
            (1801231, 66107268, 2712612)
            and tuple(selected[key] for key in ("impressions", "exposures", "positives")) ==
            (431517, 17400106, 671044), "Separate complete fitting and validation coverage required")
    head = args.artifacts / "selection.joblib"
    require(digest(head) == selected["head_sha256"], "Selection head changed")
    model = joblib.load(head)
    require(model.n_iter_ == 160, "Fixed complete booster rounds")
    score_path = args.artifacts / "validation.scores.f64"
    metric_path = score_path.with_suffix(".metrics.npy")
    require(digest(score_path) == selected["scores_sha256"]
            and digest(metric_path) == selected["per_impression_metrics_sha256"]
            and score_path.stat().st_size == 17400106 * 8, "Complete raw selection scores/metrics")
    scores = np.memmap(score_path, dtype=np.float64, mode="r", shape=(17400106,))
    recorded = np.load(metric_path, allow_pickle=False)
    require(recorded.shape == (431517, 4), "Every metric cell is required")
    values = np.empty_like(recorded)
    offsets, row, position, positive = arrays["candidate_offsets.npy"], 0, 0, 0
    maximum_error = 0.
    for batch in batches(validation, arrays, candidate_limit=131072, impression_limit=4096):
        check_deadline(deadline)
        source = np.concatenate([np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in batch])
        expected = numeric_tree_scores(model, features[source])
        actual = scores[position:position + len(source)]
        require(np.allclose(actual, expected, atol=1e-12, rtol=0), "Independent numeric-tree traversal differs")
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
        local = 0
        for index in batch:
            y = arrays["candidate_labels.npy"][offsets[index]:offsets[index + 1]]
            values[row] = metrics(y, actual[local:local + len(y)])
            positive += int(y.sum())
            local += len(y)
            row += 1
        position += len(source)
    require((row, position, positive) == (431517, 17400106, 671044)
            and np.allclose(values, recorded, atol=1e-12, rtol=0), "Complete identical metric replay")
    rebuilt = dict(zip(("auc", "mrr", "ndcg5", "ndcg10"), values.mean(axis=0).tolist()))
    require(all(abs(rebuilt[key] - selected["metrics"][key]) <= 1e-12 for key in rebuilt), "Aggregate metrics differ")
    baseline = prior["recipes"]["candidate_attention"]["metrics"]
    gain = rebuilt["auc"] - baseline["auc"]
    opens = gain >= .005 and all(rebuilt[key] >= baseline[key] for key in ("mrr", "ndcg5", "ndcg10"))
    require(result["unchanged_baseline_metrics"] == baseline and opens == result["opens_dev_gate"]
            and abs(gain - result["training_only_auc_gain"]) <= 1e-12, "Unchanged stronger-baseline gate differs")
    if opens:
        require(result["stage"] == "temporal_full_training_fitted"
                and result["final_refit_holdout_impressions"] == 0, "Fresh ALL-training refit required")
        verify_design(args.artifacts / "final.design", np.arange(len(days)), arrays, features,
                      (2232748, 83507374, 3383656), deadline)
        final = args.artifacts / "selected-full.joblib"
        require(digest(final) == result["final_refit"]["head_sha256"]
                and joblib.load(final).n_iter_ == 160, "Fixed complete final head required")
        full_dates, full_counts = independent_population(arrays, 101527, True)
        with np.load(args.artifacts / "final.population.npz", allow_pickle=False) as stored:
            require(digest(args.artifacts / "final.population.npz") == result["final_population_sha256"]
                    and np.array_equal(stored["days"], full_dates) and np.array_equal(stored["counts"], full_counts),
                    "Every all-training count must replay")
    else:
        require(result["stage"] == "temporal_training_gate_evaluated"
                and not (args.artifacts / "selected-full.joblib").exists()
                and not (args.artifacts / "final.design").exists(), "Rejected recipe cannot refit")
    check_deadline(deadline)
    require(digest(result_path) == result_sha, "Frozen result changed during replay")
    proof = {"source_revision": args.source_revision, "result_sha256": result_sha,
             "prior_training_proof_sha256": PRIOR_PROOF, "metrics": rebuilt, "training_only_auc_gain": gain,
             "opens_dev_gate": opens, "full_training_source_and_coverage_verified": opens,
             "all_source_feature_exposures_replayed": 83507374,
             "independent_causal_population_replay": True, "maximum_independent_tree_score_error": maximum_error,
             "coverage": {"impressions": row, "exposures": position, "positives": positive},
             "content_feature_implementation_reused_for_full_source_replay": True,
             "lexical_encoder_and_booster_optimization_independently_retrained": False,
             "metric_scope": "last_official_training_day_only;not_official_dev_or_test",
             "gpu_allocations": 0, "dev_or_test_contents_read": False, "competition_test_goal_achieved": False}
    with args.proof.open("x", encoding="utf-8") as handle:
        json.dump(proof, handle, indent=2, allow_nan=False)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "training-artifacts", "training-proof", "artifacts", "proof"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-seconds", type=int, default=3600)
    run(parser.parse_args())
