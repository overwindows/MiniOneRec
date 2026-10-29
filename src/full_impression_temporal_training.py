"""One full-support CPU booster with strictly earlier-day behavioral statistics."""

import argparse
from datetime import datetime, timedelta, timezone
import gc
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from full_impression_contract import check_deadline
from full_impression_evaluation import TRAINING_RESULT_SHA, load_gate
from full_impression_training import MANIFEST_SHA, PROOF_SHA, batches, load_contract
from jev_content_ranker import Content, FEATURE_NAMES, content_features, digest, write_json
from mind_utils import auc_score, mrr_score, ndcg_score


PRIOR_PROOF = "305808f97e4dc08f7d0b88fc690db655919052fc60a74fd68f15b90e3a30b128"
AGGREGATE = "2026-10-30T01:17:07.923179+00:00"
COLUMNS = (0, *range(3, 25), *range(25, 29))
NAMES = tuple(name for index, name in enumerate(FEATURE_NAMES) if index not in (1, 2)) + (
    "recent_log_exposure_rate", "recent_log_click_rate", "recent_ctr_log_odds", "recent_known_item")


def population(indices, arrays, news, deadline):
    days = np.unique(arrays["training_days.npy"][indices])
    counts = np.zeros((len(days), news + 1, 2), dtype=np.int64)
    offsets = arrays["candidate_offsets.npy"]
    for batch in batches(indices, arrays, candidate_limit=1000000, impression_limit=32768):
        check_deadline(deadline)
        positions = np.concatenate([np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in batch])
        lengths = offsets[batch + 1] - offsets[batch]
        day = np.repeat(np.searchsorted(days, arrays["training_days.npy"][batch]), lengths.astype(np.int64))
        ids = day * (news + 1) + arrays["candidate_news.npy"][positions]
        labels = arrays["candidate_labels.npy"][positions]
        counts[:, :, 0] += np.bincount(ids, minlength=len(days) * (news + 1)).reshape(len(days), news + 1)
        counts[:, :, 1] += np.bincount(ids[labels == 1], minlength=len(days) * (news + 1)).reshape(len(days), news + 1)
    if (np.any(counts[:, 0]) or np.any(counts < 0) or np.any(counts[:, :, 1] > counts[:, :, 0])
            or int(counts[:, :, 0].sum()) != int(np.sum(offsets[indices + 1] - offsets[indices]))):
        raise ValueError("Every selected source exposure/click must occur exactly once")
    return days, counts


def behavior(ids, counts, totals=None):
    exposure, clicks = counts[ids].T
    total, positive = counts.sum(axis=0) if totals is None else totals
    if total == 0:
        if positive:
            raise ValueError("Empty exposure population cannot have positives")
        prior = .04
    elif not 0 < positive < total:
        raise ValueError("Nonempty earlier-day statistics require both training classes")
    else:
        prior = positive / total
    ctr = (clicks + 20 * prior) / (exposure + 20)
    return np.column_stack((
        np.log1p(1e6 * exposure / max(int(total), 1)),
        np.log1p(1e6 * clicks / max(int(positive), 1)),
        np.log(ctr / (1 - ctr)) - np.log(prior / (1 - prior)),
        exposure > 0,
    ))


def contexts(indices, arrays, identifiers):
    offsets = arrays["candidate_offsets.npy"]
    return [{
        "impression_id": str(int(i)),
        "history": tuple(identifiers[n - 1] for n in arrays["histories.npy"][i] if n),
        "items": [(identifiers[n - 1], 0) for n in
                  arrays["candidate_news.npy"][offsets[i]:offsets[i + 1]]],
    } for i in indices]


def feature_batch(indices, arrays, identifiers, table, margins, days, counts):
    records = contexts(indices, arrays, identifiers)
    _, content = content_features(records, table)
    result = np.empty((len(content), len(NAMES)), dtype=np.float64)
    offsets, position = arrays["candidate_offsets.npy"], 0
    snapshots = {}
    for day in np.unique(arrays["training_days.npy"][indices]):
        before = np.searchsorted(days, day, side="left")
        cumulative = counts[:before].sum(axis=0)
        recent = counts[before - 1] if before else np.zeros_like(cumulative)
        snapshots[int(day)] = cumulative, recent, cumulative.sum(axis=0), recent.sum(axis=0)
    for index in indices:
        ids = arrays["candidate_news.npy"][offsets[index]:offsets[index + 1]]
        cumulative, recent, total, recent_total = snapshots[int(arrays["training_days.npy"][index])]
        end = position + len(ids)
        combined = np.column_stack((margins[ids], content[position:end],
                                    behavior(ids, cumulative, total), behavior(ids, recent, recent_total)))
        result[position:end] = combined[:, COLUMNS]
        position = end
    if position != len(result) or not np.isfinite(result).all():
        raise ValueError("Every source candidate needs one finite target-free context")
    return result


def build_features(output, arrays, identifiers, table, margins, days, counts, deadline):
    total = int(arrays["candidate_offsets.npy"][-1])
    features = np.lib.format.open_memmap(output, mode="w+", dtype=np.float64, shape=(total, len(NAMES)))
    position = impressions = 0
    for batch in batches(np.arange(len(arrays["training_days.npy"])), arrays):
        check_deadline(deadline)
        block = feature_batch(batch, arrays, identifiers, table, margins, days, counts)
        features[position:position + len(block)] = block
        position += len(block)
        impressions += len(batch)
        if impressions % 50000 < len(batch):
            print(json.dumps({"stage": "temporal_full_source_features", "impressions": impressions,
                              "candidate_exposures": position}), flush=True)
    if position != total:
        raise ValueError("Complete original source-order feature coverage required")
    features.flush()
    return features


def design(indices, arrays, features, directory, deadline):
    offsets = arrays["candidate_offsets.npy"]
    size = int(np.sum(offsets[indices + 1] - offsets[indices], dtype=np.uint64))
    directory.mkdir()
    x = np.lib.format.open_memmap(directory / "features.npy", mode="w+", dtype=np.float64,
                                 shape=(size, len(NAMES)))
    y = np.lib.format.open_memmap(directory / "labels.npy", mode="w+", dtype=np.uint8, shape=(size,))
    weight = np.lib.format.open_memmap(directory / "weights.npy", mode="w+", dtype=np.float64, shape=(size,))
    position = positives = 0
    for batch in batches(indices, arrays, candidate_limit=1000000, impression_limit=32768):
        check_deadline(deadline)
        selected = np.concatenate([np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in batch])
        end = position + len(selected)
        x[position:end], y[position:end] = features[selected], arrays["candidate_labels.npy"][selected]
        lengths = (offsets[batch + 1] - offsets[batch]).astype(np.int64)
        weight[position:end] = np.repeat(1. / lengths, lengths)
        positives += int(y[position:end].sum())
        position = end
    for value in (x, y, weight):
        value.flush()
    if position != size or abs(float(weight.sum()) - len(indices)) > 1e-6:
        raise ValueError("Every complete impression must have unit total objective weight")
    return x, y, weight, {"impressions": len(indices), "exposures": size, "positives": positives,
                          "equal_impression_weight": True, "sampling": "none"}


def fit(design_arrays, path, deadline):
    x, y, weight, coverage = design_arrays
    check_deadline(deadline)
    model = HistGradientBoostingClassifier(
        learning_rate=.06, max_iter=160, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=10., categorical_features=None, early_stopping=False, random_state=42, verbose=1,
    ).fit(x, y, sample_weight=weight)
    check_deadline(deadline)
    if model.n_iter_ != 160 or model._preprocessor is not None:
        raise ValueError("Fixed numeric full-support booster without validation split/preprocessing copies required")
    joblib.dump(model, path)
    return model, {**coverage, "iterations": 160, "head_sha256": digest(path)}


def score(model, indices, arrays, features, path, deadline):
    values = np.empty((len(indices), 4), dtype=np.float64)
    offsets, row, exposure = arrays["candidate_offsets.npy"], 0, 0
    with path.open("xb") as handle:
        for batch in batches(indices, arrays, candidate_limit=1000000, impression_limit=32768):
            check_deadline(deadline)
            selected = np.concatenate([np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in batch])
            scores = model.decision_function(features[selected]).astype(np.float64)
            if scores.shape != (len(selected),) or not np.isfinite(scores).all():
                raise ValueError("Every candidate requires one finite frozen score")
            handle.write(scores.tobytes())
            position = 0
            for index in batch:
                count = int(offsets[index + 1] - offsets[index])
                labels = arrays["candidate_labels.npy"][offsets[index]:offsets[index + 1]]
                if not 0 < int(labels.sum()) < count:
                    raise ValueError("Never drop any single-class impression")
                s = scores[position:position + count]
                values[row] = auc_score(labels, s), mrr_score(labels, s), ndcg_score(labels, s, 5), ndcg_score(labels, s, 10)
                position += count
                row += 1
            exposure += len(scores)
    np.save(path.with_suffix(".metrics.npy"), values)
    return {"metrics": dict(zip(("auc", "mrr", "ndcg5", "ndcg10"), values.mean(axis=0).tolist())),
            "impressions": row, "exposures": exposure, "scores_sha256": digest(path),
            "per_impression_metrics_sha256": digest(path.with_suffix(".metrics.npy"))}


def run(args):
    now = datetime.now(timezone.utc)
    if (not 0 < args.max_seconds <= 17400 or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(key) != "4" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
            or np.__version__ != "1.26.4" or sklearn.__version__ != "1.7.2"):
        raise ValueError("Pinned four-thread CPU-only runtime and at most17400 training seconds required")
    if now + timedelta(seconds=22200) > datetime.fromisoformat(AGGREGATE):
        raise TimeoutError("Do not extend the unchanged original aggregate budget")
    if args.output.exists():
        raise FileExistsError("One distinct temporal experiment only;no retry or overwrite")
    dispatch = {key: bool(np.core._multiarray_umath.__cpu_features__[key])
                for key in ("AVX2", "AVX512F", "AVX512_SKX", "AVX512_ICL")}
    if dispatch != {"AVX2": True, "AVX512F": False, "AVX512_SKX": False, "AVX512_ICL": False}:
        raise ValueError("Preserve the exact original tied-score metric runtime")
    deadline = time.monotonic() + args.max_seconds
    prior = load_gate(args.training_artifacts, args.training_proof, PRIOR_PROOF, TRAINING_RESULT_SHA)
    manifest, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    if (manifest["news"], manifest["impressions"], manifest["exposures"], manifest["positives"]) != (
            101527, 2232748, 83507374, 3383656):
        raise ValueError("Only the complete original large-training source is allowed")
    margins_path = args.training_artifacts / "article.margins.npy"
    if digest(margins_path) != prior["feature_extraction"]["margins_sha256"]:
        raise ValueError("Only the independently verified untouched unsupervised frozen-JEV margin")
    margins = np.load(margins_path, mmap_mode="r", allow_pickle=False)
    identifiers = json.loads((args.contract / "news.ids.json").read_bytes())
    news = {}
    with (args.contract / "news.metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            news[item["news_id"]] = {"category": item["category"], "subcategory": item["subcategory"],
                                     "text": item["title"] + " " + item["abstract"]}
    if list(news) != identifiers:
        raise ValueError("Complete original article index/metadata order required")
    args.output.mkdir()
    content = Content.fit(news)
    joblib.dump(content, args.output / "content.joblib")
    table = content.table(news)
    np.savez(args.output / "content.table.npz", vectors=table[1], categories=table[2],
             subcategories=table[3], lengths=table[4])
    days = arrays["training_days.npy"]
    fitting, validation = np.flatnonzero(days < days.max()), np.flatnonzero(days == days.max())
    if len(fitting) != 1801231 or len(validation) != 431517:
        raise ValueError("Unchanged temporary chronological selection split required")
    dates, counts = population(fitting, arrays, len(news), deadline)
    np.savez(args.output / "selection.population.npz", days=dates, counts=counts)
    features = build_features(args.output / "source.features.npy", arrays, identifiers, table,
                              margins, dates, counts, deadline)
    del news
    gc.collect()
    design_arrays = design(fitting, arrays, features, args.output / "selection.design", deadline)
    if tuple(design_arrays[3][key] for key in ("impressions", "exposures", "positives")) != (
            1801231, 66107268, 2712612):
        raise ValueError("Every earlier fitting impression/candidate/positive is required")
    model, selected = fit(design_arrays, args.output / "selection.joblib", deadline)
    del design_arrays
    gc.collect()
    selected.update(score(model, validation, arrays, features, args.output / "validation.scores.f64", deadline))
    baseline = prior["recipes"]["candidate_attention"]["metrics"]
    gain = selected["metrics"]["auc"] - baseline["auc"]
    opens = gain >= .005 and all(selected["metrics"][key] >= baseline[key] for key in ("mrr", "ndcg5", "ndcg10"))
    result = {
        "stage": "temporal_training_gate_evaluated", "source_revision": args.source_revision,
        "prior_training_proof_sha256": PRIOR_PROOF, "selection": selected,
        "unchanged_baseline_metrics": baseline, "training_only_auc_gain": gain, "opens_dev_gate": opens,
        "feature_names": NAMES, "strictly_earlier_day_statistics": True,
        "label_dependent_selection_counts_exclude_last_training_day": True,
        "selection_population_sha256": digest(args.output / "selection.population.npz"),
        "source_features_sha256": digest(args.output / "source.features.npy"),
        "content_sha256": digest(args.output / "content.joblib"),
        "content_table_sha256": digest(args.output / "content.table.npz"),
        "frozen_margins_sha256": digest(margins_path),
        "metric_scope": "last_official_training_day_only;not_official_dev_or_test",
        "head_recipe": "numeric_HGB31_leaves160_rounds_lr.06_L2=10_minleaf100;equal_impression_weight",
        "gpu_allocations": 0, "backbone_fitting": False, "dev_or_test_contents_read": False,
        "competition_test_goal_achieved": False, "independent_verification_pending": True,
        "final_refit_holdout_impressions": None,
    }
    write_json(args.output / "selection.json", result)
    print(json.dumps(result), flush=True)
    del model
    gc.collect()
    if opens:
        full = np.arange(len(days))
        final_design = design(full, arrays, features, args.output / "final.design", deadline)
        if tuple(final_design[3][key] for key in ("impressions", "exposures", "positives")) != (
                2232748, 83507374, 3383656):
            raise ValueError("No final training impression/candidate/positive may be excluded")
        final, refit = fit(final_design, args.output / "selected-full.joblib", deadline)
        final_days, final_counts = population(full, arrays, len(identifiers), deadline)
        np.savez(args.output / "final.population.npz", days=final_days, counts=final_counts)
        result.update(stage="temporal_full_training_fitted", final_refit=refit,
                      final_population_sha256=digest(args.output / "final.population.npz"),
                      final_refit_holdout_impressions=0)
        del final, final_design
    if digest(margins_path) != result["frozen_margins_sha256"]:
        raise ValueError("Original frozen article bytes changed")
    check_deadline(deadline)
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "training-artifacts", "training-proof", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-seconds", type=int, default=17400)
    run(parser.parse_args())
