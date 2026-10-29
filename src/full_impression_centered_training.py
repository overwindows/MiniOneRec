"""One CPU-only centered/whitened readout recipe; training-only selection and full refit."""

import argparse
from datetime import datetime, timedelta, timezone
import gc
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from full_impression_contract import check_deadline
from full_impression_evaluation import TRAINING_RESULT_SHA, load_gate
from full_impression_training import MANIFEST_SHA, PROOF_SHA, ModelPredictor, evaluate, fit, load_contract
from jev_content_ranker import digest, write_json


PROOF_SHA256 = "305808f97e4dc08f7d0b88fc690db655919052fc60a74fd68f15b90e3a30b128"
AGGREGATE_END = "2026-10-30T01:17:07.923179+00:00"
DIMENSION = 128
RIDGE_FRACTION = .01


def unit_rows(raw):
    block = np.array(raw, dtype=np.float64)
    norms = np.linalg.norm(block, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Every non-padding article needs a finite nonzero frozen feature")
    return block / norms[:, None]


def projection(raw, output, deadline, dimension=DIMENSION):
    if output.exists():
        raise FileExistsError("Never overwrite a frozen training projection")
    if raw.ndim != 2 or raw.dtype != np.float32 or not 1 <= dimension <= raw.shape[1]:
        raise ValueError("A valid low-rank dimension and complete FP32 frozen article matrix are required")
    if len(raw) - 1 <= dimension or raw[0].any():
        raise ValueError("Complete training articles and exact zero padding are required")
    total = np.zeros(raw.shape[1], dtype=np.float64)
    cross = np.zeros((raw.shape[1], raw.shape[1]), dtype=np.float64)
    for start in range(1, len(raw), 4096):
        check_deadline(deadline)
        block = unit_rows(raw[start:start + 4096])
        total += block.sum(axis=0)
        cross += block.T @ block
    mean = total / (len(raw) - 1)
    covariance = cross / (len(raw) - 1) - np.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2
    variance, axes = np.linalg.eigh(covariance)
    check_deadline(deadline)
    variance, axes = variance[-dimension:][::-1].copy(), axes[:, -dimension:][:, ::-1].copy()
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("The fixed retained PCA dimensions must have positive finite variance")
    signs = np.sign(axes[np.argmax(np.abs(axes), axis=0), np.arange(dimension)])
    axes *= signs
    ridge = RIDGE_FRACTION * float(variance.mean())
    np.savez(output, mean=mean, axes=axes, variance=variance, ridge=np.asarray(ridge))
    return {"articles": len(raw) - 1, "input_dimension": raw.shape[1], "dimension": dimension,
            "ridge_fraction": RIDGE_FRACTION, "ridge": ridge,
            "retained_training_variance_fraction": float(variance.sum() / np.trace(covariance)),
            "fit_scope": "ALL_official_training_article_features_only;no_labels_or_dev",
            "sha256": digest(output)}


def transform(raw, parameters, output, deadline):
    if output.exists():
        raise FileExistsError("Never overwrite transformed frozen article features")
    if raw.ndim != 2 or not len(raw) or raw.dtype != np.float32 or raw[0].any():
        raise ValueError("Only unchanged FP32 frozen article features with zero padding are allowed")
    with np.load(parameters, allow_pickle=False) as stored:
        if set(stored.files) != {"mean", "axes", "variance", "ridge"}:
            raise ValueError("Exact frozen centering/whitening parameters are required")
        mean, axes, variance, ridge = [stored[name] for name in ("mean", "axes", "variance", "ridge")]
    if (variance.ndim != 1 or not 1 <= len(variance) <= raw.shape[1]
            or mean.shape != (raw.shape[1],) or axes.shape != (raw.shape[1], len(variance))
            or any(value.dtype != np.float64 for value in (mean, axes, variance, ridge))
            or ridge.shape != () or float(ridge) <= 0
            or any(not np.isfinite(value).all() for value in (mean, axes, variance, ridge))
            or np.any(variance <= 0)
            or float(ridge) != RIDGE_FRACTION * float(variance.mean())
            or not np.allclose(axes.T @ axes, np.eye(len(variance)), atol=1e-10, rtol=0)):
        raise ValueError("Finite aligned orthonormal frozen projection and positive whitening scales required")
    matrix = axes / np.sqrt(variance + ridge)
    result = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32,
                                      shape=(len(raw), len(variance)))
    result[0] = 0
    for start in range(1, len(raw), 4096):
        check_deadline(deadline)
        projected = ((unit_rows(raw[start:start + 4096]) - mean) @ matrix).astype(np.float32)
        if not np.isfinite(projected).all() or np.any(np.linalg.norm(projected, axis=1) <= 0):
            raise ValueError("Every training article must retain one finite nonzero transformed row")
        result[start:start + len(projected)] = projected
    result.flush()
    del result


def epoch_coverage(record, impressions, exposures, positives, epochs=3):
    return len(record["epochs"]) == epochs and all(
        row["impressions"] == impressions and row["exposures"] == exposures
        and row["positives"] == positives and np.isfinite(row["mean_loss"])
        and row["optimizer_steps"] > 0 for row in record["epochs"])


def run(args):
    if (not 0 < args.max_seconds <= 21000 or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or torch.get_num_threads() != 4 or torch.cuda.is_available()
            or not re.fullmatch("[0-9a-f]{40}", args.source_revision)):
        raise ValueError("Immutable source, four CPU threads, no CUDA and at most21000 training seconds required")
    now = datetime.now(timezone.utc)
    if now + timedelta(seconds=21600 + 600) > datetime.fromisoformat(AGGREGATE_END):
        raise TimeoutError("A full six-hour CPU campaign and margin must fit the unchanged aggregate ceiling")
    if args.output.exists():
        raise FileExistsError("Preserve unique centered-training output;no automatic retry")
    deadline = time.monotonic() + args.max_seconds
    previous = load_gate(args.training_artifacts, args.training_proof, PROOF_SHA256, TRAINING_RESULT_SHA)
    manifest, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    raw_path = args.training_artifacts / "article.features.npy"
    raw_sha = previous["feature_extraction"]["features_sha256"]
    if digest(raw_path) != raw_sha or manifest["news"] != 101527:
        raise ValueError("Only unchanged complete independently verified large-training article features")
    dispatch = {name: bool(np.core._multiarray_umath.__cpu_features__[name])
                for name in ("AVX2", "AVX512F", "AVX512_SKX", "AVX512_ICL")}
    if np.__version__ != "1.26.4" or dispatch != {
            "AVX2": True, "AVX512F": False, "AVX512_SKX": False, "AVX512_ICL": False}:
        raise ValueError("Preserve the preregistered NumPy version and non-AVX512 metric tie dispatch")
    args.output.mkdir()
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if raw.shape != (101528, 2560):
        raise ValueError("All101527 unchanged training article rows and padding required")
    parameters = args.output / "projection.npz"
    transform_report = projection(raw, parameters, deadline)
    transformed = args.output / "article.centered.npy"
    transform(raw, parameters, transformed, deadline)
    features = F.normalize(torch.from_numpy(np.load(transformed, allow_pickle=False)), dim=-1)
    days = arrays["training_days.npy"]
    fitting, validation = np.flatnonzero(days < days.max()), np.flatnonzero(days == days.max())
    if len(fitting) != 1801231 or len(validation) != 431517:
        raise ValueError("Unchanged full chronological training-only cohorts required")
    model, selected = fit(features, fitting, arrays, "candidate_attention",
                          args.output / "candidate_attention.selection.pt", deadline)
    if not epoch_coverage(selected, 1801231, 66107268, 2712612):
        raise ValueError("Every earlier fitting impression/exposure/positive required in allthree epochs")
    selected.update(evaluate(validation, arrays, ModelPredictor(model),
                             args.output / "validation.candidate_attention.scores.f64", deadline))
    baseline = previous["recipes"]["candidate_attention"]
    gain = selected["metrics"]["auc"] - baseline["metrics"]["auc"]
    opens = gain >= .005 and all(selected["metrics"][key] >= baseline["metrics"][key]
                                for key in ("mrr", "ndcg5", "ndcg10"))
    result = {
        "stage": "centered_training_gate_evaluated", "source_revision": args.source_revision,
        "prior_training_proof_sha256": PROOF_SHA256, "raw_training_features_sha256": raw_sha,
        "projection": transform_report, "transformed_features_sha256": digest(transformed),
        "selection": selected, "unchanged_baseline_metrics": baseline["metrics"],
        "training_only_auc_gain": gain, "opens_dev_gate": opens,
        "metric_scope": "last_official_training_day_only;not_official_dev_or_test",
        "feature_transform": "unit_raw_article;subtract_training_mean;PCA128;ridge_whiten;L2_before_head",
        "head_recipe": "same128dim_candidate_attention;three_epochs;AdamW.001;WD.0001;seeds42_43_44",
        "gpu_allocations": 0, "backbone_fitting": False, "dev_or_test_contents_read": False,
        "competition_test_goal_achieved": False, "original_training_gpu_cutoff_not_reset": True,
        "independent_verification_pending": True, "final_refit_holdout_impressions": None,
    }
    write_json(args.output / "selection.json", result)
    print(json.dumps(result), flush=True)
    del model
    gc.collect()
    if opens:
        full = np.arange(len(days))
        final, refit = fit(features, full, arrays, "candidate_attention",
                            args.output / "selected-full.pt", deadline)
        if not epoch_coverage(refit, 2232748, 83507374, 3383656):
            raise ValueError("Final fitting must retain EVERY official training impression/candidate/positive")
        del final
        result.update(stage="centered_full_training_fitted", final_refit=refit,
                      final_refit_holdout_impressions=0, training_impressions=2232748,
                      training_exposures=83507374)
    if (digest(raw_path) != raw_sha or digest(parameters) != transform_report["sha256"]
            or digest(transformed) != result["transformed_features_sha256"]):
        raise ValueError("Frozen raw/basis/transformed training bytes changed")
    check_deadline(deadline)
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "training-artifacts", "training-proof", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-seconds", type=int, default=21000)
    torch.set_num_threads(4)
    run(parser.parse_args())
