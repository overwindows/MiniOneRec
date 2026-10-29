"""Independent complete training-only basis, frozen scores and coverage replay."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from full_impression_contract import check_deadline
from full_impression_evaluation import TRAINING_RESULT_SHA, load_gate
from full_impression_training import MANIFEST_SHA, PROOF_SHA, batch_inputs, batches, load_contract
from full_impression_verification import CachedReplay, check_scores, metrics
from jev_content_ranker import digest


PRIOR_PROOF = "305808f97e4dc08f7d0b88fc690db655919052fc60a74fd68f15b90e3a30b128"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_basis(raw, directory, deadline, dimension=128):
    require(raw.ndim == 2 and raw.dtype == np.float32 and len(raw) > dimension + 1
            and not raw[0].any(), "Complete finite FP32 articles and exact zero padding required")
    with np.load(directory / "projection.npz", allow_pickle=False) as stored:
        require(set(stored.files) == {"mean", "axes", "variance", "ridge"}, "Exact basis inventory")
        mean, axes, variance, ridge = [stored[name] for name in ("mean", "axes", "variance", "ridge")]
    require(mean.shape == (raw.shape[1],) and axes.shape == (raw.shape[1], dimension)
            and variance.shape == (dimension,) and ridge.shape == ()
            and all(value.dtype == np.float64 and np.isfinite(value).all()
                    for value in (mean, axes, variance, ridge))
            and np.all(variance > 0) and np.all(np.diff(variance) <= 0)
            and float(ridge) == .01 * float(variance.mean()), "Fixed finite descending PCA/ridge recipe")
    require(np.allclose(axes.T @ axes, np.eye(dimension), atol=1e-10, rtol=0)
            and np.all(axes[np.argmax(np.abs(axes), axis=0), np.arange(dimension)] > 0),
            "Orthonormal canonical-sign axes required")
    total = np.zeros(raw.shape[1], dtype=np.float64)
    for start in range(1, len(raw), 4096):
        check_deadline(deadline)
        block = np.array(raw[start:start + 4096], dtype=np.float64)
        norm = np.linalg.norm(block, axis=1)
        require(np.isfinite(norm).all() and np.all(norm > 0), "All original article rows must be valid")
        total += (block / norm[:, None]).sum(axis=0)
    replayed_mean = total / (len(raw) - 1)
    require(np.array_equal(mean, replayed_mean), "Complete raw-source training mean differs")
    encoded = np.load(directory / "article.centered.npy", mmap_mode="r", allow_pickle=False)
    require(encoded.dtype == np.float32 and encoded.shape == (len(raw), dimension)
            and np.all(encoded[0].view(np.uint32) == 0), "Exact transformed inventory and zero padding")
    covariance = np.zeros((raw.shape[1], raw.shape[1]), dtype=np.float64)
    matrix = axes / np.sqrt(variance + ridge)
    for start in range(1, len(raw), 4096):
        check_deadline(deadline)
        block = np.array(raw[start:start + 4096], dtype=np.float64)
        centered = block / np.linalg.norm(block, axis=1)[:, None] - replayed_mean
        covariance += centered.T @ centered
        expected = (centered @ matrix).astype(np.float32)
        require(np.isfinite(expected).all() and np.all(np.linalg.norm(expected, axis=1) > 0)
                and np.array_equal(expected.view(np.uint32),
                                   encoded[start:start + len(expected)].view(np.uint32)),
                "Every transformed article bit must replay from the unchanged raw training source")
    covariance /= len(raw) - 1
    expected_variance = np.linalg.eigvalsh(covariance)[-dimension:][::-1]
    check_deadline(deadline)
    residual = float(np.max(np.abs(covariance @ axes - axes * variance)))
    require(residual <= 1e-10 and np.allclose(expected_variance, variance, atol=1e-12, rtol=0),
            "Independent centered covariance must reproduce the TOP retained eigensystem")
    return {"articles": len(raw) - 1, "all_transformed_bits_replayed": True,
            "top_eigensystem_replayed": True, "maximum_eigen_equation_residual": residual}


def verify_epochs(record, indices, arrays, counts, deadline, expected_mode="candidate_attention",
                  seeds=(42, 43, 44)):
    require(record["mode"] == expected_mode and len(record["epochs"]) == len(seeds),
            "Exactly the registered fixed epochs and mode required")
    for seed, row in zip(seeds, record["epochs"]):
        check_deadline(deadline)
        order = np.random.default_rng(seed).permutation(indices)
        steps = sum(1 for _ in batches(order, arrays))
        require(tuple(row[key] for key in ("impressions", "exposures", "positives")) == counts
                and row["optimizer_steps"] == steps and np.isfinite(row["mean_loss"]),
                "Complete exact source/batch coverage required in each frozen epoch")


def run(args):
    deadline = time.monotonic() + args.max_seconds
    require(0 < args.max_seconds <= 600 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and torch.get_num_threads() == 4 and not torch.cuda.is_available(), "Bounded four-thread CPU replay")
    dispatch = {name: bool(np.core._multiarray_umath.__cpu_features__[name])
                for name in ("AVX2", "AVX512F", "AVX512_SKX", "AVX512_ICL")}
    require(np.__version__ == "1.26.4" and dispatch == {
        "AVX2": True, "AVX512F": False, "AVX512_SKX": False, "AVX512_ICL": False},
        "Unchanged numeric runtime and exact tied-score metric dispatch")
    require(not args.proof.exists(), "Never overwrite independent evidence")
    prior = load_gate(args.training_artifacts, args.training_proof, PRIOR_PROOF, TRAINING_RESULT_SHA)
    _, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    result_path = args.artifacts / "result.json"
    result_sha = digest(result_path)
    result = json.loads(result_path.read_bytes())
    raw_path = args.training_artifacts / "article.features.npy"
    require(result["source_revision"] == args.source_revision
            and result["prior_training_proof_sha256"] == PRIOR_PROOF
            and result["raw_training_features_sha256"] == digest(raw_path)
            == prior["feature_extraction"]["features_sha256"]
            and not result["dev_or_test_contents_read"] and not result["backbone_fitting"]
            and result["gpu_allocations"] == 0 and not result["competition_test_goal_achieved"],
            "Only the exact preregistered training-only experiment")
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    require(raw.shape == (101528, 2560), "Every unchanged original training article required")
    basis = verify_basis(raw, args.artifacts, deadline)
    for name, expected in (("projection.npz", result["projection"]["sha256"]),
                           ("article.centered.npy", result["transformed_features_sha256"])):
        require(digest(args.artifacts / name) == expected, "Frozen derived bytes changed")
    features = F.normalize(torch.from_numpy(np.load(args.artifacts / "article.centered.npy")), dim=-1)
    days = arrays["training_days.npy"]
    fitting, validation = np.flatnonzero(days < days.max()), np.flatnonzero(days == days.max())
    require(len(fitting) == 1801231 and len(validation) == 431517, "Complete chronological populations")
    selected = result["selection"]
    verify_epochs(selected, fitting, arrays, (1801231, 66107268, 2712612), deadline)
    head = args.artifacts / "candidate_attention.selection.pt"
    require(digest(head) == selected["head_sha256"], "Selection head bytes changed")
    replay = CachedReplay(features, torch.load(head, map_location="cpu", weights_only=True), "candidate_attention")
    scores_path = args.artifacts / "validation.candidate_attention.scores.f64"
    recorded_path = scores_path.with_suffix(".metrics.npy")
    require(scores_path.stat().st_size == 17400106 * 8
            and digest(scores_path) == selected["scores_sha256"]
            and digest(recorded_path) == selected["per_impression_metrics_sha256"], "Complete frozen scores/metrics")
    scores = np.memmap(scores_path, dtype=np.float64, mode="r", shape=(17400106,))
    recorded = np.load(recorded_path, mmap_mode="r", allow_pickle=False)
    require(recorded.shape == (431517, 4), "All per-impression metric cells required")
    values = np.empty((len(validation), 4), dtype=np.float64)
    row = position = positives = 0
    maximum_error = 0.
    for batch in batches(validation, arrays):
        check_deadline(deadline)
        (history, candidates, labels, groups), lengths = batch_inputs(batch, arrays, "cpu")
        actual = scores[position:position + len(candidates)]
        maximum_error = max(maximum_error, check_scores(actual, replay(history, candidates, groups)))
        targets, local = labels.numpy(), 0
        positives += int(targets.sum())
        for count in lengths:
            values[row] = metrics(targets[local:local + count], actual[local:local + count])
            local += count
            row += 1
        position += len(candidates)
        if row % 50000 < len(batch):
            print(json.dumps({"stage": "independent_centered_score_replay", "impressions": row}), flush=True)
    require((row, position, positives) == (431517, 17400106, 671044)
            and selected["impressions"] == row and selected["exposures"] == position
            and np.allclose(values, recorded, atol=1e-12, rtol=0), "Complete identical raw-score metric replay")
    rebuilt = dict(zip(("auc", "mrr", "ndcg5", "ndcg10"), values.mean(axis=0).tolist()))
    require(all(abs(rebuilt[key] - selected["metrics"][key]) <= 1e-12 for key in rebuilt),
            "Every aggregate metric must replay")
    baseline = prior["recipes"]["candidate_attention"]["metrics"]
    gain = rebuilt["auc"] - baseline["auc"]
    opens = gain >= .005 and all(rebuilt[key] >= baseline[key] for key in ("mrr", "ndcg5", "ndcg10"))
    require(result["unchanged_baseline_metrics"] == baseline
            and abs(gain - result["training_only_auc_gain"]) <= 1e-12
            and opens == result["opens_dev_gate"], "Independent stronger-baseline gate must agree")
    if opens:
        require(result["stage"] == "centered_full_training_fitted"
                and result["final_refit_holdout_impressions"] == 0, "Fresh ALL-training refit required")
        verify_epochs(result["final_refit"], np.arange(len(days)), arrays,
                      (2232748, 83507374, 3383656), deadline)
        final = args.artifacts / "selected-full.pt"
        require(digest(final) == result["final_refit"]["head_sha256"], "Final head bytes changed")
        CachedReplay(features, torch.load(final, map_location="cpu", weights_only=True), "candidate_attention")
    else:
        require(result["stage"] == "centered_training_gate_evaluated"
                and not (args.artifacts / "selected-full.pt").exists()
                and "final_refit" not in result, "Rejected gate cannot claim a final fit")
    require(digest(raw_path) == result["raw_training_features_sha256"]
            and digest(result_path) == result_sha, "Raw source/result changed during verification")
    check_deadline(deadline)
    proof = {"source_revision": args.source_revision, "result_sha256": result_sha,
             "prior_training_proof_sha256": PRIOR_PROOF, "basis": basis, "metrics": rebuilt,
             "coverage": {"impressions": row, "exposures": position, "positives": positives},
             "maximum_score_replay_error": maximum_error, "training_only_auc_gain": gain,
             "opens_dev_gate": opens, "full_training_epoch_source_and_coverage_verified": opens,
             "optimizer_training_independently_reproduced": False, "metric_tolerance": 1e-12,
             "score_tolerance": {"atol": 1e-4, "rtol": 1e-5},
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
    parser.add_argument("--max-seconds", type=int, default=600)
    torch.set_num_threads(4)
    run(parser.parse_args())
