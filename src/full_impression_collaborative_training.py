"""One bounded CPU collaborative recipe; no dev reads, sampling, search or restart."""

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

from full_impression_centered_training import AGGREGATE_END, PROOF_SHA256, epoch_coverage
from full_impression_collaborative import CollaborativeArticleRanker
from full_impression_contract import check_deadline
from full_impression_evaluation import TRAINING_RESULT_SHA, load_gate
from full_impression_training import MANIFEST_SHA, PROOF_SHA, ModelPredictor, evaluate, fit, load_contract
from jev_content_ranker import digest, write_json


BASIS_PROOF = "df35c50ec29523daa53451c7b5db15f17452f2d68802a8b2b1c1a0bcd71a9fe8"
BASIS_RESULT = "7c7a7da72cf140f318f2f0c754a23a3cd34f4c5df68dd8e57120109eef70f2bc"
MODE = "collaborative_mean"


def load_semantics(args, previous):
    if digest(args.semantic_proof) != BASIS_PROOF:
        raise ValueError("Exact completed independent unsupervised basis proof required")
    proof = json.loads(args.semantic_proof.read_bytes())
    directory = args.semantic_artifacts
    result_path = directory / "result.json"
    if digest(result_path) != BASIS_RESULT or proof["result_sha256"] != BASIS_RESULT:
        raise ValueError("Original rejected centered result must remain byte-identical")
    result = json.loads(result_path.read_bytes())
    if (not proof["basis"]["all_transformed_bits_replayed"] or not proof["basis"]["top_eigensystem_replayed"]
            or proof["basis"]["articles"] != 101527 or proof["dev_or_test_contents_read"]
            or proof["gpu_allocations"] != 0
            or digest(args.training_artifacts / "article.features.npy") != result["raw_training_features_sha256"]
            or result["raw_training_features_sha256"] != previous["feature_extraction"]["features_sha256"]
            or digest(directory / "projection.npz") != result["projection"]["sha256"]
            or digest(directory / "article.centered.npy") != result["transformed_features_sha256"]):
        raise ValueError("Reuse ONLY the independently proven unchanged training-only unsupervised transform")
    encoded = np.load(directory / "article.centered.npy", allow_pickle=False)
    if (encoded.shape != (101528, 128) or encoded.dtype != np.float32
            or not np.isfinite(encoded).all() or encoded[0].any()
            or np.any(np.linalg.norm(encoded[1:], axis=1) <= 0)):
        raise ValueError("Every finite semantic article and exact padding required")
    return F.normalize(torch.from_numpy(encoded), dim=-1), {
        "basis_proof_sha256": BASIS_PROOF, "basis_result_sha256": BASIS_RESULT,
        "transformed_features_sha256": result["transformed_features_sha256"],
        "raw_training_features_sha256": result["raw_training_features_sha256"],
        "projection_sha256": result["projection"]["sha256"],
        "reused_supervised_centered_head": False, "reused_proven_unsupervised_basis_only": True}


def runtime():
    dispatch = {name: bool(np.core._multiarray_umath.__cpu_features__[name])
                for name in ("AVX2", "AVX512F", "AVX512_SKX", "AVX512_ICL")}
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.get_num_threads() != 4
            or torch.cuda.is_available() or np.__version__ != "1.26.4" or dispatch != {
                "AVX2": True, "AVX512F": False, "AVX512_SKX": False, "AVX512_ICL": False}):
        raise ValueError("Pinned numeric/tie dispatch, four CPU threads and no CUDA required")


def run(args):
    runtime()
    if not 0 < args.max_seconds <= 9000 or not re.fullmatch("[0-9a-f]{40}", args.source_revision):
        raise ValueError("Exact immutable source and at most9000 training seconds")
    if datetime.now(timezone.utc) + timedelta(seconds=11400) > datetime.fromisoformat(AGGREGATE_END):
        raise TimeoutError("Three-hour distinct campaign plus margin must fit the ORIGINAL aggregate")
    if args.output.exists():
        raise FileExistsError("Unique collaborative campaign; no retry or overwrite")
    deadline = time.monotonic() + args.max_seconds
    previous = load_gate(args.training_artifacts, args.training_proof, PROOF_SHA256, TRAINING_RESULT_SHA)
    manifest, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    features, semantics = load_semantics(args, previous)
    days = arrays["training_days.npy"]
    fitting, validation = np.flatnonzero(days < days.max()), np.flatnonzero(days == days.max())
    if (manifest["impressions"], manifest["news"], len(fitting), len(validation)) != (
            2232748, 101527, 1801231, 431517):
        raise ValueError("Complete original official training-only populations required")
    args.output.mkdir()
    model, fitted = fit(features, fitting, arrays, MODE, args.output / "collaborative.selection.pt",
                        deadline, model_factory=CollaborativeArticleRanker)
    if not epoch_coverage(fitted, 1801231, 66107268, 2712612):
        raise ValueError("Every earlier impression/candidate/positive must occur in every epoch")
    selected = {"fitting": fitted, **evaluate(validation, arrays, ModelPredictor(model),
                args.output / "validation.collaborative.scores.f64", deadline)}
    baseline = previous["recipes"]["candidate_attention"]["metrics"]
    gain = selected["metrics"]["auc"] - baseline["auc"]
    opens = gain >= .005 and all(selected["metrics"][key] >= baseline[key]
                                for key in ("mrr", "ndcg5", "ndcg10"))
    result = {
        "stage": "collaborative_training_gate_evaluated", "source_revision": args.source_revision,
        "selection": selected, "semantics": semantics, "prior_training_proof_sha256": PROOF_SHA256,
        "unchanged_baseline_metrics": baseline, "training_only_auc_gain": gain, "opens_dev_gate": opens,
        "metric_scope": "last_official_training_day_only;not_official_dev_or_test",
        "recipe": "fixed_semantics128+zero_initialized_trainable_article_residual_and_bias;"
                  "masked_recency_mean;10cosine;three_epochs;AdamW.001;WD.0001;clip1;seeds42_43_44;"
                  "all_candidates_all_clicks_equal_impression_weight;no_sampling_or_search",
        "gpu_allocations": 0, "backbone_fitting": False, "dev_or_test_contents_read": False,
        "competition_test_goal_achieved": False, "original_training_gpu_cutoff_not_reset": True,
        "independent_verification_pending": True, "final_refit_holdout_impressions": None}
    write_json(args.output / "selection.json", result)
    print(json.dumps(result), flush=True)
    del model
    gc.collect()
    if opens:
        final, refit = fit(features, np.arange(len(days)), arrays, MODE, args.output / "selected-full.pt",
                           deadline, model_factory=CollaborativeArticleRanker)
        if not epoch_coverage(refit, 2232748, 83507374, 3383656):
            raise ValueError("Fresh final fitting must retain ALL official training impressions and exposures")
        del final
        result.update(stage="collaborative_full_training_fitted", final_refit=refit,
                      final_refit_holdout_impressions=0, training_impressions=2232748,
                      training_exposures=83507374)
    _, checked = load_semantics(args, previous)
    if checked != semantics:
        raise ValueError("Frozen semantic provenance changed")
    check_deadline(deadline)
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


def parser(description):
    result = argparse.ArgumentParser(description=description)
    for name in ("contract", "training-artifacts", "training-proof", "semantic-artifacts", "semantic-proof"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--source-revision", required=True)
    return result


if __name__ == "__main__":
    arguments = parser(__doc__)
    arguments.add_argument("--output", type=Path, required=True)
    arguments.add_argument("--max-seconds", type=int, default=9000)
    torch.set_num_threads(4)
    run(arguments.parse_args())
