"""One fixed two-epoch joint-view experiment with independent full-score replay."""

from datetime import datetime, timedelta, timezone
import gc
import json
from pathlib import Path
import re
import time

import numpy as np
from scipy import sparse
import torch

from full_impression_centered_training import AGGREGATE_END, PROOF_SHA256, epoch_coverage
from full_impression_centered_verification import require, verify_epochs
from full_impression_collaborative_training import load_semantics, runtime
from full_impression_contract import check_deadline
from full_impression_evaluation import TRAINING_RESULT_SHA, load_gate
from full_impression_multiview import MODE, MultiViewRanker, MultiViewReplay
from full_impression_training import (
    MANIFEST_SHA, PROOF_SHA, ModelPredictor, batch_inputs, batches, evaluate, fit, load_contract,
)
from full_impression_verification import check_scores, metrics
from jev_content_ranker import digest, write_json


LEXICAL_PROOF = "f679cee8e25a8f6c211c64a266299225c11ba67486de79e7699cb2aa797afa75"
LEXICAL_RESULT = "76577be4f709e4ee6a26678f3065a97ad86609f01d2d295006d44775eb6f42a0"


def load_views(args, previous):
    features, semantics = load_semantics(args, previous)
    require(digest(args.lexical_proof) == LEXICAL_PROOF
            and digest(args.lexical_artifacts / "result.json") == LEXICAL_RESULT,
            "Exact completed source/lexical-bit proof and unchanged archived result")
    proof = json.loads(args.lexical_proof.read_bytes())
    old = json.loads((args.lexical_artifacts / "result.json").read_bytes())
    require(proof["result_sha256"] == LEXICAL_RESULT
            and proof["lexical_source_and_initialization_all_bits_reproduced"]
            and proof["lexical"] == old["lexical"] and not proof["dev_or_test_contents_read"]
            and proof["gpu_allocations"] == 0, "Reuse ONLY proven unsupervised lexical initialization")
    lexical = old["lexical"]
    for name in ("lexical.joblib", "article.terms.npz", "words.initial.npy"):
        require(digest(args.lexical_artifacts / name) == lexical[name], "Pinned unsupervised lexical bytes")
    terms = sparse.load_npz(args.lexical_artifacts / "article.terms.npz")
    initial = np.load(args.lexical_artifacts / "words.initial.npy", allow_pickle=False)
    require(terms.shape == (101528, 32768) and terms.dtype == np.float32
            and initial.shape == (32768, 64) and initial.dtype == np.float32
            and np.isfinite(terms.data).all() and np.isfinite(initial).all()
            and terms[0].nnz == 0 and np.all(np.diff(terms.indptr) <= 32),
            "Complete proven lexical inventory")
    return features, terms, initial, {
        "semantics": semantics, "lexical": lexical, "lexical_proof_sha256": LEXICAL_PROOF,
        "lexical_result_sha256": LEXICAL_RESULT, "reused_supervised_heads_or_words": False,
        "prior_unsupervised_proofs_reused_not_recomputed": True}


def train(args):
    runtime()
    require(0 < args.max_seconds <= 6600 and re.fullmatch("[0-9a-f]{40}", args.source_revision),
            "Fixed immutable source and at most6600 CPU training seconds")
    require(datetime.now(timezone.utc) + timedelta(seconds=8700) <
            datetime.fromisoformat(AGGREGATE_END), "8100s campaign and margin inside ORIGINAL aggregate")
    require(not args.output.exists(), "Unique joint-view campaign;no retry or overwrite")
    deadline = time.monotonic() + args.max_seconds
    previous = load_gate(args.training_artifacts, args.training_proof, PROOF_SHA256, TRAINING_RESULT_SHA)
    _, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    features, terms, initial, views = load_views(args, previous)
    days = arrays["training_days.npy"]
    fitting, validation = np.flatnonzero(days < days.max()), np.flatnonzero(days == days.max())
    require((len(fitting), len(validation)) == (1801231, 431517), "Original complete chronological cohorts")
    args.output.mkdir()
    factory = lambda f, mode: MultiViewRanker(f, mode, terms, initial)
    model, fitted = fit(features, fitting, arrays, MODE, args.output / "multiview.selection.pt",
                        deadline, model_factory=factory, epoch_count=2)
    require(epoch_coverage(fitted, 1801231, 66107268, 2712612, epochs=2), "Every row in BOTH fitting epochs")
    selected = {"fitting": fitted, **evaluate(validation, arrays, ModelPredictor(model),
                args.output / "validation.multiview.scores.f64", deadline)}
    baseline = previous["recipes"]["candidate_attention"]["metrics"]
    gain = selected["metrics"]["auc"] - baseline["auc"]
    opens = gain >= .005 and all(selected["metrics"][key] >= baseline[key]
                                for key in ("mrr", "ndcg5", "ndcg10"))
    result = {
        "stage": "multiview_training_gate_evaluated", "source_revision": args.source_revision,
        "selection": selected, "views": views, "prior_training_proof_sha256": PROOF_SHA256,
        "unchanged_baseline_metrics": baseline, "training_only_auc_gain": gain, "opens_dev_gate": opens,
        "metric_scope": "last_official_training_day_only;not_official_dev_or_test",
        "recipe": "joint_shared_title_terms64+learned_biasfree_semantic128to64;"
                  "equal_L2_concat128;four_interest_queries;logmeanexp10cosine;"
                  "NO_item_ID_params_or_biases;TWO_fixed_epochs;AdamW.001;WD.0001;clip1;seeds42_43;"
                  "all_candidates_all_clicks_equal_impression_weight;no_recipe_or_epoch_search",
        "gpu_allocations": 0, "backbone_fitting": False, "dev_or_test_contents_read": False,
        "competition_test_goal_achieved": False, "original_training_gpu_cutoff_not_reset": True,
        "independent_verification_pending": True, "final_refit_holdout_impressions": None}
    write_json(args.output / "selection.json", result)
    print(json.dumps(result), flush=True)
    del model
    gc.collect()
    if opens:
        final, refit = fit(features, np.arange(len(days)), arrays, MODE, args.output / "selected-full.pt",
                           deadline, model_factory=factory, epoch_count=2)
        require(epoch_coverage(refit, 2232748, 83507374, 3383656, epochs=2),
                "Fresh final fitting must retain EVERY official training row in BOTH epochs")
        del final
        result.update(stage="multiview_full_training_fitted", final_refit=refit,
                      final_refit_holdout_impressions=0, training_impressions=2232748,
                      training_exposures=83507374)
    require(load_views(args, previous)[3] == views, "All proven unsupervised bytes unchanged")
    check_deadline(deadline)
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


def verify(args):
    runtime()
    require(0 < args.max_seconds <= 1200 and args.proof is not None and not args.proof.exists(),
            "Bounded unique independent proof")
    deadline = time.monotonic() + args.max_seconds
    previous = load_gate(args.training_artifacts, args.training_proof, PROOF_SHA256, TRAINING_RESULT_SHA)
    _, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    features, terms, _, views = load_views(args, previous)
    result_path = args.output / "result.json"
    result_sha = digest(result_path)
    result = json.loads(result_path.read_bytes())
    require(result["source_revision"] == args.source_revision and result["views"] == views
            and result["gpu_allocations"] == 0 and not result["dev_or_test_contents_read"]
            and not result["backbone_fitting"] and not result["competition_test_goal_achieved"],
            "Exact immutable joint-view training-only result")
    days = arrays["training_days.npy"]
    fitting, validation = np.flatnonzero(days < days.max()), np.flatnonzero(days == days.max())
    require((len(fitting), len(validation)) == (1801231, 431517), "Complete unchanged cohorts")
    selected = result["selection"]
    verify_epochs(selected["fitting"], fitting, arrays, (1801231, 66107268, 2712612),
                  deadline, MODE, seeds=(42, 43))
    head = args.output / "multiview.selection.pt"
    require(digest(head) == selected["fitting"]["head_sha256"], "Frozen selected head")
    replay = MultiViewReplay(features, terms, torch.load(head, map_location="cpu", weights_only=True))
    scores_path = args.output / "validation.multiview.scores.f64"
    metric_path = scores_path.with_suffix(".metrics.npy")
    require(scores_path.stat().st_size == 17400106 * 8 and digest(scores_path) == selected["scores_sha256"]
            and digest(metric_path) == selected["per_impression_metrics_sha256"], "All immutable score/metric bytes")
    scores = np.memmap(scores_path, dtype=np.float64, mode="r", shape=(17400106,))
    recorded = np.load(metric_path, mmap_mode="r", allow_pickle=False)
    require(recorded.shape == (431517, 4), "Every original metric cell")
    rebuilt = np.empty((len(validation), 4), dtype=np.float64)
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
            rebuilt[row] = metrics(targets[local:local + count], actual[local:local + count])
            local += count
            row += 1
        position += len(candidates)
    require((row, position, positives) == (431517, 17400106, 671044)
            and selected["impressions"] == row and selected["exposures"] == position
            and np.allclose(rebuilt, recorded, atol=1e-12, rtol=0), "All labels and metric cells replay")
    aggregate = dict(zip(("auc", "mrr", "ndcg5", "ndcg10"), rebuilt.mean(axis=0).tolist()))
    require(all(abs(aggregate[key] - selected["metrics"][key]) <= 1e-12 for key in aggregate),
            "Every aggregate metric")
    baseline = previous["recipes"]["candidate_attention"]["metrics"]
    gain = aggregate["auc"] - baseline["auc"]
    opens = gain >= .005 and all(aggregate[key] >= baseline[key] for key in ("mrr", "ndcg5", "ndcg10"))
    require(result["unchanged_baseline_metrics"] == baseline
            and abs(gain - result["training_only_auc_gain"]) <= 1e-12
            and result["opens_dev_gate"] == opens, "Independent stronger-original-attention gate")
    if opens:
        require(result["stage"] == "multiview_full_training_fitted"
                and result["final_refit_holdout_impressions"] == 0, "Fresh complete final fit")
        verify_epochs(result["final_refit"], np.arange(len(days)), arrays,
                      (2232748, 83507374, 3383656), deadline, MODE, seeds=(42, 43))
        final = args.output / "selected-full.pt"
        require(digest(final) == result["final_refit"]["head_sha256"], "Frozen final head")
        MultiViewReplay(features, terms, torch.load(final, map_location="cpu", weights_only=True))
    else:
        require(result["stage"] == "multiview_training_gate_evaluated" and "final_refit" not in result
                and not (args.output / "selected-full.pt").exists(), "Closed gate cannot claim a final fit")
    require(load_views(args, previous)[3] == views and digest(result_path) == result_sha
            and digest(head) == selected["fitting"]["head_sha256"]
            and digest(scores_path) == selected["scores_sha256"]
            and digest(metric_path) == selected["per_impression_metrics_sha256"], "Immutable bytes before/after")
    check_deadline(deadline)
    proof = {
        "source_revision": args.source_revision, "result_sha256": result_sha, "views": views,
        "metrics": aggregate, "coverage": {"impressions": row, "exposures": position, "positives": positives},
        "maximum_score_replay_error": maximum_error, "training_only_auc_gain": gain, "opens_dev_gate": opens,
        "selection_epoch_source_and_coverage_verified": True,
        "full_training_epoch_source_and_coverage_verified": opens, "fixed_epoch_count": 2,
        "every_new_model_score_independently_replayed": True,
        "independent_inference": "NumPy_semantic_projection+SciPy_CSR_term_vectors+NumPy_interest_pool",
        "optimizer_training_independently_reproduced": False,
        "prior_unsupervised_proofs_reused_not_recomputed": True,
        "metric_tolerance": 1e-12, "score_tolerance": {"atol": 1e-4, "rtol": 1e-5},
        "metric_scope": "last_official_training_day_only;not_official_dev_or_test",
        "gpu_allocations": 0, "dev_or_test_contents_read": False, "competition_test_goal_achieved": False}
    with args.proof.open("x", encoding="utf-8") as handle:
        json.dump(proof, handle, indent=2, allow_nan=False)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "training-artifacts", "training-proof", "semantic-artifacts", "semantic-proof",
                 "lexical-artifacts", "lexical-proof", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--max-seconds", type=int, required=True)
    torch.set_num_threads(4)
    arguments = parser.parse_args()
    (verify if arguments.verify else train)(arguments)
