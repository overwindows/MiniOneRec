"""Complete independent collaborative score/metric/coverage replay; no new fitting."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from full_impression_centered_training import PROOF_SHA256
from full_impression_centered_verification import require, verify_epochs
from full_impression_collaborative import CollaborativeReplay
from full_impression_collaborative_training import MODE, load_semantics, parser, runtime
from full_impression_contract import check_deadline
from full_impression_evaluation import TRAINING_RESULT_SHA, load_gate
from full_impression_training import MANIFEST_SHA, PROOF_SHA, batch_inputs, batches, load_contract
from full_impression_verification import check_scores, metrics
from jev_content_ranker import digest


def run(args):
    runtime()
    require(0 < args.max_seconds <= 1500 and not args.proof.exists(), "Bounded unique independent proof")
    deadline = time.monotonic() + args.max_seconds
    previous = load_gate(args.training_artifacts, args.training_proof, PROOF_SHA256, TRAINING_RESULT_SHA)
    _, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    features, semantics = load_semantics(args, previous)
    result_path = args.artifacts / "result.json"
    result_sha = digest(result_path)
    result = json.loads(result_path.read_bytes())
    require(result["source_revision"] == args.source_revision and result["semantics"] == semantics
            and result["prior_training_proof_sha256"] == PROOF_SHA256
            and not result["dev_or_test_contents_read"] and not result["backbone_fitting"]
            and result["gpu_allocations"] == 0 and not result["competition_test_goal_achieved"],
            "Exact immutable training-only collaborative result required")
    days = arrays["training_days.npy"]
    fitting, validation = np.flatnonzero(days < days.max()), np.flatnonzero(days == days.max())
    require((len(fitting), len(validation)) == (1801231, 431517), "Original complete chronological cohorts")
    selected = result["selection"]
    verify_epochs(selected["fitting"], fitting, arrays, (1801231, 66107268, 2712612), deadline, MODE)
    head = args.artifacts / "collaborative.selection.pt"
    require(digest(head) == selected["fitting"]["head_sha256"], "Frozen selection checkpoint")
    replay = CollaborativeReplay(features, torch.load(head, map_location="cpu", weights_only=True))
    scores_path = args.artifacts / "validation.collaborative.scores.f64"
    metric_path = scores_path.with_suffix(".metrics.npy")
    require(scores_path.stat().st_size == 17400106 * 8 and digest(scores_path) == selected["scores_sha256"]
            and digest(metric_path) == selected["per_impression_metrics_sha256"], "All immutable score/metric bytes")
    scores = np.memmap(scores_path, dtype=np.float64, mode="r", shape=(17400106,))
    recorded = np.load(metric_path, mmap_mode="r", allow_pickle=False)
    require(recorded.shape == (431517, 4), "All original validation-impression metric cells")
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
        if row % 50000 < len(batch):
            print(json.dumps({"stage": "independent_collaborative_replay", "impressions": row}), flush=True)
    require((row, position, positives) == (431517, 17400106, 671044)
            and selected["impressions"] == row and selected["exposures"] == position
            and np.allclose(rebuilt, recorded, atol=1e-12, rtol=0), "All source labels and metric cells must replay")
    aggregate = dict(zip(("auc", "mrr", "ndcg5", "ndcg10"), rebuilt.mean(axis=0).tolist()))
    require(all(abs(aggregate[key] - selected["metrics"][key]) <= 1e-12 for key in aggregate),
            "Every aggregate metric must replay")
    baseline = previous["recipes"]["candidate_attention"]["metrics"]
    gain = aggregate["auc"] - baseline["auc"]
    opens = gain >= .005 and all(aggregate[key] >= baseline[key] for key in ("mrr", "ndcg5", "ndcg10"))
    require(result["unchanged_baseline_metrics"] == baseline
            and abs(gain - result["training_only_auc_gain"]) <= 1e-12
            and result["opens_dev_gate"] == opens, "Independent stronger-original-attention gate")
    if opens:
        require(result["stage"] == "collaborative_full_training_fitted"
                and result["final_refit_holdout_impressions"] == 0, "Fresh complete final fit required")
        verify_epochs(result["final_refit"], np.arange(len(days)), arrays,
                      (2232748, 83507374, 3383656), deadline, MODE)
        final = args.artifacts / "selected-full.pt"
        require(digest(final) == result["final_refit"]["head_sha256"], "Frozen final checkpoint")
        CollaborativeReplay(features, torch.load(final, map_location="cpu", weights_only=True))
    else:
        require(result["stage"] == "collaborative_training_gate_evaluated" and "final_refit" not in result
                and not (args.artifacts / "selected-full.pt").exists(), "Closed gate cannot claim a final fit")
    _, final_semantics = load_semantics(args, previous)
    require(final_semantics == semantics and digest(result_path) == result_sha
            and digest(head) == selected["fitting"]["head_sha256"]
            and digest(scores_path) == selected["scores_sha256"]
            and digest(metric_path) == selected["per_impression_metrics_sha256"], "Immutable bytes before/after")
    check_deadline(deadline)
    proof = {
        "source_revision": args.source_revision, "result_sha256": result_sha,
        "semantics": semantics, "metrics": aggregate, "maximum_score_replay_error": maximum_error,
        "coverage": {"impressions": row, "exposures": position, "positives": positives},
        "training_only_auc_gain": gain, "opens_dev_gate": opens,
        "full_training_epoch_source_and_coverage_verified": opens,
        "selection_epoch_source_and_coverage_verified": True,
        "optimizer_training_independently_reproduced": False,
        "basis_proof_reused_not_recomputed": True, "every_new_model_score_independently_replayed": True,
        "metric_tolerance": 1e-12, "score_tolerance": {"atol": 1e-4, "rtol": 1e-5},
        "metric_scope": "last_official_training_day_only;not_official_dev_or_test",
        "gpu_allocations": 0, "dev_or_test_contents_read": False, "competition_test_goal_achieved": False}
    with args.proof.open("x", encoding="utf-8") as handle:
        json.dump(proof, handle, indent=2, allow_nan=False)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    arguments = parser(__doc__)
    arguments.add_argument("--artifacts", type=Path, required=True)
    arguments.add_argument("--proof", type=Path, required=True)
    arguments.add_argument("--max-seconds", type=int, default=1500)
    torch.set_num_threads(4)
    run(arguments.parse_args())
