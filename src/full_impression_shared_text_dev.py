"""Conditional frozen shared-text evaluation on complete previously used official dev."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy import sparse
import torch

from full_impression_centered_training import AGGREGATE_END
from full_impression_centered_verification import require
from full_impression_collaborative_training import runtime
from full_impression_contract import check_deadline
from full_impression_dev_verification import verify_source
from full_impression_shared_text import MODE, SharedTextRanker, SharedTextReplay
from full_impression_training import ModelPredictor, batch_inputs, batches, evaluate
from full_impression_verification import check_scores, metrics
from jev_content_ranker import digest


TRAINING_SOURCE = "d58ed784380034227c8fc385a1e106f7bd37d527"
PRIOR_DEV_PROOF = "9d0aa61ea8adefb637fc83f9ed481d32ac2f44c067add4e6672e3a246f425153"
PRIOR_DEV_MANIFEST = "9b138c236be99a0f09f6ba57811f118839be681c91da90f9c060769160fa5ff3"


def load_shared_gate(artifacts, proof_path, proof_sha, result_sha):
    require(digest(proof_path) == proof_sha and digest(artifacts / "result.json") == result_sha,
            "Frozen completed training proof/result required BEFORE any dev access")
    proof, result = [json.loads(path.read_bytes()) for path in (proof_path, artifacts / "result.json")]
    require(proof["result_sha256"] == result_sha
            and proof["source_revision"] == result["source_revision"] == TRAINING_SOURCE
            and proof["opens_dev_gate"] and result["opens_dev_gate"]
            and proof["full_training_epoch_source_and_coverage_verified"]
            and proof["selection_epoch_source_and_coverage_verified"]
            and proof["lexical_source_and_initialization_all_bits_reproduced"]
            and proof["every_new_model_score_independently_replayed"]
            and result["stage"] == "shared_text_full_training_fitted"
            and result["final_refit_holdout_impressions"] == 0
            and not proof["dev_or_test_contents_read"] and not result["dev_or_test_contents_read"]
            and proof["gpu_allocations"] == result["gpu_allocations"] == 0,
            "Accepted independent gate and fresh all-training fit required BEFORE dev")
    epochs = result["final_refit"]["epochs"]
    require(len(epochs) == 3 and result["final_refit"]["mode"] == MODE
            and all((row["impressions"], row["exposures"], row["positives"]) ==
                    (2232748, 83507374, 3383656) for row in epochs)
            and digest(artifacts / "selected-full.pt") == result["final_refit"]["head_sha256"],
            "Every official-training impression/candidate/click in every final epoch")
    for name in ("lexical.joblib", "article.terms.npz", "words.initial.npy"):
        require(digest(artifacts / name) == result["lexical"][name] == proof["lexical"][name],
                "Frozen training-only lexical bytes")
    return result


def capped_titles(vectorizer, titles):
    values = vectorizer.transform(titles).tocsr()
    values.sort_indices()
    data, columns, offsets = [], [], [0]
    for row in range(values.shape[0]):
        begin, end = values.indptr[row:row + 2]
        selected = np.argsort(-values.data[begin:end], kind="stable")[:32]
        selected.sort()
        data.extend(values.data[begin:end][selected])
        columns.extend(values.indices[begin:end][selected])
        offsets.append(len(data))
    return sparse.csr_matrix((np.asarray(data, dtype=np.float32), np.asarray(columns, dtype=np.int32),
                              np.asarray(offsets, dtype=np.int64)), shape=values.shape)


def run(args):
    runtime()
    require(0 < args.max_seconds <= 1200, "One bounded CPU-only evaluation")
    require(datetime.now(timezone.utc) + timedelta(seconds=1620) <
            datetime.fromisoformat(AGGREGATE_END), "1500s OS cap plus margin inside ORIGINAL aggregate")
    require(not args.output.exists(), "One unique frozen evaluation;no retry or overwrite")
    deadline = time.monotonic() + args.max_seconds
    result = load_shared_gate(args.training_artifacts, args.training_proof,
                              args.training_proof_sha256, args.training_result_sha256)
    require(digest(args.prior_dev_proof) == PRIOR_DEV_PROOF, "Previously accepted dev source/metric proof")
    prior = json.loads(args.prior_dev_proof.read_bytes())
    require(digest(args.encoded / "manifest.json") == PRIOR_DEV_MANIFEST,
            "Exact previously verified full official-dev contract")
    source = verify_source(args.dev, args.small_dev, args.train_contract, args.encoded, deadline)
    require(source == prior["complete_raw_source_replay"] and
            (source["impressions"], source["exposures"], source["positives"]) ==
            (376471, 14085557, 574845), "Every original official-dev source row/target/history")
    encoder = joblib.load(args.training_artifacts / "lexical.joblib")
    training_terms = sparse.load_npz(args.training_artifacts / "article.terms.npz")
    new_records = [json.loads(line) for line in (args.encoded / "new_articles" / "news.metadata.jsonl")
                   .read_text(encoding="utf-8").splitlines()]
    new_terms = capped_titles(encoder.vectorizer, [row["title"] for row in new_records])
    for index, row in enumerate(new_records):
        check_deadline(deadline)
        single = encoder.vectorizer.transform([row["title"]]).tocsr()
        ranked = sorted(zip(single.indices.tolist(), single.data.tolist()),
                        key=lambda item: (-item[1], item[0]))[:32]
        ranked.sort()
        start, end = new_terms.indptr[index:index + 2]
        require(np.array_equal(new_terms.indices[start:end], [column for column, _ in ranked])
                and np.array_equal(new_terms.data[start:end].view(np.uint32),
                                   np.asarray([value for _, value in ranked], dtype=np.float32).view(np.uint32)),
                "Every new dev title-term bit independently checked")
    terms = sparse.vstack((training_terms, new_terms), format="csr")
    require(training_terms.shape[0] == 101528 and len(new_records) == 2624
            and terms.shape[0] == 104152, "Complete training-prefix/new-article union")
    initial = np.load(args.training_artifacts / "words.initial.npy", allow_pickle=False)
    state = torch.load(args.training_artifacts / "selected-full.pt", map_location="cpu", weights_only=True)
    model = SharedTextRanker(torch.zeros((terms.shape[0], 1)), MODE, terms, initial)
    model.load_state_dict(state, strict=True)
    arrays = {name: np.load(args.encoded / name, mmap_mode="r", allow_pickle=False) for name in
              ("candidate_news.npy", "candidate_labels.npy", "candidate_offsets.npy", "histories.npy")}
    args.output.mkdir()
    scores_path = args.output / "official-dev.shared-text.scores.f64"
    observed = evaluate(np.arange(376471), arrays, ModelPredictor(model), scores_path, deadline)
    del model
    replay = SharedTextReplay(terms, state)
    scores = np.memmap(scores_path, dtype=np.float64, mode="r", shape=(14085557,))
    recorded = np.load(scores_path.with_suffix(".metrics.npy"), mmap_mode="r", allow_pickle=False)
    require(recorded.shape == (376471, 4), "Every complete official-dev metric cell")
    rebuilt = np.empty((376471, 4), dtype=np.float64)
    row = position = positives = 0
    maximum_error = 0.
    for batch in batches(np.arange(376471), arrays):
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
    aggregate = dict(zip(("auc", "mrr", "ndcg5", "ndcg10"), rebuilt.mean(axis=0).tolist()))
    require((row, position, positives) == (376471, 14085557, 574845)
            and np.allclose(rebuilt, recorded, atol=1e-12, rtol=0)
            and all(abs(aggregate[key] - observed["metrics"][key]) <= 1e-12 for key in aggregate),
            "Every official-dev raw-score metric cell and aggregate must replay")
    require(digest(scores_path) == observed["scores_sha256"]
            and digest(scores_path.with_suffix(".metrics.npy")) ==
            observed["per_impression_metrics_sha256"], "Scores/metrics unchanged")
    load_shared_gate(args.training_artifacts, args.training_proof,
                     args.training_proof_sha256, args.training_result_sha256)
    check_deadline(deadline)
    proof = {
        "stage": "official_dev_independently_verified", "training_source_revision": TRAINING_SOURCE,
        "evaluation_source_revision": args.source_revision,
        "training_proof_sha256": args.training_proof_sha256,
        "training_result_sha256": args.training_result_sha256,
        "final_head_sha256": result["final_refit"]["head_sha256"],
        "metrics": aggregate, "coverage": source, "maximum_score_replay_error": maximum_error,
        "scores_sha256": observed["scores_sha256"],
        "per_impression_metrics_sha256": observed["per_impression_metrics_sha256"],
        "unchanged_prior_attention_metrics": prior["metrics"]["candidate_attention"],
        "all_new_title_term_bits_verified": True, "model_fits": 0, "gpu_allocations": 0,
        "scope": "complete_previously_used_official_MIND_large_dev;not_pristine_or_competition_test",
        "competition_test_goal_achieved": False, "dev_auc_080_reached": aggregate["auc"] >= .8,
        "metric_tolerance": 1e-12, "score_tolerance": {"atol": 1e-4, "rtol": 1e-5}}
    with (args.output / "independent-verification.json").open("x", encoding="utf-8") as handle:
        json.dump(proof, handle, indent=2, allow_nan=False)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-artifacts", "training-proof", "prior-dev-proof", "train-contract",
                 "dev", "small-dev", "encoded", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("training-proof-sha256", "training-result-sha256", "source-revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-seconds", type=int, default=1200)
    torch.set_num_threads(4)
    run(parser.parse_args())
