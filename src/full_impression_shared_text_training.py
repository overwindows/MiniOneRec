"""One bounded shared-text recipe with train-only selection and complete final refit."""

from datetime import datetime, timedelta, timezone
import gc
import json
from pathlib import Path
import re
import time

import joblib
import numpy as np
from scipy import sparse
import torch

from full_impression_centered_training import AGGREGATE_END, PROOF_SHA256, epoch_coverage
from full_impression_collaborative_training import runtime
from full_impression_contract import check_deadline
from full_impression_evaluation import TRAINING_RESULT_SHA, load_gate
from full_impression_shared_text import MODE, SharedTextRanker
from full_impression_training import MANIFEST_SHA, PROOF_SHA, ModelPredictor, evaluate, fit, load_contract
from jev_content_ranker import Content, digest, write_json


def build_lexical(contract, deadline, *, expected_articles=101527):
    records = [json.loads(line) for line in (contract / "news.metadata.jsonl").read_text(
        encoding="utf-8").splitlines()]
    names = json.loads((contract / "news.ids.json").read_bytes())
    if [row["news_id"] for row in records] != names:
        raise ValueError("Every training article identity/order must match the complete verified source")
    news = {row["news_id"]: {"text": row["title"], "category": row["category"],
                            "subcategory": row["subcategory"]} for row in records}
    check_deadline(deadline)
    encoder = Content.fit(news, components=64)
    terms = encoder.vectorizer.transform([row["text"] for row in news.values()]).tocsr()
    terms.sort_indices()
    keep = np.ones(terms.nnz, dtype=bool)
    lengths = np.empty(len(records), dtype=np.int64)
    for row in range(len(records)):
        if row % 2048 == 0:
            check_deadline(deadline)
        begin, end = terms.indptr[row:row + 2]
        lengths[row] = min(end - begin, 32)
        if end - begin > 32:
            chosen = np.argsort(-terms.data[begin:end], kind="stable")[:32]
            keep[begin:end] = False
            keep[begin + chosen] = True
    capped = sparse.csr_matrix((terms.data[keep], terms.indices[keep],
                                np.r_[0, np.cumsum(lengths)]), shape=terms.shape)
    terms = sparse.vstack((sparse.csr_matrix((1, terms.shape[1]), dtype=np.float32), capped), format="csr")
    initial = encoder.svd.components_.T.astype(np.float32).copy()
    if terms.shape[0] != expected_articles + 1 or initial.shape != (terms.shape[1], 64):
        raise ValueError("Complete large-training title matrix and exactly64 shared initial dimensions required")
    check_deadline(deadline)
    return encoder, terms, initial


def run(args):
    runtime()
    if not 0 < args.max_seconds <= 8100 or not re.fullmatch("[0-9a-f]{40}", args.source_revision):
        raise ValueError("Exact immutable source and at most8100 training seconds")
    if datetime.now(timezone.utc) + timedelta(seconds=10500) > datetime.fromisoformat(AGGREGATE_END):
        raise TimeoutError("Distinct9900s CPU campaign plus margin must fit ORIGINAL aggregate")
    if args.output.exists():
        raise FileExistsError("Unique shared-text campaign;no retry or overwrite")
    deadline = time.monotonic() + args.max_seconds
    previous = load_gate(args.training_artifacts, args.training_proof, PROOF_SHA256, TRAINING_RESULT_SHA)
    manifest, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    encoder, terms, initial = build_lexical(args.contract, deadline)
    args.output.mkdir()
    joblib.dump(encoder, args.output / "lexical.joblib")
    sparse.save_npz(args.output / "article.terms.npz", terms)
    np.save(args.output / "words.initial.npy", initial)
    lexical = {name: digest(args.output / name) for name in
               ("lexical.joblib", "article.terms.npz", "words.initial.npy")}
    lexical.update(articles=101527, vocabulary=terms.shape[1], nonzero_terms=int(terms.nnz),
                   text_scope="all_official_training_titles_only;no_labels_or_dev;top32_TFIDF_terms_per_article")
    print(json.dumps({"stage": "training_only_shared_lexical_encoding_completed", **lexical}), flush=True)
    features = torch.zeros((terms.shape[0], 1))
    factory = lambda f, mode: SharedTextRanker(f, mode, terms, initial)
    days = arrays["training_days.npy"]
    fitting, validation = np.flatnonzero(days < days.max()), np.flatnonzero(days == days.max())
    if (manifest["impressions"], len(fitting), len(validation)) != (2232748, 1801231, 431517):
        raise ValueError("Complete unchanged chronological training-only cohorts")
    model, fitted = fit(features, fitting, arrays, MODE, args.output / "shared-text.selection.pt",
                        deadline, model_factory=factory)
    if not epoch_coverage(fitted, 1801231, 66107268, 2712612):
        raise ValueError("Every earlier impression/candidate/positive required in every epoch")
    selected = {"fitting": fitted, **evaluate(validation, arrays, ModelPredictor(model),
                args.output / "validation.shared-text.scores.f64", deadline)}
    baseline = previous["recipes"]["candidate_attention"]["metrics"]
    gain = selected["metrics"]["auc"] - baseline["auc"]
    opens = gain >= .005 and all(selected["metrics"][key] >= baseline[key]
                                for key in ("mrr", "ndcg5", "ndcg10"))
    result = {
        "stage": "shared_text_training_gate_evaluated", "source_revision": args.source_revision,
        "selection": selected, "lexical": lexical, "prior_training_proof_sha256": PROOF_SHA256,
        "unchanged_baseline_metrics": baseline, "training_only_auc_gain": gain, "opens_dev_gate": opens,
        "metric_scope": "last_official_training_day_only;not_official_dev_or_test",
        "recipe": "title_TFIDF32768_top32;training_only_LSA64_initial_shared_trainable_term_vectors;"
                  "four_learned_interest_queries;masked_recency_attention;logmeanexp_10cosine;"
                  "NO_item_ID_parameters_or_biases;three_epochs;AdamW.001;WD.0001;clip1;seeds42_43_44;"
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
                           deadline, model_factory=factory)
        if not epoch_coverage(refit, 2232748, 83507374, 3383656):
            raise ValueError("Fresh final refit must retain ALL official-training impressions and exposures")
        del final
        result.update(stage="shared_text_full_training_fitted", final_refit=refit,
                      final_refit_holdout_impressions=0, training_impressions=2232748,
                      training_exposures=83507374)
    for name in ("lexical.joblib", "article.terms.npz", "words.initial.npy"):
        if digest(args.output / name) != lexical[name]:
            raise ValueError("Frozen lexical bytes changed")
    check_deadline(deadline)
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


def parser(description):
    import argparse
    result = argparse.ArgumentParser(description=description)
    for name in ("contract", "training-artifacts", "training-proof"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--source-revision", required=True)
    return result


if __name__ == "__main__":
    arguments = parser(__doc__)
    arguments.add_argument("--output", type=Path, required=True)
    arguments.add_argument("--max-seconds", type=int, default=8100)
    torch.set_num_threads(4)
    run(arguments.parse_args())
