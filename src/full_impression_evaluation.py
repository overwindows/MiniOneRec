"""Frozen all-training head on every official MIND-large dev candidate; no fitting."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from full_impression_contract import check_deadline, read_news
from full_impression_ranker import FrozenArticleRanker
from full_impression_training import (
    MANIFEST_SHA, PROOF_SHA, ControlPredictor, ModelPredictor, article_features, evaluate, load_contract, popularity,
)
from jev_content_ranker import digest, write_json


TRAINING_REVISION = "ce4a8e6939eccaecefc37603e72dfc90fa0fcfdd"
TRAINING_RESULT_SHA = "5c5169a84b3a092762694a32b12b6c927dd99780d660a4db3cde99b0240d2432"
FINAL_HEAD_SHA = "c9523970501838fa165e7995400608cbc34527c5ed37a82035f81acd7bc35c6d"
ARMS = ("candidate_attention", "popularity", "semantic_mean", "frozen_article_margin")


def load_gate(artifacts, proof_path, proof_sha, result_sha):
    if digest(proof_path) != proof_sha or digest(artifacts / "result.json") != result_sha:
        raise ValueError("Frozen independent training proof/result changed")
    proof = json.loads(proof_path.read_bytes())
    result = json.loads((artifacts / "result.json").read_bytes())
    if (result_sha != TRAINING_RESULT_SHA or proof["result_sha256"] != result_sha
            or proof["source_revision"] != TRAINING_REVISION or result["source_revision"] != TRAINING_REVISION
            or not proof["opens_dev_gate"] or not result["opens_dev_gate"]
            or not proof["original16_bit_exact_parity_independently_verified"]
            or not proof["full_training_epoch_source_and_coverage_verified"]
            or proof["selected_recipe"] != "candidate_attention"
            or result["selected_recipe"] != "candidate_attention"
            or result["stage"] != "full_training_fitted"
            or result["final_refit_holdout_impressions"] != 0
            or result["final_refit"]["head_sha256"] != FINAL_HEAD_SHA
            or digest(artifacts / "selected-full.pt") != FINAL_HEAD_SHA
            or result["dev_or_test_contents_read"] or proof["official_dev_allocations"] != 0):
        raise ValueError("Independently verified unchanged all-training head is required before reading dev")
    epochs = result["final_refit"]["epochs"]
    if len(epochs) != 3 or not all(
            item["impressions"] == 2232748 and item["exposures"] == 83507374
            and item["positives"] == 3383656 for item in epochs):
        raise ValueError("Final fitting must retain every training impression/exposure/click in all3 epochs")
    return result


def behavior(fields):
    if len(fields) != 5 or not fields[0]:
        raise ValueError("Every official-dev impression requires its complete five-field source record")
    tokens = [token.rsplit("-", 1) for token in fields[4].split()]
    if not tokens or any(len(token) != 2 or token[1] not in ("0", "1") for token in tokens):
        raise ValueError("Every official-dev candidate requires an explicit binary label")
    names = [name for name, _ in tokens]
    labels = np.asarray([int(label) for _, label in tokens], dtype=np.uint8)
    if not 0 < int(labels.sum()) < len(labels):
        raise ValueError("Never silently drop a single-class official-dev impression")
    return fields[3].split()[-30:], names, labels


def source_signature(history, candidates):
    raw = json.dumps([history, sorted(candidates)], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).digest()


def small_signatures(path, deadline):
    signatures = set()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for rows, line in enumerate(handle, 1):
            if rows % 4096 == 0:
                check_deadline(deadline)
            history, candidates, _ = behavior(line.rstrip("\r\n").split("\t"))
            signatures.add(source_signature(history, candidates))
    if not rows:
        raise ValueError("Complete MIND-small dev source is required for conservative overlap disclosure")
    return signatures, rows


def prepare_dev(dev, small_dev, train_contract, output, deadline):
    if output.exists():
        raise FileExistsError("Never overwrite an official-dev contract")
    check_deadline(deadline)
    sources = {"dev_news": dev / "news.tsv", "dev_behaviors": dev / "behaviors.tsv",
               "small_dev_behaviors": small_dev / "behaviors.tsv"}
    hashes = {name: digest(path) for name, path in sources.items()}
    train_ids = json.loads((train_contract / "news.ids.json").read_bytes())
    train_rows = [json.loads(line) for line in
                  (train_contract / "news.metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    if train_ids != [row["news_id"] for row in train_rows] or len(train_ids) != len(set(train_ids)):
        raise ValueError("Exact training article index/metadata contract required")
    train_news = dict(zip(train_ids, train_rows))
    dev_news = read_news(dev / "news.tsv")
    for name in set(train_news) & set(dev_news):
        if train_news[name] != dev_news[name]:
            raise ValueError("A reused news identity has different source metadata: " + name)
    new_ids = sorted(set(dev_news) - set(train_news))
    identifiers = train_ids + new_ids
    if len(identifiers) >= np.iinfo(np.int32).max:
        raise ValueError("Article union exceeds the lossless index representation")
    lookup = {name: index + 1 for index, name in enumerate(identifiers)}
    signatures, small_rows = small_signatures(small_dev / "behaviors.tsv", deadline)
    impressions = exposures = positives = overlap = 0
    seen = set()
    with (dev / "behaviors.tsv").open(encoding="utf-8") as handle:
        for impressions, line in enumerate(handle, 1):
            if impressions % 4096 == 0:
                check_deadline(deadline)
            fields = line.rstrip("\r\n").split("\t")
            history, candidates, labels = behavior(fields)
            if fields[0] in seen:
                raise ValueError("Official-dev impression identities must be unique")
            seen.add(fields[0])
            if any(name not in lookup for name in history + candidates):
                raise ValueError("Every official-dev candidate/history article must resolve in the frozen union")
            exposures += len(candidates)
            positives += int(labels.sum())
            overlap += source_signature(history, candidates) in signatures
    if not 0 < impressions < exposures:
        raise ValueError("The complete official-dev source must be nonempty")
    output.mkdir()
    new_contract = output / "new_articles"
    new_contract.mkdir()
    write_json(output / "news.ids.json", identifiers)
    write_json(new_contract / "news.ids.json", new_ids)
    with (new_contract / "news.metadata.jsonl").open("w", encoding="utf-8") as handle:
        for name in new_ids:
            handle.write(json.dumps(dev_news[name], ensure_ascii=False, allow_nan=False) + "\n")
    specs = {"candidate_news.npy": (np.int32, (exposures,)),
             "candidate_labels.npy": (np.uint8, (exposures,)),
             "candidate_offsets.npy": (np.uint64, (impressions + 1,)),
             "histories.npy": (np.int32, (impressions, 30)),
             "small_dev_signature_overlap.npy": (np.bool_, (impressions,))}
    arrays = {name: np.lib.format.open_memmap(output / name, mode="w+", dtype=dtype, shape=shape)
              for name, (dtype, shape) in specs.items()}
    arrays["histories.npy"][:] = 0
    arrays["candidate_offsets.npy"][0] = 0
    position = encoded = encoded_positives = encoded_overlap = 0
    with (dev / "behaviors.tsv").open(encoding="utf-8") as handle, (
            output / "impression_ids.txt").open("w", encoding="utf-8") as ids:
        for index, line in enumerate(handle):
            if index % 4096 == 0:
                check_deadline(deadline)
            fields = line.rstrip("\r\n").split("\t")
            history, candidates, labels = behavior(fields)
            end = position + len(candidates)
            if index >= impressions or end > exposures:
                raise ValueError("Official-dev source counts changed between encoding passes")
            arrays["candidate_news.npy"][position:end] = [lookup[name] for name in candidates]
            arrays["candidate_labels.npy"][position:end] = labels
            arrays["candidate_offsets.npy"][index + 1] = end
            if history:
                arrays["histories.npy"][index, -len(history):] = [lookup[name] for name in history]
            matched = source_signature(history, candidates) in signatures
            arrays["small_dev_signature_overlap.npy"][index] = matched
            encoded_overlap += matched
            encoded_positives += int(labels.sum())
            ids.write(fields[0] + "\n")
            position, encoded = end, index + 1
    if (encoded != impressions or position != exposures
            or encoded_positives != positives or encoded_overlap != overlap):
        raise ValueError("Every official-dev impression/candidate must be retained")
    for array in arrays.values():
        array.flush()
    arrays.clear()
    del array
    for name, path in sources.items():
        if digest(path) != hashes[name]:
            raise ValueError("A raw official-dev/overlap source changed during encoding")
    files = [*specs, "news.ids.json", "impression_ids.txt",
             "new_articles/news.ids.json", "new_articles/news.metadata.jsonl"]
    report = {
        "stage": "complete_official_dev_encoded", "impressions": impressions, "exposures": exposures,
        "positives": positives, "negatives": exposures - positives,
        "training_articles_reused": len(train_ids), "new_dev_articles": len(new_ids),
        "dev_news": len(dev_news), "union_articles": len(identifiers),
        "history_limit": 30, "history_padding": "left_zero", "candidate_sampling": "none",
        "small_dev_source_impressions": small_rows, "small_dev_signature_overlap_impressions": overlap,
        "signature_definition": "last30_history_news_ids_in_order;sorted_candidate_news_multiset;no_labels_ids_or_time",
        "all_small_dev_is_conservatively_treated_as_previously_exposed": True,
        "signature_unmatched_does_not_guarantee_never_inspected": True,
        "source_sha256": hashes, "sha256": {name: digest(output / name) for name in files},
    }
    write_json(output / "manifest.json", report)
    return report, {name: np.load(output / name, mmap_mode="r", allow_pickle=False) for name in specs}


def all_training_prior(train_arrays, train_news, union_news, deadline):
    if union_news < train_news:
        raise ValueError("The evaluation article union cannot discard training articles")
    original = popularity(np.arange(len(train_arrays["histories.npy"])), train_arrays, train_news, deadline)
    positives = int(np.sum(train_arrays["candidate_labels.npy"], dtype=np.int64))
    probability = positives / len(train_arrays["candidate_labels.npy"])
    result = np.full(union_news + 1, np.log(probability / (1 - probability)), dtype=np.float32)
    result[:len(original)] = original
    return result


def frozen_articles(directory, articles):
    features = np.load(directory / "article.features.npy", mmap_mode="r", allow_pickle=False)
    margins = np.load(directory / "article.margins.npy", mmap_mode="r", allow_pickle=False)
    if (features.shape != (articles + 1, 2560) or margins.shape != (articles + 1,)
            or features.dtype != np.float32 or margins.dtype != np.float32
            or not np.isfinite(features).all() or not np.isfinite(margins).all()
            or features[0].any() or margins[0] != 0
            or np.any(~np.any(features[1:] != 0, axis=1))):
        raise ValueError("Every article needs a finite nonzero FP32 feature and aligned margin with zero padding")
    return features, margins


def run(args):
    if not 0 < args.max_seconds <= 21600 or not re.fullmatch("[0-9a-f]{40}", args.source_revision):
        raise ValueError("An immutable source and at most six worker hours are required")
    if args.output.exists():
        raise FileExistsError("Preserve unique official-dev output;no silent retries")
    deadline = time.monotonic() + args.max_seconds
    training = load_gate(args.training_artifacts, args.training_proof / "verification.json",
                         args.training_proof_sha256, args.training_result_sha256)
    numeric_runtime = {
        "numpy": np.__version__,
        "cpu_dispatch": {name: bool(np.core._multiarray_umath.__cpu_features__[name])
                         for name in ("AVX2", "AVX512F", "AVX512_SKX", "AVX512_ICL")},
        "tie_order": "numpy1.26.4_default_argsort_reversed;non_AVX512_dispatch",
    }
    if (numeric_runtime["numpy"] != "1.26.4" or numeric_runtime["cpu_dispatch"] != {
            "AVX2": True, "AVX512F": False, "AVX512_SKX": False, "AVX512_ICL": False}):
        raise ValueError("Official-dev metrics require the preregistered reproducible NumPy dispatch")
    manifest, train_arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    if not torch.cuda.is_available():
        raise RuntimeError("The separately registered official-dev worker requires its single bounded GPU")
    feature = training["feature_extraction"]
    for name, key in (("article.features.npy", "features_sha256"),
                      ("article.margins.npy", "margins_sha256"),
                      ("model.first-observed-sha256.json", "model_fingerprints_sha256")):
        if digest(args.training_artifacts / name) != feature[key]:
            raise ValueError("Frozen all-training feature/model bytes changed")
    args.output.mkdir()
    dev, arrays = prepare_dev(args.dev_data, args.small_data / "dev", args.contract,
                              args.output / "contract", deadline)
    new_output = args.output / "new_articles"
    new_output.mkdir()
    new_feature = article_features(
        args.output / "contract" / "new_articles", args.model_path, args.small_data,
        args.parity_cache, args.parity_sha256, new_output, min(deadline, time.monotonic() + 7200),
    )
    if json.loads((new_output / "model.first-observed-sha256.json").read_bytes()) != json.loads((
            args.training_artifacts / "model.first-observed-sha256.json").read_bytes()):
        raise ValueError("Training and official-dev frozen model fingerprints must be identical")
    union = np.lib.format.open_memmap(args.output / "article.features.npy", mode="w+",
                                     dtype=np.float32, shape=(dev["union_articles"] + 1, 2560))
    old, old_margins = frozen_articles(args.training_artifacts, manifest["news"])
    new, new_margins = frozen_articles(new_output, dev["new_dev_articles"])
    union[:len(old)] = old
    union[len(old):] = new[1:]
    union.flush()
    if (not np.array_equal(union[:len(old)].view(np.uint32), old.view(np.uint32))
            or not np.array_equal(union[len(old):].view(np.uint32), new[1:].view(np.uint32))):
        raise ValueError("Union features must preserve every original and appended article bit")
    margins = np.concatenate((old_margins, new_margins[1:]))
    np.save(args.output / "article.margins.npy", margins)
    del old, new, old_margins, new_margins, union
    gc.collect()
    features = F.normalize(torch.from_numpy(np.load(args.output / "article.features.npy")).to("cuda:0"), dim=-1)
    model = FrozenArticleRanker(features, "candidate_attention").to(features.device)
    model.load_state_dict(torch.load(args.training_artifacts / "selected-full.pt",
                                    map_location="cpu", weights_only=True), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prior = all_training_prior(train_arrays, manifest["news"], dev["union_articles"], deadline)
    predictors = {
        "candidate_attention": ModelPredictor(model),
        "popularity": ControlPredictor(features, "popularity", ctr=torch.from_numpy(prior).to(features.device)),
        "semantic_mean": ControlPredictor(features, "semantic_mean"),
        "frozen_article_margin": ControlPredictor(
            features, "frozen_article_margin", margins=torch.from_numpy(margins).to(features.device)),
    }
    indices = np.arange(dev["impressions"])
    overlap = arrays["small_dev_signature_overlap.npy"]
    result = {
        "stage": "official_dev_scored_pending_independent_verification", "source_revision": args.source_revision,
        "training_source_revision": TRAINING_REVISION, "training_result_sha256": args.training_result_sha256,
        "training_independent_proof_sha256": args.training_proof_sha256, "final_head_sha256": FINAL_HEAD_SHA,
        "metric_scope": "complete_official_MIND_large_dev;not_competition_test",
        "complete_dev": dev, "new_article_features": new_feature, "model_fits": 0,
        "numeric_runtime": numeric_runtime,
        "head_or_recipe_retuning": False, "gpu_allocations": 1, "competition_test_goal_achieved": False,
        "matched_retained_hybrid_full_dev_comparison_available": False,
        "article_features_sha256": digest(args.output / "article.features.npy"),
        "article_margins_sha256": digest(args.output / "article.margins.npy"), "arms": {},
    }
    for name in ARMS:
        scores = args.output / f"official.{name}.scores.f64"
        report = evaluate(indices, arrays, predictors[name], scores, deadline)
        metrics = np.load(scores.with_suffix(".metrics.npy"), allow_pickle=False)
        unmatched = ~overlap
        report["small_dev_source_signature_unmatched"] = {
            "impressions": int(unmatched.sum()),
            "metrics": (dict(zip(("auc", "mrr", "ndcg5", "ndcg10"),
                                 metrics[unmatched].mean(axis=0).tolist())) if unmatched.any() else None),
            "scope": "prespecified_secondary;not_guaranteed_pristine",
        }
        result["arms"][name] = report
        print(json.dumps({"stage": "complete_official_dev_arm_scored", "arm": name, **report}), flush=True)
    check_deadline(deadline)
    for name, path in (("dev_news", args.dev_data / "news.tsv"),
                       ("dev_behaviors", args.dev_data / "behaviors.tsv"),
                       ("small_dev_behaviors", args.small_data / "dev" / "behaviors.tsv")):
        if digest(path) != dev["source_sha256"][name]:
            raise ValueError("Official-dev/overlap raw source changed during scoring")
    if digest(args.training_artifacts / "selected-full.pt") != FINAL_HEAD_SHA:
        raise ValueError("Frozen all-training head changed during official-dev scoring")
    load_gate(args.training_artifacts, args.training_proof / "verification.json",
              args.training_proof_sha256, args.training_result_sha256)
    for name, key in (("article.features.npy", "features_sha256"),
                      ("article.margins.npy", "margins_sha256"),
                      ("model.first-observed-sha256.json", "model_fingerprints_sha256")):
        if digest(args.training_artifacts / name) != feature[key]:
            raise ValueError("Frozen training article/model bytes changed during scoring")
    write_json(args.output / "result.pending.json", result)
    (args.output / "result.pending.json").rename(args.output / "result.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "training-artifacts", "training-proof", "dev-data", "small-data",
                 "parity-cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("training-proof-sha256", "training-result-sha256", "parity-sha256",
                 "source-revision", "model-path"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-seconds", type=int, default=21600)
    run(parser.parse_args())
