"""Uncapped official-train readout fitting in bounded reusable feature shards."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from behavior_jev_readout import behavior_features, build_counts, save_counts
from data import MINDPointwiseSFTDataset
from evaluate_mind_jev import build_jev_state, JevScorer
from probe_jev_readout import behavior_rows, extract, fit_head, predict, save_json
from scale_jev_readout import sha256_file


SHARD_ROWS = 65536
TRAIN_ROWS = 469264
TRAIN_IMPRESSIONS = 156965
DIMENSION = 2560
DATA_SHA256 = {
    "train/news.tsv": "cd49d7da275e9c405f627d75a0de9d4ec6ad43ff8a8db5d23a447591676e09d6",
    "train/behaviors.tsv": "a424547c8fa17c9ea4879c2110221c2b2f2709f4f7615ede0b0d8e4765158658",
}
OLD_RESULT = "16771883911bf4ddf695e363db8f1839f8ae35509041478e1586e434d1ee09d1"
OLD_FEATURES = "8d4f803df205f7135a26a90058a6e5fe835549ef8010548c52685e9bf321d6f7"
HOLDOUT_FEATURES = "2e0e7dc5d247b641eeb63aee4f5f432dc8634f1f7f24ede77310a541d3c5fecc"
LOADER_PARITY = "83dd1182a732ba8e118759c05ab9d7e7dee29ff95b138eadab95aa7fe8be4ac0"
TEXT_NAME = "full_text_binary_l2_0.01"
BEHAVIOR_NAME = "full_behavior_pairwise_l2_0.01"


def identities(samples):
    grouped = defaultdict(list)
    seen = set()
    for sample in samples:
        row = {key: sample[key] for key in ("impression_id", "candidate_id", "label")}
        key = row["impression_id"], row["candidate_id"]
        if not all(key) or row["label"] not in (0, 1) or key in seen:
            raise ValueError("Full training requires unique identified binary candidate rows")
        seen.add(key)
        grouped[row["impression_id"]].append(row)
    if not grouped:
        raise ValueError("Full training cannot be empty")
    return [row for group in grouped.values() for row in group]


def legacy_prefix(rows, excluded, limit):
    grouped = defaultdict(list)
    for row in rows:
        if row["impression_id"] not in excluded:
            grouped[row["impression_id"]].append(row)
    result = []
    for group in grouped.values():
        if {row["label"] for row in group} != {0, 1}:
            continue
        if len(result) + len(group) > limit:
            break
        result.extend(group)
    return result


def key(row):
    return row["impression_id"], row["candidate_id"], row["label"]


def pending_rows(rows, covered, limit=SHARD_ROWS):
    return [row for row in rows if key(row) not in covered][:limit]


def read_cache(root, expected_rows, expected_features_sha, dimension=DIMENSION):
    if (json.loads((root / "train.rows.json").read_bytes()) != expected_rows
            or sha256_file(root / "train.features.npz") != expected_features_sha):
        raise ValueError("Full-training cache identity,order or feature checksum differs")
    with np.load(root / "train.features.npz", allow_pickle=False) as stored:
        if set(stored.files) != {"features", "margins"}:
            raise ValueError("Unexpected full-training feature cache fields")
        features, margins = stored["features"], stored["margins"]
    if (features.shape != (len(expected_rows), dimension) or margins.shape != (len(expected_rows),)
            or features.dtype != np.float32 or margins.dtype != np.float32
            or not np.isfinite(features).all() or not np.isfinite(margins).all()):
        raise ValueError("Invalid full-training cache shape,dtype or finite values")
    return features, margins


def fit_full(rows, features, margins, counts, totals, output, source_revision, provenance):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Full-training fitted output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([row["label"] for row in rows])
    groups = [row["impression_id"] for row in rows]
    text, text_loss = fit_head(features, margins, labels, groups, .01, "binary")
    text_scores = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(rows), 2048):
        end = start + 2048
        text_scores[start:end] = predict(text, features[start:end], margins[start:end])
    behavior_x = behavior_features(rows, counts, totals, cross_fitted=True)
    correction, correction_loss = fit_head(
        behavior_x, text_scores, labels, groups, .01, "pairwise", allow_unpaired=True,
    )
    hybrid_scores = predict(correction, behavior_x, text_scores)
    if not np.isfinite(hybrid_scores).all():
        raise FloatingPointError("Full-training hybrid scores are nonfinite")
    np.savez(output / (TEXT_NAME + ".head.npz"), **text)
    np.savez(output / (BEHAVIOR_NAME + ".head.npz"), **correction)
    save_counts(output / "population.npz", counts, totals)
    save_json(output / "train.rows.json", rows)
    np.savez(
        output / "train-head-margins.npz", features=behavior_x,
        frozen=margins, text=text_scores, hybrid=hybrid_scores,
    )
    indices = np.sort(np.random.default_rng(42).choice(len(rows), min(128, len(rows)), replace=False))
    np.savez(output / "fit-spotcheck.npz", indices=indices, features=features[indices])
    mixed = defaultdict(set)
    for row in rows:
        mixed[row["impression_id"]].add(row["label"])
    unpaired = sum(len(values) == 1 for values in mixed.values())
    result = {
        "source_revision": source_revision, "training_only": True, "full_training": True,
        "training_pairs": len(rows), "training_impressions": len(mixed),
        "train_holdout_impressions": 0, "artificial_pair_cap": None,
        "selected": TEXT_NAME, "behavior_selected": BEHAVIOR_NAME,
        "text_head_sha256": sha256_file(output / (TEXT_NAME + ".head.npz")),
        "behavior_head_sha256": sha256_file(output / (BEHAVIOR_NAME + ".head.npz")),
        "population_sha256": sha256_file(output / "population.npz"),
        "train_rows_sha256": sha256_file(output / "train.rows.json"),
        "train_head_margins_sha256": sha256_file(output / "train-head-margins.npz"),
        "fit_spotcheck_sha256": sha256_file(output / "fit-spotcheck.npz"),
        "text_training_loss": text_loss, "behavior_training_loss": correction_loss,
        "pairwise_supervised_impressions": len(mixed) - unpaired,
        "unpaired_impressions": unpaired, "data_sha256": DATA_SHA256,
        "sampling": {"seed": 42, "negative_ratio": 1., "hard_negative_ratio": .5},
        "regularization": .01, "folds": 5, "smoothing": 20,
        "minimum_dev_start": 31000, "cache_provenance": provenance,
        "quality_result": None,
        "limitations": [
            "Every official training impression is included;existing negative sampling is retained.",
            "All impressions train the binary text head and population;single-class impressions cannot supply pairwise contrasts.",
            "No training quality metric or train-holdout gate;quality must be established on official dev.",
        ],
    }
    save_json(output / "train-selection.json", {
        "selected": TEXT_NAME, "minimum_dev_start": 31000, "train_holdout_impressions": 0,
        "rule": "Fixed full-training recipe;no candidate selection or training holdout",
    })
    save_json(output / "result.json", result)
    return result


def run(data_root, old_cache, holdout_cache, output_root, stage, source_revision, model_path):
    if stage not in range(4):
        raise ValueError("Full training has exactly four bounded feature stages at most")
    output = output_root / "train" / f"stage_{stage:02d}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Full-training feature stage must not overwrite prior work")
    for name, expected in DATA_SHA256.items():
        if sha256_file(data_root / name) != expected:
            raise ValueError("Official full-training input checksum differs")
    if (sha256_file(old_cache / "result.json") != OLD_RESULT
            or sha256_file(old_cache / "loader-parity.json") != LOADER_PARITY):
        raise ValueError("Doubled-cache provenance differs")
    dataset = MINDPointwiseSFTDataset(
        str(data_root / "train" / "behaviors.tsv"), str(data_root / "train" / "news.tsv"), None,
        max_len=4096, sample=-1, seed=42, max_history=30, neg_ratio=1.,
        hard_neg_ratio=.5, use_abstract=True, use_chat_template=False,
    )
    rows = identities(dataset.samples)
    source_ids = [
        line.split("\t", 1)[0]
        for line in (data_root / "train" / "behaviors.tsv").open(encoding="utf-8")
    ]
    if (len(rows) != TRAIN_ROWS or len(source_ids) != TRAIN_IMPRESSIONS
            or len(set(source_ids)) != len(source_ids)
            or {row["impression_id"] for row in rows} != set(source_ids)
            or sum(row["label"] for row in rows) != 236344):
        raise ValueError("Uncapped training does not cover every official impression and positive")
    counts, totals = build_counts(data_root / "train" / "behaviors.tsv", set(), rows)
    old_holdout = behavior_rows(
        data_root / "train" / "behaviors.tsv", dataset.news, TRAIN_IMPRESSIONS - 256, 256,
    )
    holdout_rows = [row for _, group in old_holdout for row in group]
    holdout_ids = {impression for impression, _ in old_holdout}
    old_rows = legacy_prefix(rows, holdout_ids, 262144)
    old_x, old_base = read_cache(old_cache, old_rows, OLD_FEATURES)
    expected_holdout = [{name: row[name] for name in ("impression_id", "candidate_id", "label")} for row in holdout_rows]
    if (json.loads((holdout_cache / "train_holdout.rows.json").read_bytes()) != expected_holdout
            or sha256_file(holdout_cache / "train_holdout.features.npz") != HOLDOUT_FEATURES):
        raise ValueError("Former holdout cache identity or checksum differs")
    with np.load(holdout_cache / "train_holdout.features.npz", allow_pickle=False) as stored:
        hx, hb = stored["features"], stored["margins"]
    if (hx.shape != (len(holdout_rows), DIMENSION) or hb.shape != (len(holdout_rows),)
            or hx.dtype != np.float32 or hb.dtype != np.float32
            or not np.isfinite(hx).all() or not np.isfinite(hb).all()):
        raise ValueError("Invalid former holdout feature cache")
    caches = [
        (old_rows, old_x, old_base, "verified_doubled_cache", OLD_FEATURES),
        (expected_holdout, hx, hb, "former_holdout_now_training", HOLDOUT_FEATURES),
    ]
    target_keys = {key(row) for row in rows}
    covered = {key(row) for identities_, *_ in caches for row in identities_} & target_keys
    for previous in range(stage):
        previous_root = output_root / "train" / f"stage_{previous:02d}"
        manifest = json.loads((previous_root / "manifest.json").read_bytes())
        expected = pending_rows(rows, covered)
        if (manifest["source_revision"] != source_revision or manifest["stage"] != previous
                or manifest["training_complete"] or manifest["training_pairs"] != TRAIN_ROWS
                or manifest["data_sha256"] != DATA_SHA256
                or sha256_file(previous_root / "train.rows.json") != manifest["rows_sha256"]):
            raise ValueError("Previous full-training stage manifest differs")
        px, pb = read_cache(previous_root, expected, manifest["features_sha256"])
        caches.append((expected, px, pb, f"stage_{previous:02d}", manifest["features_sha256"]))
        covered.update(key(row) for row in expected)
    chosen = pending_rows(rows, covered)
    if not chosen or (stage == 3 and len(rows) - len(covered) > SHARD_ROWS):
        raise ValueError("Invalid full-training stage or bounded campaign capacity exceeded")
    samples = {key(sample): sample for sample in dataset.samples}
    holdout_states = {key(row): row["state"] for row in holdout_rows}

    def states(metadata):
        result = []
        for row in metadata:
            state = holdout_states.get(key(row))
            if state is None:
                sample = samples[key(row)]
                state = build_jev_state(sample["history"], sample["candidate"])
            result.append({**row, "state": state})
        return result

    scorer = JevScorer(model_path, prompt_contract="legacy", max_prompt_tokens=4095)
    for parameter in scorer.model.parameters():
        parameter.requires_grad_(False)
    for metadata, features, margins, role, _ in caches:
        fresh_x, fresh_base = extract(scorer, states(metadata[:16]), None, role, report_progress=False)
        if not np.array_equal(fresh_x, features[:16]) or not np.array_equal(fresh_base, margins[:16]):
            raise ValueError("Full-training cache differs from exact frozen-model replay: " + role)
    print(json.dumps({
        "stage": "full_training_extraction", "stage_index": stage, "training_rows": len(rows),
        "training_impressions": len(source_ids), "cached_rows": len(covered),
        "new_rows": len(chosen), "train_holdout_impressions": 0,
    }), flush=True)
    new_x, new_base = extract(scorer, states(chosen), None, "full_training_features")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "train.features.npz", features=new_x, margins=new_base)
    save_json(output / "train.rows.json", chosen)
    new_sha = sha256_file(output / "train.features.npz")
    caches.append((chosen, new_x, new_base, f"stage_{stage:02d}", new_sha))
    covered.update(key(row) for row in chosen)
    complete = len(covered) == len(rows)
    if complete:
        indices = {key(row): index for index, row in enumerate(rows)}
        train_x = np.empty((len(rows), DIMENSION), dtype=np.float32)
        train_base = np.empty(len(rows), dtype=np.float32)
        for metadata, features, margins, _, _ in caches:
            destinations, sources = [], []
            for index, row in enumerate(metadata):
                if key(row) in indices:
                    destinations.append(indices[key(row)])
                    sources.append(index)
            train_x[destinations] = features[sources]
            train_base[destinations] = margins[sources]
        provenance = [{"role": role, "features_sha256": digest, "rows": len(metadata)} for metadata, _, _, role, digest in caches]
        fitted = fit_full(
            rows, train_x, train_base, counts, totals, output_root / "train" / "final",
            source_revision, provenance,
        )
        (output_root / "train" / "final" / "loader-parity.json").write_bytes(
            (old_cache / "loader-parity.json").read_bytes()
        )
        print(json.dumps({"stage": "full_training_fitted", **fitted}), flush=True)
    manifest = {
        "source_revision": source_revision, "stage": stage, "training_only": True,
        "training_pairs": len(rows), "training_impressions": len(source_ids),
        "train_holdout_impressions": 0, "new_rows": len(chosen), "covered_rows": len(covered),
        "training_complete": complete, "features_sha256": new_sha,
        "rows_sha256": sha256_file(output / "train.rows.json"), "data_sha256": DATA_SHA256,
    }
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"stage": "full_training_stage_completed", **manifest}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--old-cache", type=Path, required=True)
    parser.add_argument("--holdout-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    run(args.data_root, args.old_cache, args.holdout_cache, args.output_root, args.stage,
        args.source_revision, args.model_path)
