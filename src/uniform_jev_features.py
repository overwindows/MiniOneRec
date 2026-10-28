"""Extract only missing uniform-training features, or a fresh full-candidate dev window."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import MINDPointwiseSFTDataset
from evaluate_mind_jev import JevScorer, build_jev_state
from full_jev_readout import DATA_SHA256, identities, key
from mind_utils import load_news
from probe_jev_readout import behavior_rows, extract


SHARD_ROWS = 65536
UNIFORM_SHA = "fd9f50c2668cd50ae03f918636010f84e43563c73c711997ef721ea30a9a3a84"
LEGACY_SHA = "b4dba4f73cdf5878ba1d67582d8b81899240e13c3670f38b8b26ee04605af254"
PENDING_SHA = "5e22cfab861f1d2451c79f1929582134a09c1548d877e78a60d5bcdbb5d9a1ac"
DEV_SHA256 = {
    "dev/news.tsv": "e5d144667558c449d16084fe6bfa01940c2f5d5785ea9f9110292bc3b94eb822",
    "dev/behaviors.tsv": "b6c460e33b1a8693252ded6e626da7d3ccf78920eea2ec11889020bb7d8443ef",
}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(rows):
    return (json.dumps(rows, indent=2, allow_nan=False) + "\n").encode()


def missing_rows(uniform, legacy, stage, shard_rows=SHARD_ROWS):
    if stage not in range(4) or shard_rows < 1:
        raise ValueError("Uniform training has four nonempty bounded feature shards")
    old = {key(row) for row in legacy}
    if len(old) != len(legacy) or len({key(row) for row in uniform}) != len(uniform):
        raise ValueError("Uniform and legacy identities must be unique")
    pending = [row for row in uniform if key(row) not in old]
    chosen = pending[stage * shard_rows:(stage + 1) * shard_rows]
    if not chosen:
        raise ValueError("Requested uniform feature shard is empty")
    return pending, chosen


def training_states(metadata, behaviors, news):
    needed = {row["impression_id"] for row in metadata}
    histories = {}
    with behaviors.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if fields[0] in needed:
                histories[fields[0]] = [
                    news[name] for name in fields[3].split()[-30:] if name in news
                ]
    if set(histories) != needed:
        raise ValueError("Source histories do not cover every requested candidate")
    return [
        {**row, "state": build_jev_state(histories[row["impression_id"]], news[row["candidate_id"]])}
        for row in metadata
    ]


def load_parity(directory, expected_sha):
    manifest_path = directory / "manifest.json"
    if digest(manifest_path) != expected_sha:
        raise ValueError("Frozen parity manifest checksum differs")
    manifest = json.loads(manifest_path.read_bytes())
    for name in ("rows.json", "features.npz"):
        if digest(directory / name) != manifest["sha256"][name]:
            raise ValueError("Frozen parity artifact differs: " + name)
    rows = json.loads((directory / "rows.json").read_bytes())
    with np.load(directory / "features.npz", allow_pickle=False) as stored:
        if set(stored.files) != {"features", "margins"}:
            raise ValueError("Unexpected frozen parity fields")
        x, base = stored["features"], stored["margins"]
    if (len(rows) != 16 or x.shape != (16, 2560) or base.shape != (16,)
            or x.dtype != np.float32 or base.dtype != np.float32
            or not np.isfinite(x).all() or not np.isfinite(base).all()):
        raise ValueError("Invalid exact original 16-row extraction batch")
    return rows, x, base


def check_parity(actual_x, actual_base, expected_x, expected_base):
    for actual, expected in ((actual_x, expected_x), (actual_base, expected_base)):
        if actual.dtype != np.float32 or actual.shape != expected.shape:
            raise ValueError("Frozen parity shape or dtype differs")
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def run(data_root, model_path, output_root, parity_cache, parity_sha, stage, revision):
    if stage not in range(5):
        raise ValueError("Expected feature stages0-3 or the registered dev stage4")
    output = output_root / f"stage_{stage:02d}"
    if output.exists():
        raise FileExistsError("Feature output exists; never overwrite or silently retry it")
    required = dict(DATA_SHA256)
    if stage == 4:
        required.update(DEV_SHA256)
    for name, sha in required.items():
        if digest(data_root / name) != sha:
            raise ValueError("Official source checksum differs: " + name)
    parity_rows, parity_x, parity_base = load_parity(parity_cache, parity_sha)
    news = load_news(str(data_root / "train" / "news.tsv"), use_abstract=True)
    parity_states = training_states(parity_rows, data_root / "train" / "behaviors.tsv", news)
    if stage < 4:
        legacy = MINDPointwiseSFTDataset(
            str(data_root / "train" / "behaviors.tsv"), str(data_root / "train" / "news.tsv"),
            None, seed=42, max_history=30, use_abstract=True,
        )
        old_rows = identities(legacy.samples)
        del legacy
        uniform = MINDPointwiseSFTDataset(
            str(data_root / "train" / "behaviors.tsv"), str(data_root / "train" / "news.tsv"),
            None, seed=42, max_history=30, use_abstract=True, negative_sampling="uniform",
        )
        rows = identities(uniform.samples)
        del uniform
        if (len(rows) != 472626 or len({row["impression_id"] for row in rows}) != 156965
                or hashlib.sha256(canonical(old_rows)).hexdigest() != LEGACY_SHA
                or hashlib.sha256(canonical(rows)).hexdigest() != UNIFORM_SHA):
            raise ValueError("Uniform or historical sampling differs from independent preparation")
        pending, chosen = missing_rows(rows, old_rows, stage)
        if len(pending) != 206274 or hashlib.sha256(canonical(pending)).hexdigest() != PENDING_SHA:
            raise ValueError("Missing-feature population differs")
        states = training_states(chosen, data_root / "train" / "behaviors.tsv", news)
    else:
        dev_news = load_news(str(data_root / "dev" / "news.tsv"), use_abstract=True)
        records = behavior_rows(data_root / "dev" / "behaviors.tsv", dev_news, 41000, 5000)
        states = [row for _, group in records for row in group]
        chosen = [{name: row[name] for name in ("impression_id", "candidate_id", "label")} for row in states]
    if not torch.cuda.is_available():
        raise RuntimeError("The registered frozen-feature worker requires one GPU")
    scorer = JevScorer(model_path, prompt_contract="legacy", max_prompt_tokens=4095)
    for parameter in scorer.model.parameters():
        parameter.requires_grad_(False)
    fresh_x, fresh_base = extract(scorer, parity_states, None, "exact_legacy_batch_parity", report_progress=False)
    check_parity(fresh_x, fresh_base, parity_x, parity_base)
    x, base = extract(scorer, states, None, "uniform_training_features" if stage < 4 else "frozen_dev_features")
    if (x.shape != (len(chosen), 2560) or base.shape != (len(chosen),)
            or x.dtype != np.float32 or base.dtype != np.float32):
        raise ValueError("Incomplete feature extraction")
    output.mkdir(parents=True)
    np.savez(output / "features.npz", features=x, margins=base)
    (output / "rows.json").write_bytes(canonical(chosen))
    manifest = {
        "stage": stage, "source_revision": revision, "rows": len(chosen), "dimension": 2560,
        "training_only": stage < 4, "backbone_fitting": False,
        "parity": "exact_original_contiguous_16_row_batch", "parity_manifest_sha256": parity_sha,
        "features_sha256": digest(output / "features.npz"), "rows_sha256": digest(output / "rows.json"),
        "data_sha256": required, "quality_result": None,
    }
    (output / "manifest.json").write_bytes(canonical(manifest))
    print(json.dumps({"stage": "uniform_feature_stage_completed", **manifest}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parity-cache", type=Path, required=True)
    parser.add_argument("--parity-sha256", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.data_root, args.model_path, args.output_root, args.parity_cache,
        args.parity_sha256, args.stage, args.source_revision)
