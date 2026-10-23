"""Confirm one checksum-pinned residual readout without fitting or reselection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_mind_jev import JevScorer
from mind_utils import load_news
from probe_jev_readout import behavior_rows, extract, metrics, predict, save_json


def load_head(path, expected_sha256):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Fixed readout checksum mismatch")
    with np.load(path, allow_pickle=False) as stored:
        head = dict(stored)
    if set(head) != {"mean", "scale", "weight", "bias"}:
        raise ValueError("Unexpected fixed-readout fields")
    shape = head["weight"].shape
    if (len(shape) != 1 or not shape[0]
            or head["mean"].shape != shape or head["scale"].shape != shape
            or head["bias"].shape != () or not (head["scale"] > 0).all()
            or not all(np.isfinite(value).all() for value in head.values())):
        raise ValueError("Invalid fixed-readout dimensions or numeric values")
    return head, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--head-path", type=Path, required=True)
    ap.add_argument("--head-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--dev-start", type=int, default=6000)
    ap.add_argument("--dev-impressions", type=int, default=5000)
    args = ap.parse_args()
    if args.dev_start < 6000 or args.dev_impressions < 1:
        ap.error("Confirmation must exclude dev rows already used by the pilot")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("Refusing to overwrite a completed confirmation")
    head, head_bytes = load_head(args.head_path, args.head_sha256)
    selection_path = args.head_path.parent / "train-selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if args.head_path.name != selection["selected"] + ".head.npz":
        raise ValueError("Pinned readout is not the original train-selected candidate")
    for name in ("train-selection.json", "loader-parity.json"):
        (args.output / name).write_bytes((args.head_path.parent / name).read_bytes())
    (args.output / "selected.head.npz").write_bytes(head_bytes)
    dev_dir = args.data_root / "dev"
    records = behavior_rows(
        dev_dir / "behaviors.tsv", load_news(str(dev_dir / "news.tsv"), True),
        args.dev_start, args.dev_impressions,
    )
    rows = [row for _, impression in records for row in impression]
    sample_indices = np.sort(
        np.random.default_rng(42).choice(len(rows), min(128, len(rows)), replace=False)
    )
    sample_features, sample_base, sample_scores = [], [], []
    base_parts, score_parts = [], []
    scorer = JevScorer(args.model_path, prompt_contract="legacy", max_prompt_tokens=4095)
    for parameter in scorer.model.parameters():
        parameter.requires_grad_(False)
    for start in range(0, len(rows), 128):
        chunk = rows[start:start + 128]
        features, base = extract(
            scorer, chunk, None, "confirmation", report_progress=False,
        )
        if features.shape[1:] != head["weight"].shape:
            raise ValueError("Readout dimensions do not match frozen decision features")
        scores = predict(head, features, base)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Fixed-head confirmation scores are nonfinite")
        base_parts.append(base)
        score_parts.append(scores)
        chosen = sample_indices[(sample_indices >= start) & (sample_indices < start + len(chunk))] - start
        if len(chosen):
            sample_features.append(features[chosen])
            sample_base.append(base[chosen])
            sample_scores.append(scores[chosen])
        if start % 512 == 0:
            print(json.dumps({
                "stage": "confirmation", "processed": start + len(chunk), "total": len(rows),
            }), flush=True)
    base, scores = np.concatenate(base_parts), np.concatenate(score_parts)
    np.savez(
        args.output / "readout-spotcheck.npz", indices=sample_indices,
        features=np.concatenate(sample_features), margins=np.concatenate(sample_base),
        selected=np.concatenate(sample_scores),
    )
    labels = np.asarray([row["label"] for row in rows])
    groups = [row["impression_id"] for row in rows]
    frozen = metrics(labels, base, groups)
    selected = metrics(labels, scores, groups)
    with (args.output / "dev.scores.jsonl").open("w", encoding="utf-8") as handle:
        position = 0
        for impression, candidates in records:
            end = position + len(candidates)
            handle.write(json.dumps({
                "impression_id": impression,
                "candidate_ids": [row["candidate_id"] for row in candidates],
                "labels": [row["label"] for row in candidates],
                "frozen_margins": base[position:end].tolist(),
                "selected_margins": scores[position:end].tolist(),
            }, allow_nan=False) + "\n")
            position = end
    result = {
        "source_revision": args.source_revision,
        "backbone": "frozen; fixed checksum-pinned readout; no fitting or reselection",
        "head_sha256": args.head_sha256, "selected": selection["selected"],
        "train_holdout_impressions": 256,
        "dev_start_zero_based": args.dev_start, "dev_impressions": len(records),
        "dev_candidates": len(rows), "frozen": frozen, "selected_metrics": selected,
        "delta": {key: selected[key] - frozen[key] for key in frozen},
        "data_sha256": {
            "dev/" + name: hashlib.sha256((dev_dir / name).read_bytes()).hexdigest()
            for name in ("news.tsv", "behaviors.tsv")
        },
        "limitations": [
            "Fixed frozen-backbone readout, not full-model or native two-exit JEV training.",
            "Fresh dev rows6001-11000 by default; not the old first5000 comparison.",
            "One training subset/seed; no SOTA or repeated-seed claim.",
        ],
    }
    save_json(args.output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
