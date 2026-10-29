"""Lossless, bounded-memory full-impression inputs for the next MIND-large trainer."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from jev_content_ranker import digest, write_json


HISTORY = 30
ARRAYS = {
    "candidate_news.npy": ("int32", "exposures"),
    "candidate_labels.npy": ("uint8", "exposures"),
    "candidate_offsets.npy": ("uint64", "offsets"),
    "histories.npy": ("int32", "histories"),
    "training_days.npy": ("int32", "impressions"),
}


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Full-impression contract exceeded its registered deadline")


def read_news(path):
    news = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 8 or not parts[0] or parts[0] in news:
                raise ValueError("Every training news identity must have exactly one complete record")
            news[parts[0]] = {
                "news_id": parts[0], "category": parts[1], "subcategory": parts[2],
                "title": parts[3], "abstract": parts[4],
            }
    return news


def prepare(source, source_sha, output, deadline):
    if output.exists():
        raise FileExistsError("Preserve existing full-impression output; never overwrite")
    if digest(source / "result.json") != source_sha:
        raise ValueError("Pinned complete training-source report changed")
    report = json.loads((source / "result.json").read_bytes())
    if (not report["all_training_rows_enumerated"] or not report["complete_local_bytes_independently_rehashed"]
            or report["dev_or_test_contents_read"] or report["single_class_impressions"]
            or report["unknown_candidate_news"] or report["unknown_history_news"]):
        raise ValueError("Complete, mixed-class, source-resolved training impressions are required")
    inputs = {Path(item["path"]).name: item for item in report["files"]}
    if set(inputs) != {"news.tsv", "behaviors.tsv"} or len(report["files"]) != 2:
        raise ValueError("Expected exactly the two bound training source files")
    for name, item in inputs.items():
        check_deadline(deadline)
        if (source / name).stat().st_size != item["bytes"] or digest(source / name) != item["whole_sha256"]:
            raise ValueError("Complete training source bytes differ")
    news = read_news(source / "news.tsv")
    if len(news) != report["news"]:
        raise ValueError("Complete source news coverage differs")
    identifiers = sorted(news)
    lookup = {name.encode("utf-8"): index + 1 for index, name in enumerate(identifiers)}
    impressions, exposures = report["impressions"], report["exposures"]
    if not 0 < impressions < exposures or len(news) >= np.iinfo(np.int32).max:
        raise ValueError("Unexpected source dimensions")
    shapes = {
        "exposures": (exposures,), "offsets": (impressions + 1,),
        "histories": (impressions, HISTORY), "impressions": (impressions,),
    }
    check_deadline(deadline)
    output.mkdir()
    write_json(output / "news.ids.json", identifiers)
    with (output / "news.metadata.jsonl").open("w", encoding="utf-8") as handle:
        for name in identifiers:
            handle.write(json.dumps(news[name], ensure_ascii=False, allow_nan=False) + "\n")
    arrays = {
        name: np.lib.format.open_memmap(output / name, mode="w+", dtype=dtype, shape=shapes[shape])
        for name, (dtype, shape) in ARRAYS.items()
    }
    arrays["histories.npy"][:] = 0
    arrays["candidate_offsets.npy"][0] = 0
    dates, daily = {}, {}
    position = positives = rows = maximum_candidates = maximum_history = 0
    with (source / "behaviors.tsv").open("rb") as handle, (output / "impression_ids.txt").open("xb") as ids:
        for rows, line in enumerate(handle, 1):
            if rows % 4096 == 0:
                check_deadline(deadline)
            fields = line.rstrip(b"\r\n").split(b"\t")
            if len(fields) != 5 or not fields[0] or rows > impressions:
                raise ValueError("Source impression structure or count differs")
            tokens = [token.rsplit(b"-", 1) for token in fields[4].split()]
            if not tokens or any(len(token) != 2 or token[1] not in (b"0", b"1") for token in tokens):
                raise ValueError("Malformed full candidate sequence")
            end = position + len(tokens)
            if end > exposures:
                raise ValueError("Candidate exposure count exceeds the complete source")
            try:
                candidates = np.asarray([lookup[name] for name, _ in tokens], dtype=np.int32)
                history = [lookup[name] for name in fields[3].split()[-HISTORY:]]
            except KeyError as error:
                raise ValueError("A source candidate/history news identity is unresolved") from error
            labels = np.asarray([int(label) for _, label in tokens], dtype=np.uint8)
            positive = int(labels.sum())
            if not 0 < positive < len(tokens):
                raise ValueError("Never silently drop a single-class source impression")
            day = fields[2].split(b" ", 1)[0]
            if day not in dates:
                dates[day] = datetime.strptime(day.decode("ascii"), "%m/%d/%Y").date()
            date = dates[day]
            key = date.isoformat()
            counts = daily.setdefault(key, {"impressions": 0, "exposures": 0, "positives": 0})
            counts["impressions"] += 1
            counts["exposures"] += len(tokens)
            counts["positives"] += positive
            index = rows - 1
            arrays["candidate_news.npy"][position:end] = candidates
            arrays["candidate_labels.npy"][position:end] = labels
            arrays["candidate_offsets.npy"][rows] = end
            arrays["training_days.npy"][index] = date.toordinal()
            if history:
                arrays["histories.npy"][index, -len(history):] = history
            ids.write(fields[0] + b"\n")
            positives += positive
            position = end
            maximum_candidates = max(maximum_candidates, len(tokens))
            maximum_history = max(maximum_history, len(history))
            if rows % 65536 == 0:
                print(json.dumps({"stage": "encoding_all_training_candidates", "impressions": rows,
                                  "exposures": position, "positives": positives}), flush=True)
    if (rows != impressions or position != exposures or positives != report["positives"]
            or exposures - positives != report["negatives"] or len(daily) < 2):
        raise ValueError("Every source impression, positive and negative must be retained")
    for array in arrays.values():
        array.flush()
    arrays.clear()
    del array
    files = [*ARRAYS, "news.ids.json", "news.metadata.jsonl", "impression_ids.txt"]
    hashes = {}
    for name in files:
        check_deadline(deadline)
        hashes[name] = digest(output / name)
    result = {
        "stage": "all_training_candidates_encoded", "implementation_sha256": digest(Path(__file__)),
        "source_result_sha256": source_sha, "input_sha256": {name: item["whole_sha256"] for name, item in inputs.items()},
        "impressions": impressions, "exposures": exposures, "positives": positives,
        "negatives": exposures - positives, "news": len(news), "history_limit": HISTORY,
        "maximum_candidates": maximum_candidates, "maximum_encoded_history": maximum_history,
        "padding_news_index": 0, "news_index": "one_based_lexicographically_sorted_source_ids",
        "history_padding": "left_zero_pad;retain_last30_in_source_order",
        "sampling": "none;every_source_candidate_in_original_order",
        "daily": dict(sorted(daily.items())), "temporary_validation_day": max(daily),
        "temporary_split_only": True, "final_training_requires_all_impressions": True,
        "sha256": hashes, "gpu_allocations": 0, "model_fits": 0, "dev_or_test_contents_read": False,
        "competition_test_goal_achieved": False,
    }
    write_json(output / "manifest.pending.json", result)
    (output / "manifest.pending.json").rename(output / "manifest.json")
    return result


def verify(source, output, deadline):
    manifest = json.loads((output / "manifest.json").read_bytes())
    if manifest["stage"] != "all_training_candidates_encoded" or digest(source / "result.json") != manifest["source_result_sha256"]:
        raise ValueError("Encoded source binding differs")
    if (manifest["history_limit"] != HISTORY or manifest["padding_news_index"] != 0
            or manifest["sampling"] != "none;every_source_candidate_in_original_order"
            or not manifest["temporary_split_only"] or not manifest["final_training_requires_all_impressions"]):
        raise ValueError("Full-candidate history/sampling/split contract differs")
    expected_files = set(ARRAYS) | {"news.ids.json", "news.metadata.jsonl", "impression_ids.txt"}
    if set(manifest["sha256"]) != expected_files or set(manifest["input_sha256"]) != {"news.tsv", "behaviors.tsv"}:
        raise ValueError("Encoded source/file inventory differs")
    report = json.loads((source / "result.json").read_bytes())
    inputs = {Path(item["path"]).name: item["whole_sha256"] for item in report["files"]}
    if (len(report["files"]) != 2 or inputs != manifest["input_sha256"]
            or any(manifest[key] != report[key] for key in ("news", "impressions", "exposures", "positives", "negatives"))
            or manifest["implementation_sha256"] != digest(Path(__file__))):
        raise ValueError("Encoded inputs/counts/implementation differ from the pinned source contract")
    for name, expected in manifest["input_sha256"].items():
        check_deadline(deadline)
        if digest(source / name) != expected:
            raise ValueError("Bound raw training source changed")
    for name, expected in manifest["sha256"].items():
        check_deadline(deadline)
        if digest(output / name) != expected:
            raise ValueError("Encoded complete file differs: " + name)
    news = read_news(source / "news.tsv")
    names = json.loads((output / "news.ids.json").read_bytes())
    if names != sorted(news) or len(names) != manifest["news"]:
        raise ValueError("Complete news identities/order differ")
    lookup = {name: index + 1 for index, name in enumerate(names)}
    with (output / "news.metadata.jsonl").open(encoding="utf-8") as handle:
        for name, line in zip(names, handle, strict=True):
            if json.loads(line) != news[name]:
                raise ValueError("News article metadata differs from source")
    arrays = {name: np.load(output / name, mmap_mode="r", allow_pickle=False) for name in ARRAYS}
    for name, (dtype, _) in ARRAYS.items():
        if arrays[name].dtype != np.dtype(dtype):
            raise ValueError("Encoded array dtype differs")
    candidates, labels = arrays["candidate_news.npy"], arrays["candidate_labels.npy"]
    offsets, histories, days = arrays["candidate_offsets.npy"], arrays["histories.npy"], arrays["training_days.npy"]
    if (candidates.shape != labels.shape or candidates.shape != (manifest["exposures"],)
            or offsets.shape != (manifest["impressions"] + 1,) or offsets[0] != 0
            or histories.shape != (manifest["impressions"], HISTORY) or days.shape != (manifest["impressions"],)):
        raise ValueError("Encoded dimensions differ")
    rows = positives = maximum_candidates = maximum_history = 0
    daily = {}
    with (source / "behaviors.tsv").open(encoding="utf-8") as handle, (output / "impression_ids.txt").open(encoding="utf-8") as ids:
        for index, (line, identity) in enumerate(zip(handle, ids, strict=True)):
            if index % 4096 == 0:
                check_deadline(deadline)
            fields = line.rstrip("\r\n").split("\t")
            items = [token.rsplit("-", 1) for token in fields[4].split()]
            expected_labels = np.fromiter((int(label) for _, label in items), dtype=np.uint8)
            expected_candidates = np.fromiter((lookup[name] for name, _ in items), dtype=np.int32)
            expected_history = np.zeros(HISTORY, dtype=np.int32)
            history = [lookup[name] for name in fields[3].split()[-HISTORY:]]
            if history:
                expected_history[-len(history):] = history
            date = datetime.strptime(fields[2].split(" ", 1)[0], "%m/%d/%Y").date()
            start, end = int(offsets[index]), int(offsets[index + 1])
            if (identity.rstrip("\r\n") != fields[0] or end - start != len(items)
                    or not np.array_equal(candidates[start:end], expected_candidates)
                    or not np.array_equal(labels[start:end], expected_labels)
                    or not np.array_equal(histories[index], expected_history)
                    or days[index] != date.toordinal()):
                raise ValueError("An encoded impression/candidate/label/history/day differs from raw source")
            rows += 1
            positive = int(expected_labels.sum())
            positives += positive
            counts = daily.setdefault(date.isoformat(), {"impressions": 0, "exposures": 0, "positives": 0})
            counts["impressions"] += 1
            counts["exposures"] += len(items)
            counts["positives"] += positive
            maximum_candidates = max(maximum_candidates, len(items))
            maximum_history = max(maximum_history, len(history))
            if rows % 65536 == 0:
                print(json.dumps({"stage": "replaying_all_source_candidate_origins", "impressions": rows,
                                  "positives": positives}), flush=True)
    if (rows != manifest["impressions"] or offsets[-1] != manifest["exposures"]
            or positives != manifest["positives"] or manifest["exposures"] - positives != manifest["negatives"]
            or daily != manifest["daily"] or max(daily) != manifest["temporary_validation_day"]
            or maximum_candidates != manifest["maximum_candidates"]
            or maximum_history != manifest["maximum_encoded_history"]):
        raise ValueError("Incomplete all-candidate origin replay")
    for name, expected in inputs.items():
        check_deadline(deadline)
        if digest(source / name) != expected:
            raise ValueError("Raw training source changed during origin replay")
    check_deadline(deadline)
    return {"impressions_replayed": rows, "candidate_exposures_replayed": int(offsets[-1]),
            "positives_replayed": positives, "negatives_replayed": manifest["exposures"] - positives,
            "all_histories_days_identities_and_article_metadata_verified": True,
            "temporary_validation_day": max(daily),
            "temporary_validation_impressions": daily[max(daily)]["impressions"],
            "temporary_training_impressions": rows - daily[max(daily)]["impressions"],
            "source_result_sha256": manifest["source_result_sha256"],
            "manifest_sha256": digest(output / "manifest.json"),
            "gpu_allocations": 0, "model_fits": 0, "dev_or_test_contents_read": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 1800:
        parser.error("Full preparation and source-origin replay share a hard thirty-minute maximum")
    deadline = time.monotonic() + args.max_seconds
    prepare(args.source_root, args.source_result_sha256, args.output, deadline)
    proof = verify(args.source_root, args.output, deadline)
    check_deadline(deadline)
    write_json(args.output / "verification.pending.json", proof)
    (args.output / "verification.pending.json").rename(args.output / "verification.json")
    print(json.dumps(proof), flush=True)
