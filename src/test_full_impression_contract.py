import json
import time

import numpy as np
import pytest

from full_impression_contract import digest, prepare, verify, write_json


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "news.tsv").write_text("".join(
        f"N{index}\tcat\tsub\ttitle na\u00efve{index}\tabs\thttps://example.invalid\t[]\t[]\n"
        for index in range(1, 5)
    ), encoding="utf-8")
    long_history = " ".join(f"N{index % 4 + 1}" for index in range(35))
    (root / "behaviors.tsv").write_text(
        "1\tU1\t11/12/2019 1:00:00 AM\tN4 N2\tN2-1 N3-0 N1-1 N4-0\n"
        f"2\tU2\t11/13/2019 1:00:00 AM\t{long_history}\tN4-0 N3-1\n"
        "3\tU3\t11/14/2019 1:00:00 AM\t\tN1-1 N2-0\n", encoding="utf-8",
    )
    report = {
        "all_training_rows_enumerated": True, "complete_local_bytes_independently_rehashed": True,
        "dev_or_test_contents_read": False, "single_class_impressions": 0,
        "unknown_candidate_news": 0, "unknown_history_news": 0,
        "impressions": 3, "news": 4, "exposures": 8, "positives": 4, "negatives": 4,
        "files": [{"path": str(root / name), "bytes": (root / name).stat().st_size,
                   "whole_sha256": digest(root / name)} for name in ("news.tsv", "behaviors.tsv")],
    }
    write_json(root / "result.json", report)
    return root, digest(root / "result.json")


def test_every_positive_negative_candidate_and_history_is_preserved(source, tmp_path):
    root, sha = source
    output = tmp_path / "encoded"
    manifest = prepare(root, sha, output, time.monotonic() + 60)
    np.testing.assert_array_equal(np.load(output / "candidate_labels.npy"), [1, 0, 1, 0, 0, 1, 1, 0])
    np.testing.assert_array_equal(np.load(output / "candidate_news.npy"), [2, 3, 1, 4, 4, 3, 1, 2])
    np.testing.assert_array_equal(np.load(output / "candidate_offsets.npy"), [0, 4, 6, 8])
    history = np.load(output / "histories.npy")
    np.testing.assert_array_equal(history[0, :28], 0)
    np.testing.assert_array_equal(history[0, -2:], [4, 2])
    np.testing.assert_array_equal(history[1], [index % 4 + 1 for index in range(5, 35)])
    np.testing.assert_array_equal(history[2], 0)
    assert manifest["temporary_validation_day"] == "2019-11-14"
    assert manifest["sampling"] == "none;every_source_candidate_in_original_order"
    proof = verify(root, output, time.monotonic() + 60)
    assert proof["candidate_exposures_replayed"] == 8
    assert proof["positives_replayed"] == proof["negatives_replayed"] == 4
    assert proof["temporary_training_impressions"] == 2 and proof["temporary_validation_impressions"] == 1


def test_encoding_is_deterministic_and_existing_output_is_preserved(source, tmp_path):
    root, sha = source
    one, two = tmp_path / "one", tmp_path / "two"
    a = prepare(root, sha, one, time.monotonic() + 60)
    b = prepare(root, sha, two, time.monotonic() + 60)
    assert a == b and a["sha256"] == b["sha256"]
    before = digest(one / "manifest.json")
    with pytest.raises(FileExistsError):
        prepare(root, sha, one, time.monotonic() + 60)
    assert digest(one / "manifest.json") == before


def test_second_positive_corruption_is_detected_even_after_file_digest_is_rewritten(source, tmp_path):
    root, sha = source
    output = tmp_path / "encoded"
    manifest = prepare(root, sha, output, time.monotonic() + 60)
    labels = np.load(output / "candidate_labels.npy", mmap_mode="r+")
    labels[2] = 0
    labels.flush()
    del labels
    manifest["sha256"]["candidate_labels.npy"] = digest(output / "candidate_labels.npy")
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="differs from raw source"):
        verify(root, output, time.monotonic() + 60)


def test_changed_source_is_rejected_before_output_allocation(source, tmp_path):
    root, sha = source
    with (root / "behaviors.tsv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    output = tmp_path / "encoded"
    with pytest.raises(ValueError, match="source bytes differ"):
        prepare(root, sha, output, time.monotonic() + 60)
    assert not output.exists()


def test_expired_deadline_cannot_allocate_output(source, tmp_path):
    root, sha = source
    output = tmp_path / "encoded"
    with pytest.raises(TimeoutError):
        prepare(root, sha, output, time.monotonic() - 1)
    assert not output.exists()


def test_training_day_metadata_is_independently_reconstructed(source, tmp_path):
    root, sha = source
    output = tmp_path / "encoded"
    manifest = prepare(root, sha, output, time.monotonic() + 60)
    manifest["temporary_validation_day"] = "2019-11-13"
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="origin replay"):
        verify(root, output, time.monotonic() + 60)


@pytest.mark.parametrize("kind", ["file_inventory", "source_hash", "news_count", "split", "history", "metadata"])
def test_contract_metadata_cannot_silently_change(source, tmp_path, kind):
    root, sha = source
    output = tmp_path / "encoded"
    manifest = prepare(root, sha, output, time.monotonic() + 60)
    if kind == "file_inventory":
        manifest["sha256"].pop("impression_ids.txt")
    elif kind == "source_hash":
        manifest["input_sha256"]["news.tsv"] = "0" * 64
    elif kind == "news_count":
        manifest["news"] += 1
    elif kind == "split":
        manifest["final_training_requires_all_impressions"] = False
    elif kind == "history":
        history = np.load(output / "histories.npy", mmap_mode="r+")
        history[0, -1] = 3
        history.flush()
        del history
        manifest["sha256"]["histories.npy"] = digest(output / "histories.npy")
    else:
        path = output / "news.metadata.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        records[0]["title"] = "changed"
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        manifest["sha256"]["news.metadata.jsonl"] = digest(path)
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError):
        verify(root, output, time.monotonic() + 60)


def test_unknown_news_fails_without_exposing_a_completion_marker(source, tmp_path):
    root, _ = source
    path = root / "behaviors.tsv"
    path.write_text(path.read_text(encoding="utf-8").replace("N2-1", "N5-1"), encoding="utf-8")
    report = json.loads((root / "result.json").read_bytes())
    report["files"][1]["bytes"] = path.stat().st_size
    report["files"][1]["whole_sha256"] = digest(path)
    write_json(root / "result.json", report)
    output = tmp_path / "encoded"
    with pytest.raises(ValueError, match="unresolved"):
        prepare(root, digest(root / "result.json"), output, time.monotonic() + 60)
    assert output.exists() and not (output / "manifest.json").exists()
