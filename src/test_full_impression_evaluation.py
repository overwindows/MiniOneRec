import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

import full_impression_evaluation as subject
from full_impression_contract import prepare
from full_impression_dev_verification import verify_source
from test_full_impression_contract import source


@pytest.fixture
def inputs(source, tmp_path):
    root, sha = source
    train = tmp_path / "train"
    prepare(root, sha, train, time.monotonic() + 60)
    dev, small = tmp_path / "dev", tmp_path / "small"
    dev.mkdir()
    small.mkdir()
    (dev / "news.tsv").write_text(
        (root / "news.tsv").read_text(encoding="utf-8")
        + "N5\tcat\tsub\tnew article\tabstract\thttps://example.invalid\t[]\t[]\n",
        encoding="utf-8",
    )
    long_history = " ".join(["N1"] * 31 + ["N5"])
    (dev / "behaviors.tsv").write_text(
        f"100\tU1\ttime\t{long_history}\tN5-1 N2-0 N5-0 N1-1\n"
        "101\tU2\ttime\t\tN2-0 N5-1\n", encoding="utf-8",
    )
    (small / "behaviors.tsv").write_text(
        f"other\tU9\tother time\t{long_history}\tN1-0 N5-0 N2-1 N5-1\n", encoding="utf-8",
    )
    return train, dev, small


def test_complete_dev_preserves_candidate_order_duplicates_labels_and_last30_history(inputs, tmp_path):
    train, dev, small = inputs
    output = tmp_path / "encoded"
    report, arrays = subject.prepare_dev(dev, small, train, output, time.monotonic() + 60)
    assert report["impressions"] == 2 and report["exposures"] == 6 and report["positives"] == 3
    assert report["training_articles_reused"] == 4 and report["new_dev_articles"] == 1
    assert json.loads((output / "news.ids.json").read_bytes()) == ["N1", "N2", "N3", "N4", "N5"]
    np.testing.assert_array_equal(arrays["candidate_news.npy"], [5, 2, 5, 1, 2, 5])
    np.testing.assert_array_equal(arrays["candidate_labels.npy"], [1, 0, 0, 1, 0, 1])
    np.testing.assert_array_equal(arrays["candidate_offsets.npy"], [0, 4, 6])
    np.testing.assert_array_equal(arrays["histories.npy"][0], [1] * 29 + [5])
    np.testing.assert_array_equal(arrays["histories.npy"][1], 0)
    np.testing.assert_array_equal(arrays["small_dev_signature_overlap.npy"], [True, False])
    assert (output / "impression_ids.txt").read_text() == "100\n101\n"
    assert report["signature_unmatched_does_not_guarantee_never_inspected"]
    replay = verify_source(dev, small, train, output, time.monotonic() + 60)
    assert replay["impressions"] == 2 and replay["exposures"] == 6 and replay["positives"] == 3
    with pytest.raises(FileExistsError):
        subject.prepare_dev(dev, small, train, output, time.monotonic() + 60)


def test_overlap_ignores_candidate_order_and_labels_but_retains_multiset_and_history_order():
    a = subject.source_signature(["N1", "N2"], ["N3", "N4", "N3"])
    assert a == subject.source_signature(["N1", "N2"], ["N3", "N3", "N4"])
    assert a != subject.source_signature(["N1", "N2"], ["N3", "N4"])
    assert a != subject.source_signature(["N2", "N1"], ["N3", "N4", "N3"])


@pytest.mark.parametrize("kind", ["metadata", "unknown", "duplicate", "single_class"])
def test_no_metadata_collision_unknown_article_duplicate_id_or_excluded_impression(inputs, tmp_path, kind):
    train, dev, small = inputs
    path = dev / ("news.tsv" if kind == "metadata" else "behaviors.tsv")
    text = path.read_text(encoding="utf-8")
    if kind == "metadata":
        text = text.replace("title na\u00efve1", "changed title")
    elif kind == "unknown":
        text = text.replace("N5-1", "UNKNOWN-1")
    elif kind == "duplicate":
        text = text.replace("101\t", "100\t")
    else:
        text = text.replace("N2-0 N5-1", "N2-0 N5-0")
    path.write_text(text, encoding="utf-8")
    output = tmp_path / "encoded"
    with pytest.raises(ValueError):
        subject.prepare_dev(dev, small, train, output, time.monotonic() + 60)
    assert not output.exists()


def test_expired_evaluation_contract_cannot_read_or_allocate(inputs, tmp_path):
    train, dev, small = inputs
    with pytest.raises(TimeoutError):
        subject.prepare_dev(dev, small, train, tmp_path / "encoded", time.monotonic() - 1)


def test_all_training_prior_uses_all_rows_and_prescribed_unseen_global_prior():
    arrays = {
        "histories.npy": np.zeros((2, 30), dtype=np.int32),
        "candidate_offsets.npy": np.array([0, 2, 5]),
        "candidate_news.npy": np.array([1, 2, 1, 2, 2]),
        "candidate_labels.npy": np.array([1, 0, 0, 1, 1]),
    }
    actual = subject.all_training_prior(arrays, 2, 3, time.monotonic() + 60)
    global_prior = 3 / 5
    expected = (np.array([0, 1, 2]) + 20 * global_prior) / (np.array([0, 2, 3]) + 20)
    np.testing.assert_array_equal(actual[:3], np.log(expected / (1 - expected)).astype(np.float32))
    assert actual[3] == np.float32(np.log(global_prior / (1 - global_prior)))


@pytest.mark.parametrize("kind", ["shape", "dtype", "nonfinite", "padding", "zero", "margin"])
def test_frozen_article_union_rejects_bad_features_or_margins(tmp_path, kind):
    features = np.ones((3, 2560), dtype=np.float32)
    features[0] = 0
    margins = np.array([0., 1., 2.], dtype=np.float32)
    if kind == "shape":
        features = features[:, :-1]
    elif kind == "dtype":
        features = features.astype(np.float64)
    elif kind == "nonfinite":
        features[1, 0] = np.nan
    elif kind == "padding":
        features[0, 0] = 1.
    elif kind == "zero":
        features[2] = 0
    else:
        margins = margins[:-1]
    np.save(tmp_path / "article.features.npy", features)
    np.save(tmp_path / "article.margins.npy", margins)
    with pytest.raises(ValueError, match="Every article"):
        subject.frozen_articles(tmp_path, 2)


def test_gate_failure_precedes_any_dev_or_training_contract_read(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError("fixture closed gate")

    def forbidden(*args):
        pytest.fail("A failed training gate must not read any dev or contract data")

    monkeypatch.setattr(subject, "load_gate", reject)
    monkeypatch.setattr(subject, "load_contract", forbidden)
    monkeypatch.setattr(subject, "prepare_dev", forbidden)
    args = SimpleNamespace(
        output=tmp_path / "output", source_revision="a" * 40, max_seconds=21600,
        training_artifacts=tmp_path, training_proof=tmp_path,
        training_proof_sha256="b" * 64, training_result_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="closed gate"):
        subject.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("kind", ["candidate", "target", "history", "overlap", "identity", "metadata"])
def test_independent_source_replay_rejects_rehashed_encoded_corruption(inputs, tmp_path, kind):
    train, dev, small = inputs
    output = tmp_path / "encoded"
    report, _ = subject.prepare_dev(dev, small, train, output, time.monotonic() + 60)
    if kind in ("candidate", "target", "history", "overlap"):
        name = {"candidate": "candidate_news.npy", "target": "candidate_labels.npy",
                "history": "histories.npy", "overlap": "small_dev_signature_overlap.npy"}[kind]
        array = np.load(output / name, mmap_mode="r+")
        if kind == "history":
            array[0, 0] = 2
        elif kind == "candidate":
            array[0] = 2
        else:
            array[0] = 0
        array.flush()
        del array
    elif kind == "identity":
        name = "impression_ids.txt"
        (output / name).write_text("wrong\n101\n", encoding="utf-8")
    else:
        name = "new_articles/news.metadata.jsonl"
        row = json.loads((output / name).read_text(encoding="utf-8"))
        row["title"] = "changed"
        (output / name).write_text(json.dumps(row) + "\n", encoding="utf-8")
    report["sha256"][name] = subject.digest(output / name)
    subject.write_json(output / "manifest.json", report)
    with pytest.raises(ValueError, match="raw source"):
        verify_source(dev, small, train, output, time.monotonic() + 60)
