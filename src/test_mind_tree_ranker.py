from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import mind_tree_ranker as module
from test_mind_lexical_glm import inputs


@pytest.fixture
def tree_inputs(inputs):
    original, corpus, _, news = inputs
    source = module.Source()
    for name in ("offsets", "candidates", "labels", "histories"):
        setattr(source, name, getattr(original, name).copy())
    source.days = np.asarray([0, 0, 0, 0, 1], dtype=np.int32)
    source.folds = np.asarray([0, 1, 0, 1, 0], dtype=np.uint8)
    source.times = np.arange(5, dtype=np.int32) * 3600
    source.arrays = []
    metadata = module.Metadata()
    for name in ("names", "categories", "subcategories", "metadata_sha256", "lookup"):
        setattr(metadata, name, getattr(corpus, name).copy())
    for name in ("lexical", "category", "subcategory"):
        setattr(metadata, name, getattr(corpus, name).copy())
    vocabulary = corpus.vectorizer.vocabulary_
    metadata.vocabulary = np.asarray(sorted(vocabulary, key=vocabulary.get))
    metadata.idf = corpus.vectorizer.idf_.copy()
    metadata.components = corpus.svd.components_.copy()
    metadata.title_words = np.asarray([0] + [len(news[name]["title"].split()) for name in corpus.names],
                                      dtype=np.float32)
    return source, metadata, news


def statistics(tree_inputs):
    source, metadata, _ = tree_inputs
    return module.Statistics.fit(source, metadata, np.arange(4), time.monotonic() + 30)


def test_opposite_fold_priors_and_age_exclude_all_heldout_targets(tree_inputs):
    source, metadata, _ = tree_inputs
    original = statistics(tree_inputs)
    source.labels[-3:] = [1, 1, 0]
    source.times[-1] = -10000
    changed = statistics(tree_inputs)
    np.testing.assert_array_equal(changed.tables, original.tables)
    np.testing.assert_array_equal(changed.pooled, original.pooled)
    np.testing.assert_array_equal(changed.first, original.first)
    rows, _ = source.rows(np.array([0, 2]))
    source.labels[rows] = 1 - source.labels[rows]
    changed_own_fold = statistics(tree_inputs)
    np.testing.assert_array_equal(changed_own_fold.tables[1], original.tables[1])


def test_complete_stream_source_order_duplicates_and_metadata_sampling(tree_inputs):
    source, metadata, _ = tree_inputs
    stats = statistics(tree_inputs)
    sequence = module.FeatureSequence(source, metadata, stats, np.arange(4), True, time.monotonic() + 30)
    sampled = sequence[[0, 2, 1, 2]]
    np.testing.assert_array_equal(sampled[1], sampled[3])
    assert sequence.consumed == 0
    expected = sequence[list(range(len(sequence)))]
    actual = np.r_[sequence[:5], sequence[5:]]
    np.testing.assert_array_equal(actual, expected)
    assert sequence.consumed == 11
    np.testing.assert_array_equal(actual[:3, :3], stats.tables[1][source.candidates[:3]])
    assert actual.shape == (11, 16)
    with pytest.raises(ValueError, match="exactly once"):
        sequence[:1]


def test_empty_history_and_unseen_article_age_are_explicit_and_finite(tree_inputs):
    source, metadata, _ = tree_inputs
    stats = statistics(tree_inputs)
    stats.first[4] = np.iinfo(np.int32).max
    sequence = module.FeatureSequence(source, metadata, stats, np.arange(4), False, time.monotonic() + 30)
    features = sequence.features(np.array([2]), np.array([4]))
    assert np.isfinite(features).all()
    assert features[0, 5] == 0
    np.testing.assert_array_equal(features[0, 6:13], np.zeros(7))
    assert features[0, 13] == -1


def test_history_similarity_features_and_read_again_use_real_identities(tree_inputs):
    source, metadata, _ = tree_inputs
    sequence = module.FeatureSequence(source, metadata, statistics(tree_inputs), np.arange(4),
                                      False, time.monotonic() + 30)
    row = sequence.features(np.array([0]), np.array([1]))[0]
    assert row[5] == 2 and row[8] == 1
    assert abs(float(row[10]) - 1) < 1e-6
    expected = float(metadata.lexical[1] @ metadata.lexical[2])
    assert abs(float(row[12]) - expected) < 1e-6


@pytest.mark.parametrize("index", [-1, 11, [0, -1]])
def test_invalid_exposure_positions_fail(tree_inputs, index):
    source, metadata, _ = tree_inputs
    sequence = module.FeatureSequence(source, metadata, statistics(tree_inputs), np.arange(4),
                                      False, time.monotonic() + 30)
    with pytest.raises(ValueError):
        sequence[index]


def test_times_bind_every_raw_identity_day_and_am_pm(tree_inputs, tmp_path):
    source = tree_inputs[0]
    base = datetime(2019, 11, 9).date().toordinal()
    source.days += base
    (tmp_path / "impression_ids.txt").write_text("1\n2\n3\n4\n5\n")
    raw = tmp_path / "behaviors.tsv"
    raw.write_text("".join(
        f"{i + 1}\tu\t11/{9 + int(i == 4)}/2019 12:00:00 {'AM' if i % 2 == 0 else 'PM'}\t\tN1-1 N2-0\n"
        for i in range(5)))
    module.prepare_times(source, tmp_path, raw, time.monotonic() + 30)
    np.testing.assert_array_equal(source.times, [0, 43200, 0, 43200, 86400])
    raw.write_text(raw.read_text().replace("1\tu", "X\tu", 1))
    with pytest.raises(ValueError, match="identity"):
        module.prepare_times(source, tmp_path, raw, time.monotonic() + 30)


def test_complete_native_fit_save_restore_and_ranking_metrics(tree_inputs, tmp_path, monkeypatch):
    source, metadata, _ = tree_inputs
    monkeypatch.setattr(module, "ROUNDS", 2)
    monkeypatch.setattr(module, "PARAMETERS", {
        **module.PARAMETERS, "min_data_in_leaf": 1, "min_data_in_bin": 1,
        "min_sum_hessian_in_leaf": 1e-8, "min_data_per_group": 1, "cat_smooth": 0})
    directory = tmp_path / "fit"
    groups = np.arange(4)
    model, stats = module.fit(source, metadata, groups, directory, time.monotonic() + 60)
    coverage = json.loads((directory / "fit_coverage.json").read_bytes())
    assert coverage["impressions"] == 4 and coverage["exposures"] == 11 and coverage["positives"] == 4
    assert coverage["rounds"] == 2 and coverage["every_candidate_streamed_once"]
    before = module.evaluate(model, source, metadata, stats, np.array([4]), tmp_path / "scores.jsonl",
                             time.monotonic() + 30)
    restored = module.lgb.Booster(model_file=str(directory / "model.txt"))
    after = module.evaluate(restored, source, metadata, stats, np.array([4]), tmp_path / "restored.jsonl",
                            time.monotonic() + 30)
    assert before == after
    assert before["impressions"] == 1 and before["exposures"] == 3 and before["positives"] == 1
    assert before["ranking_tie_dtype"] == "float64"
    assert (tmp_path / "scores.jsonl").read_bytes() == (tmp_path / "restored.jsonl").read_bytes()


def test_metadata_inference_preserves_ids_and_uses_unlabeled_frozen_projection(tree_inputs, tmp_path):
    _, metadata, news = tree_inputs
    path = tmp_path / "news.tsv"
    record = news["N1"]
    title = record["title"]
    path.write_text(f"N1\tsport\tball\t{title}\t{record['abstract']}\turl\t[]\t[]\n"
                    "N5\tnew\tnew\tFootball laptop\tComputer game\turl\t[]\t[]\n")
    stats = statistics(tree_inputs)
    metadata.extend(path)
    stats.extend(metadata)
    assert metadata.lookup["N1"] == 1 and metadata.lookup["N5"] == 5
    assert metadata.category[-1] == 0 and metadata.subcategory[-1] == 0
    assert stats.first[-1] == np.iinfo(np.int32).max
    assert np.isfinite(metadata.lexical).all() and np.isfinite(stats.pooled).all()
    assert stats.pooled[-1, 0] == stats.pooled[0, 1]
    path.write_text(path.read_text().replace(title, "Changed", 1))
    with pytest.raises(ValueError, match="metadata"):
        metadata.extend(path)
