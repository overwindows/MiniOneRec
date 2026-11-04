import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import mind_external_tree_ranker as module
from test_mind_tree_ranker import inputs, tree_inputs


def test_external_native_all_rows_groups_save_restore(tree_inputs, tmp_path, monkeypatch):
    source, metadata, _ = tree_inputs
    monkeypatch.setattr(module, "ROUNDS", 2)
    monkeypatch.setattr(module, "PARAMETERS", {**module.PARAMETERS, "min_child_weight": 0.})
    groups = np.arange(4)
    model, stats = module.fit(source, metadata, groups, tmp_path / "fit", time.monotonic() + 60)
    coverage = json.loads((tmp_path / "fit" / "fit_coverage.json").read_bytes())
    assert coverage["impressions"] == 4 and coverage["exposures"] == 11 and coverage["positives"] == 4
    assert all(item["exposures"] == 11 and item["positives"] == 4 for item in coverage["native_passes"])
    assert coverage["training_candidate_sampling"] == "none"
    before = module.core.evaluate(model, source, metadata, stats, np.array([4]),
                                  tmp_path / "scores.jsonl", time.monotonic() + 30)
    restored = module.Model.load(tmp_path / "fit" / "model.ubj")
    after = module.core.evaluate(restored, source, metadata, stats, np.array([4]),
                                 tmp_path / "restored.jsonl", time.monotonic() + 30)
    assert before == after
    assert (tmp_path / "scores.jsonl").read_bytes() == (tmp_path / "restored.jsonl").read_bytes()


@pytest.mark.parametrize("groups", [[0, 0], [-1], [5], [[0]], [0.5]])
def test_source_impression_guard_rejects_repeated_or_invalid_groups(tree_inputs, groups):
    source, metadata, _ = tree_inputs
    stats = module.core.Statistics.fit(source, metadata, np.arange(4), time.monotonic() + 30)
    with pytest.raises(ValueError, match="Distinct valid"):
        module.core.FeatureSequence(source, metadata, stats, np.asarray(groups), True, time.monotonic() + 30)


def test_external_iterator_rejects_incomplete_pass_and_preserves_binary_target_hash(tmp_path):
    features = np.zeros((6, 16), dtype=np.float32)
    labels = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)
    iterator = module.CachedIterator(features, labels, np.array([3, 3], dtype=np.int32),
                                     tmp_path / "native", time.monotonic() + 30)
    captured = []
    assert iterator.next(lambda **payload: captured.append(payload))
    assert not iterator.next(lambda **payload: captured.append(payload))
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0]["label"], labels)
    np.testing.assert_array_equal(captured[0]["group"], [3, 3])
    iterator.cursor = 1
    with pytest.raises(ValueError, match="Incomplete"):
        iterator.reset()
