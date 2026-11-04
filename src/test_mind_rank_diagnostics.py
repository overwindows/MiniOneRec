from pathlib import Path
import json
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from mind_rank_diagnostics import analyze, candidate_support, score_records
from test_mind_lexical_glm import inputs


def test_full_pair_partition_and_nonnegative_mixture_bound(inputs):
    source = inputs[0]
    first = [[3, [0.1, 0.9, 0.2]], [4, [0.8, 0.2, 0.1]]]
    second = [[3, [0.8, 0.1, 0.9]], [4, [0.1, 0.2, 0.9]]]
    result = analyze(source, np.arange(2), np.array([3, 4]), 5,
                     {"first": first, "second": second}, time.monotonic() + 60, 3758096384)
    assert result["evaluation_coverage"] == {"impressions": 2, "exposures": 6, "positives": 2}
    assert result["model_auc"] == {"first": .5, "second": .5}
    assert result["candidate_exposures_without_earlier_candidate_exposure"] == 2
    assert result["positives_without_earlier_candidate_exposure"] == 1
    assert result["nonnegative_convex_score_mixture_pairwise_oracle_upper_bound"] == 1
    assert result["paired_comparison"]["second_minus_first"]["impressions_improved"] == 1
    for name in ("first", "second"):
        assert sum(b["macro_auc_contribution"][name] for b in
                   result["exposure_support_pair_contributions"].values()) == .5


def test_support_uses_only_fitting_candidates_not_heldout_targets(inputs):
    source = inputs[0]
    expected, _ = candidate_support(source, np.arange(2), 5, time.monotonic() + 60, 3758096384)
    source.candidates[-3:] = [4, 4, 4]
    source.labels[-3:] = [1, 1, 0]
    actual, _ = candidate_support(source, np.arange(2), 5, time.monotonic() + 60, 3758096384)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, [False, True, True, True, False])


@pytest.mark.parametrize("kind", ["missing", "extra", "reordered", "width", "nonfinite", "overlap"])
def test_incomplete_or_invalid_population_rejected(inputs, kind):
    source = inputs[0]
    records = [[4, [0., 0., 1.]]]
    fitting = np.arange(2)
    if kind == "missing":
        records = []
    elif kind == "extra":
        records *= 2
    elif kind == "reordered":
        records[0][0] = 3
    elif kind == "width":
        records[0][1].pop()
    elif kind == "nonfinite":
        records[0][1][0] = np.nan
    else:
        fitting = np.array([0, 4])
    with pytest.raises(ValueError):
        analyze(source, fitting, np.array([4]), 5, {"model": records},
                time.monotonic() + 60, 3758096384)


def test_sharded_score_reader_preserves_positions_and_strict_schema(tmp_path):
    paths = [tmp_path / "one.jsonl", tmp_path / "two.jsonl"]
    for path, record in zip(paths, [[7, [1., 2.]], [8, [3., 4.]]]):
        path.write_text(json.dumps(record) + "\n")
    assert list(score_records(paths)) == [[7, [1., 2.]], [8, [3., 4.]]]
    paths[1].write_text("[true, [0, 1]]\n")
    with pytest.raises(ValueError):
        list(score_records(paths))
