from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import MINDPointwiseSFTDataset


def fixture(tmp_path, candidates):
    news = tmp_path / "news.tsv"
    news.write_text(
        "".join(f"{name}\t{category}\tsub\t{name}\tabs\tu\t[]\t[]\n"
                for name, category in (("P", "a"), ("Q", "a"), ("H", "a"), ("J", "a"), ("E", "b"))),
        encoding="utf-8",
    )
    behaviors = tmp_path / "behaviors.tsv"
    behaviors.write_text(f"1\tU1\t11/14/2019 1:00:00 PM\tP\t{candidates}\n", encoding="utf-8")
    return behaviors, news


def load(paths, **kwargs):
    behaviors, news = paths
    dataset = MINDPointwiseSFTDataset(str(behaviors), str(news), None, **kwargs)
    return [(row["candidate_id"], row["label"]) for row in dataset.samples]


def test_uniform_retains_negative_in_one_positive_same_category_impression(tmp_path):
    paths = fixture(tmp_path, "P-1 H-0 J-0")
    assert load(paths) == [("P", 1)]
    rows = load(paths, negative_sampling="uniform")
    assert len(rows) == 2 and sum(label == 0 for _, label in rows) == 1


def test_uniform_fills_budget_without_replacement(tmp_path):
    paths = fixture(tmp_path, "P-1 Q-1 H-0 J-0")
    assert len(load(paths)) == 3
    rows = load(paths, negative_sampling="uniform")
    assert len(rows) == 4 and len(set(rows)) == 4
    assert len(load(paths, negative_sampling="uniform", neg_ratio=10.)) == 4


def test_uniform_has_support_in_both_categories_and_ignores_hard_ratio(tmp_path):
    paths = fixture(tmp_path, "P-1 H-0 J-0 E-0")
    observed = {
        name for seed in range(64)
        for name, label in load(paths, negative_sampling="uniform", seed=seed) if not label
    }
    assert observed == {"H", "J", "E"}
    assert load(paths, negative_sampling="uniform", hard_neg_ratio=0.) == load(
        paths, negative_sampling="uniform", hard_neg_ratio=1.,
    )


def test_legacy_default_is_explicitly_reproducible_and_unknown_mode_fails(tmp_path):
    paths = fixture(tmp_path, "P-1 Q-1 H-0 J-0 E-0")
    assert load(paths) == load(paths, negative_sampling="legacy")
    assert load(paths, negative_sampling="uniform") == load(paths, negative_sampling="uniform")
    with pytest.raises(ValueError, match="negative_sampling"):
        load(paths, negative_sampling="unknown")
