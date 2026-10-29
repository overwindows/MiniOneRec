import json
import time

import numpy as np
import pytest
from scipy import sparse
import torch

from full_impression_shared_text import MODE, SharedTextRanker, SharedTextReplay
from full_impression_training import fit
from full_impression_shared_text_training import build_lexical


def fixture():
    rng = np.random.default_rng(42)
    terms = sparse.csr_matrix(rng.uniform(size=(9, 12)).astype(np.float32))
    terms.data[:terms.indptr[1]] = 0
    terms.eliminate_zeros()
    initial = rng.normal(0, .05, (12, 64)).astype(np.float32)
    history = torch.zeros((2, 30), dtype=torch.long)
    history[1, -3:] = torch.tensor([1, 2, 3])
    candidates, groups = torch.tensor([4, 5, 6, 7]), torch.tensor([0, 0, 1, 1])
    return terms, initial, history, candidates, groups


def test_complete_shared_text_training_and_independent_sparse_replay(tmp_path):
    terms, initial, history, candidates, groups = fixture()
    features = torch.zeros((9, 1))
    factory = lambda f, mode: SharedTextRanker(f, mode, terms, initial)
    arrays = {"histories.npy": history.numpy(), "candidate_news.npy": candidates.numpy(),
              "candidate_labels.npy": np.asarray([1, 0, 0, 1], dtype=np.uint8),
              "candidate_offsets.npy": np.asarray([0, 2, 4], dtype=np.uint64)}
    before = terms.data.copy()
    model, report = fit(features, np.arange(2), arrays, MODE, tmp_path / "head.pt",
                        time.monotonic() + 30, model_factory=factory)
    assert len(report["epochs"]) == 3
    assert all((row["impressions"], row["exposures"], row["positives"]) == (2, 4, 2)
               for row in report["epochs"])
    assert set(model.state_dict()) == {"words", "queries", "recency"}
    np.testing.assert_array_equal(terms.data, before)
    np.testing.assert_allclose(SharedTextReplay(terms, model.state_dict())(history, candidates, groups),
                               model(history, candidates, groups).detach().numpy(), atol=1e-4, rtol=1e-5)
    assert torch.equal(model(history, candidates, groups)[:2], torch.zeros(2))
    assert not np.array_equal(model.words.detach().numpy(), initial)


def test_invalid_padding_and_word_checkpoint_fail_closed():
    terms, initial, history, candidates, groups = fixture()
    model = SharedTextRanker(torch.zeros((9, 1)), MODE, terms, initial)
    history[0, 0] = 1
    with pytest.raises(ValueError, match="padding"):
        model(history, candidates, groups)
    state = model.state_dict()
    state["words"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        SharedTextReplay(terms, state)


def test_training_title_only_lexical_initialization_reproduces_every_bit(tmp_path):
    records = [{"news_id": f"N{row}", "title": f"token{row % 80} common token{(row + 1) % 80} topic{row % 17}",
                "abstract": "not used", "category": "news", "subcategory": "world"} for row in range(200)]
    source = tmp_path / "news.metadata.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    (tmp_path / "news.ids.json").write_text(json.dumps([row["news_id"] for row in records]))
    _, terms, initial = build_lexical(tmp_path, time.monotonic() + 30, expected_articles=200)
    for row in records:
        row["abstract"] = "different deliberately unused abstract"
    source.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    _, replayed, repeated = build_lexical(tmp_path, time.monotonic() + 30, expected_articles=200)
    np.testing.assert_array_equal(initial.view(np.uint32), repeated.view(np.uint32))
    np.testing.assert_array_equal(terms.data.view(np.uint32), replayed.data.view(np.uint32))
    np.testing.assert_array_equal(terms.indices, replayed.indices)
    np.testing.assert_array_equal(terms.indptr, replayed.indptr)
    assert terms[0].nnz == 0 and np.all(np.diff(terms.indptr) <= 32)
    (tmp_path / "news.ids.json").write_text(json.dumps(list(reversed([row["news_id"] for row in records]))))
    with pytest.raises(ValueError, match="identity"):
        build_lexical(tmp_path, time.monotonic() + 30, expected_articles=200)
