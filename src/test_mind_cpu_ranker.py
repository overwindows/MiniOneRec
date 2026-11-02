import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from mind_cpu_ranker import (
    CandidateHistoryRanker, Corpus, Priors, batch_tensors, candidate_rows,
    complete_loss, evaluate, limits, train_epochs, validate_groups,
)


def news():
    return {
        "N1": {"category": "sport", "subcategory": "ball", "title": "Football match win", "abstract": "Sports team scored goals"},
        "N2": {"category": "tech", "subcategory": "computer", "title": "Computer software update", "abstract": "New technology hardware"},
        "N3": {"category": "sport", "subcategory": "ball", "title": "Basketball team game", "abstract": "Player scored points"},
        "N4": {"category": "tech", "subcategory": "computer", "title": "Laptop processor review", "abstract": "Technology computer performance"},
    }


@pytest.fixture
def fixture_source():
    source = SimpleNamespace(
        offsets=np.asarray([0, 3, 5, 7, 10], dtype=np.uint64),
        candidates=np.asarray([1, 2, 2, 1, 4, 2, 3, 1, 2, 4], dtype=np.int32),
        labels=np.asarray([1, 0, 0, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8),
        histories=np.asarray([[0] * 28 + [1, 2], [0] * 30,
                              [0] * 29 + [1], [0] * 28 + [4, 1]], dtype=np.int32),
        folds=np.asarray([0, 1, 0, 1], dtype=np.uint8),
        manifest={"impressions": 4},
        impression_ids=["I1", "I2", "I3", "I4"],
        unchanged=lambda: None,
    )

    def counts(groups):
        rows, _, _ = candidate_rows(source.offsets, groups)
        return {"impressions": len(groups), "exposures": len(rows),
                "positives": int(source.labels[rows].sum())}

    source.counts = counts
    return source


def test_rows_preserve_source_order_duplicates_and_all_labels(fixture_source):
    rows, group_index, sizes = candidate_rows(fixture_source.offsets, np.asarray([2, 0]))
    assert rows.tolist() == [5, 6, 0, 1, 2]
    assert group_index.tolist() == [0, 0, 1, 1, 1]
    assert sizes.tolist() == [2, 3]
    assert fixture_source.candidates[rows].tolist() == [2, 3, 1, 2, 2]


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_group_loss_matches_independent_complete_impressions_and_all_gradients(weight):
    scores = torch.tensor([0.4, -0.2, -0.2, -0.1, 0.6, 0.9], requires_grad=True)
    labels = torch.tensor([1., 0., 0., 0., 1., 1.])
    sizes = torch.tensor([3, 2, 1])
    groups = torch.repeat_interleave(torch.arange(3), sizes)
    expected_ce, expected_bt = [], []
    start = 0
    for size in sizes.tolist():
        score, label = scores[start:start + size], labels[start:start + size]
        expected_ce.append(torch.nn.functional.binary_cross_entropy_with_logits(score, label))
        positive, negative = score[label == 1], score[label == 0]
        if len(positive) and len(negative):
            expected_bt.append(torch.nn.functional.softplus(
                negative[None, :] - positive[:, None]).mean())
        start += size
    expected = torch.stack(expected_ce).mean() + weight * torch.stack(expected_bt).mean()
    actual = complete_loss(scores, labels, groups, sizes, weight)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(scores.grad).all() and torch.all(scores.grad != 0)


def test_no_cross_impression_pairs_or_skipped_one_class_ce():
    labels = torch.tensor([1., 0., 1., 0., 0., 0.])
    scores = torch.tensor([0.3, -0.1, 0.8, 0.2, -0.5, 0.1])
    sizes = torch.tensor([2, 2, 2])
    groups = torch.repeat_interleave(torch.arange(3), sizes)
    before = complete_loss(scores, labels, groups, sizes, 1) - complete_loss(scores, labels, groups, sizes, 0)
    shifted = scores + torch.tensor([7., 7., -3., -3., 4., 4.])
    after = complete_loss(shifted, labels, groups, sizes, 1) - complete_loss(shifted, labels, groups, sizes, 0)
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 100])
def test_candidate_attention_chunk_outputs_and_gradients_match(fixture_source, chunk_size):
    torch.manual_seed(38)
    corpus = Corpus(news())
    priors = Priors(fixture_source, corpus, np.arange(4), time.monotonic() + 60, 10 * 2**30)
    original = CandidateHistoryRanker(corpus, priors.known, dimensions=8, id_dropout=0)
    torch.nn.init.normal_(original.head[-1].weight, std=0.1)
    original.eval()
    chunked = copy.deepcopy(original)
    tensors = batch_tensors(fixture_source, corpus, priors, np.arange(4), False)
    histories, candidates, groups, sizes, features, labels = tensors
    history_vectors = original.news_vectors(histories)
    expected = original._chunk(
        candidates, groups, features, sizes, histories, history_vectors,
        original.key(history_vectors + original.positions[None, :, :]),
        original.value(history_vectors))
    actual = chunked(histories, candidates, groups, sizes, features, chunk_size)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    complete_loss(expected, labels, groups, sizes).backward()
    complete_loss(actual, labels, groups, sizes).backward()
    for left, right in zip(original.parameters(), chunked.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            a = left.grad.to_dense() if left.grad.is_sparse else left.grad
            b = right.grad.to_dense() if right.grad.is_sparse else right.grad
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(actual).all()


def test_own_impression_click_labels_do_not_leak_into_training_priors(fixture_source):
    corpus = Corpus(news())
    deadline = time.monotonic() + 60
    first = Priors(fixture_source, corpus, np.arange(4), deadline, 10 * 2**30)
    rows, _, _ = candidate_rows(fixture_source.offsets, np.asarray([0]))
    before = first.features(fixture_source.candidates[rows], np.zeros(len(rows), dtype=np.uint8))
    fixture_source.labels[:3] = [0, 1, 0]
    second = Priors(fixture_source, corpus, np.arange(4), deadline, 10 * 2**30)
    after = second.features(fixture_source.candidates[rows], np.zeros(len(rows), dtype=np.uint8))
    np.testing.assert_array_equal(before, after)
    assert first.coverage == {"exposures": 10, "positives": 4}


def test_encoder_serialization_and_unknown_article_features(tmp_path, fixture_source):
    corpus = Corpus(news())
    corpus.save(tmp_path)
    restored = Corpus.load(tmp_path)
    np.testing.assert_array_equal(corpus.lexical, restored.lexical)
    addition = {"N5": {"category": "new", "subcategory": "new", "title": "New computer sports", "abstract": "Football technology"}}
    corpus.extend(addition)
    restored.extend(addition)
    np.testing.assert_allclose(corpus.lexical, restored.lexical, atol=1e-7, rtol=1e-6)
    assert restored.category[-1] == 0
    base = Corpus(news())
    priors = Priors(fixture_source, base, np.arange(4), time.monotonic() + 60, 10 * 2**30)
    model = CandidateHistoryRanker(base, priors.known, dimensions=8, id_dropout=0)
    priors.extend(restored)
    model.extend_features(restored, priors.known)
    model.eval()
    with torch.no_grad():
        score = model(torch.zeros((1, 30), dtype=torch.long),
                      torch.tensor([5]), torch.tensor([0]), torch.tensor([1]),
                      torch.from_numpy(priors.features(np.asarray([5]))))
    assert torch.isfinite(score).all() and model.id_map[5] == 0


def test_changed_shared_article_metadata_fails_explicitly():
    corpus = Corpus(news())
    changed = news()
    changed["N1"]["title"] = "Changed article"
    with pytest.raises(ValueError, match="changed its frozen metadata"):
        corpus.extend(changed)


def test_heldout_training_labels_never_enter_prior_fit(fixture_source):
    corpus = Corpus(news())
    groups = np.asarray([0, 1, 2])
    first = Priors(fixture_source, corpus, groups, time.monotonic() + 60, 10 * 2**30)
    fixture_source.labels[7:] = [0, 1, 1]
    second = Priors(fixture_source, corpus, groups, time.monotonic() + 60, 10 * 2**30)
    np.testing.assert_array_equal(first.pooled, second.pooled)
    for left, right in zip(first.tables, second.tables):
        np.testing.assert_array_equal(left, right)


def test_checkpoint_backward_preserves_training_dropout_rng(fixture_source):
    torch.manual_seed(73)
    corpus = Corpus(news())
    priors = Priors(fixture_source, corpus, np.arange(4), time.monotonic() + 60, 10 * 2**30)
    model = CandidateHistoryRanker(corpus, priors.known, dimensions=8, id_dropout=0.5)
    torch.nn.init.normal_(model.head[-1].weight, std=0.1)
    tensors = batch_tensors(fixture_source, corpus, priors, np.arange(4), True)
    scores = model(*tensors[:-1], chunk_size=2)
    state = torch.get_rng_state().clone()
    complete_loss(scores, tensors[-1], tensors[2], tensors[3]).backward()
    assert torch.equal(state, torch.get_rng_state())
    assert model.id_embedding.weight.grad.is_sparse
    assert torch.isfinite(model.id_embedding.weight.grad.coalesce().values()).all()


@pytest.mark.parametrize("groups", [np.asarray([0, 0]), np.asarray([]),
                                  np.asarray([-1]), np.asarray([4]), np.asarray([0.5])])
def test_duplicate_empty_and_out_of_range_impressions_rejected(fixture_source, groups):
    with pytest.raises(ValueError, match="unique"):
        validate_groups(fixture_source, groups)


def test_evaluation_ties_and_undefined_groups(fixture_source, tmp_path):
    corpus = Corpus(news())
    priors = Priors(fixture_source, corpus, np.arange(4), time.monotonic() + 60, 10 * 2**30)

    class TiedModel:
        def eval(self):
            pass

        def __call__(self, histories, news, groups, sizes, features):
            return torch.zeros(len(news))

    path = tmp_path / "scores.jsonl"
    result = evaluate(fixture_source, corpus, priors, np.arange(4), TiedModel(),
                      time.monotonic() + 60, 10 * 2**30, path)
    assert result["auc"] == 0.5
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line[0] for line in lines] == fixture_source.impression_ids
    assert [len(line[1]) for line in lines] == [3, 2, 2, 3]
    assert all(score == 0 for line in lines for score in line[1])
    fixture_source.labels[:3] = 0
    with pytest.raises(ValueError, match="undefined"):
        evaluate(fixture_source, corpus, priors, np.arange(4), TiedModel(),
                 time.monotonic() + 60, 10 * 2**30)


def test_omitted_epoch_impression_rejects_before_model_publication(tmp_path, fixture_source, monkeypatch):
    corpus = Corpus(news())
    priors = Priors(fixture_source, corpus, np.arange(4), time.monotonic() + 60, 10 * 2**30)
    monkeypatch.setattr(np.random, "default_rng",
                        lambda seed: SimpleNamespace(permutation=lambda groups: groups[:-1]))
    with pytest.raises(ValueError, match="cover every selected source row"):
        train_epochs(fixture_source, corpus, priors, np.arange(4), tmp_path, 1, 42,
                     time.monotonic() + 60, 10 * 2**30)
    assert not (tmp_path / "model.pt").exists()
    assert not (tmp_path / "epoch_coverage.json").exists()


def test_complete_two_epoch_training_persists_proofs_and_reload(tmp_path, fixture_source):
    corpus = Corpus(news())
    deadline = time.monotonic() + 120
    priors = Priors(fixture_source, corpus, np.arange(4), deadline, 10 * 2**30)
    model, ledgers = train_epochs(
        fixture_source, corpus, priors, np.arange(4), tmp_path, 2, 42, deadline, 10 * 2**30)
    assert len(ledgers) == 2
    for ledger in ledgers:
        assert (ledger["impressions"], ledger["exposures"], ledger["positives"], ledger["optimizer_steps"]) == (4, 10, 4, 1)
    assert json.loads((tmp_path / "epoch_coverage.json").read_bytes()) == ledgers
    restored = Corpus.load(tmp_path)
    restored_priors = Priors.load(tmp_path)
    restored_model = CandidateHistoryRanker(restored, restored_priors.known)
    restored_model.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    model.eval()
    restored_model.eval()
    tensors = batch_tensors(fixture_source, corpus, priors, np.arange(4), False)
    with torch.no_grad():
        torch.testing.assert_close(model(*tensors[:-1]), restored_model(*tensors[:-1]))
    metrics = evaluate(fixture_source, corpus, priors, np.arange(4), model, deadline, 10 * 2**30)
    assert (metrics["impressions"], metrics["exposures"], metrics["positives"]) == (4, 10, 4)
    scores = model(*tensors[:-1]).detach().numpy()
    aucs = []
    for index in range(4):
        start, end = fixture_source.offsets[index:index + 2]
        aucs.append(roc_auc_score(fixture_source.labels[start:end], scores[start:end]))
    assert metrics["auc"] == pytest.approx(np.mean(aucs), abs=1e-12)


def test_invalid_group_metadata_and_limits_fail():
    with pytest.raises(ValueError, match="contiguous"):
        complete_loss(torch.zeros(2), torch.tensor([0., 1.]), torch.tensor([1, 0]), torch.tensor([1, 1]))
    with pytest.raises(TimeoutError):
        limits(time.monotonic() - 1, 10 * 2**30)
    with pytest.raises(MemoryError):
        limits(time.monotonic() + 30, 1)
