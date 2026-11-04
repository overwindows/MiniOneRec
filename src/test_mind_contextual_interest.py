import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import mind_contextual_interest as module
from full_impression_ranker import full_impression_loss
from mind_cpu_ranker import Priors, candidate_rows
from test_mind_lexical_glm import inputs


def build(inputs):
    source, corpus, _, _ = inputs
    source.days = np.zeros(len(source.histories), dtype=np.int32)
    source.folds = np.arange(len(source.histories), dtype=np.uint8) % 2
    vectors = np.random.default_rng(42).normal(size=(len(corpus.names) + 1, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors[0] = 0
    priors = Priors(source, corpus, np.arange(len(source.days)), time.monotonic() + 30, module.RSS_LIMIT)
    torch.manual_seed(42)
    model = module.CandidateInterest(corpus, vectors, priors, Path("public"))
    model.prior_fold = 0
    return model


def test_complete_loss_updates_interest_not_public_vectors(inputs):
    source = inputs[0]
    model = build(inputs)
    before = model.features.clone()
    rows, groups, _ = candidate_rows(source.offsets, np.arange(4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        scores = model(torch.from_numpy(source.histories[:4].astype(np.int64)),
                       torch.from_numpy(source.candidates[rows].astype(np.int64)), torch.from_numpy(groups))
        loss = full_impression_loss(scores, torch.from_numpy(source.labels[rows].astype(np.float32)),
                                    torch.from_numpy(groups), 4)
        loss.backward()
        optimizer.step()
    assert model.projection.weight.grad.abs().sum() > 0
    assert model.recency.grad.abs().sum() > 0
    assert not model.features.requires_grad and torch.equal(model.features, before)


def test_cached_scores_preserve_duplicate_order_and_zero_padding(inputs):
    model = build(inputs).eval()
    candidates, groups = np.array([2, 1, 2]), np.zeros(3, dtype=np.int64)
    before = model.scores(inputs[0].histories[:1], candidates, groups)
    model.prepare_inference()
    np.testing.assert_allclose(model.scores(inputs[0].histories[:1], candidates, groups), before, atol=1e-5, rtol=0)
    assert torch.equal(model.cached[0], torch.zeros(64))
    assert before[0] == before[2]
    model.train()
    assert model.cached is None


def test_empty_history_starts_at_exact_pooled_prior(inputs):
    model = build(inputs).eval()
    scores = model.scores(np.zeros((1, 30), dtype=np.int32), np.array([1, 2]), np.array([0, 0]))
    np.testing.assert_array_equal(scores, model.priors.pooled[[1, 2], 0])


def test_bucketed_attention_matches_reference_values_and_gradients():
    torch.manual_seed(4)
    histories = torch.randn(4, 30, 64, requires_grad=True)
    candidates = torch.randn(11, 64, requires_grad=True)
    groups = torch.tensor([2, 0, 3, 0, 2, 3, 3, 0, 0, 3, 2])
    mask = torch.ones(4, 30, dtype=torch.bool)
    mask[0, :20] = False
    mask[2] = False
    recency = torch.randn(30, requires_grad=True)
    reference_history, reference_mask = histories[groups], mask[groups]
    logits = 8 * torch.einsum("nd,nhd->nh", candidates, reference_history) + recency
    logits = logits.masked_fill(~reference_mask, -torch.inf)
    logits = torch.where(reference_mask.any(dim=1, keepdim=True), logits, torch.zeros_like(logits))
    attention = torch.softmax(logits, dim=1) * reference_mask
    expected = torch.nn.functional.normalize((reference_history * attention[:, :, None]).sum(1), dim=1)
    actual = module.conditional_context(histories, candidates, groups, mask, recency)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    weight = torch.randn_like(actual)
    expected_grad = torch.autograd.grad((expected * weight).sum(), (histories, candidates, recency), retain_graph=True)
    actual_grad = torch.autograd.grad((actual * weight).sum(), (histories, candidates, recency))
    for left, right in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-4)


def test_training_requires_fold_and_extension_requires_inference(inputs):
    model = build(inputs)
    model.prior_fold = None
    with pytest.raises(ValueError, match="opposite-fold"):
        model(torch.zeros((1, 30), dtype=torch.long), torch.tensor([1]), torch.tensor([0]))
    with pytest.raises(ValueError, match="inference"):
        model.extend(inputs[3])


def test_extension_uses_frozen_contextual_encoder_without_parameter_growth(inputs, monkeypatch):
    model = build(inputs).eval()
    before = {name: value.shape for name, value in model.named_parameters()}

    class Encoder:
        def __init__(self, path):
            pass

        def encode(self, texts):
            vectors = np.zeros((len(texts), 384), dtype=np.float32)
            vectors[:, 0] = 1
            return vectors

    monkeypatch.setattr(module, "PublicEncoder", Encoder)
    model.extend({"N5": {"category": "never", "subcategory": "never",
                          "title": "Novel title", "abstract": "Novel abstract"}})
    assert len(model.features) == 6 and model.features[5, 0] == 1
    assert before == {name: value.shape for name, value in model.named_parameters()}
    assert np.isfinite(model.scores(np.zeros((1, 30), dtype=np.int32), np.array([5]), np.array([0]))).all()


def test_tiny_complete_fit_restore_and_epoch_ledgers(inputs, tmp_path, monkeypatch):
    source, corpus, _, _ = inputs
    initial = build(inputs)
    public = tmp_path / "vectors.npz"
    np.savez(public, vectors=initial.features.numpy(), names=np.asarray(corpus.names))
    proof = {"vectors_sha256": module.digest(public), "revision": module.REVISION, "dimensions": 384,
             "maximum_tokens": 64, "text": "title_only", "dtype": "float32",
             "pooling": "masked_mean_then_l2_normalize", "click_labels_read": False,
             "dev_or_test_contents_read": False, "news_sha256": module.NEWS_SHA, "corpus_sha256": module.CORPUS_SHA,
             "source_sha256": module.digest(Path(module.__file__).with_name("mind_public_contextual.py"))}
    public.with_suffix(".json").write_text(json.dumps(proof))
    arrays = {name: getattr(source, field) for field, name in (
        ("offsets", "candidate_offsets.npy"), ("candidates", "candidate_news.npy"),
        ("labels", "candidate_labels.npy"), ("histories", "histories.npy"))}
    model = module.fit(source, arrays, corpus, public, np.arange(4), tmp_path / "fit", time.monotonic() + 60)
    ledgers = json.loads((tmp_path / "fit" / "epoch_coverage.json").read_bytes())
    assert len(ledgers) == 2
    assert all(row["impressions"] == 4 and row["exposures"] == 11 and row["positives"] == 4 for row in ledgers)
    model.eval()
    model.prepare_inference()
    before = model.scores(source.histories[:1], np.array([1, 2]), np.array([0, 0]))
    restored = module.restore(tmp_path / "fit", corpus, public)
    np.testing.assert_array_equal(restored.scores(source.histories[:1], np.array([1, 2]), np.array([0, 0])), before)
