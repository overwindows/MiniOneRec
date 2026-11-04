from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from full_impression_ranker import full_impression_loss
from mind_cpu_ranker import Priors, candidate_rows
import mind_pretrained_news
from mind_pretrained_news import PretrainedNewsRanker, encode_titles, load_public_glove, title_vocabulary, tokenize
from test_mind_lexical_glm import inputs


def model_for(inputs):
    _, corpus, _, news = inputs
    vocabulary = title_vocabulary(corpus, news)
    tokens = encode_titles(corpus, news, vocabulary)
    embeddings = np.random.default_rng(42).normal(0, .1, (len(vocabulary), 300)).astype(np.float32)
    embeddings[0] = 0
    source = inputs[0]
    source.days = np.zeros(len(source.histories), dtype=np.int32)
    source.folds = np.arange(len(source.histories), dtype=np.uint8) % 2
    priors = Priors(source, corpus, np.arange(len(source.days)), time.monotonic() + 30, 3758096384)
    torch.manual_seed(42)
    model = PretrainedNewsRanker(corpus, tokens, embeddings, vocabulary, priors)
    model.prior_fold = 0
    return model


def test_vocabulary_title_order_and_frozen_oov_mapping(inputs):
    _, corpus, _, news = inputs
    vocabulary = title_vocabulary(corpus, news)
    tokens = encode_titles(corpus, news, vocabulary)
    assert tokenize("New, CHIP!") == ["new", ",", "chip", "!"]
    assert [vocabulary[i] for i in tokens[1] if i] == tokenize(news["N1"]["title"])
    assert not tokens[0].any()


def test_complete_listwise_gradients_reach_cnn_not_frozen_public_vectors(inputs):
    source = inputs[0]
    model = model_for(inputs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    rows, groups, _ = candidate_rows(source.offsets, np.arange(4))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        scores = model(torch.from_numpy(source.histories[:4].astype(np.int64)),
                       torch.from_numpy(source.candidates[rows].astype(np.int64)), torch.from_numpy(groups))
        loss = full_impression_loss(scores, torch.from_numpy(source.labels[rows].astype(np.float32)),
                                    torch.from_numpy(groups), 4)
        loss.backward()
        optimizer.step()
    assert not model.word_vectors.requires_grad
    assert model.word_projection.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.convolution.weight.grad).all()
    assert model.convolution.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.history_attention.in_proj_weight.grad).all()
    assert model.history_attention.in_proj_weight.grad.abs().sum() > 0


def test_cached_scores_preserve_duplicate_order_and_padding(inputs):
    model = model_for(inputs)
    with torch.no_grad():
        model.head[-1].weight.fill_(.01)
    model.eval()
    candidates, groups = np.array([2, 1, 2]), np.zeros(3, dtype=np.int64)
    histories = inputs[0].histories[:1]
    original = model.scores(histories, candidates, groups)
    model.prepare_inference()
    cached = model.scores(histories, candidates, groups)
    np.testing.assert_allclose(cached, original, rtol=0, atol=1e-5)
    assert cached[0] == cached[2]
    assert torch.equal(model.cached[0], torch.zeros(64))
    model.train()
    assert model.cached is None


def test_empty_history_and_unknown_title_are_finite_without_parameter_growth(inputs):
    model = model_for(inputs)
    model.eval()
    before = {name: parameter.shape for name, parameter in model.named_parameters()}
    model.extend({"N5": {"category": "never", "subcategory": "never",
                          "title": "zzzoovterm", "abstract": "zzzoovterm"}})
    scores = model.scores(np.zeros((1, 30), dtype=np.int32), np.array([5]), np.array([0]))
    assert np.isfinite(scores).all()
    assert scores[0] == model.priors.pooled[5, 0]
    assert before == {name: parameter.shape for name, parameter in model.named_parameters()}
    assert model.tokens[-1, 0].item() == 1


def test_shared_metadata_and_training_extension_fail(inputs):
    model = model_for(inputs)
    with pytest.raises(ValueError, match="inference"):
        model.extend(inputs[3])
    model.eval()
    with pytest.raises(ValueError, match="metadata"):
        model.extend({"N1": dict(inputs[3]["N1"], title="Changed")})


def test_word_order_is_not_reduced_to_bag_features(inputs):
    model = model_for(inputs)
    model.eval()
    original = model.encode(torch.tensor([1])).clone()
    model.tokens[1, :4] = model.tokens[1, :4].flip(0)
    changed = model.encode(torch.tensor([1]))
    assert not torch.allclose(original, changed)


@pytest.mark.parametrize("footer", [False, True])
def test_public_archive_accepts_exact_vocabulary_and_optional_reserved_footer(tmp_path, monkeypatch, footer):
    monkeypatch.setattr(mind_pretrained_news, "GLOVE_WORD_ROWS", 2)
    path = tmp_path / "vectors.zip"
    values = " ".join([".25"] * 300)
    text = f"known {values}\nother {values}\n"
    if footer:
        text += f"<unk> {values}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("glove.6B.300d.txt", text)
    embeddings, coverage = load_public_glove(path, ["<pad>", "<unk>", "known", "missing"])
    assert coverage["public_word_rows"] == 2
    assert coverage["reserved_unknown_rows"] == int(footer)
    assert coverage["matched_words"] == 1
    np.testing.assert_array_equal(embeddings[0], np.zeros(300))
    np.testing.assert_array_equal(embeddings[1:], np.full((3, 300), .25))


@pytest.mark.parametrize("words", [
    ["known"], ["known", "known"], ["known", "other", "extra"],
    ["<unk>", "known"], ["known", "other", "<unk>", "<unk>"]])
def test_public_archive_rejects_truncation_duplicates_and_unexpected_rows(tmp_path, monkeypatch, words):
    monkeypatch.setattr(mind_pretrained_news, "GLOVE_WORD_ROWS", 2)
    path = tmp_path / "vectors.zip"
    values = " ".join([".25"] * 300)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("glove.6B.300d.txt", "".join(f"{word} {values}\n" for word in words))
    with pytest.raises(ValueError):
        load_public_glove(path, ["<pad>", "<unk>", "known"])


def test_prior_baseline_uses_only_opposite_fold_and_requires_explicit_fold(inputs):
    model = model_for(inputs)
    source = inputs[0]
    candidates = torch.tensor([1, 2])
    groups = torch.zeros(2, dtype=torch.long)
    histories = torch.from_numpy(source.histories[:1].astype(np.int64))
    for fold in (0, 1):
        model.prior_fold = fold
        torch.testing.assert_close(model(histories, candidates, groups),
                                   model.fold_logits[1 - fold, candidates])
    model.prior_fold = None
    with pytest.raises(ValueError, match="explicit source fold"):
        model(histories, candidates, groups)
    model.eval()
    torch.testing.assert_close(model(histories, candidates, groups), model.pooled_logits[candidates])


def test_complete_two_epoch_fit_restore_and_source_coverage(inputs, tmp_path):
    from mind_pretrained_training import fit, restore
    import json
    source, corpus, _, news = inputs
    model = model_for(inputs)
    public = tmp_path / "public.npz"
    np.savez(public, news_names=np.asarray(corpus.names), tokens=model.tokens.numpy().astype(np.int32),
             embeddings=model.word_vectors.numpy(), vocabulary=np.asarray(model.vocabulary))
    arrays = {"candidate_offsets.npy": source.offsets, "candidate_news.npy": source.candidates,
              "candidate_labels.npy": source.labels, "histories.npy": source.histories}
    groups = np.arange(len(source.days))
    output = tmp_path / "fit"
    fitted = fit(source, arrays, corpus, public, groups, output, time.monotonic() + 30)
    coverage = json.loads((output / "epoch_coverage.json").read_text())
    assert len(coverage) == 2
    for record in coverage:
        assert all(record[key] == value for key, value in source.counts(groups).items())
        assert record["optimizer_steps"] == 2
    fitted.eval()
    restored = restore(output, corpus, public)
    rows, local, _ = candidate_rows(source.offsets, groups)
    np.testing.assert_allclose(fitted.scores(source.histories, source.candidates[rows], local),
                               restored.scores(source.histories, source.candidates[rows], local),
                               rtol=0, atol=1e-5)
    restored.train()
    restored.prior_fold = 0
    with pytest.raises(ValueError, match="opposite-fold"):
        restored(torch.from_numpy(source.histories.astype(np.int64)),
                 torch.from_numpy(source.candidates[rows].astype(np.int64)), torch.from_numpy(local))
