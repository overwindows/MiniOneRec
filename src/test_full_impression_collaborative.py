import numpy as np
import pytest
import torch
import torch.nn.functional as F

from full_impression_collaborative import CollaborativeArticleRanker, CollaborativeReplay
from full_impression_ranker import full_impression_loss
from full_impression_training import fit


def fixture():
    torch.manual_seed(42)
    features = F.normalize(torch.randn(9, 128), dim=-1)
    features[0] = 0
    histories = torch.zeros((2, 30), dtype=torch.long)
    histories[1, -3:] = torch.tensor([1, 2, 3])
    return features, histories, torch.tensor([4, 5, 6, 7]), torch.tensor([0, 0, 1, 1])


def test_collaborative_training_replay_padding_and_frozen_semantics():
    features, history, candidates, groups = fixture()
    before = features.clone()
    model = CollaborativeArticleRanker(features)
    assert torch.equal(model(history, candidates, groups)[:2], torch.zeros(2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    loss = full_impression_loss(model(history, candidates, groups), torch.tensor([1., 0, 0, 1]), groups, 2)
    loss.backward()
    optimizer.step()
    assert not torch.any(model.residual.weight[0]) and not torch.any(model.article_bias.weight[0])
    assert torch.equal(before, features) and not features.requires_grad
    state = model.state_dict()
    assert set(state) == {"residual.weight", "article_bias.weight", "recency"}
    replay = CollaborativeReplay(features, state)
    np.testing.assert_allclose(replay(history, candidates, groups),
                               model(history, candidates, groups).detach().numpy(), atol=1e-5, rtol=1e-5)
    assert torch.any(model.residual.weight[1:] != 0)
    assert not torch.any(model.residual.weight[8]) and not torch.any(model.article_bias.weight[8])


def test_collaborative_inputs_keep_the_same_strict_padding_contract():
    features, history, candidates, groups = fixture()
    history[0, 0] = 1
    with pytest.raises(ValueError, match="padding"):
        CollaborativeArticleRanker(features)(history, candidates, groups)


def test_replay_rejects_nonfinite_or_padding_updates():
    features, _, _, _ = fixture()
    state = CollaborativeArticleRanker(features).state_dict()
    state["residual.weight"][0, 0] = 1
    with pytest.raises(ValueError):
        CollaborativeReplay(features, state)


def test_shared_fit_factory_preserves_three_complete_epochs(tmp_path):
    features, history, candidates, _ = fixture()
    arrays = {"histories.npy": history.numpy(), "candidate_news.npy": candidates.numpy(),
              "candidate_labels.npy": np.asarray([1, 0, 0, 1], dtype=np.uint8),
              "candidate_offsets.npy": np.asarray([0, 2, 4], dtype=np.uint64)}
    import time
    _, report = fit(features, np.arange(2), arrays, "collaborative_mean", tmp_path / "head.pt",
                    time.monotonic() + 30, model_factory=CollaborativeArticleRanker)
    assert report["mode"] == "collaborative_mean"
    assert len(report["epochs"]) == 3
    assert all((row["impressions"], row["exposures"], row["positives"], row["optimizer_steps"]) ==
               (2, 4, 2, 1) for row in report["epochs"])


def test_fixed_semantics_and_mode_guards():
    features, _, _, _ = fixture()
    with pytest.raises(ValueError):
        CollaborativeArticleRanker(features.double())
    with pytest.raises(ValueError):
        CollaborativeArticleRanker(features, "candidate_attention")
    features[0, 0] = 1
    with pytest.raises(ValueError):
        CollaborativeArticleRanker(features)
