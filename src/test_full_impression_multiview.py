import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from full_impression_multiview import MODE, MultiViewRanker, MultiViewReplay
from full_impression_training import fit
from test_full_impression_shared_text import fixture


def test_joint_views_train_with_complete_loss_and_independent_replay(tmp_path):
    terms, initial, history, candidates, groups = fixture()
    torch.manual_seed(42)
    features = F.normalize(torch.randn(9, 128), dim=-1)
    features[0] = 0
    before = features.clone()
    arrays = {"histories.npy": history.numpy(), "candidate_news.npy": candidates.numpy(),
              "candidate_labels.npy": np.asarray([1, 0, 0, 1], dtype=np.uint8),
              "candidate_offsets.npy": np.asarray([0, 2, 4], dtype=np.uint64)}
    model, report = fit(features, np.arange(2), arrays, MODE, tmp_path / "head.pt",
                        time.monotonic() + 30,
                        model_factory=lambda f, mode: MultiViewRanker(f, mode, terms, initial), epoch_count=2)
    assert len(report["epochs"]) == 2 and torch.equal(features, before)
    assert all((row["impressions"], row["exposures"], row["positives"]) == (2, 4, 2)
               for row in report["epochs"])
    assert set(model.state_dict()) == {"words", "queries", "recency", "semantic_project.weight"}
    np.testing.assert_allclose(MultiViewReplay(features, terms, model.state_dict())(history, candidates, groups),
                               model(history, candidates, groups).detach().numpy(), atol=1e-4, rtol=1e-5)
    assert torch.equal(model(history, candidates, groups)[:2], torch.zeros(2))
    assert not np.array_equal(model.words.detach().numpy(), initial)


def test_joint_replay_rejects_nonfinite_projection_and_invalid_semantic_padding():
    terms, initial, _, _, _ = fixture()
    features = torch.zeros((9, 128))
    model = MultiViewRanker(features, MODE, terms, initial)
    state = model.state_dict()
    state["semantic_project.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        MultiViewReplay(features, terms, state)
