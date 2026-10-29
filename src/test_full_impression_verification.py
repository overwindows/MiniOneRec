import numpy as np
import pytest
import torch
import torch.nn.functional as F

from full_impression_ranker import FrozenArticleRanker
from full_impression_verification import CachedReplay, check_scores, metrics
from mind_utils import auc_score, mrr_score, ndcg_score


@pytest.mark.parametrize("ties", [False, True])
def test_independent_metrics_match_existing_official_semantics(ties):
    generator = np.random.default_rng(42)
    for size in (2, 4, 17, 300):
        labels = generator.integers(0, 2, size)
        labels[0], labels[-1] = 0, 1
        scores = generator.integers(0, 3, size).astype(float) if ties else generator.normal(size=size)
        expected = (auc_score(labels, scores), mrr_score(labels, scores),
                    ndcg_score(labels, scores, 5), ndcg_score(labels, scores, 10))
        np.testing.assert_allclose(metrics(labels, scores), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("labels,scores", [
    ([0, 0], [1., 2.]), ([1, 1], [1., 2.]), ([0, 1], [1., np.nan]),
    ([0, 2], [1., 2.]), ([0, 1], [1.]),
])
def test_invalid_or_incomplete_metrics_fail_closed(labels, scores):
    with pytest.raises(ValueError):
        metrics(labels, scores)


@pytest.mark.parametrize("mode", ["mean", "candidate_attention"])
def test_cached_independent_replay_matches_full_model_and_empty_history(mode):
    torch.manual_seed(42)
    features = F.normalize(torch.randn(9, 2560), dim=-1)
    features[0] = 0
    model = FrozenArticleRanker(features, mode)
    histories = torch.zeros((3, 30), dtype=torch.long)
    histories[1, -3:] = torch.tensor([2, 3, 4])
    histories[2, -2:] = torch.tensor([5, 6])
    candidates = torch.tensor([1, 2, 4, 3, 5, 7, 8])
    groups = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    replay = CachedReplay(features, model.state_dict(), mode)
    expected = model(histories, candidates, groups).detach().numpy()
    assert check_scores(replay(histories, candidates, groups), expected) < 1e-4
    bad = replay(histories, candidates, groups)
    bad[0] += .1
    with pytest.raises(ValueError):
        check_scores(bad, expected)


def test_feature_payload_in_head_is_rejected():
    features = torch.zeros(3, 2560)
    model = FrozenArticleRanker(features, "mean")
    with pytest.raises(ValueError):
        CachedReplay(features, {**model.state_dict(), "features": features}, "mean")
