import math

import pytest
import torch

from full_impression_ranker import FrozenArticleRanker, full_impression_loss


def features():
    torch.manual_seed(42)
    value = torch.randn(6, 8)
    value[0] = 0
    return value


def test_all_positives_receive_gradient_and_each_impression_has_equal_weight():
    scores = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1, 1, 0, 0, 1, 0])
    groups = torch.tensor([0, 0, 0, 0, 1, 1])
    loss = full_impression_loss(scores, labels, groups, 2)
    assert loss.item() == pytest.approx((math.log(4) + math.log(2)) / 2)
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.tensor([-.125, -.125, .125, .125, -.25, .25],
                                                       dtype=torch.float64))


def test_large_logits_are_finite_and_group_permutations_do_not_change_loss():
    scores = torch.tensor([10000., -10000., 10001., 9999.], requires_grad=True)
    labels = torch.tensor([1, 0, 1, 0])
    groups = torch.tensor([0, 0, 1, 1])
    loss = full_impression_loss(scores, labels, groups, 2)
    order = torch.tensor([3, 0, 2, 1])
    torch.testing.assert_close(loss, full_impression_loss(scores[order], labels[order], groups[order], 2))
    loss.backward()
    assert torch.isfinite(scores.grad).all()


@pytest.mark.parametrize("labels,groups,impressions", [
    ([1, 1], [0, 0], 1), ([0, 0], [0, 0], 1),
    ([1, 0], [0, 0], 2), ([1, 2], [0, 0], 1), ([1, 0], [0, 2], 2),
])
def test_incomplete_or_invalid_training_impressions_fail_closed(labels, groups, impressions):
    with pytest.raises(ValueError):
        full_impression_loss(torch.zeros(2), torch.tensor(labels), torch.tensor(groups), impressions)


@pytest.mark.parametrize("mode", ["mean", "candidate_attention"])
def test_empty_histories_have_explicit_candidate_only_scores_and_finite_gradients(mode):
    model = FrozenArticleRanker(features(), mode, dimension=4)
    histories = torch.zeros((2, 30), dtype=torch.long)
    candidates = torch.tensor([1, 2, 3, 4])
    groups = torch.tensor([0, 0, 1, 1])
    scores = model(histories, candidates, groups)
    expected = model.article_bias(model.features[candidates]).squeeze(-1)
    torch.testing.assert_close(scores, expected)
    full_impression_loss(scores, torch.tensor([1, 0, 1, 0]), groups, 2).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model.features.grad is None


@pytest.mark.parametrize("mode", ["mean", "candidate_attention"])
def test_padding_is_masked_despite_nonzero_projection_bias_and_batching_is_invariant(mode):
    model = FrozenArticleRanker(features(), mode, dimension=4)
    histories = torch.zeros((2, 30), dtype=torch.long)
    histories[0, -1] = 1
    histories[1, -3:] = torch.tensor([2, 3, 4])
    candidates = torch.tensor([1, 5, 2, 3])
    groups = torch.tensor([0, 0, 1, 1])
    actual = model(histories, candidates, groups)
    first = model(histories[:1], candidates[:2], torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(actual[:2], first)
    embedding = torch.nn.functional.normalize(model.project(model.features[1:2]), dim=-1)[0]
    candidate = torch.nn.functional.normalize(model.project(model.features[candidates[:2]]), dim=-1)
    expected = 10 * (candidate * embedding).sum(dim=-1)
    expected += model.article_bias(model.features[candidates[:2]]).squeeze(-1)
    torch.testing.assert_close(first, expected)
    assert "features" not in model.state_dict()


def test_invalid_padding_candidate_and_left_padding_fail_closed():
    bad = features()
    bad[0, 0] = 1
    with pytest.raises(ValueError):
        FrozenArticleRanker(bad, "mean")
    model = FrozenArticleRanker(features(), "mean")
    histories = torch.zeros((1, 30), dtype=torch.long)
    with pytest.raises(ValueError):
        model(histories, torch.tensor([0]), torch.tensor([0]))
    histories[0, 0] = 1
    with pytest.raises(ValueError, match="padding"):
        model(histories, torch.tensor([1]), torch.tensor([0]))
