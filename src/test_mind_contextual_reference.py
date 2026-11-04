import numpy as np
import pytest
import torch

from mind_contextual_reference import contextual_reference
from test_mind_contextual_recent import example
from test_mind_lexical_glm import inputs


@pytest.mark.parametrize("cached", [False, True])
def test_numpy_contextual_math_is_independent_of_cached_values_and_module_forwards(inputs, monkeypatch, cached):
    base = example(inputs).base
    torch.manual_seed(19)
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.normal_(0, .15)
        base.category.weight[0].zero_()
        base.subcategory.weight[0].zero_()
        base.semantic_scale.fill_(3)
    if cached:
        base.prepare_inference()
    else:
        base.cached = None
    histories = np.zeros((3, 30), dtype=np.int32)
    histories[1, -4:] = [1, 2, 1, 3]
    histories[2, -1] = 4
    candidates = np.array([2, 1, 2, 4, 3, 1, 2])
    groups = np.array([1, 0, 2, 1, 2, 1, 0])
    expected = base.scores(histories, candidates, groups)
    reference = np.array([contextual_reference(base, histories, candidates, groups, index)
                          for index in range(len(candidates))])
    np.testing.assert_allclose(reference, expected, rtol=1e-5, atol=2e-4)

    def forbidden(*args, **kwargs):
        raise AssertionError("The reference must not invoke native neural computation")

    if cached:
        base.cached.fill_(float("nan"))
    for module in base.modules():
        monkeypatch.setattr(module, "forward", forbidden)
    for name in ("encode", "scores", "prepare_inference"):
        monkeypatch.setattr(base, name, forbidden)
    repeated = np.array([contextual_reference(base, histories, candidates, groups, index)
                         for index in range(len(candidates))])
    np.testing.assert_array_equal(repeated, reference)
    with torch.no_grad():
        base.head[-1].bias.add_(1)
    changed = contextual_reference(base, histories, candidates, groups, 0)
    assert changed == pytest.approx(reference[0] + 1, rel=1e-6, abs=1e-6)


def test_contextual_reference_rejects_training_mode(inputs):
    base = example(inputs).base.train()
    with pytest.raises(ValueError, match="Frozen evaluation"):
        contextual_reference(base, np.zeros((1, 30), dtype=np.int32),
                             np.array([1]), np.array([0]), 0)
