from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import mind_public_contextual as module


def test_masked_public_pool_excludes_padding_and_normalizes():
    hidden = torch.tensor([[[1., 0.], [0., 1.], [100., 100.]],
                           [[-1., 0.], [100., 100.], [100., 100.]]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    actual = module.masked_pool(hidden, mask).numpy()
    np.testing.assert_allclose(actual, [[2 ** -.5, 2 ** -.5], [-1, 0]], rtol=1e-6)


def test_all_padding_public_pool_is_zero_not_nan():
    output = module.masked_pool(torch.ones(2, 4, 3), torch.zeros(2, 4, dtype=torch.int64))
    assert torch.isfinite(output).all() and not output.count_nonzero()


def test_public_file_pin_precedes_model_loading(tmp_path):
    (tmp_path / "config.json").write_bytes(b"bad")
    with pytest.raises(ValueError, match="length changed"):
        module.PublicEncoder(tmp_path)
