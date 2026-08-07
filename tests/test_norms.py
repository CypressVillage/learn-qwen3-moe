import pytest
import torch

from qwen3_moe import RMSNorm


def test_rms_norm_matches_the_reference_formula():
    layer = RMSNorm(4, eps=1e-6)
    hidden = torch.tensor(
        [[[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -0.5, -1.5]]]
    )
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))

    actual = layer(hidden)
    expected = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
    expected = expected * layer.weight

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert actual.shape == hidden.shape
    assert torch.isfinite(actual).all()


def test_rms_norm_rejects_wrong_width_and_integer_input():
    layer = RMSNorm(4)

    with pytest.raises(ValueError, match="last dimension"):
        layer(torch.randn(2, 3, 5))
    with pytest.raises(ValueError, match="floating point"):
        layer(torch.ones(2, 3, 4, dtype=torch.long))
