import torch
import torch.nn.functional as F

from qwen3_moe import DenseConfig, SwiGLU


def test_swiglu_matches_explicit_pytorch_primitives():
    torch.manual_seed(7)
    config = DenseConfig(
        vocab_size=11,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    mlp = SwiGLU(config)
    hidden = torch.randn(2, 3, 8)

    actual, debug = mlp(hidden, return_debug=True)
    gate = F.linear(hidden, mlp.gate_proj.weight)
    up = F.linear(hidden, mlp.up_proj.weight)
    expected_product = F.silu(gate) * up
    expected = F.linear(expected_product, mlp.down_proj.weight)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(debug["product"], expected_product, atol=1e-6, rtol=1e-6)
    assert debug["gate"].shape == debug["up"].shape == (2, 3, 12)
