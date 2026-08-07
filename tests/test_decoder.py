import torch

from qwen3_moe import DenseConfig, DenseDecoderLayer


def test_dense_decoder_uses_two_correct_residual_bases():
    torch.manual_seed(7)
    config = DenseConfig(
        vocab_size=11,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rope_theta=100.0,
    )
    layer = DenseDecoderLayer(config)
    hidden = torch.randn(2, 4, 8)
    positions = torch.arange(4).view(1, 4).expand(2, 4)

    output, debug = layer(hidden, positions, return_debug=True)

    torch.testing.assert_close(
        debug["after_attention"],
        hidden + debug["attention_update"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        output,
        debug["after_attention"] + debug["mlp_update"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert debug["attention_norm"].shape == hidden.shape
    assert debug["mlp_norm"].shape == hidden.shape
    assert torch.isfinite(output).all()
