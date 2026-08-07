import pytest

from qwen3_moe import DenseConfig


def test_dense_config_accepts_a_valid_tiny_model():
    config = DenseConfig(
        vocab_size=11,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )

    assert config.hidden_size == config.num_attention_heads * config.head_dim


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"hidden_size": 7}, "hidden_size"),
        ({"num_attention_heads": 3}, "hidden_size"),
        ({"num_attention_heads": 4, "head_dim": 2, "num_key_value_heads": 3}, "divisible"),
        ({"head_dim": 3, "hidden_size": 6}, "even"),
        ({"vocab_size": 0}, "vocab_size"),
        ({"rms_norm_eps": 0.0}, "rms_norm_eps"),
    ],
)
def test_dense_config_rejects_invalid_dimensions(overrides, expected):
    values = {
        "vocab_size": 11,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=expected):
        DenseConfig(**values)
