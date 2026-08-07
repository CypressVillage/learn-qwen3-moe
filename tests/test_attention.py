import pytest
import torch

from qwen3_moe import DenseConfig, GroupedQueryAttention


def make_config(**overrides):
    values = {
        "vocab_size": 11,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "rope_theta": 100.0,
    }
    values.update(overrides)
    return DenseConfig(**values)


def test_causal_gqa_masks_future_tokens_and_maps_head_groups():
    torch.manual_seed(7)
    attention = GroupedQueryAttention(make_config())
    hidden = torch.randn(2, 4, 8)
    positions = torch.arange(4).view(1, 4).expand(2, 4)

    update, debug = attention(hidden, positions, return_debug=True)

    probabilities = debug["probabilities"]
    future_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    future = probabilities.masked_select(future_mask.view(1, 1, 4, 4))
    assert update.shape == hidden.shape
    assert probabilities.shape == (2, 4, 4, 4)
    assert torch.equal(future, torch.zeros_like(future))
    torch.testing.assert_close(
        probabilities.sum(-1), torch.ones(2, 4, 4), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(debug["repeated_key"][:, 0], debug["key"][:, 0])
    torch.testing.assert_close(debug["repeated_key"][:, 1], debug["key"][:, 0])
    torch.testing.assert_close(debug["repeated_key"][:, 2], debug["key"][:, 1])
    torch.testing.assert_close(debug["repeated_key"][:, 3], debug["key"][:, 1])


def test_attention_rejects_non_broadcastable_or_fully_blocked_masks():
    attention = GroupedQueryAttention(make_config())
    hidden = torch.randn(1, 3, 8)
    positions = torch.arange(3).view(1, 3)

    with pytest.raises(ValueError, match="cannot broadcast"):
        attention(hidden, positions, torch.zeros(3, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="fully blocked"):
        attention(hidden, positions, torch.ones(1, 1, 3, 3, dtype=torch.bool))
