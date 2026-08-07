import torch

from qwen3_moe import RotaryEmbedding, apply_rotary_pos_emb


def test_rope_preserves_shape_and_pairwise_norm():
    rotary = RotaryEmbedding(head_dim=4, theta=100.0)
    positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, 3, 4)
    cos, sin = rotary(positions, dtype=query.dtype)

    rotated_query, rotated_key = apply_rotary_pos_emb(query, key, cos, sin)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    torch.testing.assert_close(
        rotated_query.square().sum(-1),
        query.square().sum(-1),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        rotated_key.square().sum(-1),
        key.square().sum(-1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_rope_position_zero_is_identity_and_later_positions_rotate():
    rotary = RotaryEmbedding(head_dim=4, theta=100.0)
    positions = torch.tensor([[0, 1]], dtype=torch.long)
    query = torch.ones(1, 1, 2, 4)
    key = query.clone()
    cos, sin = rotary(positions, dtype=query.dtype)

    rotated_query, _ = apply_rotary_pos_emb(query, key, cos, sin)

    torch.testing.assert_close(rotated_query[:, :, 0], query[:, :, 0])
    assert not torch.allclose(rotated_query[:, :, 1], query[:, :, 1])
    assert "inv_freq" not in rotary.state_dict()
