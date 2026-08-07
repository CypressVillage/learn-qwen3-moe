"""Rotary position embedding primitives."""

import torch
from torch import Tensor, nn

from qwen3_moe.debug import require_finite, require_floating


def rotate_half(hidden_states: Tensor) -> Tensor:
    if hidden_states.shape[-1] % 2 != 0:
        raise ValueError("split-half RoPE requires an even last dimension")
    half = hidden_states.shape[-1] // 2
    return torch.cat((-hidden_states[..., half:], hidden_states[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    """Build split-half RoPE cosines and sines on the active device."""

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if theta <= 0:
            raise ValueError("theta must be positive")

        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.head_dim = head_dim
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, positions: Tensor, *, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if positions.ndim != 2 or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must be a rank-2 integer tensor")
        if positions.device != self.inv_freq.device:
            raise ValueError("positions and rotary buffer must be on the same device")
        if (positions < 0).any():
            raise ValueError("positions must be non-negative")
        if not dtype.is_floating_point:
            raise ValueError("RoPE output dtype must be floating point")

        angles = positions.float().unsqueeze(-1) * self.inv_freq
        full_angles = torch.cat((angles, angles), dim=-1)
        cos = full_angles.cos().unsqueeze(1).to(dtype=dtype)
        sin = full_angles.sin().unsqueeze(1).to(dtype=dtype)
        return cos, sin


def apply_rotary_pos_emb(
    query: Tensor, key: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    require_floating("query", query)
    require_floating("key", key)
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must be rank-4 [B,H,S,Dh] tensors")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query and key must share B, S, and Dh dimensions")
    expected = (query.shape[0], 1, query.shape[2], query.shape[3])
    if cos.shape != expected or sin.shape != expected:
        raise ValueError(f"cos and sin must have shape {expected}")
    tensors = (query, key, cos, sin)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("query, key, cos, and sin must share a device")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("query, key, cos, and sin must share a dtype")

    rotated_query = query * cos + rotate_half(query) * sin
    rotated_key = key * cos + rotate_half(key) * sin
    require_finite("rotated query", rotated_query)
    require_finite("rotated key", rotated_key)
    return rotated_query, rotated_key
