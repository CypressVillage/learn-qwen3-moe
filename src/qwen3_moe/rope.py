"""Rotary position embeddings for Qwen3 attention."""

from __future__ import annotations

import numpy as np


class RotaryEmbedding:
    """Build the cosine and sine values used to rotate attention heads."""

    def __init__(self, head_dim: int, theta: float = 1_000_000.0) -> None:
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError("RoPE head dimension must be a positive integer")
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        if theta <= 0:
            raise ValueError("RoPE theta must be positive")
        dimensions = np.arange(0, head_dim, 2, dtype=np.float32)
        self.inverse_frequencies = 1.0 / (theta ** (dimensions / head_dim))

    def __call__(self, position_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position_ids = np.asarray(position_ids)
        if position_ids.ndim != 2:
            raise ValueError("position IDs must have shape [B,S]")
        if not np.issubdtype(position_ids.dtype, np.integer):
            raise ValueError("position IDs must be integers")
        if np.any(position_ids < 0):
            raise ValueError("position IDs must be non-negative")
        angles = position_ids.astype(np.float32)[..., None] * self.inverse_frequencies
        angles = np.concatenate((angles, angles), axis=-1)
        return np.cos(angles), np.sin(angles)


def _rotate_half(hidden_states: np.ndarray) -> np.ndarray:
    first_half, second_half = np.split(hidden_states, 2, axis=-1)
    return np.concatenate((-second_half, first_half), axis=-1)


def apply_rotary_position_embedding(
    query: np.ndarray,
    key: np.ndarray,
    cosine: np.ndarray,
    sine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate Query and Key while allowing different Q and KV head counts."""

    query = np.asarray(query)
    key = np.asarray(key)
    cosine = np.asarray(cosine)
    sine = np.asarray(sine)
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("Query and Key must have shape [B,H,S,Hd]")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("Query and Key must share batch, sequence, and head dimensions")
    expected = (query.shape[0], query.shape[2], query.shape[3])
    if cosine.shape != expected or sine.shape != expected:
        raise ValueError("RoPE cosine and sine must have shape [B,S,Hd]")
    cosine = cosine[:, None, :, :]
    sine = sine[:, None, :, :]
    rotated_query = query * cosine + _rotate_half(query) * sine
    rotated_key = key * cosine + _rotate_half(key) * sine
    return rotated_query, rotated_key
