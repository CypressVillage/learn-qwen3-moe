"""Grouped-query attention for Qwen3 MoE prefill."""

from __future__ import annotations

import numpy as np

from qwen3_moe.cache import KVCache
from qwen3_moe.config import Qwen3MoeConfig
from qwen3_moe.layers import Linear, RMSNorm
from qwen3_moe.rope import RotaryEmbedding, apply_rotary_position_embedding


class Qwen3Attention:
    """Read earlier prompt tokens with Qwen3 grouped-query attention."""

    def __init__(
        self,
        config: Qwen3MoeConfig,
        q_proj_weight: np.ndarray,
        k_proj_weight: np.ndarray,
        v_proj_weight: np.ndarray,
        o_proj_weight: np.ndarray,
        q_norm_weight: np.ndarray,
        k_norm_weight: np.ndarray,
    ) -> None:
        query_size = config.num_attention_heads * config.head_dim
        key_value_size = config.num_key_value_heads * config.head_dim
        expected_shapes = {
            "q_proj": (query_size, config.hidden_size),
            "k_proj": (key_value_size, config.hidden_size),
            "v_proj": (key_value_size, config.hidden_size),
            "o_proj": (config.hidden_size, query_size),
            "q_norm": (config.head_dim,),
            "k_norm": (config.head_dim,),
        }
        weights = {
            "q_proj": q_proj_weight,
            "k_proj": k_proj_weight,
            "v_proj": v_proj_weight,
            "o_proj": o_proj_weight,
            "q_norm": q_norm_weight,
            "k_norm": k_norm_weight,
        }
        for name, weight in weights.items():
            if weight.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} weight must have shape {expected_shapes[name]}"
                )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = Linear(q_proj_weight)
        self.k_proj = Linear(k_proj_weight)
        self.v_proj = Linear(v_proj_weight)
        self.o_proj = Linear(o_proj_weight)
        self.q_norm = RMSNorm(q_norm_weight, config.rms_norm_eps)
        self.k_norm = RMSNorm(k_norm_weight, config.rms_norm_eps)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)

    def _project_query_key_value(
        self, hidden_states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).reshape(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).reshape(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(0, 2, 1, 3)
        key = self.k_norm(key).transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        return query, key, value

    def _apply_positions(
        self,
        query: np.ndarray,
        key: np.ndarray,
        position_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cosine, sine = self.rope(position_ids)
        return apply_rotary_position_embedding(query, key, cosine, sine)

    def _scaled_dot_product_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
    ) -> np.ndarray:
        key = np.repeat(key, self.num_key_value_groups, axis=1)
        value = np.repeat(value, self.num_key_value_groups, axis=1)
        scores = query.astype(np.float32) @ key.astype(np.float32).swapaxes(-1, -2)
        scores *= self.head_dim**-0.5

        sequence_length = query.shape[2]
        future_tokens = np.triu(
            np.ones((sequence_length, sequence_length), dtype=bool), k=1
        )
        if key.shape[2] != sequence_length:
            cached_length = key.shape[2] - sequence_length
            future_tokens = np.triu(
                np.ones((sequence_length, key.shape[2]), dtype=bool),
                k=cached_length + 1,
            )
        scores = np.where(future_tokens[None, None, :, :], -np.inf, scores)
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        return probabilities @ value.astype(np.float32)

    def __call__(
        self, hidden_states: np.ndarray, position_ids: np.ndarray
    ) -> np.ndarray:
        hidden_states = np.asarray(hidden_states)
        position_ids = np.asarray(position_ids)
        if hidden_states.ndim != 3:
            raise ValueError("attention input must have shape [B,S,D]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("attention input hidden size does not match config")
        if hidden_states.shape[1] == 0:
            raise ValueError("attention requires at least one token")
        if position_ids.shape != hidden_states.shape[:2]:
            raise ValueError("position IDs must match attention batch and sequence")

        query, key, value = self._project_query_key_value(hidden_states)
        query, key = self._apply_positions(query, key, position_ids)
        attended = self._scaled_dot_product_attention(query, key, value)
        batch_size, _, sequence_length, _ = attended.shape
        merged = attended.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        return self.o_proj(merged)

    def cached(
        self,
        hidden_states: np.ndarray,
        position_ids: np.ndarray,
        cache: KVCache,
        layer_index: int,
    ) -> np.ndarray:
        """Attend with Key/Value states appended to one layer's cache."""
        hidden_states = np.asarray(hidden_states)
        position_ids = np.asarray(position_ids)
        if hidden_states.ndim != 3:
            raise ValueError("attention input must have shape [B,S,D]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("attention input hidden size does not match config")
        if hidden_states.shape[1] == 0:
            raise ValueError("attention requires at least one token")
        if position_ids.shape != hidden_states.shape[:2]:
            raise ValueError("position IDs must match attention batch and sequence")

        query, key, value = self._project_query_key_value(hidden_states)
        query, key = self._apply_positions(query, key, position_ids)
        key, value = cache.update(layer_index, key, value)
        attended = self._scaled_dot_product_attention(query, key, value)
        batch_size, _, sequence_length, _ = attended.shape
        merged = attended.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        return self.o_proj(merged)
