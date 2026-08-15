"""Decoder layers and causal language model assembly for Qwen3 MoE."""

from __future__ import annotations

import numpy as np

from qwen3_moe.attention import Qwen3Attention
from qwen3_moe.config import Qwen3MoeConfig
from qwen3_moe.layers import RMSNorm
from qwen3_moe.moe import Qwen3SparseMoeBlock


class Qwen3DecoderLayer:
    """Combine pre-norm Attention and sparse MoE with residual connections."""

    def __init__(
        self,
        config: Qwen3MoeConfig,
        input_layernorm_weight: np.ndarray,
        q_proj_weight: np.ndarray,
        k_proj_weight: np.ndarray,
        v_proj_weight: np.ndarray,
        o_proj_weight: np.ndarray,
        q_norm_weight: np.ndarray,
        k_norm_weight: np.ndarray,
        post_attention_layernorm_weight: np.ndarray,
        router_weight: np.ndarray,
        gate_up_proj_weight: np.ndarray,
        down_proj_weight: np.ndarray,
    ) -> None:
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(
            input_layernorm_weight, config.rms_norm_eps
        )
        self.self_attention = Qwen3Attention(
            config,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            o_proj_weight,
            q_norm_weight,
            k_norm_weight,
        )
        self.post_attention_layernorm = RMSNorm(
            post_attention_layernorm_weight, config.rms_norm_eps
        )
        self.mlp = Qwen3SparseMoeBlock(
            config,
            router_weight,
            gate_up_proj_weight,
            down_proj_weight,
        )

    def _attention_block(
        self, hidden_states: np.ndarray, position_ids: np.ndarray
    ) -> np.ndarray:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attention(hidden_states, position_ids)
        return residual + hidden_states

    def _moe_block(self, hidden_states: np.ndarray) -> np.ndarray:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def __call__(
        self, hidden_states: np.ndarray, position_ids: np.ndarray
    ) -> np.ndarray:
        hidden_states = np.asarray(hidden_states)
        if hidden_states.ndim != 3:
            raise ValueError("decoder layer input must have shape [B,S,D]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("decoder layer input hidden size does not match config")

        hidden_states = self._attention_block(hidden_states, position_ids)
        return self._moe_block(hidden_states)
