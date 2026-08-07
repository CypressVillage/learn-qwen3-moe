"""Causal grouped-query attention for the cumulative dense model."""

import math

import torch
from torch import Tensor, nn

from qwen3_moe.config import DenseConfig
from qwen3_moe.debug import require_finite, require_floating
from qwen3_moe.norms import RMSNorm
from qwen3_moe.rope import RotaryEmbedding, apply_rotary_pos_emb


class GroupedQueryAttention(nn.Module):
    """Reference causal GQA implementation with explicit head expansion."""

    def __init__(self, config: DenseConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.group_size = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        attention_mask: Tensor | None = None,
        *,
        return_debug: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        require_floating("attention input", hidden_states)
        if hidden_states.ndim != 3:
            raise ValueError("attention input must be rank-3 [B,S,D]")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"attention expected hidden size {self.hidden_size}, got {hidden_size}"
            )
        if positions.shape != (batch_size, sequence_length):
            raise ValueError(
                f"positions must have shape {(batch_size, sequence_length)}"
            )
        if positions.device != hidden_states.device:
            raise ValueError("positions and hidden states must share a device")

        query_projected = self.q_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        )
        key_projected = self.k_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )

        query_pre_rope = self.q_norm(query_projected).transpose(1, 2)
        key_pre_rope = self.k_norm(key_projected).transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = self.rotary(positions, dtype=hidden_states.dtype)
        query, key = apply_rotary_pos_emb(
            query_pre_rope, key_pre_rope, cos, sin
        )

        repeated_key = key.repeat_interleave(self.group_size, dim=1)
        repeated_value = value.repeat_interleave(self.group_size, dim=1)
        scores = torch.matmul(query, repeated_key.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                    device=hidden_states.device,
                ),
                diagonal=1,
            ).view(1, 1, sequence_length, sequence_length)
        if attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must have bool dtype")
        if attention_mask.device != hidden_states.device:
            raise ValueError("attention_mask and hidden states must share a device")
        try:
            expanded_mask = attention_mask.expand_as(scores)
        except RuntimeError as exc:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} cannot broadcast "
                f"to scores {tuple(scores.shape)}"
            ) from exc
        if expanded_mask.all(dim=-1).any():
            raise ValueError("attention_mask contains a fully blocked query row")

        masked_scores = scores.masked_fill(expanded_mask, float("-inf"))
        probabilities = torch.softmax(masked_scores.float(), dim=-1).to(scores.dtype)
        context = torch.matmul(probabilities, repeated_value)
        merged = context.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.hidden_size
        )
        update = self.out_proj(merged)

        require_finite("attention scores", scores)
        require_finite("attention probabilities", probabilities)
        require_finite("attention update", update)
        if not return_debug:
            return update
        return update, {
            "query_projected": query_projected,
            "key_projected": key_projected,
            "query_pre_rope": query_pre_rope,
            "key_pre_rope": key_pre_rope,
            "query": query,
            "key": key,
            "value": value,
            "repeated_key": repeated_key,
            "repeated_value": repeated_value,
            "scores": scores,
            "probabilities": probabilities,
        }
