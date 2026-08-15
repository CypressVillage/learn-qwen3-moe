"""Decoder layers and causal language model assembly for Qwen3 MoE."""

from __future__ import annotations

import numpy as np

from qwen3_moe.attention import Qwen3Attention
from qwen3_moe.cache import KVCache
from qwen3_moe.checkpoint import SafetensorsCheckpoint
from qwen3_moe.config import Qwen3MoeConfig
from qwen3_moe.layers import Embedding, Linear
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

    def _cached_attention_block(
        self,
        hidden_states: np.ndarray,
        position_ids: np.ndarray,
        cache: KVCache,
        layer_index: int,
    ) -> np.ndarray:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attention.cached(
            hidden_states, position_ids, cache, layer_index
        )
        return residual + hidden_states

    def cached(
        self,
        hidden_states: np.ndarray,
        position_ids: np.ndarray,
        cache: KVCache,
        layer_index: int,
    ) -> np.ndarray:
        """Run one decoder layer while updating its Key/Value cache."""
        hidden_states = np.asarray(hidden_states)
        if hidden_states.ndim != 3:
            raise ValueError("decoder layer input must have shape [B,S,D]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("decoder layer input hidden size does not match config")

        hidden_states = self._cached_attention_block(
            hidden_states, position_ids, cache, layer_index
        )
        return self._moe_block(hidden_states)


class Qwen3MoeForCausalLM:
    """Run full-prompt Qwen3 MoE inference and return vocabulary logits."""

    def __init__(
        self,
        config: Qwen3MoeConfig,
        embed_tokens_weight: np.ndarray,
        layers: list[Qwen3DecoderLayer],
        norm_weight: np.ndarray,
        lm_head_weight: np.ndarray,
    ) -> None:
        if embed_tokens_weight.shape != (config.vocab_size, config.hidden_size):
            raise ValueError("embedding weight does not match config")
        if len(layers) != config.num_hidden_layers:
            raise ValueError("decoder layer count does not match config")
        if norm_weight.shape != (config.hidden_size,):
            raise ValueError("final norm weight does not match config")
        if lm_head_weight.shape != (config.vocab_size, config.hidden_size):
            raise ValueError("LM Head weight does not match config")

        self.embed_tokens = Embedding(embed_tokens_weight)
        self.layers = layers
        self.norm = RMSNorm(norm_weight, config.rms_norm_eps)
        self.lm_head = Linear(lm_head_weight)

    @classmethod
    def from_checkpoint(
        cls,
        config: Qwen3MoeConfig,
        checkpoint: SafetensorsCheckpoint,
    ) -> "Qwen3MoeForCausalLM":
        embed_tokens_weight = checkpoint.load_tensor("model.embed_tokens.weight")
        layers = [
            cls._load_decoder_layer(config, checkpoint, layer_index)
            for layer_index in range(config.num_hidden_layers)
        ]
        norm_weight = checkpoint.load_tensor("model.norm.weight")
        lm_head_weight = (
            embed_tokens_weight
            if config.tie_word_embeddings
            else checkpoint.load_tensor("lm_head.weight")
        )
        return cls(
            config,
            embed_tokens_weight,
            layers,
            norm_weight,
            lm_head_weight,
        )

    @staticmethod
    def _load_decoder_layer(
        config: Qwen3MoeConfig,
        checkpoint: SafetensorsCheckpoint,
        layer_index: int,
    ) -> Qwen3DecoderLayer:
        prefix = f"model.layers.{layer_index}"
        load = checkpoint.load_tensor
        return Qwen3DecoderLayer(
            config,
            load(f"{prefix}.input_layernorm.weight"),
            load(f"{prefix}.self_attn.q_proj.weight"),
            load(f"{prefix}.self_attn.k_proj.weight"),
            load(f"{prefix}.self_attn.v_proj.weight"),
            load(f"{prefix}.self_attn.o_proj.weight"),
            load(f"{prefix}.self_attn.q_norm.weight"),
            load(f"{prefix}.self_attn.k_norm.weight"),
            load(f"{prefix}.post_attention_layernorm.weight"),
            load(f"{prefix}.mlp.gate.weight"),
            load(f"{prefix}.mlp.experts.gate_up_proj"),
            load(f"{prefix}.mlp.experts.down_proj"),
        )

    @staticmethod
    def _position_ids(token_ids: np.ndarray) -> np.ndarray:
        batch_size, sequence_length = token_ids.shape
        positions = np.arange(sequence_length, dtype=np.int64)
        return np.broadcast_to(positions, (batch_size, sequence_length))

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 2:
            raise ValueError("Causal LM input must have shape [B,S]")
        if token_ids.shape[1] == 0:
            raise ValueError("Causal LM requires at least one token")

        position_ids = self._position_ids(token_ids)
        hidden_states = self.embed_tokens(token_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    @staticmethod
    def _cached_position_ids(
        token_ids: np.ndarray, cache: KVCache
    ) -> np.ndarray:
        batch_size, sequence_length = token_ids.shape
        start = cache.sequence_length
        positions = np.arange(start, start + sequence_length, dtype=np.int64)
        return np.broadcast_to(positions, (batch_size, sequence_length))

    def cached(self, token_ids: np.ndarray, cache: KVCache) -> np.ndarray:
        """Run prefill or decode while appending every layer's Key/Value state."""
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 2:
            raise ValueError("Causal LM input must have shape [B,S]")
        if token_ids.shape[1] == 0:
            raise ValueError("Causal LM requires at least one token")

        position_ids = self._cached_position_ids(token_ids, cache)
        hidden_states = self.embed_tokens(token_ids)
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer.cached(
                hidden_states, position_ids, cache, layer_index
            )
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
