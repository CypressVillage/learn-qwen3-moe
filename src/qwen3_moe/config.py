"""Qwen3 MoE architecture configuration."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Qwen3MoeConfig:
    """The config fields that shape Qwen3 MoE inference."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    max_position_embeddings: int
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    norm_topk_prob: bool = True
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    rope_scaling: dict[str, Any] | None = None
    model_type: str = "qwen3_moe"
    extra_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        positive_fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_experts",
            "num_experts_per_tok",
            "max_position_embeddings",
            "decoder_sparse_step",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must divide into KV head groups")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3MoeConfig":
        """Read the fields we use and keep the rest for later chapters."""

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config.json must contain an object")

        known_fields = set(cls.__dataclass_fields__) - {"extra_fields"}
        values = {name: raw[name] for name in known_fields if name in raw}
        values["mlp_only_layers"] = tuple(raw.get("mlp_only_layers", ()))
        values["extra_fields"] = {
            name: value for name, value in raw.items() if name not in known_fields
        }
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(
                f"config.json is missing or mis-shapes a field: {error}"
            ) from error
