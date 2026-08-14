"""Qwen3 MoE architecture configuration."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Qwen3MoeConfig:
    """Architecture fields needed to inspect a Qwen3 MoE checkpoint."""

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
        integer_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "max_position_embeddings": self.max_position_embeddings,
            "decoder_sparse_step": self.decoder_sparse_step,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for split-half RoPE")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok must not exceed num_experts")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        for name, value in {
            "norm_topk_prob": self.norm_topk_prob,
            "tie_word_embeddings": self.tie_word_embeddings,
        }.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
        if self.rope_scaling is not None and not isinstance(self.rope_scaling, dict):
            raise ValueError("rope_scaling must be an object or null")
        if not isinstance(self.model_type, str) or not self.model_type:
            raise ValueError("model_type must be a non-empty string")

        seen_layers: set[int] = set()
        for layer in self.mlp_only_layers:
            if isinstance(layer, bool) or not isinstance(layer, int):
                raise ValueError("mlp_only_layers must contain integers")
            if layer < 0 or layer >= self.num_hidden_layers:
                raise ValueError(
                    f"mlp_only_layers contains out-of-range layer {layer}"
                )
            if layer in seen_layers:
                raise ValueError(f"mlp_only_layers contains duplicate layer {layer}")
            seen_layers.add(layer)

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3MoeConfig":
        """Read and validate architecture fields from ``config.json``."""

        config_path = Path(path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"config file does not exist: {config_path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"config file is not valid JSON at line {error.lineno}: {config_path}"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError("config root must be a JSON object")

        required = {
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
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"config is missing required field: {missing[0]}")

        known = {item.name for item in cls.__dataclass_fields__.values()}
        values = {name: value for name, value in raw.items() if name in known}
        try:
            values["mlp_only_layers"] = tuple(raw.get("mlp_only_layers", ()))
        except TypeError as error:
            raise ValueError("mlp_only_layers must be an array") from error
        values["extra_fields"] = {
            name: value for name, value in raw.items() if name not in known
        }
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"config field has an invalid type: {error}") from error
