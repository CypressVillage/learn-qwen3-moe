"""Teaching-oriented building blocks for Qwen3-style MoE inference."""

from qwen3_moe.attention import GroupedQueryAttention
from qwen3_moe.config import DenseConfig
from qwen3_moe.decoder import DenseDecoderLayer
from qwen3_moe.mlp import SwiGLU
from qwen3_moe.model import TinyDenseCausalLM
from qwen3_moe.norms import RMSNorm
from qwen3_moe.rope import RotaryEmbedding, apply_rotary_pos_emb

__all__ = [
    "DenseConfig",
    "DenseDecoderLayer",
    "GroupedQueryAttention",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "TinyDenseCausalLM",
    "apply_rotary_pos_emb",
]
