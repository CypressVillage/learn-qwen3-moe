"""Small NumPy layers shared by Qwen3 MoE inference modules."""

from __future__ import annotations

import numpy as np


class Embedding:
    """Look up one hidden vector for every token ID."""

    def __init__(self, weight: np.ndarray) -> None:
        if weight.ndim != 2:
            raise ValueError("embedding weight must have shape [V,D]")
        self.weight = weight

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        token_ids = np.asarray(token_ids)
        if not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("token IDs must be integers")
        if np.any(token_ids < 0) or np.any(token_ids >= self.weight.shape[0]):
            raise ValueError("token ID is outside the embedding vocabulary")
        return self.weight[token_ids]


class RMSNorm:
    """Scale hidden states by their root-mean-square magnitude."""

    def __init__(self, weight: np.ndarray, epsilon: float) -> None:
        if weight.ndim != 1:
            raise ValueError("RMSNorm weight must have shape [D]")
        if epsilon <= 0:
            raise ValueError("RMSNorm epsilon must be positive")
        self.weight = weight
        self.epsilon = epsilon

    def __call__(self, hidden_states: np.ndarray) -> np.ndarray:
        hidden_states = np.asarray(hidden_states)
        if hidden_states.shape[-1] != self.weight.shape[0]:
            raise ValueError("hidden size does not match RMSNorm weight")
        values = hidden_states.astype(np.float32, copy=False)
        mean_square = np.mean(np.square(values), axis=-1, keepdims=True)
        return values * np.reciprocal(np.sqrt(mean_square + self.epsilon)) * self.weight


class Linear:
    """Apply a learned projection to the last tensor dimension."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None) -> None:
        if weight.ndim != 2:
            raise ValueError("linear weight must have shape [O,I]")
        if bias is not None and bias.shape != (weight.shape[0],):
            raise ValueError("linear bias must have shape [O]")
        self.weight = weight
        self.bias = bias

    def __call__(self, hidden_states: np.ndarray) -> np.ndarray:
        hidden_states = np.asarray(hidden_states)
        if hidden_states.shape[-1] != self.weight.shape[1]:
            raise ValueError("input size does not match linear weight")
        output = hidden_states @ self.weight.T
        return output if self.bias is None else output + self.bias
