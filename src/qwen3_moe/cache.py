"""Per-layer Key/Value storage for cached Qwen3 MoE decoding."""

from __future__ import annotations

import numpy as np


class KVCache:
    """Store and append grouped-query Key/Value tensors for every layer."""

    def __init__(self, num_hidden_layers: int) -> None:
        if isinstance(num_hidden_layers, bool) or not isinstance(
            num_hidden_layers, int
        ) or num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be a positive integer")
        self._keys: list[np.ndarray | None] = [None] * num_hidden_layers
        self._values: list[np.ndarray | None] = [None] * num_hidden_layers

    def _validate_layer_index(self, layer_index: int) -> None:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer")
        if not 0 <= layer_index < len(self._keys):
            raise IndexError("layer_index is outside the KV Cache")

    @staticmethod
    def _validate_key_value(key: np.ndarray, value: np.ndarray) -> None:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("cached Key and Value must have shape [B,Hkv,S,Dh]")
        if key.shape != value.shape:
            raise ValueError("cached Key and Value shapes must match")
        if key.shape[2] == 0:
            raise ValueError("cached Key and Value must contain at least one token")

    def get(self, layer_index: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return one layer's cached tensors, or None before its first update."""
        self._validate_layer_index(layer_index)
        key = self._keys[layer_index]
        value = self._values[layer_index]
        if key is None or value is None:
            return None
        return key, value

    def update(
        self,
        layer_index: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Append new Key/Value states and return the complete layer cache."""
        self._validate_layer_index(layer_index)
        key = np.asarray(key)
        value = np.asarray(value)
        self._validate_key_value(key, value)

        cached = self.get(layer_index)
        if cached is not None:
            cached_key, cached_value = cached
            if key.shape[:2] + key.shape[3:] != cached_key.shape[:2] + cached_key.shape[3:]:
                raise ValueError("new Key/Value dimensions do not match the layer cache")
            key = np.concatenate((cached_key, key), axis=2)
            value = np.concatenate((cached_value, value), axis=2)

        self._keys[layer_index] = key
        self._values[layer_index] = value
        return key, value

    @property
    def sequence_length(self) -> int:
        """Return the shared cached length after all populated layers agree."""
        lengths = {key.shape[2] for key in self._keys if key is not None}
        if not lengths:
            return 0
        if len(lengths) != 1:
            raise ValueError("KV Cache layers do not have the same sequence length")
        return lengths.pop()
