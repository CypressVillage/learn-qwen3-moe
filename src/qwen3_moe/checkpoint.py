"""Read named tensors from a Safetensors checkpoint."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_DTYPE_WIDTHS = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I32": 4,
    "I64": 8,
}

_NUMPY_DTYPES = {
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "I32": np.dtype("<i4"),
    "I64": np.dtype("<i8"),
}


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """Where one tensor lives inside one Safetensors shard."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    data_offsets: tuple[int, int]
    data_start: int


class SafetensorsCheckpoint:
    """A name-to-tensor view over one or more Safetensors files."""

    def __init__(self, directory: Path, tensors: dict[str, TensorInfo]) -> None:
        self.directory = directory
        self._tensors = tensors

    @classmethod
    def from_directory(cls, directory: str | Path) -> "SafetensorsCheckpoint":
        root = Path(directory)
        index_files = sorted(root.glob("*.safetensors.index.json"))
        if index_files:
            tensors = _read_index(root, index_files[0])
        else:
            shards = sorted(root.glob("*.safetensors"))
            if len(shards) != 1:
                raise ValueError("expected one Safetensors file or one index")
            tensors = _read_shard(shards[0])
        return cls(root, tensors)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._tensors))

    def tensor_info(self, name: str) -> TensorInfo:
        try:
            return self._tensors[name]
        except KeyError as error:
            raise KeyError(f"unknown tensor: {name}") from error

    def load_tensor(self, name: str) -> np.ndarray:
        """Read one tensor payload instead of loading the whole checkpoint."""

        info = self.tensor_info(name)
        start, end = info.data_offsets
        with (self.directory / info.shard).open("rb") as handle:
            handle.seek(info.data_start + start)
            payload = handle.read(end - start)
        if len(payload) != end - start:
            raise ValueError(f"tensor payload is truncated: {name}")

        if info.dtype == "BF16":
            words = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
            array = (words << 16).view(np.float32)
        else:
            array = np.frombuffer(payload, dtype=_NUMPY_DTYPES[info.dtype]).copy()
        return array.reshape(info.shape)


def _read_index(root: Path, path: Path) -> dict[str, TensorInfo]:
    index = json.loads(path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Safetensors index must contain weight_map")

    shard_headers = {
        shard_name: _read_shard(root / shard_name)
        for shard_name in sorted(set(weight_map.values()))
    }
    tensors: dict[str, TensorInfo] = {}
    for name, shard_name in weight_map.items():
        try:
            tensors[name] = shard_headers[shard_name][name]
        except KeyError as error:
            raise ValueError(f"index points to a missing tensor: {name}") from error
    return tensors


def _read_shard(path: Path) -> dict[str, TensorInfo]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"invalid Safetensors file: {path.name}")
        (header_length,) = struct.unpack("<Q", prefix)
        header = json.loads(handle.read(header_length))

    data_start = 8 + header_length
    data_size = path.stat().st_size - data_start
    tensors: dict[str, TensorInfo] = {}
    for name, metadata in header.items():
        if name != "__metadata__":
            tensors[name] = _tensor_info(
                name, metadata, path.name, data_start, data_size
            )
    return tensors


def _tensor_info(
    name: str,
    metadata: Any,
    shard: str,
    data_start: int,
    data_size: int,
) -> TensorInfo:
    dtype = metadata["dtype"]
    shape = tuple(metadata["shape"])
    start, end = metadata["data_offsets"]
    if dtype not in _DTYPE_WIDTHS:
        raise ValueError(f"unsupported Safetensors dtype: {dtype}")
    if start < 0 or end > data_size:
        raise ValueError(f"tensor offsets are outside {shard}: {name}")
    expected_bytes = math.prod(shape) * _DTYPE_WIDTHS[dtype]
    if end - start != expected_bytes:
        raise ValueError(f"tensor shape does not match its byte range: {name}")
    return TensorInfo(name, dtype, shape, shard, (start, end), data_start)
