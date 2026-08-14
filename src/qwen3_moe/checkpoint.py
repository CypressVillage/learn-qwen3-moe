"""Small, strict Safetensors metadata and named-tensor reader."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

import numpy as np


_DTYPE_WIDTHS = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

_NUMPY_DTYPES = {
    "BOOL": np.dtype("?"),
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "I16": np.dtype("<i2"),
    "U16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "I32": np.dtype("<i4"),
    "U32": np.dtype("<u4"),
    "F32": np.dtype("<f4"),
    "I64": np.dtype("<i8"),
    "U64": np.dtype("<u8"),
    "F64": np.dtype("<f8"),
}


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """Validated metadata for one tensor in one Safetensors shard."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    data_offsets: tuple[int, int]
    data_start: int

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


class SafetensorsCheckpoint:
    """Inspect a single-file or indexed sharded Safetensors checkpoint."""

    def __init__(self, directory: Path, tensors: dict[str, TensorInfo]) -> None:
        self.directory = directory
        self._tensors = dict(sorted(tensors.items()))

    @classmethod
    def from_directory(cls, directory: str | Path) -> "SafetensorsCheckpoint":
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f"checkpoint directory does not exist: {root}")

        index_files = sorted(root.glob("*.safetensors.index.json"))
        if len(index_files) > 1:
            raise ValueError("checkpoint directory contains multiple Safetensors indexes")
        if index_files:
            return cls(root, _read_indexed_checkpoint(root, index_files[0]))

        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            raise ValueError("checkpoint directory contains no Safetensors file")
        if len(shards) > 1:
            raise ValueError("multiple Safetensors files require an index")
        return cls(root, _read_shard(shards[0]))

    def keys(self) -> tuple[str, ...]:
        return tuple(self._tensors)

    def tensor_info(self, name: str) -> TensorInfo:
        try:
            return self._tensors[name]
        except KeyError as error:
            raise KeyError(f"unknown tensor: {name}") from error

    def load_tensor(self, name: str) -> np.ndarray:
        """Load exactly one named tensor as a CPU NumPy array."""

        info = self.tensor_info(name)
        path = self.directory / info.shard
        start, end = info.data_offsets
        with path.open("rb") as handle:
            handle.seek(info.data_start + start)
            payload = handle.read(end - start)
        if len(payload) != info.nbytes:
            raise ValueError(f"tensor payload is truncated: {name}")

        if info.dtype == "BF16":
            words = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
            array = (words << 16).view(np.float32)
        else:
            try:
                dtype = _NUMPY_DTYPES[info.dtype]
            except KeyError as error:
                raise ValueError(
                    f"dtype {info.dtype} cannot be represented by the NumPy reader"
                ) from error
            array = np.frombuffer(payload, dtype=dtype).copy()
        return array.reshape(info.shape)


def _read_indexed_checkpoint(root: Path, index_path: Path) -> dict[str, TensorInfo]:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Safetensors index is not valid JSON: {index_path.name}") from error
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("Safetensors index must contain an object weight_map")

    weight_map = index["weight_map"]
    shard_cache: dict[str, dict[str, TensorInfo]] = {}
    tensors: dict[str, TensorInfo] = {}
    for name, shard_name in sorted(weight_map.items()):
        if not isinstance(name, str) or not isinstance(shard_name, str):
            raise ValueError("Safetensors weight_map keys and values must be strings")
        _validate_shard_name(shard_name)
        shard_path = root / shard_name
        if not shard_path.is_file():
            raise ValueError(f"index references missing shard: {shard_name}")
        if shard_name not in shard_cache:
            shard_cache[shard_name] = _read_shard(shard_path)
        shard_tensors = shard_cache[shard_name]
        if name not in shard_tensors:
            raise ValueError(f"index maps {name} to {shard_name}, but the tensor is absent")
        tensors[name] = shard_tensors[name]
    return tensors


def _validate_shard_name(name: str) -> None:
    path = PurePath(name)
    if path.is_absolute() or len(path.parts) != 1 or name in {"", ".", ".."}:
        raise ValueError(f"unsafe shard path in Safetensors index: {name!r}")
    if not name.endswith(".safetensors"):
        raise ValueError(f"shard does not use .safetensors suffix: {name}")


def _read_shard(path: Path) -> dict[str, TensorInfo]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Safetensors header is truncated: {path.name}")
        (header_length,) = struct.unpack("<Q", prefix)
        if header_length <= 0 or header_length > file_size - 8:
            raise ValueError(f"Safetensors header length is out of bounds: {path.name}")
        try:
            header = json.loads(handle.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Safetensors header is not valid JSON: {path.name}") from error
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header must be an object: {path.name}")

    data_start = 8 + header_length
    data_size = file_size - data_start
    tensors: dict[str, TensorInfo] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        info = _parse_tensor_info(name, metadata, path.name, data_start, data_size)
        tensors[name] = info
        ranges.append((info.data_offsets[0], info.data_offsets[1], name))

    for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                f"tensor byte ranges overlap in {path.name}: {previous[2]} and {current[2]}"
            )
    return tensors


def _parse_tensor_info(
    name: str,
    metadata: Any,
    shard: str,
    data_start: int,
    data_size: int,
) -> TensorInfo:
    if not isinstance(name, str) or not name:
        raise ValueError(f"Safetensors tensor names must be non-empty strings: {shard}")
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata for {name} must be an object")
    dtype = metadata.get("dtype")
    shape = metadata.get("shape")
    offsets = metadata.get("data_offsets")
    if dtype not in _DTYPE_WIDTHS:
        raise ValueError(f"unsupported Safetensors dtype for {name}: {dtype!r}")
    if not isinstance(shape, list):
        raise ValueError(f"shape for {name} must be an array")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape):
        raise ValueError(f"shape for {name} must contain non-negative integers")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
    ):
        raise ValueError(f"data_offsets for {name} must contain two integers")
    start, end = offsets
    if start < 0 or end < start or end > data_size:
        raise ValueError(f"data_offsets for {name} are out of bounds")

    expected_nbytes = math.prod(shape) * _DTYPE_WIDTHS[dtype]
    if end - start != expected_nbytes:
        raise ValueError(
            f"shape and dtype for {name} require {expected_nbytes} bytes, "
            f"but offsets span {end - start}"
        )
    return TensorInfo(
        name=name,
        dtype=dtype,
        shape=tuple(shape),
        shard=shard,
        data_offsets=(start, end),
        data_start=data_start,
    )
