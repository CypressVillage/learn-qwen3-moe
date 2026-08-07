"""Small validation helpers shared by the teaching modules."""

import torch
from torch import Tensor


def require_floating(name: str, value: Tensor) -> None:
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")


def require_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def require_same_tensor_contract(name: str, left: Tensor, right: Tensor) -> None:
    if left.shape != right.shape:
        raise ValueError(
            f"{name} shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    if left.dtype != right.dtype:
        raise ValueError(f"{name} dtype mismatch: {left.dtype} vs {right.dtype}")
    if left.device != right.device:
        raise ValueError(f"{name} device mismatch: {left.device} vs {right.device}")
