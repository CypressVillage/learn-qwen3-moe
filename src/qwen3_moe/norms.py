"""Normalization layers used by the dense and future MoE decoders."""

import torch
from torch import Tensor, nn

from qwen3_moe.debug import require_finite, require_floating


class RMSNorm(nn.Module):
    """RMS normalization with FP32 statistics and no mean subtraction."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        require_floating("RMSNorm input", hidden_states)
        if hidden_states.shape[-1] != self.weight.numel():
            raise ValueError(
                f"RMSNorm expected last dimension {self.weight.numel()}, "
                f"got {hidden_states.shape[-1]}"
            )
        if hidden_states.device != self.weight.device:
            raise ValueError("RMSNorm input and weight must be on the same device")
        if hidden_states.dtype != self.weight.dtype:
            raise ValueError("RMSNorm input and weight must have the same dtype")

        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        output = normalized.to(input_dtype) * self.weight
        require_finite("RMSNorm output", output)
        return output
