"""Dense feed-forward blocks used as the basis for future experts."""

import torch.nn.functional as F
from torch import Tensor, nn

from qwen3_moe.config import DenseConfig
from qwen3_moe.debug import require_finite, require_floating


class SwiGLU(nn.Module):
    """Bias-free SwiGLU feed-forward block."""

    def __init__(self, config: DenseConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(
        self, hidden_states: Tensor, *, return_debug: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        require_floating("SwiGLU input", hidden_states)
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"SwiGLU expected hidden size {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )

        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        product = F.silu(gate) * up
        update = self.down_proj(product)
        require_finite("SwiGLU product", product)
        require_finite("SwiGLU update", update)
        if not return_debug:
            return update
        return update, {"gate": gate, "up": up, "product": product}
