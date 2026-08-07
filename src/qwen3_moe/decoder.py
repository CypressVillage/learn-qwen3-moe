"""Pre-norm dense decoder layer."""

from torch import Tensor, nn

from qwen3_moe.attention import GroupedQueryAttention
from qwen3_moe.config import DenseConfig
from qwen3_moe.debug import require_finite, require_same_tensor_contract
from qwen3_moe.mlp import SwiGLU
from qwen3_moe.norms import RMSNorm


class DenseDecoderLayer(nn.Module):
    """Combine causal GQA and SwiGLU with two pre-norm residual blocks."""

    def __init__(self, config: DenseConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = SwiGLU(config)

    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        attention_mask: Tensor | None = None,
        *,
        return_debug: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        attention_norm = self.input_norm(hidden_states)
        if return_debug:
            attention_update, attention_debug = self.self_attn(
                attention_norm,
                positions,
                attention_mask,
                return_debug=True,
            )
        else:
            attention_update = self.self_attn(
                attention_norm, positions, attention_mask
            )
        require_same_tensor_contract(
            "attention residual", hidden_states, attention_update
        )
        after_attention = hidden_states + attention_update

        mlp_norm = self.post_attention_norm(after_attention)
        if return_debug:
            mlp_update, mlp_debug = self.mlp(mlp_norm, return_debug=True)
        else:
            mlp_update = self.mlp(mlp_norm)
        require_same_tensor_contract("MLP residual", after_attention, mlp_update)
        output = after_attention + mlp_update
        require_finite("decoder output", output)

        if not return_debug:
            return output
        debug = {
            "attention_norm": attention_norm,
            "attention_update": attention_update,
            "after_attention": after_attention,
            "mlp_norm": mlp_norm,
            "mlp_update": mlp_update,
            "output": output,
        }
        debug.update(
            {f"attention.{name}": value for name, value in attention_debug.items()}
        )
        debug.update({f"mlp.{name}": value for name, value in mlp_debug.items()})
        return output, debug
