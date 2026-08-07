"""A tiny dense causal language model assembled from reusable modules."""

import torch
from torch import Tensor, nn

from qwen3_moe.config import DenseConfig
from qwen3_moe.debug import require_finite
from qwen3_moe.decoder import DenseDecoderLayer
from qwen3_moe.norms import RMSNorm


class TinyDenseCausalLM(nn.Module):
    """CPU-friendly correctness model that can later move to CUDA unchanged."""

    def __init__(self, config: DenseConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DenseDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(
        self, input_ids: Tensor, *, return_debug: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be a rank-2 integer tensor")
        if input_ids.shape[0] <= 0 or input_ids.shape[1] <= 0:
            raise ValueError("batch and sequence dimensions must be positive")
        if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
            raise ValueError("input_ids contain a token outside the vocabulary")
        if input_ids.device != self.embedding.weight.device:
            raise ValueError("input_ids and model parameters must share a device")

        batch_size, sequence_length = input_ids.shape
        positions = torch.arange(
            sequence_length, dtype=torch.long, device=input_ids.device
        ).view(1, sequence_length).expand(batch_size, sequence_length)
        attention_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=input_ids.device,
            ),
            diagonal=1,
        ).view(1, 1, sequence_length, sequence_length)

        hidden_states = self.embedding(input_ids)
        debug: dict[str, Tensor] = {}
        if return_debug:
            debug["embedding"] = hidden_states
        for index, layer in enumerate(self.layers):
            if return_debug:
                hidden_states, layer_debug = layer(
                    hidden_states,
                    positions,
                    attention_mask,
                    return_debug=True,
                )
                debug.update(
                    {
                        f"layers.{index}.{name}": value
                        for name, value in layer_debug.items()
                    }
                )
            else:
                hidden_states = layer(
                    hidden_states, positions, attention_mask
                )

        normalized = self.final_norm(hidden_states)
        logits = self.lm_head(normalized)
        require_finite("logits", logits)
        if not return_debug:
            return logits
        debug["final_norm"] = normalized
        debug["logits"] = logits
        return logits, debug
