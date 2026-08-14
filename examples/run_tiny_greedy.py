"""Show the autoregressive loop around a randomly initialized tiny model."""

import torch
from torch import Tensor, nn

from qwen3_moe import DenseConfig, TinyDenseCausalLM


def greedy_generate(
    model: nn.Module, input_ids: Tensor, max_new_tokens: int
) -> tuple[Tensor, list[Tensor]]:
    """Append one greedy token per step and expose each step for teaching."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    generated = input_ids
    new_tokens: list[Tensor] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            new_tokens.append(next_token)
            generated = torch.cat((generated, next_token), dim=1)
    return generated, new_tokens


def main() -> None:
    torch.manual_seed(7)
    config = DenseConfig(
        vocab_size=11,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rope_theta=100.0,
    )
    model = TinyDenseCausalLM(config).eval().to("cpu")
    input_ids = torch.tensor(
        [[1, 2, 3], [4, 5, 6]], dtype=torch.long, device="cpu"
    )

    generated, new_tokens = greedy_generate(model, input_ids, max_new_tokens=3)

    print("Random weights: token IDs demonstrate control flow, not language quality.")
    print(f"input_ids: {input_ids.tolist()}")
    for step, token in enumerate(new_tokens, start=1):
        print(f"step {step}: next_token={token.squeeze(-1).tolist()}")
    print(f"generated: {generated.tolist()}")


if __name__ == "__main__":
    main()
