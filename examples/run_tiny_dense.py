"""Run the cumulative dense model on a deterministic CPU example."""

import torch

from qwen3_moe import DenseConfig, TinyDenseCausalLM


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
        [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long, device="cpu"
    )

    with torch.inference_mode():
        logits = model(input_ids)

    print(f"input_ids: {tuple(input_ids.shape)} {input_ids.dtype} {input_ids.device}")
    print(f"logits: {tuple(logits.shape)} {logits.dtype} {logits.device}")
    print(f"finite: {bool(torch.isfinite(logits).all())}")


if __name__ == "__main__":
    main()
