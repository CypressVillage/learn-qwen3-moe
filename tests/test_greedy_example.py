from pathlib import Path
from runpy import run_path

import pytest
import torch

from qwen3_moe import DenseConfig, TinyDenseCausalLM


greedy_generate = run_path(
    str(Path(__file__).parents[1] / "examples" / "run_tiny_greedy.py")
)["greedy_generate"]


def make_model(seed: int = 7) -> TinyDenseCausalLM:
    torch.manual_seed(seed)
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
    return TinyDenseCausalLM(config).eval().to("cpu")


def test_greedy_generate_appends_one_valid_token_per_step():
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    generated, new_tokens = greedy_generate(model, input_ids, max_new_tokens=3)

    assert generated.shape == (2, 6)
    assert torch.equal(generated[:, :3], input_ids)
    assert len(new_tokens) == 3
    for step, token in enumerate(new_tokens, start=1):
        assert token.shape == (2, 1)
        assert token.dtype == torch.long
        assert (token >= 0).all()
        assert (token < model.config.vocab_size).all()
        assert torch.equal(generated[:, 3 + step - 1 : 3 + step], token)


def test_greedy_generate_is_deterministic_for_same_model_and_input():
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    first, first_tokens = greedy_generate(make_model(), input_ids, 3)
    second, second_tokens = greedy_generate(make_model(), input_ids, 3)

    assert torch.equal(first, second)
    assert all(torch.equal(a, b) for a, b in zip(first_tokens, second_tokens))


def test_greedy_generate_rejects_negative_length():
    with pytest.raises(ValueError, match="non-negative"):
        greedy_generate(make_model(), torch.tensor([[1, 2, 3]]), -1)
