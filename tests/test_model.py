import copy

import pytest
import torch

from qwen3_moe import DenseConfig, TinyDenseCausalLM


def make_config(**overrides):
    values = {
        "vocab_size": 11,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rope_theta": 100.0,
    }
    values.update(overrides)
    return DenseConfig(**values)


def test_tiny_dense_model_runs_end_to_end_on_cpu():
    torch.manual_seed(7)
    model = TinyDenseCausalLM(make_config()).eval().to("cpu")
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    with torch.inference_mode():
        logits, debug = model(input_ids, return_debug=True)

    assert logits.shape == (2, 4, 11)
    assert debug["embedding"].shape == (2, 4, 8)
    assert debug["layers.0.output"].shape == (2, 4, 8)
    assert debug["layers.1.output"].shape == (2, 4, 8)
    assert all(value.device.type == "cpu" for value in debug.values())
    assert all(torch.isfinite(value).all() for value in debug.values())
    assert model.layers[0] is not model.layers[1]
    assert model.layers[0].input_norm is not model.layers[0].post_attention_norm


def test_causal_prefix_logits_do_not_depend_on_a_future_token():
    torch.manual_seed(7)
    model = TinyDenseCausalLM(make_config()).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[1, 2, 3, 9]])

    with torch.inference_mode():
        first_logits = model(first)
        second_logits = model(second)

    torch.testing.assert_close(
        first_logits[:, :3], second_logits[:, :3], atol=1e-6, rtol=1e-6
    )


def test_copy_is_identical_then_active_layer_change_reaches_logits():
    torch.manual_seed(7)
    model = TinyDenseCausalLM(make_config()).eval()
    copied = copy.deepcopy(model)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.inference_mode():
        baseline = model(input_ids)
        copied_baseline = copied(input_ids)
    torch.testing.assert_close(baseline, copied_baseline, atol=0, rtol=0)

    with torch.no_grad():
        copied.layers[1].mlp.down_proj.weight[0, 0] += 0.25
    with torch.inference_mode():
        changed = copied(input_ids)
    assert not torch.allclose(baseline, changed)


def test_tied_embeddings_share_the_same_parameter():
    model = TinyDenseCausalLM(make_config(tie_word_embeddings=True))

    assert model.embedding.weight is model.lm_head.weight


@pytest.mark.parametrize(
    "input_ids, expected",
    [
        (torch.zeros(1, 2), "integer"),
        (torch.tensor([[0, 11]]), "vocabulary"),
        (torch.empty(0, 2, dtype=torch.long), "positive"),
    ],
)
def test_model_rejects_invalid_token_ids(input_ids, expected):
    model = TinyDenseCausalLM(make_config())

    with pytest.raises(ValueError, match=expected):
        model(input_ids)
