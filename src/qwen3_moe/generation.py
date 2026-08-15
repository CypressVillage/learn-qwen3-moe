"""Next-token selection from Causal LM logits."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qwen3_moe.cache import KVCache

def last_token_logits(logits: np.ndarray) -> np.ndarray:
    """Return the vocabulary logits after the final input token."""
    logits = np.asarray(logits)
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B,S,V]")
    if logits.shape[1] == 0:
        raise ValueError("logits must contain at least one sequence position")
    if logits.shape[2] == 0:
        raise ValueError("logits must contain at least one vocabulary entry")
    return logits[:, -1, :]


def greedy_next_token(logits: np.ndarray) -> np.ndarray:
    """Choose the highest-logit token for each batch item."""
    return np.argmax(last_token_logits(logits), axis=-1)


def next_token_probabilities(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert final-position logits to probabilities with temperature."""
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a finite positive number")

    scaled_logits = last_token_logits(logits) / temperature
    shifted_logits = scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
    unnormalized = np.exp(shifted_logits)
    return unnormalized / np.sum(unnormalized, axis=-1, keepdims=True)


def sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample one token ID per batch item from final-position logits."""
    probabilities = next_token_probabilities(logits, temperature)
    rng = np.random.default_rng() if rng is None else rng
    vocabulary_size = probabilities.shape[-1]
    return np.asarray(
        [
            rng.choice(vocabulary_size, p=batch_probabilities)
            for batch_probabilities in probabilities
        ],
        dtype=np.int64,
    )


def generate_token_ids(
    model: object,
    prompt_token_ids: np.ndarray,
    max_new_tokens: int,
    *,
    eos_token_id: int | None = None,
    temperature: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate one continuation with cached prefill and decode."""
    prompt_token_ids = np.asarray(prompt_token_ids)
    if prompt_token_ids.ndim != 2 or prompt_token_ids.shape[0] != 1:
        raise ValueError("generation expects one prompt with shape [1,S]")
    if prompt_token_ids.shape[1] == 0:
        raise ValueError("generation requires at least one prompt token")
    if not np.issubdtype(prompt_token_ids.dtype, np.integer):
        raise ValueError("prompt token IDs must be integers")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    generated = prompt_token_ids.astype(np.int64, copy=True)
    if max_new_tokens == 0:
        return generated

    layers = getattr(model, "layers", None)
    if not isinstance(layers, list) or not layers:
        raise ValueError("model must expose a non-empty decoder layer list")
    cache = KVCache(len(layers))
    logits = model.cached(generated, cache)

    for step in range(max_new_tokens):
        next_token = (
            greedy_next_token(logits)
            if temperature is None
            else sample_next_token(logits, temperature, rng)
        )
        generated = np.concatenate((generated, next_token[:, None]), axis=1)
        if eos_token_id is not None and next_token[0] == eos_token_id:
            break
        if step + 1 == max_new_tokens:
            break
        logits = model.cached(next_token[:, None], cache)
    return generated


def _load_model_directory(model_directory: str | Path) -> tuple[object, object]:
    from qwen3_moe.checkpoint import SafetensorsCheckpoint
    from qwen3_moe.config import Qwen3MoeConfig
    from qwen3_moe.model import Qwen3MoeForCausalLM
    from qwen3_moe.tokenizer import Qwen3Tokenizer

    model_directory = Path(model_directory)
    config = Qwen3MoeConfig.from_json(model_directory / "config.json")
    checkpoint = SafetensorsCheckpoint.from_directory(model_directory)
    tokenizer = Qwen3Tokenizer.from_file(model_directory / "tokenizer.json")
    model = Qwen3MoeForCausalLM.from_checkpoint(config, checkpoint)
    return tokenizer, model


def generate_text(
    model_directory: str | Path,
    prompt: str,
    max_new_tokens: int,
    *,
    eos_token_id: int | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    skip_special_tokens: bool = True,
) -> str:
    """Load a Qwen3 MoE directory and generate decoded text from one prompt."""
    tokenizer, model = _load_model_directory(model_directory)
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")

    rng = np.random.default_rng(seed) if temperature is not None else None
    generated_ids = generate_token_ids(
        model,
        np.asarray([prompt_ids], dtype=np.int64),
        max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        rng=rng,
    )
    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=skip_special_tokens,
    )
