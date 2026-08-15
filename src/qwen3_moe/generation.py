"""Next-token selection from Causal LM logits."""

from __future__ import annotations

import numpy as np


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
