"""Sparse mixture-of-experts feed-forward block for Qwen3 MoE."""

from __future__ import annotations

import numpy as np

from qwen3_moe.config import Qwen3MoeConfig


class Qwen3MoeRouter:
    """Choose the highest-probability experts for every token."""

    def __init__(self, config: Qwen3MoeConfig, weight: np.ndarray) -> None:
        expected_shape = (config.num_experts, config.hidden_size)
        if weight.shape != expected_shape:
            raise ValueError(f"router weight must have shape {expected_shape}")

        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = weight

    def __call__(self, hidden_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden_states = np.asarray(hidden_states)
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("router input must have shape [T,D]")

        values = hidden_states.astype(np.float32, copy=False)
        weight = self.weight.astype(np.float32, copy=False)
        logits = values @ weight.T
        logits -= np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)

        ranked_experts = np.argsort(probabilities, axis=-1)[:, ::-1]
        selected_experts = ranked_experts[:, : self.top_k]
        routing_weights = np.take_along_axis(
            probabilities, selected_experts, axis=-1
        )
        if self.norm_topk_prob:
            routing_weights /= np.sum(routing_weights, axis=-1, keepdims=True)
        return selected_experts, routing_weights


class Qwen3MoeExperts:
    """Run the selected SwiGLU experts and merge their weighted outputs."""

    def __init__(
        self,
        config: Qwen3MoeConfig,
        gate_up_proj_weight: np.ndarray,
        down_proj_weight: np.ndarray,
    ) -> None:
        expected_gate_up_shape = (
            config.num_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
        )
        expected_down_shape = (
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )
        if gate_up_proj_weight.shape != expected_gate_up_shape:
            raise ValueError(
                f"expert gate_up_proj weight must have shape {expected_gate_up_shape}"
            )
        if down_proj_weight.shape != expected_down_shape:
            raise ValueError(
                f"expert down_proj weight must have shape {expected_down_shape}"
            )

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.gate_up_proj_weight = gate_up_proj_weight
        self.down_proj_weight = down_proj_weight

    def __call__(
        self,
        hidden_states: np.ndarray,
        selected_experts: np.ndarray,
        routing_weights: np.ndarray,
    ) -> np.ndarray:
        hidden_states = np.asarray(hidden_states)
        selected_experts = np.asarray(selected_experts)
        routing_weights = np.asarray(routing_weights)
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("expert input must have shape [T,D]")
        if selected_experts.shape != routing_weights.shape:
            raise ValueError("selected experts and routing weights must have the same shape")
        if selected_experts.ndim != 2 or selected_experts.shape[0] != hidden_states.shape[0]:
            raise ValueError("expert selections must have shape [T,K]")
        if not np.issubdtype(selected_experts.dtype, np.integer):
            raise ValueError("selected expert IDs must be integers")
        if np.any(selected_experts < 0) or np.any(selected_experts >= self.num_experts):
            raise ValueError("selected expert ID is outside the expert range")

        values = hidden_states.astype(np.float32, copy=False)
        output = np.zeros_like(values)
        for expert_index in range(self.num_experts):
            token_indices, top_k_positions = np.where(
                selected_experts == expert_index
            )
            if token_indices.size == 0:
                continue

            expert_input = values[token_indices]
            gate_up_weight = self.gate_up_proj_weight[expert_index].astype(
                np.float32, copy=False
            )
            gate_up = expert_input @ gate_up_weight.T
            gate, up = np.split(gate_up, 2, axis=-1)
            exp_negative_abs = np.exp(-np.abs(gate))
            sigmoid = np.where(
                gate >= 0.0,
                1.0 / (1.0 + exp_negative_abs),
                exp_negative_abs / (1.0 + exp_negative_abs),
            )
            activated = gate * sigmoid * up
            down_weight = self.down_proj_weight[expert_index].astype(
                np.float32, copy=False
            )
            expert_output = activated @ down_weight.T
            weights = routing_weights[token_indices, top_k_positions, None]
            output[token_indices] += expert_output * weights
        return output


class Qwen3SparseMoeBlock:
    """Route normalized hidden states through a sparse set of experts."""

    def __init__(
        self,
        config: Qwen3MoeConfig,
        router_weight: np.ndarray,
        gate_up_proj_weight: np.ndarray,
        down_proj_weight: np.ndarray,
    ) -> None:
        self.hidden_size = config.hidden_size
        self.router = Qwen3MoeRouter(config, router_weight)
        self.experts = Qwen3MoeExperts(
            config, gate_up_proj_weight, down_proj_weight
        )

    def __call__(self, hidden_states: np.ndarray) -> np.ndarray:
        hidden_states = np.asarray(hidden_states)
        if hidden_states.ndim != 3:
            raise ValueError("MoE input must have shape [B,S,D]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("MoE input hidden size does not match config")

        batch_size, sequence_length, _ = hidden_states.shape
        tokens = hidden_states.reshape(-1, self.hidden_size)
        selected_experts, routing_weights = self.router(tokens)
        output = self.experts(tokens, selected_experts, routing_weights)
        return output.reshape(batch_size, sequence_length, self.hidden_size)
