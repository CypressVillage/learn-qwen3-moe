"""Generate the Step 11 cumulative Cached Attention checkpoints."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step11.json"
ATTENTION_PATH = "src/qwen3_moe/attention.py"

CACHE_IMPORT = "from qwen3_moe.cache import KVCache\n"
CACHED_METHOD = '''
    def cached(
        self,
        hidden_states: np.ndarray,
        position_ids: np.ndarray,
        cache: KVCache,
        layer_index: int,
    ) -> np.ndarray:
        """Attend with Key/Value states appended to one layer's cache."""
        hidden_states = np.asarray(hidden_states)
        position_ids = np.asarray(position_ids)
        if hidden_states.ndim != 3:
            raise ValueError("attention input must have shape [B,S,D]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("attention input hidden size does not match config")
        if hidden_states.shape[1] == 0:
            raise ValueError("attention requires at least one token")
        if position_ids.shape != hidden_states.shape[:2]:
            raise ValueError("position IDs must match attention batch and sequence")

        query, key, value = self._project_query_key_value(hidden_states)
        query, key = self._apply_positions(query, key, position_ids)
        key, value = cache.update(layer_index, key, value)
        attended = self._scaled_dot_product_attention(query, key, value)
        batch_size, _, sequence_length, _ = attended.shape
        merged = attended.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        return self.o_proj(merged)
'''
RECTANGULAR_MASK = '''        if key.shape[2] != sequence_length:
            cached_length = key.shape[2] - sequence_length
            future_tokens = np.triu(
                np.ones((sequence_length, key.shape[2]), dtype=bool),
                k=cached_length + 1,
            )
'''


def _source_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _remove(source: str, *blocks: str) -> str:
    for block in blocks:
        source = source.replace(block, "")
    return source


def _focus_range(content: str, marker: str, line_count: int) -> dict[str, object]:
    lines = content.splitlines()
    start = next(index for index, line in enumerate(lines, start=1) if marker in line)
    return {
        "start": start,
        "end": min(len(lines), start + line_count - 1),
        "symbol": marker,
    }


def _snapshot(source_files: list[str], contents: dict[str, str]) -> dict[str, object]:
    return {
        "files": [
            {"path": relative_path, "content": contents[relative_path]}
            for relative_path in source_files
        ]
    }


def generate() -> None:
    source_files = [
        "src/qwen3_moe/config.py",
        "src/qwen3_moe/checkpoint.py",
        "src/qwen3_moe/tokenizer.py",
        "src/qwen3_moe/layers.py",
        "src/qwen3_moe/rope.py",
        ATTENTION_PATH,
        "src/qwen3_moe/moe.py",
        "src/qwen3_moe/model.py",
        "src/qwen3_moe/generation.py",
        "src/qwen3_moe/cache.py",
        "src/qwen3_moe/__init__.py",
    ]
    current_attention = _source_text(ATTENTION_PATH)
    initial_attention = _remove(
        current_attention,
        CACHE_IMPORT,
        CACHED_METHOD,
        RECTANGULAR_MASK,
    )
    initial = {
        relative_path: (
            initial_attention
            if relative_path == ATTENTION_PATH
            else _source_text(relative_path)
        )
        for relative_path in source_files
    }
    cache_entry = {
        **initial,
        ATTENTION_PATH: _remove(
            current_attention,
            RECTANGULAR_MASK,
        ),
    }
    mask = {**cache_entry, ATTENTION_PATH: current_attention}
    staged = [
        ("step11-ready", "从独立 KV Cache 进入 Attention 缓存读取", ATTENTION_PATH, initial, "class Qwen3Attention", 8, "initial", []),
        ("step11-cache-update", "用 cached 入口追加并读取完整 Key/Value", ATTENTION_PATH, cache_entry, "def cached", 31, "insert", [ATTENTION_PATH]),
        ("step11-causal-mask", "为历史 Key 构造矩形 causal mask", ATTENTION_PATH, mask, "cached_length", 10, "insert", [ATTENTION_PATH]),
    ]
    checkpoints = []
    for checkpoint_id, label, active_file, contents, marker, count, kind, files in staged:
        checkpoints.append(
            {
                "id": checkpoint_id,
                "step": "step11",
                "label": label,
                "active_file": active_file,
                "focus_range": _focus_range(contents[active_file], marker, count),
                "repository_snapshot": _snapshot(source_files, contents),
                "diff": {"kind": kind, "files": files},
            }
        )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps(
            {"schema_version": 1, "checkpoints": checkpoints},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    generate()
