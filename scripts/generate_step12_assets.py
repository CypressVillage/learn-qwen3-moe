"""Generate the Step 12 cumulative full-model Cached Decode checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe.model import Qwen3DecoderLayer, Qwen3MoeForCausalLM


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step12.json"
MODEL_PATH = "src/qwen3_moe/model.py"
CACHE_IMPORT = "from qwen3_moe.cache import KVCache\n"


def _source_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _method_source(symbol: object) -> str:
    return "\n" + inspect.getsource(symbol)


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
        "src/qwen3_moe/attention.py",
        "src/qwen3_moe/moe.py",
        MODEL_PATH,
        "src/qwen3_moe/generation.py",
        "src/qwen3_moe/cache.py",
        "src/qwen3_moe/__init__.py",
    ]
    current_model = _source_text(MODEL_PATH)
    decoder_block = _method_source(Qwen3DecoderLayer._cached_attention_block)
    decoder_cached = _method_source(Qwen3DecoderLayer.cached)
    position_ids = _method_source(Qwen3MoeForCausalLM._cached_position_ids)
    model_cached = _method_source(Qwen3MoeForCausalLM.cached)
    initial_model = _remove(
        current_model,
        CACHE_IMPORT,
        decoder_block,
        decoder_cached,
        position_ids,
        model_cached,
    )
    initial = {
        relative_path: initial_model if relative_path == MODEL_PATH else _source_text(relative_path)
        for relative_path in source_files
    }
    decoder = {
        **initial,
        MODEL_PATH: _remove(current_model, position_ids, model_cached),
    }
    positions = {
        **decoder,
        MODEL_PATH: _remove(current_model, model_cached),
    }
    full_model = {**positions, MODEL_PATH: current_model}
    staged = [
        ("step12-ready", "从单层 Cached Attention 进入完整模型", MODEL_PATH, initial, "class Qwen3DecoderLayer", 8, "initial", []),
        ("step12-decoder-layer", "让 Decoder Layer 传递 cache 与 layer index", MODEL_PATH, decoder, "def _cached_attention_block", 42, "insert", [MODEL_PATH]),
        ("step12-positions", "从缓存长度生成接续的绝对位置", MODEL_PATH, positions, "def _cached_position_ids", 10, "insert", [MODEL_PATH]),
        ("step12-model", "串起全部层的 prefill 与 cached decode", MODEL_PATH, full_model, "def cached", 22, "insert", [MODEL_PATH]),
    ]
    checkpoints = []
    for checkpoint_id, label, active_file, contents, marker, count, kind, files in staged:
        checkpoints.append(
            {
                "id": checkpoint_id,
                "step": "step12",
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
