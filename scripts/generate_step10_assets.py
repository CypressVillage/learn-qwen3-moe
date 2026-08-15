"""Generate the Step 10 cumulative KV Cache checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe.attention import Qwen3Attention
from qwen3_moe.cache import KVCache
from qwen3_moe.generation import sample_next_token
from qwen3_moe.model import Qwen3DecoderLayer, Qwen3MoeForCausalLM


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step10.json"
PATH_IMPORT = "from pathlib import Path\n\n"
CACHE_IMPORT = "from qwen3_moe.cache import KVCache\n"
RECTANGULAR_MASK = '''        if key.shape[2] != sequence_length:
            cached_length = key.shape[2] - sequence_length
            future_tokens = np.triu(
                np.ones((sequence_length, key.shape[2]), dtype=bool),
                k=cached_length + 1,
            )
'''


def _source_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _through(symbol: object) -> str:
    relative_path = "src/qwen3_moe/cache.py"
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, start = inspect.getsourcelines(symbol)
    symbol_lines = inspect.getsource(symbol).splitlines(keepends=True)
    return "".join(lines[: start - 1 + len(symbol_lines)])


def _source_through(relative_path: str, symbol: object) -> str:
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, start = inspect.getsourcelines(symbol)
    symbol_lines = inspect.getsource(symbol).splitlines(keepends=True)
    return "".join(lines[: start - 1 + len(symbol_lines)])


def _prefill_attention() -> str:
    source = _source_through("src/qwen3_moe/attention.py", Qwen3Attention.__call__)
    return source.replace(CACHE_IMPORT, "").replace(RECTANGULAR_MASK, "")


def _prefill_model() -> str:
    source = _source_text("src/qwen3_moe/model.py")
    for symbol in (
        Qwen3DecoderLayer._cached_attention_block,
        Qwen3DecoderLayer.cached,
        Qwen3MoeForCausalLM._cached_position_ids,
        Qwen3MoeForCausalLM.cached,
    ):
        source = source.replace("\n" + inspect.getsource(symbol), "")
    return source.replace(CACHE_IMPORT, "")


def _step09_generation() -> str:
    source = _source_through("src/qwen3_moe/generation.py", sample_next_token)
    return source.replace(PATH_IMPORT, "").replace(CACHE_IMPORT, "")


def _package_before_step13() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "generate_token_ids" not in line and "generate_text" not in line
    )


def _package_without_cache() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "qwen3_moe.cache" not in line
        and '"KVCache"' not in line
        and "generate_token_ids" not in line
        and "generate_text" not in line
    )


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
        "src/qwen3_moe/model.py",
        "src/qwen3_moe/generation.py",
        "src/qwen3_moe/cache.py",
        "src/qwen3_moe/__init__.py",
    ]
    cache_path = "src/qwen3_moe/cache.py"
    package_path = "src/qwen3_moe/__init__.py"
    initial = {
        relative_path: (
            ""
            if relative_path == cache_path
            else _package_without_cache()
            if relative_path == package_path
            else _prefill_attention()
            if relative_path == "src/qwen3_moe/attention.py"
            else _prefill_model()
            if relative_path == "src/qwen3_moe/model.py"
            else _step09_generation()
            if relative_path == "src/qwen3_moe/generation.py"
            else _source_text(relative_path)
        )
        for relative_path in source_files
    }
    storage = {**initial, cache_path: _through(KVCache.__init__)}
    lookup = {**storage, cache_path: _through(KVCache.get)}
    append = {**lookup, cache_path: _through(KVCache.update)}
    length = {**append, cache_path: _source_text(cache_path)}
    package_export = {**length, package_path: _package_before_step13()}
    staged = [
        ("step10-ready", "从 next token 进入每层 KV 状态保存", cache_path, initial, "", 0, "initial", []),
        ("step10-storage", "为每个 Decoder Layer 建立 Key/Value 槽位", cache_path, storage, "class KVCache", 17, "insert", [cache_path]),
        ("step10-lookup", "读取某一层已经保存的 Key 和 Value", cache_path, lookup, "def get", 11, "insert", [cache_path]),
        ("step10-append", "沿 sequence 维追加新 token 的 Key/Value", cache_path, append, "def update", 29, "insert", [cache_path]),
        ("step10-length", "暴露所有层一致的缓存序列长度", cache_path, length, "def sequence_length", 10, "insert", [cache_path]),
        ("step10-package", "从包入口导出 KV Cache", package_path, package_export, "KVCache", 18, "insert", [package_path]),
    ]
    checkpoints = []
    for checkpoint_id, label, active_file, contents, marker, count, kind, files in staged:
        checkpoints.append(
            {
                "id": checkpoint_id,
                "step": "step10",
                "label": label,
                "active_file": active_file,
                "focus_range": (
                    {"start": 0, "end": 0, "symbol": ""}
                    if not marker
                    else _focus_range(contents[active_file], marker, count)
                ),
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
