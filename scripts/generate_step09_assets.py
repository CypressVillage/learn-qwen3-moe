"""Generate the Step 09 cumulative next-token selection checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe.attention import Qwen3Attention
from qwen3_moe.generation import (
    greedy_next_token,
    last_token_logits,
    next_token_probabilities,
    sample_next_token,
)
from qwen3_moe.model import Qwen3DecoderLayer, Qwen3MoeForCausalLM


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step09.json"
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
    relative_path = "src/qwen3_moe/generation.py"
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, start = inspect.getsourcelines(symbol)
    symbol_lines = inspect.getsource(symbol).splitlines(keepends=True)
    source = "".join(lines[: start - 1 + len(symbol_lines)])
    return source.replace(PATH_IMPORT, "").replace(CACHE_IMPORT, "")


def _method_source(symbol: object) -> str:
    return "\n" + inspect.getsource(symbol)


def _prefill_attention() -> str:
    lines = _source_text("src/qwen3_moe/attention.py").splitlines(keepends=True)
    _, start = inspect.getsourcelines(Qwen3Attention.__call__)
    method_lines = inspect.getsource(Qwen3Attention.__call__).splitlines(keepends=True)
    source = "".join(lines[: start - 1 + len(method_lines)])
    return source.replace(CACHE_IMPORT, "").replace(RECTANGULAR_MASK, "")


def _prefill_model() -> str:
    source = _source_text("src/qwen3_moe/model.py")
    for block in (
        CACHE_IMPORT,
        _method_source(Qwen3DecoderLayer._cached_attention_block),
        _method_source(Qwen3DecoderLayer.cached),
        _method_source(Qwen3MoeForCausalLM._cached_position_ids),
        _method_source(Qwen3MoeForCausalLM.cached),
    ):
        source = source.replace(block, "")
    return source


def _step09_generation() -> str:
    return _through(sample_next_token)


def _package_through_generation() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "generate_token_ids" not in line
        and "generate_text" not in line
    )


def _package_without_generation() -> str:
    symbols = (
        "greedy_next_token",
        "last_token_logits",
        "next_token_probabilities",
        "sample_next_token",
        "generate_token_ids",
        "generate_text",
    )
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "qwen3_moe.generation" not in line
        and "qwen3_moe.cache" not in line
        and '"KVCache"' not in line
        and not any(f'"{symbol}"' in line for symbol in symbols)
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
        "src/qwen3_moe/__init__.py",
    ]
    generation_path = "src/qwen3_moe/generation.py"
    package_path = "src/qwen3_moe/__init__.py"
    initial = {
        relative_path: (
            ""
            if relative_path == generation_path
            else _package_without_generation()
            if relative_path == package_path
            else _prefill_attention()
            if relative_path == "src/qwen3_moe/attention.py"
            else _prefill_model()
            if relative_path == "src/qwen3_moe/model.py"
            else _source_text(relative_path)
        )
        for relative_path in source_files
    }
    final_logits = {
        **initial,
        generation_path: _through(last_token_logits),
    }
    greedy = {
        **final_logits,
        generation_path: _through(greedy_next_token),
    }
    probabilities = {
        **greedy,
        generation_path: _through(next_token_probabilities),
    }
    sampling = {
        **probabilities,
        generation_path: _step09_generation(),
    }
    package_export = {
        **sampling,
        package_path: _package_through_generation(),
    }
    staged = [
        ("step09-ready", "从完整 vocabulary logits 进入 token 选择", generation_path, initial, "", 0, "initial", []),
        ("step09-final-logits", "只取读完整段 prompt 后的 logits", generation_path, final_logits, "def last_token_logits", 13, "insert", [generation_path]),
        ("step09-greedy", "用 argmax 选择分数最高的 token", generation_path, greedy, "def greedy_next_token", 4, "insert", [generation_path]),
        ("step09-temperature", "用 temperature 和稳定 softmax 得到概率", generation_path, probabilities, "def next_token_probabilities", 16, "insert", [generation_path]),
        ("step09-sampling", "从 categorical distribution 采样 token", generation_path, sampling, "def sample_next_token", 20, "insert", [generation_path]),
        ("step09-package", "从包入口导出 next-token selection", package_path, package_export, "greedy_next_token", 22, "insert", [package_path]),
    ]
    checkpoints = []
    for checkpoint_id, label, active_file, contents, marker, count, kind, files in staged:
        checkpoints.append(
            {
                "id": checkpoint_id,
                "step": "step09",
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
