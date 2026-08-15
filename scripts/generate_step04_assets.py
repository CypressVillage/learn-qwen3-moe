"""Generate the Step 04 cumulative RoPE checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe.rope import (
    RotaryEmbedding,
    _rotate_half,
    apply_rotary_position_embedding,
)


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step04.json"


def _source_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _through(symbol: object) -> str:
    relative_path = "src/qwen3_moe/rope.py"
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, start = inspect.getsourcelines(symbol)
    symbol_lines = inspect.getsource(symbol).splitlines(keepends=True)
    return "".join(lines[: start - 1 + len(symbol_lines)])


def _package_without_rope() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "qwen3_moe.rope" not in line
        and "qwen3_moe.attention" not in line
        and "qwen3_moe.moe" not in line
        and "qwen3_moe.model" not in line
        and not any(
            f'"{name}"' in line
            for name in (
                "RotaryEmbedding",
                "Qwen3Attention",
                "Qwen3DecoderLayer",
                "Qwen3MoeExperts",
                "Qwen3MoeForCausalLM",
                "Qwen3MoeRouter",
                "Qwen3SparseMoeBlock",
                "apply_rotary_position_embedding",
            )
        )
    )


def _package_through_rope() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "qwen3_moe.attention" not in line
        and "qwen3_moe.moe" not in line
        and "qwen3_moe.model" not in line
        and '"Qwen3Attention"' not in line
        and not any(
            f'"{name}"' in line
            for name in (
                "Qwen3DecoderLayer",
                "Qwen3MoeExperts",
                "Qwen3MoeForCausalLM",
                "Qwen3MoeRouter",
                "Qwen3SparseMoeBlock",
            )
        )
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
        "src/qwen3_moe/__init__.py",
    ]
    config_path, checkpoint_path, tokenizer_path, layers_path, rope_path, package_path = source_files
    initial = {
        config_path: _source_text(config_path),
        checkpoint_path: _source_text(checkpoint_path),
        tokenizer_path: _source_text(tokenizer_path),
        layers_path: _source_text(layers_path),
        rope_path: "",
        package_path: _package_without_rope(),
    }
    inverse_frequencies = {
        **initial,
        rope_path: _through(RotaryEmbedding.__init__),
    }
    position_angles = {
        **inverse_frequencies,
        rope_path: _through(RotaryEmbedding.__call__),
    }
    rotate_half = {
        **position_angles,
        rope_path: _through(_rotate_half),
    }
    apply_rotation = {
        **rotate_half,
        rope_path: _source_text(rope_path),
    }
    package_export = {
        **apply_rotation,
        package_path: _package_through_rope(),
    }
    staged = [
        ("step04-ready", "带着 projected hidden states 来到空 RoPE", rope_path, initial, "", 0, "initial", []),
        ("step04-inverse-frequencies", "为每个旋转平面建立逆频率", rope_path, inverse_frequencies, "class RotaryEmbedding", 16, "insert", [rope_path]),
        ("step04-position-angles", "把 position IDs 展开成 cosine 与 sine", rope_path, position_angles, "def __call__", 14, "insert", [rope_path]),
        ("step04-rotate-half", "把 head vector 的两半配成旋转平面", rope_path, rotate_half, "def _rotate_half", 4, "insert", [rope_path]),
        ("step04-apply", "把位置信息同时写入 Query 和 Key", rope_path, apply_rotation, "def apply_rotary_position_embedding", 31, "insert", [rope_path]),
        ("step04-package", "从包入口导出 RoPE 能力", package_path, package_export, "qwen3_moe.rope", 13, "insert", [package_path]),
    ]
    checkpoints = []
    for checkpoint_id, label, active_file, contents, marker, count, kind, files in staged:
        focus = (
            {"start": 0, "end": 0, "symbol": ""}
            if not marker
            else _focus_range(contents[active_file], marker, count)
        )
        checkpoints.append(
            {
                "id": checkpoint_id,
                "step": "step04",
                "label": label,
                "active_file": active_file,
                "focus_range": focus,
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
