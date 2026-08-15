"""Generate the Step 07 cumulative decoder layer checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe.model import Qwen3DecoderLayer


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step07.json"


def _source_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _through(symbol: object) -> str:
    relative_path = "src/qwen3_moe/model.py"
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, start = inspect.getsourcelines(symbol)
    symbol_lines = inspect.getsource(symbol).splitlines(keepends=True)
    source = "".join(lines[: start - 1 + len(symbol_lines)])
    return "".join(
        line
        for line in source.splitlines(keepends=True)
        if "qwen3_moe.checkpoint" not in line
        and "from qwen3_moe.layers import Embedding, Linear" not in line
    )


def _package_without_decoder_layer() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "qwen3_moe.model" not in line
        and "qwen3_moe.generation" not in line
        and "qwen3_moe.cache" not in line
        and '"Qwen3DecoderLayer"' not in line
        and '"Qwen3MoeForCausalLM"' not in line
        and not any(
            f'"{name}"' in line
            for name in (
                "greedy_next_token",
                "last_token_logits",
                "next_token_probabilities",
                "sample_next_token",
                "KVCache",
            )
        )
    )


def _package_with_decoder_layer() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "Qwen3MoeForCausalLM" not in line
        and "qwen3_moe.generation" not in line
        and "qwen3_moe.cache" not in line
        and not any(
            f'"{name}"' in line
            for name in (
                "greedy_next_token",
                "last_token_logits",
                "next_token_probabilities",
                "sample_next_token",
                "KVCache",
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
        "src/qwen3_moe/attention.py",
        "src/qwen3_moe/moe.py",
        "src/qwen3_moe/model.py",
        "src/qwen3_moe/__init__.py",
    ]
    model_path = "src/qwen3_moe/model.py"
    package_path = "src/qwen3_moe/__init__.py"
    initial = {
        relative_path: (
            ""
            if relative_path == model_path
            else _package_without_decoder_layer()
            if relative_path == package_path
            else _source_text(relative_path)
        )
        for relative_path in source_files
    }
    components = {
        **initial,
        model_path: _through(Qwen3DecoderLayer.__init__),
    }
    attention_residual = {
        **components,
        model_path: _through(Qwen3DecoderLayer._attention_block),
    }
    moe_residual = {
        **attention_residual,
        model_path: _through(Qwen3DecoderLayer._moe_block),
    }
    forward = {
        **moe_residual,
        model_path: _through(Qwen3DecoderLayer.__call__),
    }
    package_export = {
        **forward,
        package_path: _package_with_decoder_layer(),
    }
    staged = [
        ("step07-ready", "带着 Attention 与 MoE 积木进入空 model.py", model_path, initial, "", 0, "initial", []),
        ("step07-components", "接住一层权重并组装四个子模块", model_path, components, "class Qwen3DecoderLayer", 52, "insert", [model_path]),
        ("step07-attention-residual", "完成 pre-norm Attention 与第一条 residual", model_path, attention_residual, "def _attention_block", 9, "insert", [model_path]),
        ("step07-moe-residual", "完成 pre-norm MoE 与第二条 residual", model_path, moe_residual, "def _moe_block", 6, "insert", [model_path]),
        ("step07-forward", "串起完整 Decoder Layer 前向", model_path, forward, "def __call__", 15, "insert", [model_path]),
        ("step07-package", "从包入口导出 Decoder Layer", package_path, package_export, "qwen3_moe.model", 15, "insert", [package_path]),
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
                "step": "step07",
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
