"""Generate the Step 01 cumulative source checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe import Qwen3MoeConfig, SafetensorsCheckpoint


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step01.json"


def _source_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _source_without_symbol(relative_path: str, symbol: object) -> str:
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, start = inspect.getsourcelines(symbol)
    symbol_lines = inspect.getsource(symbol).splitlines(keepends=True)
    return "".join(lines[: start - 1] + lines[start - 1 + len(symbol_lines) :])


def _config_contract_source() -> str:
    relative_path = "src/qwen3_moe/config.py"
    lines = _source_text(relative_path).splitlines(keepends=True)
    qwen_source = inspect.getsource(Qwen3MoeConfig).splitlines(keepends=True)
    validation_start = next(
        index for index, line in enumerate(qwen_source) if "def __post_init__" in line
    )
    return "".join(lines[:8] + qwen_source[:validation_start])


def _checkpoint_index_source() -> str:
    relative_path = "src/qwen3_moe/checkpoint.py"
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, load_start = inspect.getsourcelines(SafetensorsCheckpoint.load_tensor)
    return "".join(lines[: load_start - 1])


def _checkpoint_abstraction_source() -> str:
    relative_path = "src/qwen3_moe/checkpoint.py"
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, discovery_start = inspect.getsourcelines(
        SafetensorsCheckpoint.from_directory
    )
    return "".join(lines[: discovery_start - 1])


def _focus_range(
    content: str, marker: str, line_count: int, occurrence: int = 1
) -> dict[str, object]:
    lines = content.splitlines()
    matches = [index for index, line in enumerate(lines, start=1) if marker in line]
    start = matches[occurrence - 1]
    return {
        "start": start,
        "end": min(len(lines), start + line_count - 1),
        "symbol": marker,
    }


def _snapshot(source_files: list[str], contents: dict[str, str]) -> dict[str, object]:
    return {
        "files": [
            {"path": relative_path, "content": contents.get(relative_path, "")}
            for relative_path in source_files
        ]
    }


def generate() -> None:
    source_files = [
        "src/qwen3_moe/config.py",
        "src/qwen3_moe/checkpoint.py",
    ]
    config_path, checkpoint_path = source_files
    empty = {relative_path: "" for relative_path in source_files}
    config_contract = {**empty, config_path: _config_contract_source()}
    config_validation = {**config_contract, config_path: _source_text(config_path)}
    checkpoint_abstraction = {
        **config_validation,
        checkpoint_path: _checkpoint_abstraction_source(),
    }
    index_discovery = {
        **checkpoint_abstraction,
        checkpoint_path: _checkpoint_index_source(),
    }
    header_validation = {
        **index_discovery,
        checkpoint_path: _source_without_symbol(
            checkpoint_path, SafetensorsCheckpoint.load_tensor
        ),
    }
    tensor_loading = {
        **header_validation,
        checkpoint_path: _source_text(checkpoint_path),
    }
    staged_checkpoints = [
        {
            "id": "step01-empty-workspace",
            "label": "从两个空文件开始",
            "active_file": config_path,
            "focus_range": {"start": 0, "end": 0, "symbol": ""},
            "contents": empty,
            "diff": {"kind": "initial", "files": source_files},
        },
        {
            "id": "step01-config-contract",
            "label": "声明 Qwen3 MoE 配置字段",
            "active_file": config_path,
            "focus_range": _focus_range(
                config_contract[config_path],
                '"""Qwen3 MoE architecture configuration.',
                32,
            ),
            "contents": config_contract,
            "diff": {"kind": "insert", "files": [config_path]},
        },
        {
            "id": "step01-config-validation",
            "label": "守住结构关系并读取 config.json",
            "active_file": config_path,
            "focus_range": _focus_range(
                config_validation[config_path], "def __post_init__", 53
            ),
            "contents": config_validation,
            "diff": {"kind": "insert", "files": [config_path]},
        },
        {
            "id": "step01-checkpoint-abstraction",
            "label": "建立按名字访问权重的抽象",
            "active_file": checkpoint_path,
            "focus_range": _focus_range(
                checkpoint_abstraction[checkpoint_path],
                '"""Read named tensors from a Safetensors checkpoint.',
                49,
            ),
            "contents": checkpoint_abstraction,
            "diff": {"kind": "insert", "files": [checkpoint_path]},
        },
        {
            "id": "step01-index-discovery",
            "label": "用 index 找到参数所在分片",
            "active_file": checkpoint_path,
            "focus_range": _focus_range(
                index_discovery[checkpoint_path], "@classmethod", 21
            ),
            "contents": index_discovery,
            "diff": {"kind": "insert", "files": [checkpoint_path]},
        },
        {
            "id": "step01-header-validation",
            "label": "读取 Safetensors header",
            "active_file": checkpoint_path,
            "focus_range": _focus_range(
                header_validation[checkpoint_path], "def _read_index", 56
            ),
            "contents": header_validation,
            "diff": {"kind": "insert", "files": [checkpoint_path]},
        },
        {
            "id": "step01-load-tensor",
            "label": "按名字只读取一个 tensor",
            "active_file": checkpoint_path,
            "focus_range": _focus_range(
                tensor_loading[checkpoint_path], "def load_tensor", 17
            ),
            "contents": tensor_loading,
            "diff": {"kind": "insert", "files": [checkpoint_path]},
        },
    ]
    checkpoint_data = {
        "schema_version": 1,
        "checkpoints": [
            {
                "id": item["id"],
                "step": "step01",
                "label": item["label"],
                "active_file": item["active_file"],
                "focus_range": item["focus_range"],
                "repository_snapshot": _snapshot(source_files, item["contents"]),
                "diff": item["diff"],
            }
            for item in staged_checkpoints
        ],
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps(checkpoint_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    generate()
