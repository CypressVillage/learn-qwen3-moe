"""Generate the Step 03 cumulative basic-layer checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe.layers import Embedding, Linear, RMSNorm


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step03.json"


def _source_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _through(method: object) -> str:
    relative_path = "src/qwen3_moe/layers.py"
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, start = inspect.getsourcelines(method)
    method_lines = inspect.getsource(method).splitlines(keepends=True)
    return "".join(lines[: start - 1 + len(method_lines)])


def _package_without_layers() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "qwen3_moe.layers" not in line
        and "qwen3_moe.rope" not in line
        and "qwen3_moe.attention" not in line
        and "qwen3_moe.moe" not in line
        and "qwen3_moe.model" not in line
        and "qwen3_moe.generation" not in line
        and "qwen3_moe.cache" not in line
        and not any(
            f'"{name}"' in line
            for name in (
                "Embedding",
                "Linear",
                "RMSNorm",
                "RotaryEmbedding",
                "Qwen3Attention",
                "Qwen3DecoderLayer",
                "Qwen3MoeExperts",
                "Qwen3MoeForCausalLM",
                "Qwen3MoeRouter",
                "Qwen3SparseMoeBlock",
                "apply_rotary_position_embedding",
                "greedy_next_token",
                "last_token_logits",
                "next_token_probabilities",
                "sample_next_token",
                "KVCache",
            )
        )
    )


def _package_through_layers() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "qwen3_moe.rope" not in line
        and "qwen3_moe.attention" not in line
        and "qwen3_moe.moe" not in line
        and "qwen3_moe.model" not in line
        and "qwen3_moe.generation" not in line
        and "qwen3_moe.cache" not in line
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
        "src/qwen3_moe/__init__.py",
    ]
    config_path, checkpoint_path, tokenizer_path, layers_path, package_path = source_files
    initial = {
        config_path: _source_text(config_path),
        checkpoint_path: _source_text(checkpoint_path),
        tokenizer_path: _source_text(tokenizer_path),
        layers_path: "",
        package_path: _package_without_layers(),
    }
    embedding_weight = {
        **initial,
        layers_path: _through(Embedding.__init__),
    }
    embedding_lookup = {
        **embedding_weight,
        layers_path: _through(Embedding.__call__),
    }
    rmsnorm = {
        **embedding_lookup,
        layers_path: _through(RMSNorm.__call__),
    }
    linear = {
        **rmsnorm,
        layers_path: _source_text(layers_path),
    }
    package_export = {
        **linear,
        package_path: _package_through_layers(),
    }
    staged = [
        ("step03-ready", "带着 token IDs 来到空基础层", layers_path, initial, "", 0, "initial", []),
        ("step03-embedding-weight", "接住真实 Embedding 权重", layers_path, embedding_weight, "class Embedding", 10, "insert", [layers_path]),
        ("step03-embedding-lookup", "用 token IDs 查出 hidden states", layers_path, embedding_lookup, "def __call__", 10, "insert", [layers_path]),
        ("step03-rmsnorm", "按最后一维执行 RMSNorm", layers_path, rmsnorm, "class RMSNorm", 21, "insert", [layers_path]),
        ("step03-linear", "把最后一维投影到新空间", layers_path, linear, "class Linear", 20, "insert", [layers_path]),
        ("step03-package", "从包入口导出三个基础层", package_path, package_export, "qwen3_moe.layers", 12, "insert", [package_path]),
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
                "step": "step03",
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
