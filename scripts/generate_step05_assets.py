"""Generate the Step 05 cumulative GQA Attention checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe.attention import Qwen3Attention


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step05.json"


def _source_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _through(symbol: object) -> str:
    relative_path = "src/qwen3_moe/attention.py"
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, start = inspect.getsourcelines(symbol)
    symbol_lines = inspect.getsource(symbol).splitlines(keepends=True)
    return "".join(lines[: start - 1 + len(symbol_lines)])


def _package_without_attention() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "qwen3_moe.attention" not in line
        and "qwen3_moe.moe" not in line
        and "qwen3_moe.model" not in line
        and "qwen3_moe.generation" not in line
        and '"Qwen3Attention"' not in line
        and not any(
            f'"{name}"' in line
            for name in (
                "Qwen3DecoderLayer",
                "Qwen3MoeExperts",
                "Qwen3MoeForCausalLM",
                "Qwen3MoeRouter",
                "Qwen3SparseMoeBlock",
                "greedy_next_token",
                "last_token_logits",
                "next_token_probabilities",
                "sample_next_token",
            )
        )
    )


def _package_through_attention() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "qwen3_moe.moe" not in line
        and "qwen3_moe.model" not in line
        and "qwen3_moe.generation" not in line
        and not any(
            f'"{name}"' in line
            for name in (
                "Qwen3DecoderLayer",
                "Qwen3MoeExperts",
                "Qwen3MoeForCausalLM",
                "Qwen3MoeRouter",
                "Qwen3SparseMoeBlock",
                "greedy_next_token",
                "last_token_logits",
                "next_token_probabilities",
                "sample_next_token",
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
        "src/qwen3_moe/__init__.py",
    ]
    attention_path = "src/qwen3_moe/attention.py"
    package_path = "src/qwen3_moe/__init__.py"
    initial = {
        relative_path: (
            ""
            if relative_path == attention_path
            else _package_without_attention()
            if relative_path == package_path
            else _source_text(relative_path)
        )
        for relative_path in source_files
    }
    weights = {
        **initial,
        attention_path: _through(Qwen3Attention.__init__),
    }
    projections = {
        **weights,
        attention_path: _through(Qwen3Attention._project_query_key_value),
    }
    positions = {
        **projections,
        attention_path: _through(Qwen3Attention._apply_positions),
    }
    attention = {
        **positions,
        attention_path: _through(Qwen3Attention._scaled_dot_product_attention),
    }
    forward = {
        **attention,
        attention_path: _source_text(attention_path),
    }
    package_export = {
        **forward,
        package_path: _package_through_attention(),
    }
    staged = [
        ("step05-ready", "带着带位置信号的 Q/K 积木来到空 Attention", attention_path, initial, "", 0, "initial", []),
        ("step05-weights", "接住一层 Attention 的真实权重", attention_path, weights, "class Qwen3Attention", 61, "insert", [attention_path]),
        ("step05-projections", "投影 Q/K/V 并执行 QK Norm", attention_path, projections, "def _project_query_key_value", 22, "insert", [attention_path]),
        ("step05-positions", "在点积前把 RoPE 接回 Q/K", attention_path, positions, "def _apply_positions", 9, "insert", [attention_path]),
        ("step05-causal-attention", "共享 KV heads 并只读取当前位置以前", attention_path, attention, "def _scaled_dot_product_attention", 28, "insert", [attention_path]),
        ("step05-forward", "合并 heads 并回到 hidden size", attention_path, forward, "def __call__", 31, "insert", [attention_path]),
        ("step05-package", "从包入口导出 GQA Attention", package_path, package_export, "qwen3_moe.attention", 13, "insert", [package_path]),
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
                "step": "step05",
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
