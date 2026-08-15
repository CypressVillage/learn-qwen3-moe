"""Generate the Step 08 cumulative Causal LM prefill checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe.model import Qwen3MoeForCausalLM


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step08.json"


def _source_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _through(symbol: object) -> str:
    relative_path = "src/qwen3_moe/model.py"
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, start = inspect.getsourcelines(symbol)
    symbol_lines = inspect.getsource(symbol).splitlines(keepends=True)
    return "".join(lines[: start - 1 + len(symbol_lines)])


def _model_without_causal_lm() -> str:
    source = _source_text("src/qwen3_moe/model.py")
    marker = "\n\nclass Qwen3MoeForCausalLM:"
    return "".join(
        line
        for line in (source.split(marker, maxsplit=1)[0] + "\n").splitlines(
            keepends=True
        )
        if "qwen3_moe.checkpoint" not in line
        and "from qwen3_moe.layers import Embedding, Linear" not in line
    )


def _without_checkpoint_import(source: str) -> str:
    return "".join(
        line
        for line in source.splitlines(keepends=True)
        if "qwen3_moe.checkpoint" not in line
    )


def _package_without_causal_lm() -> str:
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
            _model_without_causal_lm()
            if relative_path == model_path
            else _package_without_causal_lm()
            if relative_path == package_path
            else _source_text(relative_path)
        )
        for relative_path in source_files
    }
    components = {
        **initial,
        model_path: _without_checkpoint_import(
            _through(Qwen3MoeForCausalLM.__init__)
        ),
    }
    checkpoint_loading = {
        **components,
        model_path: _through(Qwen3MoeForCausalLM.from_checkpoint),
    }
    decoder_loading = {
        **checkpoint_loading,
        model_path: _through(Qwen3MoeForCausalLM._load_decoder_layer),
    }
    positions = {
        **decoder_loading,
        model_path: _through(Qwen3MoeForCausalLM._position_ids),
    }
    forward = {
        **positions,
        model_path: _source_text(model_path),
    }
    package_export = {
        **forward,
        package_path: "".join(
            line
            for line in _source_text(package_path).splitlines(keepends=True)
            if "qwen3_moe.generation" not in line
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
        ),
    }
    staged = [
        ("step08-ready", "从单层输出进入完整模型组装", model_path, initial, "class Qwen3DecoderLayer", 8, "initial", []),
        ("step08-components", "接住 Embedding、层列表、final norm 与 LM Head", model_path, components, "class Qwen3MoeForCausalLM", 35, "insert", [model_path]),
        ("step08-checkpoint-loading", "从真实 checkpoint 装载模型级权重", model_path, checkpoint_loading, "def from_checkpoint", 29, "insert", [model_path]),
        ("step08-decoder-loading", "按层号装载全部 Decoder Layer 权重", model_path, decoder_loading, "def _load_decoder_layer", 30, "insert", [model_path]),
        ("step08-positions", "为整段 prompt 建立 position IDs", model_path, positions, "def _position_ids", 7, "insert", [model_path]),
        ("step08-forward", "串起 Embedding、全部层、final norm 与 logits", model_path, forward, "def __call__", 18, "insert", [model_path]),
        ("step08-package", "从包入口导出完整 Causal LM", package_path, package_export, "Qwen3MoeForCausalLM", 16, "insert", [package_path]),
    ]
    checkpoints = []
    for checkpoint_id, label, active_file, contents, marker, count, kind, files in staged:
        checkpoints.append(
            {
                "id": checkpoint_id,
                "step": "step08",
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
