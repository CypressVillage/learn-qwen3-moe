"""Generate the Step 06 cumulative sparse MoE checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe.attention import Qwen3Attention
from qwen3_moe.moe import Qwen3MoeExperts, Qwen3MoeRouter


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step06.json"
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
    relative_path = "src/qwen3_moe/moe.py"
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, start = inspect.getsourcelines(symbol)
    symbol_lines = inspect.getsource(symbol).splitlines(keepends=True)
    return "".join(lines[: start - 1 + len(symbol_lines)])


def _prefill_attention() -> str:
    lines = _source_text("src/qwen3_moe/attention.py").splitlines(keepends=True)
    _, start = inspect.getsourcelines(Qwen3Attention.__call__)
    method_lines = inspect.getsource(Qwen3Attention.__call__).splitlines(keepends=True)
    source = "".join(lines[: start - 1 + len(method_lines)])
    return source.replace(CACHE_IMPORT, "").replace(RECTANGULAR_MASK, "")


def _package_without_moe() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "qwen3_moe.moe" not in line
        and "qwen3_moe.model" not in line
        and "qwen3_moe.generation" not in line
        and "qwen3_moe.cache" not in line
        and '"Qwen3MoeExperts"' not in line
        and '"Qwen3MoeRouter"' not in line
        and '"Qwen3SparseMoeBlock"' not in line
        and '"Qwen3DecoderLayer"' not in line
        and '"Qwen3MoeForCausalLM"' not in line
        and not any(
            f'"{name}"' in line
            for name in (
                "greedy_next_token",
                "last_token_logits",
                "next_token_probabilities",
                "sample_next_token",
                "generate_token_ids",
                "generate_text",
                "KVCache",
            )
        )
    )


def _package_through_moe() -> str:
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
                "generate_token_ids",
                "generate_text",
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
        "src/qwen3_moe/__init__.py",
    ]
    moe_path = "src/qwen3_moe/moe.py"
    package_path = "src/qwen3_moe/__init__.py"
    initial = {
        relative_path: (
            ""
            if relative_path == moe_path
            else _package_without_moe()
            if relative_path == package_path
            else _prefill_attention()
            if relative_path == "src/qwen3_moe/attention.py"
            else _source_text(relative_path)
        )
        for relative_path in source_files
    }
    router_weights = {
        **initial,
        moe_path: _through(Qwen3MoeRouter.__init__),
    }
    routing = {
        **router_weights,
        moe_path: _through(Qwen3MoeRouter.__call__),
    }
    expert_weights = {
        **routing,
        moe_path: _through(Qwen3MoeExperts.__init__),
    }
    experts = {
        **expert_weights,
        moe_path: _through(Qwen3MoeExperts.__call__),
    }
    block = {
        **experts,
        moe_path: _source_text(moe_path),
    }
    package_export = {
        **block,
        package_path: _package_through_moe(),
    }
    staged = [
        ("step06-ready", "带着 Attention 输出进入空 MoE 模块", moe_path, initial, "", 0, "initial", []),
        ("step06-router-weights", "接住 Router 权重与 top-k 配置", moe_path, router_weights, "class Qwen3MoeRouter", 24, "insert", [moe_path]),
        ("step06-routing", "为每个 token 选择 top-k experts", moe_path, routing, "def __call__", 29, "insert", [moe_path]),
        ("step06-expert-weights", "检查融合的 SwiGLU 专家权重", moe_path, expert_weights, "class Qwen3MoeExperts", 44, "insert", [moe_path]),
        ("step06-experts", "只运行被选中的专家并加权累加", moe_path, experts, "selected_experts: np.ndarray", 55, "insert", [moe_path]),
        ("step06-block", "组装完整 Sparse MoE 数据流", moe_path, block, "class Qwen3SparseMoeBlock", 34, "insert", [moe_path]),
        ("step06-package", "从包入口导出 Router、Experts 与 MoE", package_path, package_export, "qwen3_moe.moe", 16, "insert", [package_path]),
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
                "step": "step06",
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
