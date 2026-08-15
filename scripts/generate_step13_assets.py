"""Generate the Step 13 cumulative autoregressive generation checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe.generation import generate_token_ids


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step13.json"
GENERATION_PATH = "src/qwen3_moe/generation.py"
PACKAGE_PATH = "src/qwen3_moe/__init__.py"
CACHE_IMPORT = "from qwen3_moe.cache import KVCache\n"
PACKAGE_IMPORT = "from qwen3_moe.generation import generate_token_ids\n"
PACKAGE_EXPORT = '    "generate_token_ids",\n'


def _source_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


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
        GENERATION_PATH,
        "src/qwen3_moe/cache.py",
        PACKAGE_PATH,
    ]
    generation_source = _source_text(GENERATION_PATH)
    function_source = "\n\n" + inspect.getsource(generate_token_ids) + "\n"
    initial_generation = generation_source.replace(CACHE_IMPORT, "").replace(
        function_source, ""
    )
    initial_package = _source_text(PACKAGE_PATH).replace(PACKAGE_IMPORT, "").replace(
        PACKAGE_EXPORT, ""
    )
    initial = {
        relative_path: (
            initial_generation
            if relative_path == GENERATION_PATH
            else initial_package
            if relative_path == PACKAGE_PATH
            else _source_text(relative_path)
        )
        for relative_path in source_files
    }
    loop = {**initial, GENERATION_PATH: generation_source}
    package_export = {**loop, PACKAGE_PATH: _source_text(PACKAGE_PATH)}
    staged = [
        ("step13-ready", "从单步 cached decode 进入重复生成", GENERATION_PATH, initial, "def sample_next_token", 16, "initial", []),
        ("step13-loop", "连接 prefill、token selection 与 cached decode", GENERATION_PATH, loop, "def generate_token_ids", 49, "insert", [GENERATION_PATH]),
        ("step13-package", "从包入口导出自回归生成循环", PACKAGE_PATH, package_export, "generate_token_ids", 20, "insert", [PACKAGE_PATH]),
    ]
    checkpoints = []
    for checkpoint_id, label, active_file, contents, marker, count, kind, files in staged:
        checkpoints.append(
            {
                "id": checkpoint_id,
                "step": "step13",
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
