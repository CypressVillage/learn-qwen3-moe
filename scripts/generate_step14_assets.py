"""Generate the Step 14 cumulative end-to-end inference checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe.generation import _load_model_directory, generate_text


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step14.json"
GENERATION_PATH = "src/qwen3_moe/generation.py"
PACKAGE_PATH = "src/qwen3_moe/__init__.py"
PATH_IMPORT = "from pathlib import Path\n\n"
PACKAGE_IMPORT = "from qwen3_moe.generation import generate_text\n"
PACKAGE_EXPORT = '    "generate_text",\n'


def _source_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _function_source(symbol: object) -> str:
    return "\n\n" + inspect.getsource(symbol) + "\n"


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
    loading_source = _function_source(_load_model_directory)
    text_source = _function_source(generate_text)
    initial_generation = generation_source.replace(PATH_IMPORT, "").replace(
        loading_source, ""
    ).replace(text_source, "")
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
    loading = {
        **initial,
        GENERATION_PATH: generation_source.replace(text_source, ""),
    }
    text = {**loading, GENERATION_PATH: generation_source}
    package_export = {**text, PACKAGE_PATH: _source_text(PACKAGE_PATH)}
    staged = [
        ("step14-ready", "从 token ID 循环回到文本入口", GENERATION_PATH, initial, "def generate_token_ids", 14, "initial", []),
        ("step14-loading", "从模型目录组装 config、权重、Tokenizer 与模型", GENERATION_PATH, loading, "def _load_model_directory", 15, "insert", [GENERATION_PATH]),
        ("step14-text", "编码 prompt、生成 token IDs 并解码文本", GENERATION_PATH, text, "def generate_text", 39, "insert", [GENERATION_PATH]),
        ("step14-package", "从包入口导出端到端文本生成", PACKAGE_PATH, package_export, "generate_text", 22, "insert", [PACKAGE_PATH]),
    ]
    checkpoints = []
    for checkpoint_id, label, active_file, contents, marker, count, kind, files in staged:
        checkpoints.append(
            {
                "id": checkpoint_id,
                "step": "step14",
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
