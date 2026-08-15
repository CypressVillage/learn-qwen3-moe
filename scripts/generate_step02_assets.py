"""Generate the Step 02 cumulative tokenizer checkpoints."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from qwen3_moe.tokenizer import (
    Qwen3Tokenizer,
    _byte_alphabet,
    _find_split_pattern,
    _merge_token,
)


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step02.json"


def _source_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _class_through(method: object) -> str:
    relative_path = "src/qwen3_moe/tokenizer.py"
    lines = _source_text(relative_path).splitlines(keepends=True)
    _, start = inspect.getsourcelines(method)
    method_lines = inspect.getsource(method).splitlines(keepends=True)
    return "".join(lines[: start - 1 + len(method_lines)])


def _helpers(*symbols: object) -> str:
    sources = (inspect.getsource(symbol).rstrip("\n") for symbol in symbols)
    return "\n\n" + "\n\n".join(sources) + "\n"


def _package_before_tokenizer() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "qwen3_moe.tokenizer" not in line
        and "qwen3_moe.layers" not in line
        and '"Qwen3Tokenizer"' not in line
        and not any(f'"{name}"' in line for name in ("Embedding", "Linear", "RMSNorm"))
    )


def _package_through_tokenizer() -> str:
    return "".join(
        line
        for line in _source_text("src/qwen3_moe/__init__.py").splitlines(keepends=True)
        if "qwen3_moe.layers" not in line
        and not any(f'"{name}"' in line for name in ("Embedding", "Linear", "RMSNorm"))
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
        "src/qwen3_moe/__init__.py",
    ]
    config_path, checkpoint_path, tokenizer_path, package_path = source_files
    package_before = _package_before_tokenizer()
    initial = {
        config_path: _source_text(config_path),
        checkpoint_path: _source_text(checkpoint_path),
        tokenizer_path: "",
        package_path: package_before,
    }
    assets = {
        **initial,
        tokenizer_path: _class_through(Qwen3Tokenizer.from_file),
    }
    byte_alphabet = {
        **assets,
        tokenizer_path: assets[tokenizer_path] + _helpers(_byte_alphabet),
    }
    pretokenizer = {
        **byte_alphabet,
        tokenizer_path: byte_alphabet[tokenizer_path] + _helpers(_find_split_pattern),
    }
    bpe = {
        **pretokenizer,
        tokenizer_path: assets[tokenizer_path]
        + _helpers(_byte_alphabet, _merge_token, _find_split_pattern),
    }
    encoding = {
        **bpe,
        tokenizer_path: _class_through(Qwen3Tokenizer._encode_text)
        + _helpers(_byte_alphabet, _merge_token, _find_split_pattern),
    }
    decoding = {
        **encoding,
        tokenizer_path: _source_text(tokenizer_path),
    }
    package_export = {
        **decoding,
        package_path: _package_through_tokenizer(),
    }
    staged = [
        ("step02-ready", "带着上一章地基来到空 Tokenizer", tokenizer_path, initial, "", 0, "initial", []),
        ("step02-assets", "从 tokenizer.json 读取词表与 merge 规则", tokenizer_path, assets, "def from_file", 42, "insert", [tokenizer_path]),
        ("step02-byte-alphabet", "建立 UTF-8 byte 与可见字符映射", tokenizer_path, byte_alphabet, "def _byte_alphabet", 22, "insert", [tokenizer_path]),
        ("step02-pretokenizer", "读取 Qwen3 的正则预切分规则", tokenizer_path, pretokenizer, "def _find_split_pattern", 16, "insert", [tokenizer_path]),
        ("step02-bpe", "按 rank 反复合并相邻片段", tokenizer_path, bpe, "def _merge_token", 28, "insert", [tokenizer_path]),
        ("step02-encode", "把普通文本和 special token 编成 IDs", tokenizer_path, encoding, "def encode", 41, "insert", [tokenizer_path]),
        ("step02-decode", "把 token IDs 重新拼回 UTF-8 文本", tokenizer_path, decoding, "def decode", 42, "insert", [tokenizer_path]),
        ("step02-package", "从包入口导出 Qwen3Tokenizer", package_path, package_export, "Qwen3Tokenizer", 8, "insert", [package_path]),
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
                "step": "step02",
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
