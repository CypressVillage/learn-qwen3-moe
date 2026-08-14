"""Generate Step 00 checkpoints that reveal the empty framework one file at a time."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CHECKPOINT_PATH = ROOT / "lessons" / "checkpoints" / "step00.json"
MODULES = [
    ("step00-config", "建立模型配置边界", "src/qwen3_moe/config.py"),
    ("step00-checkpoint", "建立权重读取边界", "src/qwen3_moe/checkpoint.py"),
    ("step00-tokenizer", "建立文本与 token IDs 边界", "src/qwen3_moe/tokenizer.py"),
    ("step00-layers", "建立基础张量层边界", "src/qwen3_moe/layers.py"),
    ("step00-rope", "建立旋转位置编码边界", "src/qwen3_moe/rope.py"),
    ("step00-attention", "建立 Attention 边界", "src/qwen3_moe/attention.py"),
    ("step00-moe", "建立稀疏 MoE 边界", "src/qwen3_moe/moe.py"),
    ("step00-cache", "建立 KV Cache 边界", "src/qwen3_moe/cache.py"),
    ("step00-model", "建立完整模型组装边界", "src/qwen3_moe/model.py"),
    ("step00-generation", "建立自回归生成边界", "src/qwen3_moe/generation.py"),
    ("step00-package", "建立包公共入口", "src/qwen3_moe/__init__.py"),
]


def _checkpoint(
    checkpoint_id: str,
    label: str,
    active_file: str,
    visible_files: list[str],
    added_files: list[str],
    kind: str = "insert",
) -> dict[str, object]:
    return {
        "id": checkpoint_id,
        "step": "step00",
        "label": label,
        "active_file": active_file,
        "focus_range": {"start": 0, "end": 0, "symbol": ""},
        "repository_snapshot": {
            "files": [{"path": path, "content": ""} for path in visible_files]
        },
        "diff": {"kind": kind, "files": added_files},
    }


def generate() -> None:
    checkpoints = [
        _checkpoint(
            "step00-empty-repository",
            "从空仓库开始认识推理工程",
            "",
            [],
            [],
            kind="initial",
        )
    ]
    visible_files: list[str] = []
    for checkpoint_id, label, path in MODULES:
        visible_files.append(path)
        checkpoints.append(
            _checkpoint(checkpoint_id, label, path, visible_files.copy(), [path])
        )
    checkpoints.append(
        _checkpoint(
            "step00-module-flow",
            "把模块连接成完整推理数据流",
            "src/qwen3_moe/model.py",
            visible_files.copy(),
            [],
            kind="context",
        )
    )

    checkpoint_data = {"schema_version": 1, "checkpoints": checkpoints}
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps(checkpoint_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    generate()
