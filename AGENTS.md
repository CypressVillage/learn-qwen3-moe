# AGENTS.md

本仓库只维护三类交付物：

1. `src/qwen3_moe/` 中的 Qwen3 MoE 累计推理框架。
2. `lessons/` 中的教程正文和源码 checkpoint。
3. `web/`、`scripts/` 中的静态教程生成与渲染代码。

不要重新引入旧 Dense 基线、旧教程、设计文档、测试目录、实验脚本、trace、fixture 或环境检查器。

## 当前状态

- 当前分支：`rewrite/visual-qwen3-moe-course`
- Step 01：完成。
- 当前源码：`config.py`、`checkpoint.py`。
- 当前正文：`lessons/step01-overview-config-weights.md`。
- 当前阅读资产：6 个 insert-only checkpoint，位于 `lessons/checkpoints/step01.json`。
- 当前网站：Vite/React 静态入口 `/` 与 `/step01/`，右侧只显示源码阅读器。
- 下一步：Step 02 Qwen Tokenizer。

## 工作流程

开始前：

1. 运行 `git status --short --branch`。
2. 运行 `git log --oneline -10`。
3. 阅读当前任务涉及的源码、正文、checkpoint 和前端文件。

发生实质性进展后更新本文件的“当前状态”。

## 内容边界

- `src/qwen3_moe/` 是推理实现的唯一来源。
- `lessons/*.md` 是教程正文的唯一来源。
- `lessons/checkpoints/` 保存由脚本生成的累计源码快照。
- `web/` 只渲染生成内容，不维护第二份 Python 实现。
- checkpoint 与 Markdown 中的引用必须由 `scripts/validate_course_assets.py` 校验。

## 实现原则

- 按真实数据流逐步累计代码，不预先加入未讲解模块。
- 右侧只展示当前课程涉及的推理框架源码。
- checkpoint 必须保持 insert-only，最终快照必须等于当前源码。
- 使用小张量和 CPU 优先保证实现清晰正确。
- 不实现训练、反向传播或生产级推理服务。

## 验证命令

```bash
uv run python scripts/generate_step01_assets.py
uv run python scripts/validate_course_assets.py
uv run python -c "from qwen3_moe import Qwen3MoeConfig, SafetensorsCheckpoint"
```

`web/` 下：

```bash
npm run build
```
