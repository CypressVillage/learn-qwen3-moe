# AGENTS.md

本仓库只维护三类交付物：

1. `src/qwen3_moe/` 中的 Qwen3 MoE 累计推理框架。
2. `lessons/` 中的教程正文和源码 checkpoint。
3. `web/`、`scripts/` 中的静态教程生成与渲染代码。

不要重新引入旧 Dense 基线、旧教程、设计文档、测试目录、实验脚本、trace、fixture 或环境检查器。

## 项目目标与课程结构

- 本项目的核心目标是让用户从零学会 Qwen3 MoE 的完整推理过程，并最终能够理解和实现对应的推理代码。
- 项目形式仿照 [`SaladDay/pi-from-scratch`](https://github.com/SaladDay/pi-from-scratch)：页面左侧展示教程正文，右侧同步展示当前章节对应的源码。
- 第 0 章必须先给出一张完整的 Qwen3 MoE 推理地图，说明从输入文本到输出 token 的全过程及其中涉及的主要环节，让用户在开始实现前建立总体概念。
- 第 0 章应同时在右侧创建完整推理框架所需的空文件骨架。此处只建立模块边界和文件结构，不提前填入后续章节才会讲解的实现。
- 第 0 章之后，每一章按照真实推理数据流循序渐进地实现一个或一组相关模块，并让右侧代码与左侧讲解保持同步。
- 教程以代码讲解为主，重点说明当前代码写了什么、为什么这样写、它在完整推理链路中的位置，并在需要时穿插必要的原理讲解。
- 每一章都应帮助用户把局部实现重新连接到第 0 章的完整推理地图，避免只见模块、不见整体。

### 主线主题

- Step 00：完整推理地图与源码骨架。
- Step 01：配置与 Safetensors 权重读取。
- Step 02：Qwen3 Tokenizer。
- Step 03：Embedding、RMSNorm 与 Linear 基础层。
- Step 04：RoPE 位置编码。
- Step 05：GQA Attention。
- Step 06：Sparse MoE Router 与 Experts。
- Step 07：Decoder Layer 与 residual 连接。
- Step 08：完整 Causal LM prefill 与 logits。
- Step 09：从 logits 选择 next token。
- Step 10：KV Cache 数据结构。
- Step 11：支持 prefill 与 decode 的 Cached Attention。
- Step 12：完整模型的 Cached Decode。
- Step 13：自回归生成循环。
- Step 14：端到端 Qwen3 MoE 推理。

## 当前状态

- 当前分支：`master`
- Step 00、Step 01、Step 02、Step 03、Step 04、Step 05：完成；六章都从 `Qwen/Qwen3-30B-A3B` 真实仓库资产切入，始终沿 prompt 到 next token 的数据流推进。
- 当前源码：`config.py`、`checkpoint.py`、`tokenizer.py`、`layers.py`、`rope.py`、`attention.py` 已实现并按教学主线精简；Attention 使用纯 NumPy 完成真实 Q/K/V/O 投影、按 head 的 QK Norm、RoPE、GQA KV 共享、causal mask、稳定 softmax 与 Value 加权，当前边界为无 KV Cache 的 prefill；其余推理模块为空文件骨架。
- 当前正文：`lessons/step00-inference-map.md`、`lessons/step01-overview-config-weights.md`、`lessons/step02-tokenizer.md`、`lessons/step03-basic-layers.md`、`lessons/step04-rope.md`、`lessons/step05-gqa-attention.md`；Step 05 围绕“让每个 token 只读取已经出现的上下文”展开，依次解释真实 Attention 权重、Q/K/V head shape、QK Norm、RoPE 接入、GQA 分组、causal softmax、Value 加权与输出投影。
- 当前阅读资产：Step 00 有 13 个逐文件建立空骨架的 checkpoint，Step 01 有 6 个 insert-only checkpoint，Step 02 有 8 个 insert-only checkpoint，Step 03 有 6 个 insert-only checkpoint，Step 04 有 6 个 insert-only checkpoint，Step 05 有 7 个 insert-only checkpoint。
- 当前网站：Vite/React 静态入口 `/`、`/step00/`、`/step01/`、`/step02/`、`/step03/`、`/step04/` 与 `/step05/`；顶部课程进度可展开并在现有 Step 文章间导航，正文底部提供上一章、下一章入口；左侧源码阅读器包含可折叠文件夹树、文件导航和 Python 语法高亮，右侧展示教程，并支持持久化的深浅色主题切换与可读性优化的代码字体；`master` 推送后由 GitHub Actions 构建并部署到 GitHub Pages。
- 当前规划：主线固定为 Step 00 至 Step 14；推理优化内容独立放入后日谈，不计入主线进度。
- 下一步：实现 Step 06 MoE，从真实 Router 与 Expert 权重开始，为每个 token 选择 top-k experts，并按 routing probability 合并专家输出。

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

- 除第 0 章创建空文件骨架外，按真实数据流逐步累计代码，不预先加入未讲解模块的实现。
- 右侧只展示当前课程涉及的推理框架源码。
- checkpoint 必须保持 insert-only，最终快照必须等于当前源码。
- 使用小张量和 CPU 优先保证实现清晰正确。
- 不实现训练、反向传播或生产级推理服务。

## 验证命令

```bash
uv run python scripts/generate_step00_assets.py
uv run python scripts/generate_step01_assets.py
uv run python scripts/generate_step02_assets.py
uv run python scripts/generate_step03_assets.py
uv run python scripts/generate_step04_assets.py
uv run python scripts/generate_step05_assets.py
uv run python scripts/validate_course_assets.py
uv run python -c "from qwen3_moe import Embedding, Linear, Qwen3Attention, Qwen3MoeConfig, Qwen3Tokenizer, RMSNorm, RotaryEmbedding, SafetensorsCheckpoint, apply_rotary_position_embedding"
```

`web/` 下：

```bash
npm run build
```
