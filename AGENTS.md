# AGENTS.md

本仓库只维护三类交付物：

1. `src/qwen3_moe/` 中的 Qwen3 MoE 累计推理框架。
2. `lessons/` 中的教程正文和源码 checkpoint。
3. `web/`、`scripts/` 中的静态教程生成与渲染代码。

不要重新引入旧 Dense 基线、旧教程、设计文档、测试目录、实验脚本、trace、fixture 或环境检查器。

## 项目目标与课程结构

- 本项目的核心目标是让用户从零学会 Qwen3 MoE 的完整推理过程，并最终能够理解和实现对应的推理代码。
- 项目形式仿照 [`SaladDay/pi-from-scratch`](https://github.com/SaladDay/pi-from-scratch)：页面左侧展示当前章节对应的源码，右侧展示教程正文。
- 第 0 章必须先给出一张完整的 Qwen3 MoE 推理地图，说明从输入文本到输出 token 的全过程及其中涉及的主要环节，让用户在开始实现前建立总体概念。
- 第 0 章应同时在左侧创建完整推理框架所需的空文件骨架。此处只建立模块边界和文件结构，不提前填入后续章节才会讲解的实现。
- 第 0 章之后，每一章按照真实推理数据流循序渐进地实现一个或一组相关模块，并让左侧代码与右侧讲解保持同步。
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
- Step 00 至 Step 14：完成；主线课程已经从文本输入闭合到生成文本输出。
- 当前源码：`generate_text()` 从模型目录加载 config、Safetensors、Tokenizer 与完整模型，编码单条 prompt，调用 cached 自回归循环，再将完整 token 序列解码回文本。
- 当前正文：已完成 `lessons/step00-inference-map.md` 至 `lessons/step14-end-to-end-inference.md`；Step 00 开头已补充教程目标、学习路径与实现边界，Step 14 汇总模型资产、文本入口、生成参数、解码出口与完整推理地图。
- 当前阅读资产：Step 14 有 4 个 insert-only checkpoint，最终快照等于当前源码；历史 checkpoint 保持逐行可累积。
- 当前网站：Vite/React 静态入口 `/` 与 `/step00/` 至 `/step14/`；课程导航与源码阅读器已接入全部主线章节，顶部章节目录支持视口内滚动；源码高亮覆盖完整横向滚动宽度，EXPLORER 支持折叠扩宽代码区；`scripts/build_site.py` 提供课程资产生成、校验和前端构建的一条命令入口。
- Step 01 阅读资产：`step01-config-contract` 的高亮范围从 `config.py` 文件首行开始，包含模块说明、imports、`@dataclass` 与完整配置字段；配置正文按源码顺序先讲 `__post_init__()`，再讲 `from_json()`；权重部分先解释课程自定义的 `SafetensorsCheckpoint` 抽象，再依次高亮基础状态、index 发现、header 解析和 tensor payload 读取，避免代码未讲解先出现。
- 当前规划：主线固定为 Step 00 至 Step 14；推理优化内容独立放入后日谈，不计入主线进度。
- 下一步：主线完成；推理优化内容只在后日谈中继续，不计入 Step 00 至 Step 14。

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
- 左侧只展示当前课程涉及的推理框架源码。
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
uv run python scripts/generate_step06_assets.py
uv run python scripts/generate_step07_assets.py
uv run python scripts/generate_step08_assets.py
uv run python scripts/generate_step09_assets.py
uv run python scripts/generate_step10_assets.py
uv run python scripts/generate_step11_assets.py
uv run python scripts/generate_step12_assets.py
uv run python scripts/generate_step13_assets.py
uv run python scripts/generate_step14_assets.py
uv run python scripts/validate_course_assets.py
uv run python -c "from qwen3_moe import Embedding, KVCache, Linear, Qwen3Attention, Qwen3DecoderLayer, Qwen3MoeConfig, Qwen3MoeExperts, Qwen3MoeForCausalLM, Qwen3MoeRouter, Qwen3SparseMoeBlock, Qwen3Tokenizer, RMSNorm, RotaryEmbedding, SafetensorsCheckpoint, apply_rotary_position_embedding, generate_text, generate_token_ids, greedy_next_token, last_token_logits, next_token_probabilities, sample_next_token"
```

`web/` 下：

```bash
npm run build
```
