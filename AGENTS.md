# AGENTS.md

本文件是本仓库所有后续会话的项目状态入口与执行约束。开始工作前先读取本文件，再按需读取设计文档和相关源码；结束工作前根据实际证据更新本文件中的状态区。

## 项目目标

将现有 Dense 推理学习仓库并行重写为一套 14 步、沿真实 LLM 推理数据流推进的可视化 Qwen3 MoE 课程：

```text
文本 -> token IDs -> hidden states -> Qwen3 MoE Transformer
     -> logits -> next token -> 自回归生成
```

最终交付包括：

- 可安装、可测试的累计 Python 实现。
- NumPy 与 PyTorch 教学后端。
- 14 篇累计课程正文。
- 由真实 Python 执行生成的 checkpoint 与离线 trace。
- 桌面和移动端可用的静态交互式教程网站。
- 微型 checkpoint 严格对齐，以及至少一个可承载兼容 checkpoint 的逐层对齐证据。

权威设计文档：`docs/superpowers/specs/2026-08-14-visual-qwen3-moe-course-design.md`。

## 会话启动流程

每次会话开始时按以下顺序建立上下文，不要仅凭上次对话记忆判断进度：

1. 阅读本文件的“当前状态”“下一步”和“阻塞/决策”部分。
2. 运行 `git status --short --branch`，确认当前分支和未提交改动。
3. 运行 `git log --oneline -10`，检查本文件记录后是否已有新提交。
4. 检查当前任务涉及的源码、测试、课程、checkpoint、trace 和网站文件是否真实存在。
5. 对准备标记为完成的范围运行最小相关测试；不能只根据文件名或文字描述判定完成。
6. 选择“下一步”中优先级最高且未被阻塞的一项继续，除非用户明确指定其他任务。

如果仓库事实与本文件冲突，以仓库、测试和生成产物为准，并在本次会话中修正本文件。

## 会话结束流程

发生实质性进展时，在结束会话前更新“当前状态”：

- 更新“最后核对”日期、分支、测试结果和当前阶段。
- 只在验收证据完整时将状态改为“完成”。
- 记录本次新增的关键文件、可复现命令和测试证据。
- 更新“下一步”，保证下一会话有一个明确的首要任务。
- 将未解决问题写入“阻塞/决策”，不要把猜测写成既定结论。
- 不记录访问 token、模型凭据、机器绝对数据路径或不可再分发输入。

仅修改状态文字不代表功能完成。完成状态必须由代码、测试、checkpoint、trace 或静态构建等仓库证据支持。

## 状态定义

- `未开始`：新课程路径中没有可验收产物。
- `进行中`：已有部分产物，但尚未满足该 Step 或里程碑的完整契约。
- `可迁移基线`：旧 Dense 仓库已有相关正确实现，但尚未迁移到新双后端、课程、checkpoint 和 trace 体系。
- `阻塞`：存在明确外部依赖或待决策事项，当前无法合理继续。
- `完成`：正文、累计代码 checkpoint、trace、测试、受控错误和运行命令均满足设计契约。

“可迁移基线”不得等同于新课程 Step 完成。

## 当前状态

最后核对：2026-08-14

- 当前分支：`rewrite/visual-qwen3-moe-course`
- 当前阶段：阶段一“建立新骨架”尚未开始实现。
- 当前里程碑：M1 课程基础设施，未开始。
- 测试基线：`uv run pytest`，52 passed（2026-08-14）。
- 设计基线：`docs/superpowers/specs/2026-08-14-visual-qwen3-moe-course-design.md` 是当前重写的权威设计文档。
- 当前包名：`qwen3_moe`。是否改为 `mini_qwen` 尚未决定。

### 已有可复用基础

- `src/qwen3_moe/` 已有纯 PyTorch Dense 教学实现：配置、RMSNorm、RoPE、QK Norm、GQA、SwiGLU、Dense Decoder 和 Tiny Causal LM。
- `examples/run_tiny_dense.py` 可执行微型 Dense forward。
- `examples/run_tiny_greedy.py` 可执行基于整数 token 的无 cache greedy 循环。
- `tests/` 已覆盖现有 Dense 模块、端到端模型、示例和课程源码片段。
- `docs/course/` 与 `docs/tutorials/` 是迁移期间必须保留的旧课程与深入实验。

### 尚未建立

- `lessons/` 新 14 步课程正文。
- `tutorial/checkpoints/` 累计代码快照与 schema。
- `tutorial/traces/` 离线 trace 与 schema。
- `web/` 静态交互网站。
- `src/qwen3_moe/backends/` 或等价 NumPy/PyTorch 双后端。
- Qwen tokenizer、Safetensors 元数据读取器和严格权重加载器。
- MoE Router、Top-K、专家分发和聚合。
- KV Cache、prefill/decode 和完整 sampling 模块。
- 课程内容生成、同步校验和网站构建 CI。

## 14 步进度

| Step | 主题 | 状态 | 当前证据/缺口 |
| --- | --- | --- | --- |
| 01 | 推理全景、配置和权重目录 | 未开始 | 仅有设计文档；缺新课程骨架、配置读取、Safetensors index、按名读取和网站原型 |
| 02 | Qwen Tokenizer | 未开始 | 缺 tokenizer 资源解析、自有 encode/decode、oracle 对照和 trace |
| 03 | Linear、Embedding、RMSNorm | 可迁移基线 | PyTorch Embedding/RMSNorm 已有；缺双后端、教学 Linear/Parameter、权重加载、checkpoint 和 trace |
| 04 | RoPE | 可迁移基线 | `src/qwen3_moe/rope.py` 已有 Dense 路径实现；缺 backend 对照、配置驱动 scaling 接口和新课程产物 |
| 05 | 单头 Causal Attention | 可迁移基线 | GQA 实现包含相关计算；缺独立单头教学步骤、手算 trace 和新课程测试契约 |
| 06 | Multi-Head、GQA、QK Norm | 可迁移基线 | `attention.py` 已实现无 cache PyTorch GQA/QK Norm/RoPE；缺双后端、课程 checkpoint 和 trace |
| 07 | SwiGLU Expert | 可迁移基线 | `mlp.py` 已有 Dense SwiGLU；缺 expert 语义、双后端和新课程产物 |
| 08 | MoE Router 和 Top-K | 未开始 | 无 Router 实现 |
| 09 | 专家分发、执行和聚合 | 未开始 | 无 Sparse MoE block |
| 10 | 完整 Qwen3 MoE Transformer | 未开始 | 当前仅 Tiny Dense LM；缺配置驱动 Dense/MoE 层、严格权重映射和逐层对齐 |
| 11 | 生成第一个 Token | 可迁移基线 | 旧示例可从整数 ID 取 greedy token；缺真实 tokenizer、权重和可解码首 token |
| 12 | 自回归生成 | 可迁移基线 | 旧示例有无 cache greedy 循环；缺 tokenizer、EOS、batch 停止和正式 generation API |
| 13 | Prefill、Decode 和 KV Cache | 未开始 | 无 cache contract 或 cached/uncached 对齐 |
| 14 | Temperature、Top-K 和 Top-P | 未开始 | 无正式 sampling 模块和 trace |

## 里程碑进度

| 里程碑 | 范围 | 状态 | 完成门槛 |
| --- | --- | --- | --- |
| M1 课程基础设施 | schema、checkpoint、trace、网站壳、Step 01 | 未开始 | Step 01 桌面与移动端完整可读 |
| M2 双后端基础 | Step 02-04 | 未开始 | Tokenizer、Embedding、RMSNorm、RoPE 对照通过 |
| M3 Attention 与 Dense | Step 05-07 | 可迁移基线 | 新课程中的单头、GQA、QK Norm、SwiGLU trace 可用 |
| M4 MoE | Step 08-09 | 未开始 | Router、dispatch、聚合与朴素参考一致 |
| M5 完整模型 | Step 10-11 | 未开始 | 微型 checkpoint 输出 logits 并生成首 token |
| M6 生成系统 | Step 12-14 | 未开始 | 自回归、KV Cache、采样完整通过 |
| M7 真实对齐与发布 | 对齐、部署、入口迁移 | 未开始 | 可承载 checkpoint 逐层对齐，静态网站发布 |

## 下一步

当前首要任务是完成 M1/Step 01 的最小端到端垂直切片，而不是先扩写后续数学模块。

建议执行顺序：

1. 决定新累计包继续使用 `qwen3_moe` 还是改为 `mini_qwen`；无外部消费者证据时不要增加兼容层。
2. 决定网站使用 Next.js static export 还是 Vite/React 静态方案。
3. 建立 `lessons/`、`tutorial/checkpoints/`、`tutorial/traces/` 和 `web/` 骨架。
4. 定义 checkpoint 与 trace 的可校验 schema。
5. 实现配置读取、Safetensors index 元数据检查和 `load_tensor(name)` 边界。
6. 编写 Step 01 正文、固定微型 trace、测试和受控错误。
7. 完成 Step 01 桌面/移动端网站原型并通过静态构建。

任何单项实现都应服务于该垂直切片；避免先创建无法被正文、trace 或网站使用的抽象层。

## 阻塞/决策

以下事项尚未确定，但不改变 14 步结构：

- 包名：继续使用 `qwen3_moe`，或更名为 `mini_qwen`。
- 网站框架：Next.js static export，或 Vite/React 静态方案。
- Safetensors：标准库教学读取器作为默认，或官方库作为可选快速路径。
- 真实对齐目标：首先选择哪个可承载的 Qwen3 MoE checkpoint。

若当前任务直接依赖其中一项，应优先采用最小正确实现；存在明显长期影响或多个同等方案时向用户询问，不要暗中决定后再增加兼容层。

## 不可漂移的架构约束

- 目标架构是 Qwen3 MoE decoder-only causal LM，具体行为以目标 checkpoint 配置和参考实现为权威。
- Attention 使用独立无 bias Q/K/V/O projection、GQA、head-dim QK Norm，并在 QK Norm 后应用 RoPE。
- KV Cache 每层保存旋转后的 K 和 V，head 数保持 `Hkv`，不保存重复到 `Hq` 的副本。
- Decoder Layer 使用两次 pre-norm residual。
- Dense MLP 和 Expert 共用无 bias SwiGLU 计算核心。
- Router 输入 `[N,D]`、输出 logits `[N,E]`，Top-K indices/weights 为 `[N,K]`。
- `norm_topk_prob` 决定选中专家权重是否重新归一化。
- Dense/MoE 层布局由配置决定，不写死某个 checkpoint 的层分布。
- 纯推理主线不实现训练、反向传播或 Router auxiliary loss。
- 不在浏览器中运行 PyTorch，不把真实权重打入网站产物。
- 不声称单张 A10 24GB 可加载 Qwen3-30B-A3B 全量 BF16 权重。

## 统一形状记号

| 记号 | 含义 |
| --- | --- |
| `B` | batch size |
| `S` | 本次 forward 的 query token 数 |
| `P` | 调用前缓存的历史 token 数 |
| `T` | key/value 总长度，`T=P+S` |
| `V` | vocabulary size |
| `D` | hidden size |
| `L` | decoder layer 数 |
| `Hq` | query head 数 |
| `Hkv` | key/value head 数 |
| `Dh` | head dimension |
| `G` | 每个 KV head 服务的 query head 数，`G=Hq/Hkv` |
| `I` | Dense MLP intermediate size |
| `Imoe` | 单个 MoE expert intermediate size |
| `E` | routed expert 总数 |
| `K` | 每 token 选择的 expert 数 |
| `N` | 展平 token 数，`N=B*S` |

所有具体数值来自微型配置或当前 `config.json`，不得把某篇文章或单个 checkpoint 的数字写成整个模型族的固定事实。

## 内容与源码边界

- `lessons/stepXX-*.md`：新课程正文的唯一来源。
- `src/qwen3_moe/` 或最终确定的包目录：累计实现的权威源码。
- `tutorial/checkpoints/`：脚本生成或严格验证的阅读节点快照。
- `tutorial/traces/`：真实 Python 运行生成的静态 trace。
- `web/`：只渲染生成内容，不维护第二份手写 Python 实现。
- `docs/course/` 与 `docs/tutorials/`：新课程验收前保留，不直接删除或覆盖。

checkpoint、trace、Markdown 和源码引用必须能自动校验。网站不得根据公式伪造运行结果。

## 单步完成契约

每个 Step 必须同时具备：

- 当前数据流位置与缺失能力说明。
- 最小输入、输出和 shape ledger。
- 可手算的小张量示例。
- 累计代码 checkpoint 与相对上一 checkpoint 的结构化 diff。
- 来自真实源码执行的固定 trace。
- 至少一个正常测试和一个边界/错误测试。
- 与 NumPy、PyTorch 原语或参考实现的数值对照。
- 一个受控错误及 contract 解释。
- 可运行命令、当前限制、变更文件、验收证据和下一步说明。

缺少其中任何一项时，Step 最多标记为“进行中”。

## 实现原则

- 先小张量、固定种子和 CPU 正确性，再接真实权重或优化。
- 先实现清晰参考路径，再讨论向量化、预分配或高性能 kernel。
- NumPy/PyTorch 只共享仓库自有的最小教学 API，不伪装完整 `torch` 兼容。
- 新 backend API 必须有 NumPy 与 PyTorch 两个调用方及 differential test。
- 外部权重 key 到内部 key 使用显式映射；shape 不符立即报错，不用静默 reshape/transpose 掩盖问题。
- Trace 只保存教学所需的小 tensor、切片、统计或聚合，不保存真实权重和敏感信息。
- 采用最小正确修改，避免无调用方的抽象和无证据的兼容代码。

## 验证命令

当前仓库基线：

```bash
uv sync --locked --python 3.11.15
uv run python scripts/check_environment.py
uv run pytest
uv run python examples/run_tiny_dense.py
uv run python examples/run_tiny_greedy.py
```

后续建立新系统后，应补充并维护以下可执行入口：

- Python 模块与示例测试。
- checkpoint/content/trace 重新生成并检查无漂移。
- Web lint、组件测试和静态构建。
- Markdown、源码引用和站内链接检查。

测试失败时记录具体失败范围，不要将整个里程碑标记为完成。真实大模型或远程下载不得成为普通 CI 的必要条件。

## 状态更新记录

只保留对后续会话有价值的简短记录；详细设计和实现历史留在 Git 与对应文档中。

| 日期 | 变更 | 证据 |
| --- | --- | --- |
| 2026-08-14 | 将可视化课程设计纳入版本控制；删除两份已明确被取代的环境计划 | 权威环境说明保留在 `docs/environment.md`，Dense 基线设计保留在 `docs/superpowers/specs/2026-08-07-installable-dense-foundation-design.md` |
| 2026-08-14 | 建立会话状态入口；确认旧 Dense 基线与新课程缺口 | `uv run pytest`：52 passed；当前分支 `rewrite/visual-qwen3-moe-course` |
