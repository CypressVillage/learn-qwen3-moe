# LLM 推理教程主线重构设计

## 背景

当前仓库已经有七篇细致的周教程，以及从这些教程提取出的可安装 Dense 推理实现。局部内容重视小张量、形状推导、受控错误和数值验证，但首次学习路径仍有三个明显断点：完整 LLM 推理地图出现较晚，教程正文没有把概念直接映射到 `src/qwen3_moe/` 和 `tests/`，一次模型 forward 与连续自回归生成之间缺少可运行的教学闭环。

本次重构不删除已有高质量教程，而是增加一条更短、更连续、始终对应真实源码的主线课程。学习者先自顶向下建立整机地图，再自底向上理解关键模块，最后重新组装模型并观察生成控制流。

## 目标学习者与成功标准

目标学习者会 Python 基础语法、函数和类，但没有系统学习 PyTorch，也没有稳定的 decoder-only LLM 推理心智模型。

首个主线章节应支持学习者在约 60 分钟内：

- 解释 Tokenizer、模型 forward、next-token 选择和生成循环的边界。
- 说出 `Embedding -> Decoder Layers -> Final Norm -> LM Head` 的整体路径。
- 指出 `src/qwen3_moe/` 中每个核心文件位于推理链的什么位置。
- 区分完整 Decoder Layer、完整单次模型 forward 和完整自回归生成。
- 在暂时不理解 Attention 或 RoPE 公式的情况下，知道后续每个模块为什么需要学习。

完成 Dense 主线后，学习者应能沿真实源码跟踪 `[B,S] -> [B,S,D] -> [B,S,V] -> [B,1]`，运行最小生成循环，并说明 MoE 将替换 Decoder Layer 中的 MLP 插槽，而不改变模型外部接口。

## 教学架构

仓库提供两条职责不同、相互链接的学习轨道：

```text
README
  |
  +-> 主线课程 docs/course/
  |     按认知依赖顺序阅读
  |     负责动机、整机位置、源码导航和最小闭环
  |
  +-> 深入实验 docs/tutorials/
        按主线章节需要选读
        负责 PyTorch API、公式推导、手算、受控错误和大量练习
```

主线不再按周次组织。现有 `docs/tutorials/week01-...` 至 `week07-...` 保留原路径，避免丢失内容和破坏历史链接，并重新定位为深入实验库。

课程采用三次经过同一模型的螺旋结构：

1. 第一遍只看组件职责和端到端边界，不展开内部公式。
2. 第二遍沿 `TinyDenseCausalLM.forward()` 下钻，理解张量、Decoder、Attention 和 MLP。
3. 第三遍把单次 forward 放入生成循环，再说明 KV Cache、真实 Tokenizer、sampling 和 MoE 的后续位置。

## 主线目录

新增 `docs/course/`：

```text
docs/course/
├── README.md
├── 00-inference-overview.md
├── 01-model-shell.md
├── 02-decoder-layer.md
├── 03-tensor-flow.md
├── 04-attention.md
├── 05-mlp.md
├── 06-assemble-dense-model.md
├── 07-autoregressive-generation.md
└── 08-dense-to-moe.md
```

### 课程入口

`docs/course/README.md` 说明目标学习者、阅读顺序、主线与深入实验的关系、统一形状记号、运行命令和学习产出。它明确主线必须按顺序阅读，深入实验按需选读。

### 00：LLM 推理整机导览

在约 60 分钟内完成一次黑盒到白盒的初步导览：

```text
文本
-> Tokenizer
-> token IDs
-> TinyDenseCausalLM forward
-> logits
-> next-token 选择
-> 追加 token
-> 重复
```

本章运行现有 Dense forward 示例，阅读 `TinyDenseCausalLM` 的组件树和 `forward()` 调用顺序，但不进入 Attention、RoPE 或 SwiGLU 的公式。章节必须明确当前随机初始化模型只用于观察数据流，不能生成有意义文本。

### 01：模型外壳

解释 token IDs、Embedding、hidden states、final norm、LM Head 和 logits，建立 `[B,S] -> [B,S,D] -> [B,S,V]`。Decoder Layers 暂时作为黑盒。源码落点是 `model.py::TinyDenseCausalLM` 的构造函数、Embedding 和 LM Head。

### 02：Decoder Layer

先解释模块职责，再解释顺序：

```text
h = x + Attention(RMSNorm(x))
y = h + MLP(RMSNorm(h))
```

本章重点是 residual contract、上下文混合与逐 token 变换的分工，不提前展开 Attention 内部数学。源码落点是 `decoder.py::DenseDecoderLayer`。

### 03：真实模型中的张量流

围绕 `src` 中实际出现的操作讲解 rank、shape、reshape、transpose、broadcasting、matmul、dtype 和 device。每个 PyTorch 概念都必须说明它服务于哪个模型边界，例如 Attention head 布局为什么需要 transpose。更完整的 stride、contiguous、内存和受控错误练习链接到原 Week 1-2 教程。

### 04：Attention

从“当前位置怎样读取左侧上下文”开始，依次解释 causal mask、Q/K/V、multi-head、GQA、QK Norm 和 RoPE。章节持续维护 shape ledger，并对应 `attention.py`、`rope.py`、`norms.py` 及相关测试。

### 05：MLP

解释 Attention 负责跨 token 混合，MLP 负责每个 token 位置上的特征变换。介绍 SwiGLU 的三条投影路径和输入输出 contract，并把 MLP 明确标记为未来替换成 MoE 的插槽。

### 06：组装 Dense 模型

重新从 `input_ids` 走到 logits，对照 `model.py` 和 `decoder.py` 阅读真实累计实现。学习者记录每层边界，运行 debug trace，并用前缀因果性测试理解模型正确性不只等于 shape 正确。

### 07：自回归生成

加入最小、确定性的 greedy 生成循环：

```text
input_ids
-> model
-> logits[:, -1, :]
-> argmax
-> next_token [B,1]
-> append
-> repeat
```

本章明确该循环没有真实 Tokenizer、EOS、随机 sampling 或 KV Cache，只用于揭示控制流。完整生产式生成能力仍由后续课程逐步实现。

### 08：从 Dense 到 MoE

通过 Decoder Layer 的稳定外部 contract 解释 Dense MLP 到 MoE 的替换边界：

```text
Dense: hidden -> SwiGLU -> update
MoE:   hidden -> Router -> Top-K Experts -> weighted aggregation -> update
```

本章只建立后续路线，不在本轮实现 Router 或 Experts。

## 统一章节契约

每个主线章节使用一致结构：

1. 为什么需要这个模块。
2. 它位于整机的什么位置。
3. 输入、输出和关键中间 shape。
4. 这段真实代码在整机中承担的任务。
5. 直接展示当前 `src` 中的关键完整函数；构造函数只摘录理解依赖关系所需的部分。
6. 按执行顺序分段讲解输入 contract、核心计算、shape 变化、输出和 debug 分支。
7. 一个可独立运行的最小实验。
8. 对应测试及其验证的行为。
9. 常见误解和受控错误。
10. 完整源码文件和深入实验链接。
11. 回到完整推理流程后的当前进度。

每章结束要求学习者留下四类证据：位置图、shape ledger、一次真实源码调用、一个测试或受控错误。验收问题优先要求解释“为什么需要”和“删除后会怎样”，避免只考 API 记忆。

“对应真实源码符号”不能作为独立索引块代替教学正文。源码链接只用于查看完整文件上下文；关键代码必须进入正文，并在代码后立即解释。教程嵌入的源码片段应标注来源，并由自动化测试校验其与当前 `src` 实现保持同步，避免实现变化后教程继续展示过期代码。

## 旧教程改造

七篇旧教程保留正文和文件路径，只在顶部增加简短导航块：

- 本篇在新主线中的位置。
- 建议何时阅读，以及哪些内容可以第一次跳过。
- 对应 `src` 文件、主要符号和测试文件。
- 返回主线课程的链接。

旧教程不再承担首次建立完整推理地图的职责，也不需要复制新主线的整机说明。

## README 与路线图

根 `README.md` 把 `docs/course/00-inference-overview.md` 设为首次学习的主要入口。原七周链接移动到“深入实验”区域，并说明它们适合在主线需要时选读。

`docs/roadmap.md` 保留 16 周时间规划和实践验收价值，但明确它是进度计划，不是最推荐的首次概念阅读顺序。路线图链接到相应主线章节和深入实验。

## 最小生成示例

保留 `examples/run_tiny_dense.py` 作为单次 forward 示例。新增单独的最小 greedy 生成示例，避免一个文件同时承担两个概念：

- 使用微型 `DenseConfig` 和固定随机种子。
- 输入手工整数 token IDs。
- 每一步运行模型、取最后位置 logits、argmax、追加 token。
- 打印每一步新增 token 和最终 token ID 序列。
- 不下载 Tokenizer、模型权重或数据集。
- 明确输出没有语言语义。

生成循环可保留在示例中，不提前扩展稳定公共 API。只有后续出现多个调用方或真实生成需求时，才提取 `generation.py`。

## 测试与验证

现有模块测试必须继续通过。新增测试只覆盖最小生成示例新增的行为，不测试随机初始化模型的文本质量：

- 每一步序列长度增加 1。
- 新 token shape 为 `[B,1]`。
- 新 token 位于配置词表范围内。
- 固定随机种子和相同输入产生确定结果。
- batch 维保持不变。

文档验证包括：

- 所有新增 Markdown 相对链接有效。
- README、主线、路线图和旧教程之间可双向导航。
- 源码符号引用与实际文件一致。
- 文档不把一次 forward 描述为完整生成。
- 文档不声称随机初始化模型具有语言能力。

实施完成后运行：

```bash
uv run pytest
uv run python examples/run_tiny_dense.py
uv run python examples/run_tiny_greedy.py
```

## 范围外事项

本轮不实现：

- 真实 Qwen Tokenizer 或 chat template。
- EOS、temperature、top-k、top-p 等完整 sampling 策略。
- KV Cache、prefill/decode 分离或性能优化。
- MoE Router、专家分发和聚合。
- 真实 checkpoint、权重映射或数值对齐。
- 对七篇旧教程内部全部代码围栏进行重写。

## 验收标准

- 新学习者从根 README 能直接进入 60 分钟整机导览。
- 主线按认知阶段而不是周次组织，并完整覆盖 Dense forward 与最小生成控制流。
- 每个主线章节都有动机、整机位置、shape、源码、测试和深入实验映射。
- 七篇旧教程保留原路径，并明确转为深入实验。
- 学习者能够从 `TinyDenseCausalLM` 导航到 Decoder、Attention、Norm、RoPE 和 MLP。
- 单次 forward 与自回归生成在文档和示例层面清楚分离。
- 最小 greedy 示例和全部测试在 CPU 环境通过。
- 本轮没有提前实现 Tokenizer、KV Cache、sampling、MoE 或真实权重。
