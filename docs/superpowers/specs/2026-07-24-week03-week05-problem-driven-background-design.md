# 第 3-5 周问题驱动背景层改造设计

## 背景

现有教程对公式、形状、代码和验证讲解较完整，但经常在给出学习目标后直接进入实现。读者能够跟随步骤运行代码，却不一定先建立以下认识：模块解决什么问题、位于 LLM 推理链的哪里、缺少它会怎样、有哪些相似方案，以及 Qwen3 和本教程为何采用当前方案。

本次只改造第 3-5 周，采用问题驱动重写，而不是在原文中零散插入补充说明。

## 目标

- 每周开始时先说明当前已经具备的能力、尚未解决的问题和本周完成后推理链新增的能力。
- 每个核心模块先回答“为什么需要”，再进入公式和代码。
- 同时解释真实 Qwen3 架构选型与教学实现选型，避免把二者混为一谈。
- 比较功能相同或相近的方案，但只讲理解当前选型所需的差异，不扩展为框架或 kernel 性能综述。
- 保留现有可手算张量、Predict-Run-Explain、练习编号、提示、答案和验收闭环。

## 统一叙事结构

每周正文在学习目标前增加“本周在完整推理中的位置”，按以下顺序展开：

1. 前几周已经实现了什么。
2. 当前系统仍然做不到什么。
3. 本周模块解决哪个具体问题。
4. 本周结束后，完整推理链向前推进到哪里。
5. 本周明确不解决哪些后续问题。

每个核心模块按以下顺序组织：

1. **问题场景**：先用输入、输出或失败案例描述问题，不先给 API。
2. **模块职责**：说明模块接收什么、产生什么，以及它不负责什么。
3. **推理链位置**：给出模块前后的局部数据流和关键 shape。
4. **没有它会怎样**：说明删除或替换不当造成的能力缺失或错误。
5. **相近方案与取舍**：比较 1-3 个最相关方案。
6. **Qwen3 的选择**：只陈述可由目标架构或参考实现确认的选择，配置数字仍以实际 `config.json` 为准。
7. **教程的选择**：解释为何使用微型、显式、低性能但易验证的参考路径。
8. **原理与实现**：再进入公式、shape 推导、代码和断言。
9. **回到总体图**：模块结尾说明它为下一模块提供了什么。

不是每个基础操作都需要完整九段。`argmax`、拆头、广播等局部操作可合并说明；Embedding、LM Head、Attention、GQA、RMSNorm、QK Norm 和 RoPE 等架构模块必须完整覆盖。

## 第 3 周设计

### 核心问题

神经网络不能直接计算文本；即使已经得到隐藏向量，也需要把它转换为对词表中下一个 token 的可比较分数。

### 总体数据流

```text
文本
-> Tokenizer（模型 forward 外部）
-> token_ids [B,S]
-> Embedding
-> hidden [B,S,D]
-> 后续 Decoder（本周暂缺）
-> LM Head
-> logits [B,S,V]
-> 选择规则
-> next_token_ids [B,1]
```

### 重点比较

- Tokenizer 与 Embedding：离散文本切分、整数编号与连续向量表示的职责边界。
- Embedding 查表与 one-hot 矩阵乘法：数学上可对应，但查表更直接且避免构造巨大 one-hot Tensor。
- 独立 LM Head 与权重绑定 LM Head：说明共享权重的参数优势和架构约束，本周使用显式独立模块便于观察。
- logits、softmax 概率与 greedy/sampling：区分模型输出与生成策略。
- teacher forcing 与自回归生成：区分训练数据组织方式和推理循环。

### 叙事终点

读者应明确本周只建立输入端和输出端接口，中间尚无跨 token 的上下文建模，因此不能把微型 `Embedding + LM Head` 当成完整语言模型。

## 第 4 周设计

### 核心问题

第 3 周的每个 token 位置彼此独立。当前位置需要读取此前 token 的信息，同时不能读取未来位置。

### 总体数据流

```text
hidden [B,S,D]
-> Q/K/V projections
-> heads
-> QK 相似度 scores [B,Hq,S,T]
-> causal mask
-> probabilities
-> 对 V 加权汇总
-> merge heads + output projection
-> context-aware hidden [B,S,D]
```

### 重点比较

- Attention 与固定窗口卷积、循环网络：解释动态内容寻址、并行性和全局依赖的差异，不展开训练历史。
- Self-Attention 与 Cross-Attention：本项目是 decoder-only causal LM，因此当前只实现 self-attention。
- 单头与多头：多头允许在不同表示子空间中形成不同关系。
- MHA、GQA 与 MQA：比较 query/KV head 数、KV Cache 成本和表达能力；Qwen3 使用 GQA，本周同时保留 MHA 作为过渡基线。
- 物理复制与逻辑分组 GQA：前者作为易验证 oracle，后者说明避免显式复制的计算组织。
- 手写 attention 与 PyTorch SDPA/fused kernel：本周手写是为了观察中间 Tensor，不代表生产实现应采用该路径。

### 叙事终点

读者应能回答 Attention 为什么存在、causal 约束为何必须在 softmax 前处理，以及 GQA 为何是在 MHA 与 MQA 之间的折中，而不只是复述矩阵公式。

## 第 5 周设计

### 核心问题

Attention 已能混合上下文，但深层网络中的激活尺度需要稳定，纯内容相似度也没有可靠表示 token 顺序。目标 Qwen3 attention 还要求 Q/K 在进入 RoPE 前按 head 归一化。

### 总体数据流

```text
hidden [B,S,D]
-> input RMSNorm
-> Q/K/V projections
-> reshape [B,S,H,Dh]
-> QK Norm（仅 Q/K）
-> transpose [B,H,S,Dh]
-> RoPE（仅 Q/K）
-> causal GQA
-> output [B,S,D]
```

### 重点比较

- RMSNorm 与 LayerNorm：是否减均值、参数与计算差异，以及 Qwen3 采用 RMSNorm 的事实边界。
- input RMSNorm 与 QK Norm：两者统计对象、所在位置和职责不同，不能互相替代。
- 不使用位置编码、绝对位置 embedding、相对位置 bias 与 RoPE：比较位置注入位置、长度泛化直觉及对 attention score 的影响。
- split-half 与 adjacent-pair RoPE：二者都是布局约定，必须与权重实现一致；本教程选择与目标参考实现一致的 split-half。
- 先 QK Norm 后 RoPE 与交换顺序：用非均匀权重说明它们通常不等价。
- 显式 FP32 参考计算与 fused/低精度实现：教学主线优先稳定、可验证的数值基线。

### 叙事终点

读者应理解 normalization 解决尺度问题、RoPE 解决顺序信息问题、QK Norm 约束 attention 内部 Q/K 尺度，三者不是一组可任意互换的“数学技巧”。

## 改写边界

- 不新增真实模型下载或运行要求。
- 不把教程扩展为 Transformer 历史综述。
- 不对未经配置或参考实现确认的 Qwen3 细节作推断。
- 不提前实现 Decoder、SwiGLU、MoE、KV Cache 或生成循环。
- 不删除现有正确性实验；只在叙事需要时移动代码块位置或补充前后解释。
- 尽量保持现有锚点和练习编号，减少已有链接失效。

## 验收标准

- 阅读每周前两节后，不看代码也能说出本周模块在完整 LLM 推理中的位置。
- 每个核心架构模块都明确回答用途、缺失后果、相近方案、Qwen3 选型和教学选型。
- 所有比较都服务于当前模块选择，不形成无边界的方案清单。
- 原有代码示例仍可独立运行，原有断言、练习、提示和答案保持一致或同步更新。
- 第 3-5 周之间形成连续叙事：接口建立 -> 上下文混合 -> 尺度与位置信息。
