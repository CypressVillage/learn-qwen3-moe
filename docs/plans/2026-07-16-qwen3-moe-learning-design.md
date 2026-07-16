# Qwen3 MoE 端到端推理学习项目设计

## 学习者画像

- Python 基础：会基础语法，尚未系统使用 PyTorch。
- 每周投入：6-10 小时。
- 本地设备：RTX 4060 Laptop，8GB 专用显存。
- 服务器设备：单张 NVIDIA A10，24GB 显存。
- 主目标：从原理出发，用 PyTorch 实现可加载真实权重的教学版 Qwen3-MoE 推理器。
- 进阶目标：掌握 KV Cache、量化适配、显存分析和基准测试；CUDA/Triton 作为选修。

## 教学策略

采用双线并行方式：每个主题先通过极小张量和短代码理解公式，再到真实 Qwen3 实现中定位对应模块。每个主题都完成“讲义、实验、正式实现、数值验证、真实模型观察、学习记录”的闭环。

项目不复刻完整的 Transformers 框架。主线只实现一条最小而完整的推理数据流：

```text
文本 -> Token IDs -> Embedding -> Decoder Layers -> LM Head
     -> Logits -> Sampling -> New Token -> KV Cache Decode Loop
```

Decoder Layer 中重点实现：

```text
RMSNorm -> Q/K/V Projection -> QK Norm -> RoPE -> GQA Attention
        -> Residual -> RMSNorm -> MoE Router -> Top-K Experts
        -> Weighted Aggregation -> Residual
```

## 仓库结构

```text
learn-qwen3-moe/
├── README.md
├── pyproject.toml
├── docs/
│   ├── roadmap.md
│   ├── environment.md
│   ├── progress.md
│   └── plans/
├── exercises/
├── notebooks/
├── src/learn_qwen3_moe/
│   ├── foundations/
│   ├── dense/
│   ├── moe/
│   ├── model/
│   └── generation/
├── scripts/
├── configs/
└── tests/
```

核心实现放在可测试的 Python 模块中。Notebook 仅用于可视化和交互观察，不作为唯一实现，以避免代码难以复用和验证。

## 16 周阶段

1. 第 1-2 周：PyTorch 张量、广播、矩阵乘法、模块、参数、dtype、device 和显存估算。
2. 第 3-4 周：Token、因果语言建模、Softmax、Self-Attention、GQA 和逐 token 解码。
3. 第 5-7 周：RMSNorm、RoPE、QK Norm、SwiGLU、Causal Mask 和 Dense Decoder Layer。
4. 第 8-10 周：MoE Router、Top-K、专家分发、专家 MLP、加权聚合和负载观察。
5. 第 11-13 周：配置、权重映射、Tokenizer、Prefill、Decode、KV Cache、采样和停止条件。
6. 第 14-16 周：FP32/FP16/BF16/INT8/INT4 对比、显存核算、吞吐和延迟分析。

## 验证策略

每个组件使用三级验证：

1. 形状验证：断言 shape、dtype 和 device，显式记录每一维的语义。
2. 数值验证：固定随机种子，与公式、PyTorch 原语或 Hugging Face 参考模块比较。
3. 端到端验证：逐层比较 hidden states、logits 和 KV Cache，定位首次偏差。

测试遵循小步 TDD。浮点测试根据 dtype 使用明确的 `atol` 和 `rtol`，不以最终生成文本相似代替中间数值正确性。

## 硬件分工

- RTX 4060 8GB：日常开发、微型模型、单元测试和可视化实验。
- A10 24GB：BF16 小模型、Qwen3-30B-A3B 量化运行、显存和吞吐测试。
- CPU：Tokenizer、权重检查和高精度数值参考。

Qwen3-30B-A3B 的 BF16 权重约需 60GB，单张 A10 无法完整加载。真实大模型阶段采用 4-bit 量化或 CPU offload，并把量化加载与教学版纯 PyTorch 权重加载明确分开。

## 完成标准

学习者能够：

- 画出 Qwen3-MoE 从文本到新 token 的完整数据流。
- 解释关键张量每一维的含义及变化。
- 独立实现教学版 Qwen3-MoE 前向传播和 KV Cache 解码。
- 将兼容的官方权重映射到自己的模块，并进行逐层数值对齐。
- 估算并实测权重、激活、KV Cache 和临时张量的显存。
- 清楚说明 MoE 的总参数、激活参数、路由机制及性能瓶颈。
