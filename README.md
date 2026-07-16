# 从零实现 Qwen3-30B-A3B MoE 推理

这是一个面向初学者的 16 周学习仓库。目标不是调用现成的 `generate()`，而是逐步理解并独立实现一条可验证的 Qwen3-30B-A3B MoE 端到端推理链路。

## 适合谁

默认学习者：

- 会 Python 基础语法、函数、类和虚拟环境，但尚未系统学习 PyTorch。
- 每周可投入 6-10 小时，持续 16 周。
- 本地有 RTX 4060 Laptop 8GB，另有一台单卡 A10 24GB Linux 服务器。
- 愿意先在小张量和微型配置上验证正确性，再接触真实模型权重。

## 16 周后你将能够

- 解释从文本、Token IDs、Embedding、Decoder、Logits 到下一个 token 的完整数据流。
- 对关键张量写出形状，并说明每一维代表 batch、序列、头、专家还是隐藏维度。
- 用 PyTorch 实现 RMSNorm、RoPE、QK Norm、GQA、SwiGLU、MoE Router、Top-K 专家分发与聚合。
- 组合 Dense Decoder 和 MoE Decoder，完成 prefill、逐 token decode、KV Cache、采样与停止条件。
- 读取模型配置和权重索引，把兼容的 Qwen3-MoE 权重映射到自己的模块，并逐层做数值对齐。
- 估算和实测权重、激活、临时张量与 KV Cache 的显存占用。
- 解释 BF16、FP16、INT8、INT4 的精度与性能取舍，并对量化运行做基准测试。
- 在不依赖 Transformers 高层模型实现的前提下，独立维护一条教学用途的端到端推理路径。

## 两台机器如何分工

| 设备 | 主要职责 | 不适合承担的工作 |
| --- | --- | --- |
| RTX 4060 Laptop 8GB | 日常编码、CPU/CUDA 单元测试、微型模型、形状实验、单层基准 | 加载 Qwen3-30B-A3B 全量 BF16 权重 |
| A10 24GB | 较大微型配置、量化后的真实模型实验、显存与吞吐测试、CPU offload 实验 | 单卡完整加载约 60GB 权重体积的 BF16 30B 模型 |
| CPU | Tokenizer、配置与权重清单检查、高精度参考、小规模正确性验证 | 追求大模型交互式生成速度 |

日常流程是“本地写代码和跑小测试，服务器只跑确实需要更多显存的实验”。代码通过 SSH 和版本控制同步；模型文件留在各机器的缓存或数据盘中，不提交到仓库。

> Qwen3-30B-A3B 的“约 3B 激活参数”不等于只需存储 3B 参数。推理时仍要保存全部专家权重。仅 BF16 权重理论体积就约为 60GB，未计入 KV Cache、激活、框架开销和临时缓冲区，因此单张 24GB A10 无法完整加载全量 BF16 模型。真实 30B 阶段应使用合适的 4-bit 量化方案或 CPU offload，并如实记录性能代价。

## 仓库地图

```text
learn-qwen3-moe/
├── README.md                         # 项目入口与首次学习流程
├── docs/
│   ├── roadmap.md                    # 16 周逐周路线
│   ├── environment.md                # 本地与服务器环境
│   ├── progress.md                   # 可复用学习周记模板
│   └── plans/                        # 学习项目设计说明
└── .gitignore                        # 本地环境、模型权重和输出忽略规则
```

当前仓库只包含教学大纲和学习准备文档。代码、练习、测试与模型权重将在正式开始对应周次时逐步创建。

## 第一次学习会话

当前可以直接执行以下命令；它们只验证基础环境和学习文档，不下载模型：

```bash
cd /home/zbc/learn-qwen3-moe
python3 --version
command -v nvidia-smi >/dev/null && nvidia-smi || printf '%s\n' '未检测到 nvidia-smi；可以先使用 CPU 阅读和学习。'
test -f README.md && test -f docs/environment.md && test -f docs/roadmap.md && test -f docs/progress.md && printf '%s\n' '文档检查通过：请先阅读 docs/environment.md，再查看 docs/roadmap.md 第 1 周。'
```

随后按 [环境配置](docs/environment.md) 创建 Python 3.11 虚拟环境，并阅读 [第 1 周路线](docs/roadmap.md#第-1-周张量形状与显存)。本次会话不下载 Qwen3-30B-A3B，也不需要登录 Hugging Face。

## 学习方法

每周遵循同一闭环：

1. **先预测**：运行代码前写出每个输入、输出和中间张量的形状。
2. **小张量推导**：用能手算的数字验证公式，不一开始就使用大模型。
3. **独立实现**：核心逻辑放在可测试的 Python 模块中，Notebook 只用于观察。
4. **三级验证**：依次检查 shape/dtype/device、数值误差、端到端首个偏差位置。
5. **做对照实验**：每次只改变一个变量，记录随机种子、设备、dtype 和峰值显存。
6. **留下证据**：把命令、输出摘要、断言或图表写入周记，而不是仅勾选“看懂了”。

建议每周时间分配：2 小时概念与手算，3-5 小时实现和测试，1-2 小时实验、复盘与记录。完整安排见 [16 周路线](docs/roadmap.md)。

## 项目边界

本仓库会做：

- 教学优先、形状显式、可逐层验证的纯 PyTorch 推理实现。
- 单 batch 和小 batch 的正确性路径，再逐步讨论性能。
- 官方/兼容权重的名称映射、分片读取和数值对齐方法。
- KV Cache、量化适配接口、显存分析和基础 benchmark。

本仓库不会把以下内容当作主线目标：

- 训练、微调、反向传播、分布式训练或数据集工程。
- 重写完整 Transformers、vLLM 或生产级推理服务框架。
- 从零实现高性能 CUDA/Triton kernel；它们仅作为第 16 周选修。
- 承诺单张 A10 以 BF16 装下 Qwen3-30B-A3B，或把量化权重加载伪装成纯 PyTorch BF16 权重加载。
- 在理解形状与数值验证之前追求生成速度或“能输出文字就算正确”。

## 从这里继续

- 配置机器：[docs/environment.md](docs/environment.md)
- 查看每周任务与验收标准：[docs/roadmap.md](docs/roadmap.md)
- 复制周记模板开始记录：[docs/progress.md](docs/progress.md)
