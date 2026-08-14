# Qwen3 风格 LLM 推理主线课程

这条主线面向会 Python 基础语法、函数和类，但不熟悉 PyTorch，也还没有 decoder-only LLM 推理整体图的学习者。课程始终对照仓库中的真实 `src/qwen3_moe/` 实现；你不会先读完一大堆公式，再猜它们应放在模型哪里。

## 怎样使用这套课程

请按编号顺序阅读主线。每章都要求留下四类证据：

1. 一张“当前模块在整机哪里”的位置图。
2. 一份写清轴含义的 shape ledger。
3. 阅读并运行正文中的真实关键函数。
4. 一个已运行的测试或受控错误。

主线负责动机、整机位置、源码导航和最小闭环；[深入实验库](../tutorials/)负责 API 细节、公式推导、手算和大量练习。第一次阅读时不要试图把深入教程全部读完，只在主线明确链接时按需选读。

## 阅读顺序

| 章节 | 核心问题 | 完成后的产出 |
| --- | --- | --- |
| [00：LLM 推理整机导览](00-inference-overview.md) | 文本怎样经过模型并产生下一个 token？ | 区分 Tokenizer、一次 forward、一次选择和生成循环 |
| [01：模型外壳](01-model-shell.md) | `[B,S]` 怎样变成 `[B,S,D]`，再变成 `[B,S,V]`？ | 能读 `TinyDenseCausalLM` 的构造函数和外层 forward |
| [02：Decoder Layer](02-decoder-layer.md) | 为什么每层有 Attention、MLP、两次 Norm 和两次 residual？ | 能写出单层正确执行顺序并核对真实 debug 边界 |
| [03：真实模型中的张量流](03-tensor-flow.md) | `view`、`transpose`、广播和 `matmul` 在模型里各解决什么问题？ | 能沿真实 attention 代码追踪 rank、shape、dtype、device |
| [04：Attention](04-attention.md) | 一个位置怎样读取允许看到的上下文？ | 能追踪 causal GQA、QK Norm 和 RoPE 的 shape |
| [05：MLP](05-mlp.md) | Attention 之后为什么还需要逐 token 变换？ | 能解释 SwiGLU，并指出未来的 MoE 替换插槽 |
| [06：组装 Dense 模型](06-assemble-dense-model.md) | 已验证的零件怎样组成一次完整 forward？ | 能沿 `model.py` 和 `decoder.py` 定位首个偏差 |
| [07：自回归生成](07-autoregressive-generation.md) | 一次 forward 怎样变成不断增长的 token 序列？ | 能运行并解释最小 greedy 循环 |
| [08：从 Dense 到 MoE](08-dense-to-moe.md) | MoE 改变模型的哪一部分？ | 能说明 Router 和 Experts 将如何替换 MLP 插槽 |

第一次学习先完成 00-07，建立 Dense forward 与最小生成闭环；第 08 章用于把这套心智模型连接到后续 MoE 实现。

## 统一形状记号

| 记号 | 含义 | 本课程常见位置 |
| --- | --- | --- |
| `B` | batch size，一次处理的序列数 | `input_ids [B,S]` |
| `S` | sequence length，每条序列的 token 数 | `hidden_states [B,S,D]` |
| `V` | vocabulary size，词表条目数 | `logits [B,S,V]` |
| `D` | hidden size，每个 token 的隐藏宽度 | residual 主干 `[B,S,D]` |
| `L` | decoder layer 数 | `model.layers` |
| `I` | MLP intermediate size | SwiGLU 中间张量 `[B,S,I]` |
| `Hq` | query head 数 | query `[B,Hq,S,Dh]` |
| `Hkv` | key/value head 数 | key/value `[B,Hkv,S,Dh]` |
| `Dh` | 每个 attention head 的宽度 | `D = Hq * Dh` |
| `G` | 每个 KV head 服务的 query head 数 | `G = Hq / Hkv` |

读 shape 时必须同时说出轴的含义。`[2,4,8]` 只是一组长度；`[B,S,D]=[2,4,8]` 才说明它是一批 2 条、每条 4 个位置、每个位置宽度 8 的 hidden states。

## 环境与运行约定

所有命令都从仓库根目录运行：

```bash
uv sync --locked --python 3.11.15
uv run python scripts/check_environment.py
uv run pytest
```

主线实验使用 CPU、小 shape 和固定随机种子，不下载 Tokenizer、checkpoint 或数据集。代码围栏可以直接通过 heredoc 运行，也可以只阅读后执行章节给出的仓库脚本。

## 真实源码地图

| 推理位置 | 真实文件与符号 |
| --- | --- |
| 模型结构与单次 forward | [`model.py::TinyDenseCausalLM`](../../src/qwen3_moe/model.py) |
| 单个 Decoder Layer | [`decoder.py::DenseDecoderLayer`](../../src/qwen3_moe/decoder.py) |
| 因果 GQA | [`attention.py::GroupedQueryAttention`](../../src/qwen3_moe/attention.py) |
| RMSNorm | [`norms.py::RMSNorm`](../../src/qwen3_moe/norms.py) |
| RoPE | [`rope.py::RotaryEmbedding`](../../src/qwen3_moe/rope.py) 与 [`apply_rotary_pos_emb`](../../src/qwen3_moe/rope.py) |
| Dense MLP | [`mlp.py::SwiGLU`](../../src/qwen3_moe/mlp.py) |
| 结构配置 | [`config.py::DenseConfig`](../../src/qwen3_moe/config.py) |
| shape/dtype/device/有限性守卫 | [`debug.py`](../../src/qwen3_moe/debug.py) |

## 完成主线课程后你应能做到

- 口述 `文本 -> Tokenizer -> input_ids -> model forward -> logits -> next-token 选择 -> 追加 -> 重复`。
- 指出当前仓库实现的是随机初始化微型 Dense 模型，不具有可用语言能力。
- 沿真实源码跟踪 `[B,S] -> [B,S,D] -> L x [B,S,D] -> [B,S,V]`。
- 区分完整 Decoder Layer、完整单次模型 forward 和完整自回归生成。
- 在暂时不会 Attention 或 RoPE 公式时，仍知道每个模块为什么存在，以及下一步应进入哪份源码或深入教程。

从 [00：LLM 推理整机导览](00-inference-overview.md) 开始。
