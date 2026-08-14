# 08：从 Dense 到 MoE，只改变 Decoder 内部插槽

Dense 主线已经完整覆盖单次 forward 和最小 greedy 循环。本章只建立下一阶段的路线：MoE 替换 Decoder Layer 中的 MLP 插槽，而不是推翻模型外壳、Attention 或生成循环。

## 为什么引入 MoE

当前每个 token 都经过同一个 `SwiGLU`，并激活这组 Dense MLP 的全部参数。MoE 的核心想法是准备多个“专家”MLP，再让 Router 为每个 token 选择少量专家，将它们的输出按权重聚合。

概念对比：

```text
Dense: hidden -> SwiGLU -> update
MoE:   hidden -> Router -> Top-K Experts -> weighted aggregation -> update
```

MoE 增加可用参数和路由行为，但为了接回现有 residual，外部仍应像一个 MLP 模块：输入 `[B,S,D]`，输出 `[B,S,D]`。

## 它在整机中的位置

```text
TinyDenseCausalLM
  -> Embedding
  -> Decoder Layer
       -> input_norm
       -> Attention
       -> residual add
       -> post_attention_norm
       -> self.mlp                 <- 唯一替换边界
       -> residual add
  -> Final Norm
  -> LM Head
  -> logits
```

当前替换插槽在 `DenseDecoderLayer.__init__()` 中直接可见：

<!-- source-sync: src/qwen3_moe/decoder.py::DenseDecoderLayer.__init__ -->
```python
def __init__(self, config: DenseConfig) -> None:
    super().__init__()
    self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    self.self_attn = GroupedQueryAttention(config)
    self.post_attention_norm = RMSNorm(
        config.hidden_size, config.rms_norm_eps
    )
    self.mlp = SwiGLU(config)
```

### 1. 初始化 nn.Module 外壳

`super().__init__()` 先建立 PyTorch 模块的参数、子模块和状态管理基础。Dense 与未来 MoE Decoder Layer 都必须保留这一步。

### 2. 保留 Attention 前的 RMSNorm

`self.input_norm` 是第一个 pre-norm，负责在 Self-Attention 前规范化 hidden states。MLP 是否为 Dense 或 MoE 不改变这条 Attention 路径。

### 3. 保留 GroupedQueryAttention

`self.self_attn = GroupedQueryAttention(config)` 连同其 causal mask、QK Norm、RoPE 和 GQA 数据流保持不变。MoE 不替换 Attention。

### 4. 保留 MLP 前的 RMSNorm

`self.post_attention_norm` 在 Attention residual 之后、MLP 插槽之前执行。它仍把 `[B,S,D]` 的 `after_attention` 变为同 shape 的 `mlp_norm`，也是未来 Router 实际接收的浮点 hidden states。

### 5. 只替换 self.mlp 插槽

`self.mlp = SwiGLU(config)` 是 Dense 到 MoE 的直接替换点：当前绑定 Dense `SwiGLU`，未来才会绑定遵守同一输入输出 contract 的 MoE 模块。替换模块仍必须接收 `[B,S,D]` 并返回 `[B,S,D]`，这样 `forward()` 中的 `output = after_attention + mlp_update` 才能保持不变。

因此不变的组件包括 `input_norm`、`self_attn`、`post_attention_norm`、两次 pre-norm residual 外壳、模型级 Embedding/positions/causal mask、Final Norm、LM Head，以及第 07 章的 greedy 循环。只要未来 MoE 遵守 `self.mlp` contract，`TinyDenseCausalLM.forward()` 的输入输出也不需要改变。

查看完整文件：[`src/qwen3_moe/decoder.py`](../../src/qwen3_moe/decoder.py)。当前插槽实现见 [`src/qwen3_moe/mlp.py`](../../src/qwen3_moe/mlp.py)，上层模型边界见 [`src/qwen3_moe/model.py`](../../src/qwen3_moe/model.py)，residual contract 检查见 [`src/qwen3_moe/debug.py`](../../src/qwen3_moe/debug.py)。

## Shape ledger：稳定边界与未来内部

### 当前 Dense MLP

| 阶段 | shape |
| --- | --- |
| `mlp_norm` | `[B,S,D]` |
| `gate/up/product` | `[B,S,M]` |
| `mlp_update` | `[B,S,D]` |
| residual 输出 | `[B,S,D]` |

### 未来 MoE 的概念 shape

记专家数为 `E`、每个 token 选择 `K` 个专家。这里仅用于认识边界，不规定具体实现：

| 概念阶段 | 可能的 shape |
| --- | --- |
| MoE 输入 | `[B,S,D]` |
| Router 分数 | `[B,S,E]` |
| Top-K 专家索引/权重 | `[B,S,K]` |
| 被选专家更新 | 概念上每个 token 有 `K` 个 `[D]` 更新 |
| 加权聚合结果 | `[B,S,D]` |
| residual 输出 | `[B,S,D]` |

内部可以增加 `E` 和 `K` 维，但进入模块和离开模块时必须回到 `[B,S,D]`。这就是稳定替换边界。

## 哪些保持不变

- `input_ids [B,S]` 的模型输入 contract 不变。
- Embedding、positions 和 causal mask 的构造不变。
- `GroupedQueryAttention`、QK Norm 和 RoPE 不因 MLP 替换而改变。
- pre-norm 顺序与第二次 residual 基准 `after_attention` 不变。
- Final Norm 和 LM Head 不变。
- 模型输出仍是 `logits [B,S,V]`。
- greedy 循环仍取 `logits[:, -1, :]`，得到 `[B,1]` 后追加。

## 哪些会在后续改变

- `DenseDecoderLayer.__init__()` 中 `self.mlp = SwiGLU(config)` 将有新的模块选择。
- 配置需要专家数、Top-K 等 MoE 参数。
- MLP 内部从“一套参数处理全部 token”变成 Router、专家选择、分发、专家计算和加权聚合。
- debug 信息和测试会增加路由概率、专家索引、负载等行为。

本章**不实现**这些改变，也不定义 Router、Experts 或 dispatch 代码。这里的目标只是让你能指出改动应发生在哪里，以及哪些外部 contract 必须受到保护。

注意当前仓库真实符号仍是 `TinyDenseCausalLM`、`DenseDecoderLayer` 和 `SwiGLU`。不要把上面的 `Router`、`Top-K Experts` 当成已经存在的类名。

## 最小实验：观察替换 contract

本实验只观察现有 MLP 插槽的输入、输出和 residual，不创建或模拟 MoE：

```bash
uv run python - <<'PY'
import torch
from qwen3_moe import DenseConfig, DenseDecoderLayer

torch.manual_seed(7)
config = DenseConfig(
    vocab_size=11,
    hidden_size=8,
    intermediate_size=12,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
    rope_theta=100.0,
)
layer = DenseDecoderLayer(config).eval()
hidden = torch.randn(2, 4, 8)
positions = torch.arange(4).view(1, 4).expand(2, 4)

with torch.inference_mode():
    output, debug = layer(hidden, positions, return_debug=True)

print("MLP input:", debug["mlp_norm"].shape)
print("MLP update:", debug["mlp_update"].shape)
print("layer output:", output.shape)
print(
    "residual contract:",
    torch.allclose(output, debug["after_attention"] + debug["mlp_update"]),
)
PY
```

实验结果应该显示三个张量都是 `[2,4,8]`。未来 MoE 首先必须通过同一项外部检查，然后才谈路由是否正确。

受控思考：如果未来模块返回 `[B,S,M]`，第二次 residual add 为什么不能成立？因为 `after_attention` 是 `[B,S,D]`，最后一维 contract 已被破坏。不能靠随意广播掩盖这个错误，模块必须在内部聚合并投影回 `D`。

## 对应测试

- [`tests/test_decoder.py`](../../tests/test_decoder.py) 的 `test_dense_decoder_uses_two_correct_residual_bases()`：定义替换后仍必须保持的核心行为，`output = after_attention + mlp_update`。
- [`tests/test_mlp.py`](../../tests/test_mlp.py) 的 `test_swiglu_matches_explicit_pytorch_primitives()`：记录当前 Dense 插槽的基线语义和 `D -> M -> D` shape。
- [`tests/test_model.py`](../../tests/test_model.py) 的 `test_tiny_dense_model_runs_end_to_end_on_cpu()`：保护 Decoder 外部的 `[B,S,D]` 累计边界和最终 `[B,S,V]` 接口。
- 同文件的 `test_causal_prefix_logits_do_not_depend_on_a_future_token()`：MLP 被替换后，整机仍不能破坏因果性。

未来实现 MoE 时，需要新增 Router 选择、Top-K、聚合、专家路由和边界错误测试；本章不提前编写这些测试或实现。

## 常见误解

1. **“MoE 替换整个模型。”** 在这条路线中，它只替换每个 Decoder Layer 的 MLP 插槽。
2. **“有多个专家就要输出多个 hidden states。”** 专家结果必须在模块内部聚合成一个 `[B,S,D]` 更新，才能接回 residual。
3. **“Router 在 token ID 上选择专家。”** 替换边界接收的是规范化后的浮点 hidden states `[B,S,D]`，不是 `[B,S]` token IDs。
4. **“MoE 会改变 logits 或生成循环接口。”** 内部替换后，外部仍应返回 `[B,S,V]`；greedy 控制流不依赖 MLP 的具体内部实现。
5. **“本章的概念图说明 Router 已经实现。”** 当前 `src` 中没有 Router 或 Experts；本章明确只建立后续路线。
6. **“只要 shape 一样，MoE 就正确。”** shape 是第一道 contract；未来还要验证路由权重、Top-K、聚合数值和 token/专家映射。

## 深入教程

- [Week 6：SwiGLU Dense MLP](../tutorials/week06-swiglu-dense-mlp.md)：理解未来每个 Expert 最可能复用的 Dense MLP 基础。
- [Week 7：Dense Decoder](../tutorials/week07-dense-decoder.md)：复习稳定 residual 外壳和模块组合测试。
- [05：MLP](05-mlp.md)：回看当前 `self.mlp` 插槽的职责与 shape。
- [06：组装 Dense 模型](06-assemble-dense-model.md)：回看为什么 Decoder 内部替换不应改变整机接口。

## 回到整体

课程主线到这里形成两层稳定 contract：

```text
模型外部：input_ids [B,S] -> model -> logits [B,S,V]
层内插槽：mlp_norm [B,S,D] -> Dense MLP 或未来 MoE -> update [B,S,D]
```

因此从 Dense 走向 MoE 的第一原则不是立即编写 Router，而是先保护替换边界。你应留下四项证据：MoE 在整机中的唯一位置图、稳定/内部 shape ledger、一次真实 `DenseDecoderLayer` debug 调用，以及 residual contract 测试或错误输出宽度的受控推演。

你现在可以返回 [主线课程目录](README.md) 复盘完整路径：整机导览 -> 模型外壳 -> Decoder -> 张量流 -> Attention -> MLP -> Dense 组装 -> greedy 生成 -> MoE 替换边界。
