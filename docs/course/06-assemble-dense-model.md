# 06：组装完整 Dense 模型

前几章分别看过模型外壳、Decoder、张量变形、Attention 和 MLP。本章第三次经过同一台模型，不再把核心模块当黑盒，而是沿真实 `TinyDenseCausalLM.forward()` 完成一次端到端追踪。

## 为什么要重新组装

理解每个零件不等于理解整机。组装时最容易出现的错误往往不是某个公式完全写错，而是边界接错：

- positions 或 causal mask 没有覆盖当前序列长度；
- residual 加错基准张量；
- 多层没有真正依次执行；
- final norm 或 LM Head 被漏掉；
- shape 正确，但未来 token 泄漏到了前缀 logits。

因此本章同时检查数据流、模块独立性、有限数值和因果行为。

## 整机位置图

```text
input_ids [B,S]
  -> positions [B,S]
  -> causal attention_mask [1,1,S,S]
  -> embedding
hidden_states [B,S,D]
  -> DenseDecoderLayer 0
  -> DenseDecoderLayer 1
  -> ...
  -> DenseDecoderLayer L-1
hidden_states [B,S,D]
  -> final_norm [B,S,D]
  -> lm_head [B,S,V]
logits [B,S,V]
```

这是**一次模型 forward**。它为输入序列的每个位置都计算词表分数，但不会自己选择 token，也不会追加 token。生成循环留到下一章。

## Shape ledger

设 `L = num_hidden_layers`、`V = vocab_size`：

| 边界 | 真实符号 | shape / dtype |
| --- | --- | --- |
| 模型输入 | `input_ids` | `[B,S]`，`int32` 或 `int64` |
| 位置编号 | `positions` | `[B,S]`，`long` |
| 因果 mask | `attention_mask` | `[1,1,S,S]`，`bool` |
| 词向量 | `hidden_states = embedding(input_ids)` | `[B,S,D]`，浮点 |
| 第 `i` 层输入/输出 | `hidden_states` | `[B,S,D]` |
| 最终规范化 | `normalized` | `[B,S,D]` |
| 词表分数 | `logits` | `[B,S,V]` |

每层内部继续遵守：

```text
attention_norm   [B,S,D]
attention_update [B,S,D]
after_attention  [B,S,D]
mlp_norm         [B,S,D]
mlp_update       [B,S,D]
output           [B,S,D]
```

`D` 在所有 Decoder Layer 边界保持不变，使 `ModuleList` 中任意数量的层可以串联。内部 Attention 会短暂变为 `[B,H,S,Dh]`，MLP 会短暂变为 `[B,S,M]`，但两者在返回外壳前都恢复 `[B,S,D]`。

## 对照真实 forward 阅读

下面完整嵌入当前 `TinyDenseCausalLM.forward()`。同步测试会把这个代码块与真实符号经过 `textwrap.dedent` 后的文本逐字比较，源码变化时本章也必须同步更新。

<!-- source-sync: src/qwen3_moe/model.py::TinyDenseCausalLM.forward -->
```python
def forward(
    self, input_ids: Tensor, *, return_debug: bool = False
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be a rank-2 integer tensor")
    if input_ids.shape[0] <= 0 or input_ids.shape[1] <= 0:
        raise ValueError("batch and sequence dimensions must be positive")
    if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
        raise ValueError("input_ids contain a token outside the vocabulary")
    if input_ids.device != self.embedding.weight.device:
        raise ValueError("input_ids and model parameters must share a device")

    batch_size, sequence_length = input_ids.shape
    positions = torch.arange(
        sequence_length, dtype=torch.long, device=input_ids.device
    ).view(1, sequence_length).expand(batch_size, sequence_length)
    attention_mask = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=input_ids.device,
        ),
        diagonal=1,
    ).view(1, 1, sequence_length, sequence_length)

    hidden_states = self.embedding(input_ids)
    debug: dict[str, Tensor] = {}
    if return_debug:
        debug["embedding"] = hidden_states
    for index, layer in enumerate(self.layers):
        if return_debug:
            hidden_states, layer_debug = layer(
                hidden_states,
                positions,
                attention_mask,
                return_debug=True,
            )
            debug.update(
                {
                    f"layers.{index}.{name}": value
                    for name, value in layer_debug.items()
                }
            )
        else:
            hidden_states = layer(
                hidden_states, positions, attention_mask
            )

    normalized = self.final_norm(hidden_states)
    logits = self.lm_head(normalized)
    require_finite("logits", logits)
    if not return_debug:
        return logits
    debug["final_norm"] = normalized
    debug["logits"] = logits
    return logits, debug
```

### 1. 先验证输入 contract

函数先拒绝非 rank-2、非整数、空 batch/sequence、越界 token，以及与模型参数不在同一 device 的输入。这些检查发生在 Embedding 前，因此边界错误不会变成更难定位的索引或设备异常。

### 2. 构造 positions

从 `input_ids.shape` 取得 `batch_size, sequence_length` 后，`torch.arange(sequence_length)` 先产生 `0..S-1`，再经 `view` 和 `expand` 得到每个 batch 共用的位置编号 `[B,S]`。dtype 固定为 `long`，device 跟随输入。

### 3. 构造 causal mask

`torch.ones(S,S,dtype=bool)` 经 `torch.triu(..., diagonal=1)` 后，只把主对角线以上的未来位置标为 `True`，最后变成 `[1,1,S,S]`，供所有 Decoder Layers 广播使用。

### 4. 执行 Embedding

`self.embedding(input_ids)` 把离散 ID `[B,S]` 映射为浮点 `hidden_states [B,S,D]`。若开启 debug，这也是第一项轨迹 `debug["embedding"]`。

### 5. 依次执行所有 layers

循环把同一份 `positions` 和 `attention_mask` 传给每个独立的 `DenseDecoderLayer`。每轮都覆盖 `hidden_states`，所以第 `i+1` 层接收第 `i` 层的输出。debug 分支还会把层内键加上 `layers.{index}.` 前缀，例如 `layers.0.attention.probabilities`、`layers.0.mlp.product` 和 `layers.0.output`。

### 6. 执行 final norm

所有层结束后，`self.final_norm(hidden_states)` 仍保持 `[B,S,D]`，对送入词表投影前的最终 hidden states 做 RMSNorm。

### 7. 执行 LM Head 并检查有限值

`self.lm_head(normalized)` 把最后一维从 `D` 投影到 `V`，得到 `logits [B,S,V]`；随后 `require_finite("logits", logits)` 立即阻止 NaN 或 Inf 离开模型。`logits[b,s,v]` 是 batch `b`、位置 `s` 对词表项 `v` 的未归一化分数。

### 8. 返回普通结果或 debug 轨迹

默认只返回 `logits`。开启 `return_debug` 时，函数在逐层轨迹后补入 `final_norm` 和 `logits`，再返回 `(logits, debug)`。随机初始化模型的 argmax token 没有可靠语言语义，debug 的用途是检查真实执行数据流。

查看完整文件：[`src/qwen3_moe/model.py`](../../src/qwen3_moe/model.py)。相关组合模块见 [`src/qwen3_moe/decoder.py`](../../src/qwen3_moe/decoder.py)、[`src/qwen3_moe/attention.py`](../../src/qwen3_moe/attention.py)、[`src/qwen3_moe/mlp.py`](../../src/qwen3_moe/mlp.py) 和 [`src/qwen3_moe/norms.py`](../../src/qwen3_moe/norms.py)；一次 forward 的可运行入口见 [`examples/run_tiny_dense.py`](../../examples/run_tiny_dense.py)。

## 最小实验：打印整机 trace

```bash
uv run python - <<'PY'
import torch
from qwen3_moe import DenseConfig, TinyDenseCausalLM

torch.manual_seed(7)
config = DenseConfig(
    vocab_size=11,
    hidden_size=8,
    intermediate_size=12,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
    rope_theta=100.0,
)
model = TinyDenseCausalLM(config).eval()
input_ids = torch.tensor([[1, 2, 3, 4]])

with torch.inference_mode():
    logits, debug = model(input_ids, return_debug=True)

for name in (
    "embedding",
    "layers.0.attention_update",
    "layers.0.mlp_update",
    "layers.0.output",
    "layers.1.output",
    "final_norm",
    "logits",
):
    value = debug[name]
    print(f"{name:30s} shape={tuple(value.shape)} finite={bool(torch.isfinite(value).all())}")
print("normal return shape:", tuple(logits.shape))
PY
```

也可运行仓库示例：

```bash
uv run python examples/run_tiny_dense.py
```

受控实验：把 `input_ids` 中某个值改成 `config.vocab_size`（这里是 11）。模型会在 Embedding 前报告 token 超出词表，而不是产生难懂的索引错误。

## Shape 正确还不够：前缀因果性

比较两条只在最后一个 token 不同的序列：

```text
[1, 2, 3, 4]
[1, 2, 3, 9]
```

因果模型在前 3 个位置的 logits 必须一致，因为这些位置不能看到未来的第 4 个 token。最后位置的 logits 可以不同。这个测试能发现“shape 全部正确但 mask 方向反了或漏了”的错误。

## 对应测试

- [`tests/test_model.py`](../../tests/test_model.py) 的 `test_tiny_dense_model_runs_end_to_end_on_cpu()`：验证 `[B,S] -> [B,S,V]`、逐层 debug shape、CPU、有限数值和层实例独立。
- 同文件的 `test_causal_prefix_logits_do_not_depend_on_a_future_token()`：验证前缀因果性。
- 同文件的 `test_copy_is_identical_then_active_layer_change_reaches_logits()`：验证活动层的参数修改确实传到 logits。
- 同文件的 `test_tied_embeddings_share_the_same_parameter()`：验证可选的 Embedding/LM Head 权重绑定。
- 同文件的 `test_model_rejects_invalid_token_ids()`：验证 rank、整数类型、非空 batch/sequence 和词表范围边界。
- [`tests/test_decoder.py`](../../tests/test_decoder.py)：验证每层两次 residual 使用正确基准。

运行完整 Dense 路径测试：

```bash
uv run pytest tests/test_model.py tests/test_decoder.py tests/test_attention.py tests/test_mlp.py tests/test_norms.py tests/test_rope.py
```

## 常见误解

1. **“`logits [B,S,V]` 就是生成结果。”** logits 是一次 forward 的分数；还没有选 token、追加或重复执行。
2. **“只需要最后位置 logits，前面位置可以随便算。”** 无 KV Cache 的最小实现每轮重算整个序列；而前缀位置还参与后续位置的上下文计算。
3. **“多个 Decoder Layers 可以复用同一个对象。”** 层结构相同，但参数应是独立实例；测试明确检查这一点。
4. **“输出 shape 对就说明模型正确。”** 错误的 residual 基准或 causal mask 都可能保留 shape，必须做数值关系和因果行为测试。
5. **“随机初始化模型的 logits 可以解释成有意义文本。”** 没有训练权重和 Tokenizer 映射时，只能把它当数据流实验。

## 深入教程

- [Week 7：Dense Decoder](../tutorials/week07-dense-decoder.md)：完整组装、pre-norm residual、测试策略和累计实现。
- [Week 3：Token、logits 与 Causal LM](../tutorials/week03-token-logits-causal-lm.md)：Embedding、LM Head、词表维和 next-token 分数。
- [Week 1：张量、shape 与内存](../tutorials/week01-tensors-shapes-memory.md)：如果 trace 中的 rank、dtype、device 仍不稳固，回到这里复习。

## 回到整体

现在你已经能沿真实源码完整解释一次 forward：

```text
[B,S] token IDs
-> [B,S,D] hidden states
-> L 个保持 [B,S,D] 的 Decoder Layers
-> [B,S,V] logits
```

你应留下四项证据：完整位置图、跨层 shape ledger、一次 `TinyDenseCausalLM(..., return_debug=True)` 调用，以及一次端到端或前缀因果性测试。

下一章把这次 forward 放入 [最小自回归生成循环](07-autoregressive-generation.md)，完成 `[B,S] -> [B,S,V] -> [B,1] -> append` 的控制流闭环。
