# 00：LLM 推理整机导览

本章先把整台机器跑一遍。你不需要先理解 Attention、RoPE 或 SwiGLU 的公式，只需要知道数据经过哪些边界，以及哪些工作不属于模型的一次 `forward()`。

## 为什么先看整机

如果一开始只学 `matmul` 或 softmax，很容易会算局部，却不知道它为什么出现在这里。先建立整机地图，后续每学一个模块都能回答两件事：它解决了哪一段问题；删除它后整机缺少什么能力。

## 它位于整机的什么位置

完整的 decoder-only LLM 生成控制流可以先画成：

```text
文本
-> Tokenizer
-> input_ids [B,S]
-> TinyDenseCausalLM.forward()
-> logits [B,S,V]
-> 取 logits[:, -1, :] [B,V]
-> greedy 或 sampling 选 next_token [B,1]
-> 追加到 input_ids
-> 重复模型调用，直到停止
```

这里有四个不同边界：

| 边界 | 输入与输出 | 当前仓库状态 |
| --- | --- | --- |
| Tokenizer | 文本 `<->` token IDs | 尚未实现，实验手工写整数 ID |
| 模型单次 forward | `[B,S] -> [B,S,V]` | 已实现 `TinyDenseCausalLM` |
| next-token 选择 | `[B,V] -> [B,1]` | 可用 `argmax` 演示，但不属于模型 forward |
| 自回归生成循环 | 追加 token 并重复 | 后续主线单独学习 |

**一次 forward 不是完整生成。** forward 为输入中的每个位置计算词表分数；生成还要选择、追加、停止，并可能管理 KV Cache。

## 模型内部的第一张地图

把 `TinyDenseCausalLM.forward()` 暂时看成白盒外壳：

```text
input_ids [B,S]
-> Embedding [B,S,D]
-> DenseDecoderLayer 0 [B,S,D]
-> ...
-> DenseDecoderLayer L-1 [B,S,D]
-> Final RMSNorm [B,S,D]
-> LM Head [B,S,V]
-> logits
```

Decoder Layer 内部再分为两项职责：Attention 让当前位置读取允许看到的上下文；MLP 在每个 token 位置变换特征。现在只记职责，不展开公式。

## Shape ledger

本仓库示例使用 `B=2, S=4, V=11, D=8, L=2`：

| 边界 | shape | 轴含义 |
| --- | --- | --- |
| `input_ids` | `[2,4]` | 2 条序列，每条 4 个 token ID |
| `embedding` | `[2,4,8]` | 每个 token ID 变成宽度 8 的向量 |
| 每层 `output` | `[2,4,8]` | Decoder 更新 values，但保持 residual 主干 shape |
| `final_norm` | `[2,4,8]` | 调整数值尺度，不改 shape |
| `logits` | `[2,4,11]` | 每个位置对 11 个词表候选给分 |
| `last_logits` | `[2,11]` | 每条序列最后位置的候选分数 |
| `next_token` | `[2,1]` | 每条序列选出的一个新 ID |

## 先读真实代码：一次 forward 怎样跑完整机

先带着一个任务读下面的函数：从 `input_ids [B,S]` 出发，找出输入在哪里被拒绝、Attention 所需的辅助张量在哪里建立、Decoder Layers 何时依次执行，以及 logits 最终在哪里产生。暂时不用进入 Attention 和 MLP 内部。

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

函数先守住入口契约：`input_ids` 必须是 rank-2 整数 Tensor，`B/S` 都为正，ID 落在 `[0,V)`，并且输入与模型参数位于同一设备。任何一项不满足都会在查 Embedding 前明确失败，因此后续模块可以按合法的 `[B,S]` 输入推理。

接着从 shape 中取出 `B/S`，建立 `positions [B,S]` 和上三角因果 `attention_mask [1,1,S,S]`。Embedding 把整数 ID 查表成 `hidden_states [B,S,D]`；循环随后把同一份位置和 mask 交给每个 Decoder Layer，每层接收并返回 `[B,S,D]`，所以 residual 主干能够连续穿过整个 layer stack。

所有层执行完后，`final_norm` 仍保持 `[B,S,D]`，`lm_head` 才把最后一维从 `D` 投影到词表大小 `V`，得到 `logits [B,S,V]`。`require_finite` 在返回前检查整机结果没有 NaN 或 Inf；这里没有选 next token，也没有追加输入，因此它仍只是一次 forward。

`return_debug=False` 走正常推理路径，只返回 logits；开启后，各层会返回自己的中间量，外壳再用 `layers.{index}.` 前缀汇总，并补上 Embedding、Final Norm 和 logits。这样 debug 路径保留真实模块调用和 shape，而不是另写一套容易与模型漂移的演示计算。

完整文件：[`src/qwen3_moe/model.py`](../../src/qwen3_moe/model.py)、[`src/qwen3_moe/decoder.py`](../../src/qwen3_moe/decoder.py)、[`src/qwen3_moe/config.py`](../../src/qwen3_moe/config.py)、[`examples/run_tiny_dense.py`](../../examples/run_tiny_dense.py)。

## 最小实验：先运行，再打开外壳

从仓库根目录运行：

```bash
uv run python examples/run_tiny_dense.py
```

预期只关注三件事：`input_ids` 是 `(2, 4)` 的整数 Tensor；`logits` 是 `(2, 4, 11)` 的浮点 Tensor；所有 logits 都是有限值。

再运行一次真实 debug 路径，观察组件树和调用边界：

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
input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

with torch.inference_mode():
    logits, debug = model(input_ids, return_debug=True)

print(model)
for name in ("embedding", "layers.0.output", "layers.1.output", "final_norm", "logits"):
    value = debug[name]
    print(name, tuple(value.shape), value.dtype, value.device)

last_logits = logits[:, -1, :]
next_token = last_logits.argmax(dim=-1, keepdim=True)
print("last/next:", tuple(last_logits.shape), tuple(next_token.shape), next_token.tolist())
PY
```

最后三行演示了**一次选择**，仍没有追加和重复，所以仍不是完整生成。随机种子使本机重复运行便于对照，但输出 token 没有语言语义。

## 对应测试

- [`tests/test_model.py::test_tiny_dense_model_runs_end_to_end_on_cpu`](../../tests/test_model.py)：验证整机输出 shape、每层 debug shape、CPU 设备、有限值和层实例独立。
- [`tests/test_model.py::test_causal_prefix_logits_do_not_depend_on_a_future_token`](../../tests/test_model.py)：只改未来 token，较早位置 logits 不变，验证因果边界。
- [`tests/test_model.py::test_model_rejects_invalid_token_ids`](../../tests/test_model.py)：验证浮点 ID、越界 ID、空 batch 等输入会明确失败。

只运行本章对应测试：

```bash
uv run pytest tests/test_model.py
```

## 常见误解与受控错误

- **“logits 就是生成文本。”** logits 只是候选分数；还没有选择 token，更没有 Tokenizer 解码。
- **“模型有 `CausalLM` 名字，所以随机权重也会说话。”** 架构接口正确不等于权重学过语言。当前输出只用于观察数据流。
- **“每个位置都有 logits，所以模型一次生成了 S 个新 token。”** forward 返回输入各位置的预测分数；自回归 decode 通常只取最后位置生成一个新 token。
- **“调用两层就是调用模型两次。”** 两层发生在同一次 forward 内；生成循环才会再次调用模型。

观察一个入口受控错误：

```bash
uv run python - <<'PY'
import torch
from qwen3_moe import DenseConfig, TinyDenseCausalLM

config = DenseConfig(11, 8, 12, 1, 2, 1, 4)
model = TinyDenseCausalLM(config)
try:
    model(torch.tensor([[0.0, 1.0]]))
except ValueError as error:
    print(error)
PY
```

错误应指出 `input_ids` 必须是 rank-2 整数 Tensor。不要把浮点 ID 强转为整数来掩盖上游输入错误。

## 深入教程

- [第三周：Token、Logits 与因果语言建模](../tutorials/week03-token-logits-causal-lm.md)：深入 Tokenizer 边界、Embedding、LM Head、softmax 和一次 next-token 选择。
- [第七周：完整 Dense Decoder](../tutorials/week07-dense-decoder.md)：深入完整模型组装、trace、因果前缀实验和首次偏差定位。

## 回到完整推理流程

现在你已经走过：

```text
文本 -> [Tokenizer 尚未实现] -> input_ids
     -> [真实 TinyDenseCausalLM 单次 forward 已运行]
     -> logits -> [一次 argmax 已观察]
     -> [追加与重复尚未学习]
```

请留下本章证据：整机位置图、上面的 shape ledger、一次 `TinyDenseCausalLM(..., return_debug=True)` 调用，以及 `tests/test_model.py` 的运行结果。下一章进入 [01：模型外壳](01-model-shell.md)，只拆 Embedding、Decoder 黑盒、Final Norm 和 LM Head。
