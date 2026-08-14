# 05：MLP，在每个 token 内变换特征

Attention 让一个位置读取其他位置，但读取到的信息还需要在 hidden 维度上重新组合。Decoder Layer 的第二个核心模块 MLP 就承担这项工作。

## 为什么需要 MLP

把序列想成一个 `[B,S,D]` 表格：

- Attention 沿 `S` 方向混合，让不同 token 位置交换信息。
- MLP 沿最后的 `D` 方向变换，每个 token 独立使用同一组参数。

如果删除 Attention，位置之间不能交流；如果删除 MLP，模型仍能搬运上下文，却少了强大的逐位置非线性特征变换。两者分工不同，不能互相替代。

## 它在整机中的位置

```text
input_ids
  -> Embedding
  -> Decoder Layer
       -> RMSNorm -> Attention -> residual add
       -> RMSNorm -> SwiGLU MLP -> residual add   <- 本章
  -> Final Norm
  -> LM Head
  -> logits
```

对应 Decoder 公式的第二行：

```text
h = x + Attention(RMSNorm(x))
y = h + MLP(RMSNorm(h))
```

MLP 的输入是 `mlp_norm [B,S,D]`，输出是 `mlp_update [B,S,D]`。shape 相同是 residual add 能成立的必要条件。

## SwiGLU 的三条投影路径

仓库实现的 `SwiGLU` 不使用 bias：

```text
gate    = gate_proj(x)
up      = up_proj(x)
product = SiLU(gate) * up
update  = down_proj(product)
```

记 `M = intermediate_size`：

| 阶段 | 真实符号 | shape |
| --- | --- | --- |
| 输入 | `hidden_states` | `[B,S,D]` |
| 门控投影 | `gate` | `[B,S,M]` |
| 内容投影 | `up` | `[B,S,M]` |
| 激活并逐元素相乘 | `product` | `[B,S,M]` |
| 降回 hidden size | `update` | `[B,S,D]` |

`gate_proj` 和 `up_proj` 都从 `D` 升到 `M`，但参数不同。`SiLU(gate)` 像一个可学习的、连续的门，逐元素调节 `up` 路径。`down_proj` 再把宽度降回 `D`。

这里没有对 `S` 维做矩阵乘法。对于固定权重，改变某个 token 位置的输入不会直接改变其他位置的 MLP 输出；跨位置影响已经由前面的 Attention 提供。

## 真实 `forward()` 与执行顺序

下面是模型实际执行的完整 `SwiGLU.forward()`。代码很短，但输入检查、五步计算、有限性检查和 debug 返回都属于函数真实 contract。

<!-- source-sync: src/qwen3_moe/mlp.py::SwiGLU.forward -->
```python
def forward(
    self, hidden_states: Tensor, *, return_debug: bool = False
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    require_floating("SwiGLU input", hidden_states)
    if hidden_states.shape[-1] != self.hidden_size:
        raise ValueError(
            f"SwiGLU expected hidden size {self.hidden_size}, "
            f"got {hidden_states.shape[-1]}"
        )

    gate = self.gate_proj(hidden_states)
    up = self.up_proj(hidden_states)
    product = F.silu(gate) * up
    update = self.down_proj(product)
    require_finite("SwiGLU product", product)
    require_finite("SwiGLU update", update)
    if not return_debug:
        return update
    return update, {"gate": gate, "up": up, "product": product}
```

### 1. Gate

函数先要求输入是浮点 Tensor，并检查最后一维确实是 `D = hidden_size`。`gate_proj` 随后对每个 token 的 `[D]` 向量使用同一组权重，产生 `gate [B,S,M]`。它升高特征宽度，但不混合 batch 轴或 sequence 轴。

### 2. Up

`up_proj` 从同一个 `hidden_states [B,S,D]` 独立产生 `up [B,S,M]`。它与 gate shape 相同，却使用另一组参数；这条路径提供将被门控的内容，而不是重复 gate 的计算。

### 3. SiLU

`F.silu(gate)` 对 gate 的每个元素独立应用 `x * sigmoid(x)`，shape 仍为 `[B,S,M]`。SiLU 引入非线性，使门控强度不再等价于把两个线性层直接合并成一个线性变换。

### 4. Product

源码在同一行计算 `product = F.silu(gate) * up`。这里的 `*` 是逐元素乘法，两侧 shape 都是 `[B,S,M]`，所以 product 也保持 `[B,S,M]`；它没有矩阵收缩，也不会跨 token 读取信息。紧随其后的有限性检查会在非有限值进入 residual 主干前失败。

### 5. Down

`down_proj` 把 product 的最后一维从 `M` 投影回 `D`，得到 `update [B,S,D]`。只有回到 residual 主干宽度后，它才能与 `h [B,S,D]` 相加。默认只返回 update；`return_debug=True` 时附带 gate、up、product，计算结果本身不变。

## 完整文件链接

- [`mlp.py`](../../src/qwen3_moe/mlp.py)：完整 `SwiGLU` 模块及三条投影层定义。
- [`decoder.py`](../../src/qwen3_moe/decoder.py)：`post_attention_norm`、MLP 调用和第二次 residual add。
- [`config.py`](../../src/qwen3_moe/config.py)：`hidden_size` 与 `intermediate_size`。
- [`debug.py`](../../src/qwen3_moe/debug.py)：浮点、有限性和 residual contract 检查。

请特别观察 `return_debug=True` 时返回的 `gate`、`up` 和 `product`。`update` 作为正常返回值已经在外层拿到，所以没有重复放进 debug 字典。

## 最小实验

这个实验运行真实 `SwiGLU`，并验证不同序列位置是逐位置处理的：只改变位置 1，位置 0 和 2 的输出保持不变。

```bash
uv run python - <<'PY'
import torch
from qwen3_moe import DenseConfig, SwiGLU

torch.manual_seed(7)
config = DenseConfig(
    vocab_size=11,
    hidden_size=8,
    intermediate_size=12,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
)
mlp = SwiGLU(config).eval()
first = torch.randn(1, 3, 8)
second = first.clone()
second[:, 1] += 10

with torch.inference_mode():
    first_update, debug = mlp(first, return_debug=True)
    second_update = mlp(second)

print("input:", first.shape)
print("gate/up/product:", debug["gate"].shape, debug["up"].shape, debug["product"].shape)
print("update:", first_update.shape)
print("position 0 unchanged:", torch.allclose(first_update[:, 0], second_update[:, 0]))
print("position 1 changed:", not torch.allclose(first_update[:, 1], second_update[:, 1]))
print("position 2 unchanged:", torch.allclose(first_update[:, 2], second_update[:, 2]))
PY
```

受控错误：把输入改成 `torch.randn(1, 3, 7)`。`SwiGLU.forward()` 会明确报告期望 hidden size 为 8，而不是让错误延迟到某个难以定位的矩阵乘法。

## 对应测试

- [`tests/test_mlp.py`](../../tests/test_mlp.py) 的 `test_swiglu_matches_explicit_pytorch_primitives()`：用 `F.linear`、`F.silu` 和逐元素乘法独立重算，验证真实实现及 `[B,S,M]` 中间 shape。
- [`tests/test_decoder.py`](../../tests/test_decoder.py) 的 `test_dense_decoder_uses_two_correct_residual_bases()`：验证 `mlp_update` 加到 `after_attention`，而不是错误地加回原始 `hidden_states`。
- [`tests/test_model.py`](../../tests/test_model.py) 的 `test_copy_is_identical_then_active_layer_change_reaches_logits()`：修改某层 `mlp.down_proj` 后 logits 确实变化，说明 MLP 真正处于端到端计算路径上。

运行：

```bash
uv run pytest tests/test_mlp.py tests/test_decoder.py tests/test_model.py
```

## 常见误解

1. **“MLP 会混合 token。”** 线性层作用于最后一维，同一套参数分别处理每个 `[D]` 向量；它不直接跨 `S` 混合。
2. **“`gate_proj` 和 `up_proj` 是重复计算。”** 两者 shape 相同但权重和职责不同，一个形成门控，一个提供被门控的内容。
3. **“SwiGLU 只是一层 Linear。”** 它有三次投影、一个 SiLU 激活和一次逐元素乘法。
4. **“`intermediate_size` 必须等于 `hidden_size`。”** 中间宽度 `M` 可以不同；只有最终 `down_proj` 必须回到 `D`，才能参与 residual add。
5. **“MoE 会替换整个 Decoder Layer。”** 后续替换边界是当前 `self.mlp` 插槽；Attention、两次 Norm 和 residual 外壳仍可保持稳定。

## 深入教程

- [Week 6：SwiGLU Dense MLP](../tutorials/week06-swiglu-dense-mlp.md)：公式、参数量、手算、逐位置性质和更多受控错误。
- [Week 2：矩阵乘法与 `nn.Module`](../tutorials/week02-matmul-nn-module.md)：线性层、参数注册和模块调用基础。
- [Week 7：Dense Decoder](../tutorials/week07-dense-decoder.md)：MLP 如何进入完整 pre-norm residual layer。

## 回到整体

至此，一个 Dense Decoder Layer 的两个更新分支都已展开：

```text
上下文混合：h = x + Attention(RMSNorm(x))
逐位置变换：y = h + SwiGLU(RMSNorm(h))
```

你应留下四项证据：MLP 在 Decoder 中的位置图、`D -> M -> D` shape ledger、一次真实 `SwiGLU(..., return_debug=True)` 调用，以及显式 PyTorch 原语对齐测试或错误宽度实验。

下一章将这些模块重新组装成 [完整 Dense 模型](06-assemble-dense-model.md)，从 `input_ids` 一直追踪到 logits。
