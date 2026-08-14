# 02：Decoder Layer

模型外壳已经把 token IDs 变成 hidden states，并最终投影成 logits。现在打开中间反复堆叠的 `DenseDecoderLayer`，理解它为什么必须同时包含上下文混合、逐位置变换、归一化和 residual connection。

## 为什么需要 Decoder Layer

仅有 Embedding 和 LM Head 时，同一个 token ID 总得到同一行 Embedding，左侧上下文不会改变它。Decoder Layer 填补这个缺口：

- Attention 负责跨 token 位置读取允许看到的上下文。
- MLP 负责在每个 token 位置内部变换特征。
- RMSNorm 在子层计算前控制数值尺度。
- Residual connection 保留已有表示，并把子层结果作为更新量加回主干。

只保留 Attention 会缺少强大的逐位置特征变换；只保留 MLP 则各位置不能交换上下文。删除 identity skip 后，子层不再是“对现有 hidden 的更新”，深层组合也失去本课程要维护的稳定 contract。

## 它位于整机的什么位置

```text
input_ids [B,S]
-> Embedding [B,S,D]
-> DenseDecoderLayer 0
-> DenseDecoderLayer 1
-> ...
-> DenseDecoderLayer L-1
-> Final Norm
-> LM Head
-> logits [B,S,V]
```

单层采用 pre-norm 和两条 residual：

```text
x
-> attention_norm = input_norm(x)
-> attention_update = self_attn(attention_norm)
-> h = x + attention_update
-> mlp_norm = post_attention_norm(h)
-> mlp_update = mlp(mlp_norm)
-> y = h + mlp_update
```

压缩写法是：

```text
h = x + Attention(RMSNorm(x))
y = h + MLP(RMSNorm(h))
```

第二次相加的 residual 基准是 `h`，不是最初的 `x`。

## Shape ledger

以真实示例 `B=2, S=4, D=8, I=12` 为例：

| 边界 | shape | 为什么 |
| --- | --- | --- |
| `hidden_states` / `x` | `[2,4,8]` | 单层输入 contract |
| `attention_norm` | `[2,4,8]` | RMSNorm 不改 shape |
| `attention_update` | `[2,4,8]` | 必须与 `x` 完全一致才能 residual add |
| `after_attention` / `h` | `[2,4,8]` | 逐元素相加 |
| `mlp_norm` | `[2,4,8]` | 第二个 Norm 作用于 `h` |
| MLP `gate/up/product` | `[2,4,12]` | 临时进入中间宽度 `I` |
| `mlp_update` | `[2,4,8]` | down projection 回到 `D` |
| `output` / `y` | `[2,4,8]` | 可直接送给下一层 |

Residual contract 不允许“能广播就算”。例如 `[B,S,1]` 虽可能广播加到 `[B,S,D]`，但它不是对每个 hidden 通道一一对应的更新。真实实现用 `require_same_tensor_contract` 同时检查 shape、dtype 和 device。

## 先读真实代码：一层怎样完成两次更新

先带着一个任务读下面的函数：沿执行顺序找出两次 pre-norm、Attention 和 MLP 的调用边界、两条 residual 各自使用的基准，以及 contract 与 debug 检查插在什么位置。

<!-- source-sync: src/qwen3_moe/decoder.py::DenseDecoderLayer.forward -->
```python
def forward(
    self,
    hidden_states: Tensor,
    positions: Tensor,
    attention_mask: Tensor | None = None,
    *,
    return_debug: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    attention_norm = self.input_norm(hidden_states)
    if return_debug:
        attention_update, attention_debug = self.self_attn(
            attention_norm,
            positions,
            attention_mask,
            return_debug=True,
        )
    else:
        attention_update = self.self_attn(
            attention_norm, positions, attention_mask
        )
    require_same_tensor_contract(
        "attention residual", hidden_states, attention_update
    )
    after_attention = hidden_states + attention_update

    mlp_norm = self.post_attention_norm(after_attention)
    if return_debug:
        mlp_update, mlp_debug = self.mlp(mlp_norm, return_debug=True)
    else:
        mlp_update = self.mlp(mlp_norm)
    require_same_tensor_contract("MLP residual", after_attention, mlp_update)
    output = after_attention + mlp_update
    require_finite("decoder output", output)

    if not return_debug:
        return output
    debug = {
        "attention_norm": attention_norm,
        "attention_update": attention_update,
        "after_attention": after_attention,
        "mlp_norm": mlp_norm,
        "mlp_update": mlp_update,
        "output": output,
    }
    debug.update(
        {f"attention.{name}": value for name, value in attention_debug.items()}
    )
    debug.update({f"mlp.{name}": value for name, value in mlp_debug.items()})
    return output, debug
```

这一层没有重复模型入口的 token ID 检查，因为它接收的是上游已经建立好的 hidden states、positions 和 mask。第一步先对 `hidden_states [B,S,D]` 做 `input_norm`，再调用 Attention；只有 Attention 收到 `positions [B,S]` 和 `attention_mask`，因为跨位置读取与位置编码属于它的职责。

Attention 必须返回 `attention_update [B,S,D]`。相加前，`require_same_tensor_contract` 明确检查它与 `hidden_states` 的 shape、dtype 和 device 完全一致，防止广播掩盖错误；第一条 residual 随后得到 `after_attention = hidden_states + attention_update`，shape 仍是 `[B,S,D]`。

第二个 pre-norm 作用于已经包含 Attention 更新的 `after_attention`，然后 MLP 只接收 `mlp_norm`，不接收 positions 或 mask。MLP 内部可暂时扩展到 `[B,S,I]`，但返回层边界前必须降回 `[B,S,D]`；第二次 contract 检查通过后，`output = after_attention + mlp_update`，所以这条 residual 的基准是 `after_attention`，不是原始输入。

`require_finite` 在层输出处拦截 NaN 或 Inf。正常路径只返回 output；debug 路径沿用相同的 Attention、MLP 和 residual 计算，同时记录六个层级张量，并给子模块明细加上 `attention.` 与 `mlp.` 前缀，便于从层边界继续向内定位首次偏差。

完整文件：[`src/qwen3_moe/decoder.py`](../../src/qwen3_moe/decoder.py)、[`src/qwen3_moe/attention.py`](../../src/qwen3_moe/attention.py)、[`src/qwen3_moe/mlp.py`](../../src/qwen3_moe/mlp.py)、[`src/qwen3_moe/debug.py`](../../src/qwen3_moe/debug.py)。

## 最小实验：验证两条 residual 的基准

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
x = torch.randn(2, 4, 8)
positions = torch.arange(4).view(1, 4).expand(2, 4)

with torch.inference_mode():
    output, debug = layer(x, positions, return_debug=True)

for name in (
    "attention_norm",
    "attention_update",
    "after_attention",
    "mlp_norm",
    "mlp_update",
    "output",
):
    print(name, tuple(debug[name].shape))

torch.testing.assert_close(
    debug["after_attention"], x + debug["attention_update"], atol=1e-6, rtol=1e-6
)
torch.testing.assert_close(
    output,
    debug["after_attention"] + debug["mlp_update"],
    atol=1e-6,
    rtol=1e-6,
)
print("two residual bases are correct")
PY
```

实验故意不手算 Attention 和 SwiGLU 的内部 values，只检查组合 contract。局部公式将在后续章节展开。

## 对应测试

- [`tests/test_decoder.py::test_dense_decoder_uses_two_correct_residual_bases`](../../tests/test_decoder.py)：直接验证 `h=x+attention_update` 和 `y=h+mlp_update`，并检查两个 Norm 的 shape 与输出有限性。
- [`tests/test_model.py::test_tiny_dense_model_runs_end_to_end_on_cpu`](../../tests/test_model.py)：验证多个 `DenseDecoderLayer` 实例独立，且每层输出保持 `[B,S,D]`。
- [`tests/test_model.py::test_copy_is_identical_then_active_layer_change_reaches_logits`](../../tests/test_model.py)：修改第二层 MLP 的活跃权重后，最终 logits 必须改变，说明层不是只有 shape、没有数值作用。
- [`tests/test_attention.py::test_causal_gqa_masks_future_tokens_and_maps_head_groups`](../../tests/test_attention.py)：验证 Attention update 的因果性和 GQA head 映射。

```bash
uv run pytest tests/test_decoder.py tests/test_model.py tests/test_attention.py
```

## 常见误解与受控错误

- **“Residual 就是把原始 `x` 加两次。”** 第二条 identity path 从 `h` 开始，因为 `h` 已经包含 Attention 更新。
- **“Norm 在 residual 相加之后，所以这是 post-norm。”** 本实现先 Norm，再把子层 update 加回未归一化的 residual 基准，因此是 pre-norm。
- **“单层输入输出 shape 相同，说明层什么也没做。”** residual 架构通常保持 shape，但改变 values；测试通过修改活跃层权重观察 logits 变化。
- **“Attention 和 MLP 都是矩阵乘法，所以职责相同。”** Attention 跨位置混合；MLP 对每个位置独立应用同一组变换。
- **“能广播相加就满足 residual contract。”** 真实实现要求 shape、dtype、device 全部相同。

直接观察 contract 守卫：

```bash
uv run python - <<'PY'
import torch
from qwen3_moe.debug import require_same_tensor_contract

residual = torch.zeros(2, 4, 8)
bad_update = torch.zeros(2, 4, 1)
try:
    require_same_tensor_contract("demo residual", residual, bad_update)
except ValueError as error:
    print(error)
PY
```

这里 PyTorch 普通加法可能广播成功，但 Decoder contract 主动拒绝，防止静默改变更新语义。

## 深入教程

- [第七周：完整 Dense Decoder](../tutorials/week07-dense-decoder.md)：pre-norm、分支开关、完整层、stack、trace 和首次偏差。
- [第四周：Self-Attention 与 GQA](../tutorials/week04-self-attention-gqa.md)：深入“跨 token 混合”如何实现。
- [第六周：SwiGLU Dense MLP](../tutorials/week06-swiglu-dense-mlp.md)：深入逐位置特征变换和三条投影。
- [第五周：RMSNorm、RoPE 与 QK Norm](../tutorials/week05-rmsnorm-rope-qk-norm.md)：深入两类 Norm 与位置旋转。

## 回到完整推理流程

中间黑盒现在打开了一层：

```text
input_ids -> Embedding
-> [L 次：Norm -> Attention -> residual -> Norm -> MLP -> residual]
-> Final Norm -> LM Head -> logits
```

请留下本章证据：单层位置图、八个边界的 shape ledger、一次 `DenseDecoderLayer(..., return_debug=True)` 调用，以及 Decoder 测试结果。下一章进入 [03：真实模型中的张量流](03-tensor-flow.md)，学习读懂这些源码操作所需的最小 PyTorch 语言。
