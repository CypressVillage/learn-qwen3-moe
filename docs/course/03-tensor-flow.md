# 03：真实模型中的张量流

这一章不按 PyTorch API 字母表学习，而是只解释真实模型中已经出现的 Tensor 操作：每个操作都必须回答“它服务于哪个模型边界”。

## 为什么需要张量流

模型源码中的变量不是普通嵌套列表。Tensor 同时带有 rank、shape、dtype、device 和存储布局。很多错误并非公式错，而是轴顺序、收缩维、广播方向、整数/浮点类型或 CPU/GPU 位置不符合模块 contract。

学习目标不是背 `view`、`transpose`、`matmul`，而是看到一行源码时能先预测输入输出 shape，并解释为什么必须这样变换。

## 它位于整机的什么位置

```text
input_ids [B,S]                    rank 2, integer
-> Embedding
hidden_states [B,S,D]              rank 3, floating
-> Attention head split/transpose
Q [B,Hq,S,Dh], K/V [B,Hkv,S,Dh]   rank 4, floating
-> scores / probabilities
[B,Hq,S,S]                         rank 4, floating
-> merge heads
attention_update [B,S,D]           rank 3, floating
-> residual + MLP
hidden_states [B,S,D]
-> LM Head
logits [B,S,V]                     rank 3, floating
```

## Shape ledger：真实操作逐行对应

设 `B=2, S=4, D=8, Hq=2, Hkv=1, Dh=4, I=12, V=11`。

| 真实操作 | 输入 | 输出 | 服务的边界 |
| --- | --- | --- | --- |
| `embedding(input_ids)` | `[2,4]` | `[2,4,8]` | ID 查表进入 hidden 空间 |
| `q_proj(hidden)` | `[2,4,8]` | `[2,4,8]` | 为 2 个 query heads 生成特征 |
| `.view(B,S,Hq,Dh)` | `[2,4,8]` | `[2,4,2,4]` | 把合并宽度拆成 head 轴与 head 宽度 |
| `.transpose(1,2)` | `[2,4,2,4]` | `[2,2,4,4]` | 把 head 放到 matmul 需要的位置 |
| `repeat_interleave(G, dim=1)` | K/V `[2,1,4,4]` | `[2,2,4,4]` | 让每个 KV head 服务一组 query heads |
| `query @ key.transpose(-1,-2)` | `[2,2,4,4] @ [2,2,4,4]` | scores `[2,2,4,4]` | `Dh` 收缩，得到每个 query/key 位置对 |
| `mask.expand_as(scores)` | `[1,1,4,4]` | 逻辑 `[2,2,4,4]` | 同一因果规则广播到 batch 与 heads |
| `probabilities @ value` | `[2,2,4,4] @ [2,2,4,4]` | context `[2,2,4,4]` | 对允许读取的 value 加权汇总 |
| `.transpose(1,2).contiguous().view(B,S,D)` | `[2,2,4,4]` | `[2,4,8]` | 合并 heads，回到 residual contract |
| `gate_proj/up_proj` | `[2,4,8]` | `[2,4,12]` | 进入 MLP 中间宽度 `I` |
| `down_proj` | `[2,4,12]` | `[2,4,8]` | 回到 residual 主干宽度 `D` |
| `lm_head` | `[2,4,8]` | `[2,4,11]` | 从 hidden 空间进入词表空间 |

## 读懂六类关键概念

### 1. Rank 与 shape

Rank 是轴的数量，PyTorch 中看 `tensor.ndim`；shape 是每个轴的长度。`[B,S,D]` 是 rank 3，不是因为 `D` 的值可能是 3，而是因为它有三个轴。

轴名只存在于人的解释中。PyTorch 看到的是长度，因此 `[B,S,D]=[2,4,8]` 和某个完全不同语义的 `[H,M,N]=[2,4,8]` 在 shape 上相同，却不能随意互换。

### 2. `view` / `reshape`：重组轴

`q_proj` 输出最后一维 `Hq*Dh`，源码用 `view(B,S,Hq,Dh)` 把它拆开。元素总数必须保持：`B*S*(Hq*Dh) == B*S*Hq*Dh`。

`view` 还要求当前 size/stride 与目标布局兼容；`reshape` 可以在需要时复制。真实 attention 在转置后先调用 `.contiguous()`，再 `view(B,S,D)`，明确请求适合合并轴的连续布局。

### 3. `transpose`：交换轴

投影后 query 是 `[B,S,Hq,Dh]`。Attention 分数需要每个 head 独立做 `[S,Dh] @ [Dh,S]`，因此先交换序列轴和 head 轴，得到 `[B,Hq,S,Dh]`。

`transpose` 通常改变逻辑视图和 stride，不会自动重排全部底层数据。这就是合并 heads 前出现 `contiguous()` 的原因。

### 4. Broadcasting：复用长度为 1 的轴

因果 mask 是 `[1,1,S,S]`，scores 是 `[B,Hq,S,S]`。前两个长度 1 的轴可分别广播到 `B` 与 `Hq`。同一份“未来位置不可见”规则因此用于所有 batch 和 heads。

广播只按 shape 从右向左判断，不认识轴名。shape 能广播不保证语义正确，所以真实代码还检查 bool dtype、device，并拒绝完全屏蔽某一 query 行的 mask。

### 5. `matmul`：找到收缩维

在 `query @ key.T` 中，最后的 `Dh` 相等并被收缩；保留 batch/head/query-position，并产生 key-position 轴：

```text
[B,Hq,S,Dh] @ [B,Hq,Dh,S] -> [B,Hq,S,S]
```

在 LM Head 中：

```text
[B,S,D] @ [D,V] -> [B,S,V]
```

先找收缩维，再写输出；不要把逐元素 `*` 当成矩阵乘法。

### 6. dtype 与 device

`input_ids` 和 `positions` 是整数索引；hidden states、权重和 logits 是浮点数；`attention_mask` 是 bool。普通数值运算的 Tensor 还必须位于兼容 device。真实 `TinyDenseCausalLM.forward()`、`RMSNorm` 和 Attention 都有明确守卫，错误会在靠近根因的边界失败。

## 在真实 RoPE 中合起来读这些概念

下面不是简化伪代码，而是模型实际调用的 `RotaryEmbedding.forward()`。它位于 Attention 的 head 拆分之后、Q/K 旋转之前；本章用它集中观察 `unsqueeze`、广播、dtype、device 和 shape，但整机张量流仍然是从 `[B,S]` 一路走到 `[B,S,V]`，RoPE 只是其中一个保持 rank-4 Attention contract 的局部步骤。

<!-- source-sync: src/qwen3_moe/rope.py::RotaryEmbedding.forward -->
```python
def forward(
    self, positions: Tensor, *, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    if positions.ndim != 2 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be a rank-2 integer tensor")
    if positions.device != self.inv_freq.device:
        raise ValueError("positions and rotary buffer must be on the same device")
    if (positions < 0).any():
        raise ValueError("positions must be non-negative")
    if not dtype.is_floating_point:
        raise ValueError("RoPE output dtype must be floating point")

    angles = positions.float().unsqueeze(-1) * self.inv_freq
    full_angles = torch.cat((angles, angles), dim=-1)
    cos = full_angles.cos().unsqueeze(1).to(dtype=dtype)
    sin = full_angles.sin().unsqueeze(1).to(dtype=dtype)
    return cos, sin
```

执行顺序如下。

1. **先守住输入 contract。** `positions` 必须是 rank 2 的整数 `[B,S]`，因为每个 batch 的每个 token 位置都需要一个位置编号；负位置没有定义。它还必须与 `inv_freq` buffer 位于同一 device，否则后面的乘法会跨 CPU/GPU 失败。调用者传入的输出 `dtype` 必须是浮点类型。
2. **第一次 `unsqueeze` 为广播补轴。** `positions.float()` 是 `[B,S]`，`.unsqueeze(-1)` 后是 `[B,S,1]`；`inv_freq` 是 `[Dh/2]`。从右对齐广播后，两者相乘得到 `angles [B,S,Dh/2]`。新增的长度 1 轴不是新数据，而是明确告诉广播规则把每个位置编号分别乘上全部旋转频率。
3. **拼回完整 head 宽度。** `torch.cat((angles, angles), dim=-1)` 把最后一维从 `Dh/2` 变为 `Dh`，得到 `full_angles [B,S,Dh]`，与待旋转 Q/K 的最后一维一致。
4. **第二次 `unsqueeze` 为 head 广播补轴。** `cos()` 和 `sin()` 后仍是 `[B,S,Dh]`，`.unsqueeze(1)` 产生 `[B,1,S,Dh]`。后续它们与 Q `[B,Hq,S,Dh]`、K `[B,Hkv,S,Dh]` 运算时，长度为 1 的 head 轴分别广播到 `Hq` 和 `Hkv`，同一位置旋转参数无需真实复制多份。
5. **最后对齐计算 dtype。** 位置先转为 float32 计算角度，再用 `.to(dtype=dtype)` 把 `cos/sin` 转回 hidden states 的浮点 dtype。device 没有在这里偷偷迁移：前面的同设备检查保证结果留在当前模型 device 上。

把局部步骤放回整机：Embedding 先产生 hidden `[B,S,D]`；Attention 投影并拆成 Q/K `[B,H,S,Dh]`；本函数只根据 `positions [B,S]` 生成可广播的 `cos/sin [B,1,S,Dh]`，不改变 Q/K shape；Attention 合并 heads 后回到 `[B,S,D]`，MLP 继续保持 residual contract，最终 LM Head 才把最后一维映射为词表 `[B,S,V]`。因此，读懂 RoPE 的广播是在练习整机共同遵守的轴、dtype 与 device contract，而不是把整章缩成位置编码专题。

## 完整文件链接

- [`model.py`](../../src/qwen3_moe/model.py)：完整模型入口、Embedding、因果 mask 和 LM Head。
- [`attention.py`](../../src/qwen3_moe/attention.py)：head 拆分、矩阵乘法、mask 广播与 head merge。
- [`mlp.py`](../../src/qwen3_moe/mlp.py)：`[B,S,D] -> [B,S,I] -> [B,S,D]` 的逐位置变换。
- [`rope.py`](../../src/qwen3_moe/rope.py)：RoPE 参数生成、旋转与相关 contract。
- [`debug.py`](../../src/qwen3_moe/debug.py)：浮点、有限性和 residual tensor contract。

## 最小实验：从真实 Attention 读取 shape

```bash
uv run python - <<'PY'
import torch

from qwen3_moe import DenseConfig, GroupedQueryAttention

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
attention = GroupedQueryAttention(config).eval()
hidden = torch.randn(2, 4, 8)
positions = torch.arange(4).view(1, 4).expand(2, 4)

with torch.inference_mode():
    update, debug = attention(hidden, positions, return_debug=True)

for name in (
    "query_projected",
    "key_projected",
    "query_pre_rope",
    "key_pre_rope",
    "value",
    "repeated_key",
    "scores",
    "probabilities",
):
    value = debug[name]
    print(name, tuple(value.shape), value.dtype, value.device)
print("update", tuple(update.shape))

assert update.shape == hidden.shape
torch.testing.assert_close(
    debug["probabilities"].sum(-1), torch.ones(2, 2, 4), atol=1e-6, rtol=0
)
PY
```

运行前先预测每行 shape。运行后重点解释三次变化：`[B,S,H,Dh] -> [B,H,S,Dh]`；K/V head 从 `Hkv` 扩展到 `Hq`；最后 update 回到 `[B,S,D]`。

再观察为什么转置后不能盲目 `view`：

```bash
uv run python - <<'PY'
import torch

x = torch.arange(24).view(2, 3, 4)
y = x.transpose(1, 2)
print("shape/stride/contiguous:", y.shape, y.stride(), y.is_contiguous())
try:
    y.view(2, 12)
except RuntimeError as error:
    print("controlled view error:", str(error).splitlines()[-1])
print("fixed:", y.contiguous().view(2, 12).shape)
PY
```

## 对应测试

- [`test_causal_gqa_masks_future_tokens_and_maps_head_groups`](../../tests/test_attention.py)：验证 update shape、probabilities shape、未来概率为 0、每行概率和为 1，以及 GQA 连续 head 分组。
- [`test_attention_rejects_non_broadcastable_or_fully_blocked_masks`](../../tests/test_attention.py)：验证 mask 广播错误和完全屏蔽 query 行会明确失败。
- [`test_dense_decoder_uses_two_correct_residual_bases`](../../tests/test_decoder.py)：验证 `[B,S,D]` residual contract。
- [`test_rms_norm_rejects_wrong_width_and_integer_input`](../../tests/test_norms.py)：验证最后一维和浮点 dtype contract。
- [`test_model_rejects_invalid_token_ids`](../../tests/test_model.py)：验证模型入口 rank、dtype、范围和非空 shape。

```bash
uv run pytest tests/test_attention.py tests/test_decoder.py tests/test_norms.py tests/test_model.py
```

## 常见误解与受控错误

- **“Rank 3 就是某一维长度为 3。”** Rank 只数轴；shape 才记录轴长度。
- **“`transpose` 会复制并按新顺序连续存储。”** 它通常只改变逻辑视图与 stride；需要连续布局时显式 `contiguous()`。
- **“`reshape` 和 `view` 永远一样。”** 元素数约束相同，但 `view` 还受当前 size/stride 兼容性限制。
- **“广播会按 `B/S/D` 名字匹配。”** PyTorch 不知道这些名字，只从右对齐长度。
- **“`*` 和 `matmul` 都是乘法，可以互换。”** `*` 逐元素且走广播；`matmul` 收缩内维并可能产生新轴。
- **“shape 相同就能计算。”** dtype 或 device 不同仍可能失败；整数 hidden 也不符合 Norm/Attention 的数值 contract。
- **“mask shape 正确就一定安全。”** 若某个 query 行全部为 `True`，softmax 输入全是 `-inf`，真实实现主动拒绝。

## 深入教程

- [第一周：张量、形状与内存](../tutorials/week01-tensors-shapes-memory.md)：rank、索引、reshape/view、stride、transpose、contiguous、广播、dtype 和内存。
- [第二周：矩阵乘法与 `nn.Module`](../tutorials/week02-matmul-nn-module.md)：matmul/bmm/einsum、线性层权重方向、模块状态和 dtype/device。
- [第四周：Self-Attention 与 GQA](../tutorials/week04-self-attention-gqa.md)：把本章 shape ledger 连接到 Attention 数学。

## 回到完整推理流程

现在你不只知道模块名称，还能读它们之间的 Tensor 语言：

```text
[B,S] 整数 ID
-> [B,S,D] 浮点 hidden
-> [B,H,S,Dh] attention head 布局
-> [B,H,S,S] 因果概率
-> [B,S,D] residual 主干
-> [B,S,V] logits
```

请留下本章证据：整机张量位置图、真实 attention shape ledger、一次 `GroupedQueryAttention(..., return_debug=True)` 调用，以及 Attention/Decoder/Norm/Model 测试结果。回到 [课程入口](README.md) 复述前半主线，再进入后续 Attention 深入阶段。
