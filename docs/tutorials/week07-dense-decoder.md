# 第七周教程：完整 Dense Decoder

> **深入实验导航**
> - **主线位置：** [第 02 章：Decoder Layer](../course/02-decoder-layer.md)与[第 06 章：组装 Dense 模型](../course/06-assemble-dense-model.md)，深入 pre-norm residual contract、层堆叠和逐层调试。
> - **建议何时读：** 先完成主线第 02-05 章，在第 06 章组装并验证完整 Dense forward 时阅读；首次可跳过重复手写组件、长篇参考答案和扩展实验。
> - **源码与测试：** [`decoder.py::DenseDecoderLayer`](../../src/qwen3_moe/decoder.py)、[`model.py::TinyDenseCausalLM`](../../src/qwen3_moe/model.py)；[`test_decoder.py`](../../tests/test_decoder.py)、[`test_model.py`](../../tests/test_model.py)。
> - **返回主线：** [主线课程目录](../course/README.md)

> 本周目标：把已经验证过的 normalization、causal GQA、RoPE、QK Norm 和 SwiGLU 按正确的 pre-norm residual 顺序组合，完成可逐层追踪的微型 Dense Causal LM forward。

## 目录

1. [本周在完整推理中的位置](#inference-position)
2. [学习目标](#goals)
3. [学习方式](#method)
4. [环境、记号与边界](#environment)
5. [模块 1：Pre-norm 与 residual contract](#module-1)
6. [模块 2：Attention residual block](#module-2)
7. [模块 3：SwiGLU residual block](#module-3)
8. [模块 4：完整 Dense Decoder Layer](#module-4)
9. [模块 5：Decoder stack 与首次偏差](#module-5)
10. [模块 6：Embedding、final norm 与 LM Head](#module-6)
11. [综合任务](#capstone)
12. [最终验收](#acceptance)
13. [提示](#hints)
14. [参考答案](#answers)
15. [数据流、形状与调试速查](#glossary)
16. [Week 8 预告](#next-week)

<a id="inference-position"></a>
## 本周在完整推理中的位置

第三周建立了 `token IDs -> embedding -> logits` 的外壳；第四、五周完成 causal GQA、RMSNorm、RoPE 和 QK Norm；第六周完成 SwiGLU。现在的问题不再是“某个公式会不会算”，而是：**这些局部正确的组件怎样在 Decoder 中按正确边界组合，并在最终 logits 不一致时定位第一个错误？**

```text
token IDs [B,S]
-> embedding [B,S,D]
-> L x Dense Decoder Layer [B,S,D]
-> final RMSNorm [B,S,D]
-> LM Head [B,S,V]
```

单层内部采用 pre-norm：

```text
x
-> x + Attention(RMSNorm(x))                    = h
-> h + SwiGLU(RMSNorm(h))                       = y
```

第二条 residual 的基准是 `h`，不是原始 `x`。这个看似很小的区别，会让 shape 全部正确但 values 从第一层开始持续偏离。

<a id="goals"></a>
## 学习目标

完成后，你应该能：

1. 区分 pre-norm、子层更新和 identity skip。
2. 写出两次 norm、两次子层更新和两次 residual 的正确顺序。
3. 把 causal GQA 与 SwiGLU 组合成一个 Dense Decoder Layer。
4. 在每个边界检查 shape、dtype、device 和有限性。
5. 用 `nn.ModuleList` 堆叠参数互不共享的层。
6. 比较 1 层与多层 hidden states，而不假设范数单调增加。
7. 完成 `[B,S] -> logits [B,S,V]` 的 CPU forward。
8. 用因果前缀实验验证未来 token 不影响较早位置 logits。
9. 记录稳定 trace，并从 logits 偏差定位首个不一致张量。
10. 说明第八周如何保持外部接口不变，把 Dense MLP 边界替换为 MoE。

<a id="method"></a>
## 学习方式：Predict-Run-Explain

每个实验仍按同一闭环进行：

1. **Predict**：先画数据流，写出 residual 基准、shape 和预期第一个变化位置。
2. **Run**：独立运行当前 Python 围栏，观察断言、范数和 trace。
3. **Explain**：脱离代码说明为什么当前更新必须加回这个 residual，以及偏差为什么从该边界开始传播。

推荐调试顺序：`token IDs -> embedding -> layer 0 attention -> layer 0 MLP -> ... -> final norm -> LM Head`。在每个边界内部先查 `shape -> dtype -> device -> finite -> value`，不要只盯最终 logits。

<a id="environment"></a>
## 环境、记号与边界

仓库锁定环境可用时，从根目录运行：

```bash
uv sync --locked --python 3.11.15
uv run python scripts/check_environment.py
uv run pytest
```

所有 Python 围栏都自行 import、定义依赖并构造固定数据；必修路径只使用 CPU FP32，不下载模型、Tokenizer、checkpoint 或数据集。

统一记号：

```text
B    batch size              S    sequence length
V    vocabulary size         D    hidden width
I    MLP intermediate width  L    decoder layer count
Hq   query head count        Hkv  key/value head count
Dh   head width              G    Hq/Hkv
```

本周不加入训练、KV Cache、生成循环、padding mask、MoE、fused kernel 或真实权重。Embedding 与 LM Head 在主线中使用独立参数；权重绑定只作可选扩展。

<a id="module-1"></a>
## 模块 1：Pre-norm 与 residual contract

### 1.1 三条路径不能混写

Pre-norm block 写作：

```text
normalized = norm(x)
update = F(normalized)
output = x + update
```

`x` 是 identity skip；`update` 是子层建议写入 hidden state 的变化量。关闭子层实验应令 `update=0`，而不是删除 `x`。Residual 相加是逐元素运算，因此两侧不仅 shape 要完全相同，dtype 和 device 也应一致。

### 1.2 Predict-Run-Explain：顺序、开关与范数

运行前预测：pre-norm 和 post-norm 是否一般相等；`alpha=0` 时输出是什么；相加后范数是否一定增大。

```python
import torch

def rms_norm(x, weight, eps=1e-6):
    if x.shape[-1] != weight.numel():
        raise ValueError(f"norm width mismatch: x={tuple(x.shape)}, weight={tuple(weight.shape)}")
    if x.dtype != weight.dtype or x.device != weight.device:
        raise ValueError("norm input and weight must share dtype/device")
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight

def residual_add(residual, update, name="residual"):
    if residual.shape != update.shape:
        raise ValueError(f"{name} shape mismatch: {tuple(residual.shape)} vs {tuple(update.shape)}")
    if residual.dtype != update.dtype or residual.device != update.device:
        raise ValueError(f"{name} dtype/device mismatch")
    if not torch.isfinite(update).all():
        raise ValueError(f"{name} update contains NaN or Inf")
    return residual + update

x = torch.tensor([[[1.0, 2.0], [-2.0, 1.0]]])
weight = torch.tensor([0.8, 1.2])
w = torch.tensor([[0.5, -0.25], [0.25, 0.5]])

normalized = rms_norm(x, weight)
raw_update = normalized @ w
pre_norm = residual_add(x, raw_update)
disabled = residual_add(x, torch.zeros_like(raw_update))
post_norm = rms_norm(x + x @ w, weight)

assert torch.equal(disabled, x)
assert pre_norm.shape == x.shape
assert not torch.allclose(pre_norm, post_norm)

smaller = residual_add(x, -0.5 * x)
larger = residual_add(x, 0.5 * x)
assert torch.linalg.vector_norm(smaller) < torch.linalg.vector_norm(x)
assert torch.linalg.vector_norm(larger) > torch.linalg.vector_norm(x)

print("input norm:", torch.linalg.vector_norm(x).item())
print("pre-norm output norm:", torch.linalg.vector_norm(pre_norm).item())
print("contract checks passed")
```

范数变化取决于 `x` 与 `update` 的方向和大小。Residual connection 提供 identity path，不承诺每次相加都放大范数。

<a id="m1-e1-question"></a>
**M1-E1：** 若 `x=[B,S,D]`、`F(norm(x))=[B,S,1]`，PyTorch 可能广播相加。为什么 Decoder block 仍应拒绝？

<a id="m1-e2-question"></a>
**M1-E2：** attention 更新关闭、MLP 更新开启时，MLP 的 residual 基准和 norm 输入分别是什么？

<a id="module-2"></a>
## 模块 2：Attention residual block

### 2.1 从第五周子层到第一条 residual

第五周的 attention 路径接收已经归一化的 hidden。本周在它外部加入 input RMSNorm 和 residual：

```text
x [B,S,D]
-> input RMSNorm
-> Q/K/V projection -> QK Norm(Q/K) -> RoPE(Q/K)
-> causal GQA -> output projection = attention update [B,S,D]
-> x + attention update
```

Q/K 做 QK Norm 和 RoPE，V 不做；causal mask 必须在 softmax 前作用于每一层 scores。

### 2.2 Predict-Run-Explain：完整 attention update

```python
import math
import torch

torch.manual_seed(7)

def rms_norm(x, weight, eps=1e-6):
    if x.shape[-1] != weight.numel():
        raise ValueError("RMSNorm width mismatch")
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight

def rotate_half(x):
    if x.shape[-1] % 2:
        raise ValueError("RoPE requires even Dh")
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def attention_residual(x, weights, positions, mask, hq=2, hkv=1, enabled=True):
    b, s, d = x.shape
    dh = d // hq
    if d != hq * dh or hq % hkv or dh % 2:
        raise ValueError("invalid D/Hq/Hkv/Dh configuration")
    if mask.shape != (1, 1, s, s) or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [1,1,S,S]")

    normalized = rms_norm(x, weights["input_norm"])
    q = (normalized @ weights["q"]).view(b, s, hq, dh)
    k = (normalized @ weights["k"]).view(b, s, hkv, dh)
    v = (normalized @ weights["v"]).view(b, s, hkv, dh)
    q = rms_norm(q, weights["q_norm"]).transpose(1, 2)
    k = rms_norm(k, weights["k_norm"]).transpose(1, 2)
    v = v.transpose(1, 2)

    inv_freq = 1.0 / (100.0 ** (torch.arange(0, dh, 2).float() / dh))
    angles = positions.float().unsqueeze(-1) * inv_freq
    full_angles = torch.cat((angles, angles), dim=-1)
    cos, sin = full_angles.cos().unsqueeze(1), full_angles.sin().unsqueeze(1)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    repeated_k = k.repeat_interleave(hq // hkv, dim=1)
    repeated_v = v.repeat_interleave(hq // hkv, dim=1)
    scores = torch.einsum("bhsd,bhtd->bhst", q, repeated_k) / math.sqrt(dh)
    probabilities = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
    context = torch.einsum("bhst,bhtd->bhsd", probabilities, repeated_v)
    merged = context.transpose(1, 2).contiguous().view(b, s, d)
    raw_update = merged @ weights["out"]
    if raw_update.shape != x.shape or raw_update.dtype != x.dtype or raw_update.device != x.device:
        raise ValueError("raw attention update violates the residual contract")
    update = raw_update if enabled else torch.zeros_like(raw_update)
    output = x + update
    return output, {
        "normalized": normalized, "q": q, "k": k, "v": v,
        "scores": scores, "probabilities": probabilities,
        "raw_update": raw_update, "update": update,
    }

B, S, D, Hq, Hkv = 1, 4, 8, 2, 1
Dh = D // Hq
x = torch.randn(B, S, D)
positions = torch.arange(S).view(1, S)
mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1).view(1, 1, S, S)
weights = {
    "input_norm": torch.ones(D), "q_norm": torch.ones(Dh), "k_norm": torch.ones(Dh),
    "q": torch.randn(D, Hq * Dh) * 0.2,
    "k": torch.randn(D, Hkv * Dh) * 0.2,
    "v": torch.randn(D, Hkv * Dh) * 0.2,
    "out": torch.randn(D, D) * 0.2,
}

output, debug = attention_residual(x, weights, positions, mask, Hq, Hkv, True)
disabled, disabled_debug = attention_residual(x, weights, positions, mask, Hq, Hkv, False)

assert debug["q"].shape == (1, 2, 4, 4)
assert debug["k"].shape == debug["v"].shape == (1, 1, 4, 4)
assert debug["scores"].shape == debug["probabilities"].shape == (1, 2, 4, 4)
future = debug["probabilities"].masked_select(mask.expand_as(debug["probabilities"]))
assert torch.equal(future, torch.zeros_like(future))
torch.testing.assert_close(debug["probabilities"].sum(-1), torch.ones(1, 2, 4), atol=1e-6, rtol=0)
torch.testing.assert_close(output, x + debug["update"])
assert torch.equal(disabled, x)
assert torch.count_nonzero(disabled_debug["update"]) == 0
assert all(torch.isfinite(value).all() for value in debug.values())
print("attention residual checks passed")
```

这里的物理 GQA 用 `repeat_interleave(G, dim=1)`，保持每 `G` 个连续 query heads 共用一个 KV head。完整模型的每一层都必须重新计算并应用自己的 causal attention。

<a id="m2-e1-question"></a>
**M2-E1：** 对 `B=2,S=5,D=12,Hq=3,Hkv=1`，写出 Q、K/V、scores 和 attention update shape。

<a id="m2-e2-question"></a>
**M2-E2：** 为什么只在模型入口创建一次 causal mask 可以，但只在第 0 层应用一次不可以？

<a id="module-3"></a>
## 模块 3：SwiGLU residual block

### 3.1 第二条 residual 的基准已经改变

Attention residual 输出记为 `h`。MLP block 必须执行：

```text
normalized = post_attention_norm(h)
product = SiLU(gate(normalized)) * up(normalized)
mlp_update = down(product)
y = h + mlp_update
```

如果写成 `x + mlp_update`，shape 仍是 `[B,S,D]`，但 attention 已写入 `h` 的信息没有作为第二条 identity path 保留下来。

### 3.2 Predict-Run-Explain：MLP 更新开关

```python
import torch
import torch.nn.functional as F

torch.manual_seed(7)

def rms_norm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight

def swiglu_residual(h, norm_weight, gate_weight, up_weight, down_weight, enabled=True):
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate/up weights must have identical shape")
    d, intermediate = gate_weight.shape
    if h.shape[-1] != d or down_weight.shape != (intermediate, d):
        raise ValueError("SwiGLU projection shape mismatch")
    normalized = rms_norm(h, norm_weight)
    gate = normalized @ gate_weight
    up = normalized @ up_weight
    if gate.shape != up.shape:
        raise ValueError("gate/up outputs must match exactly")
    product = F.silu(gate) * up
    raw_update = product @ down_weight
    if raw_update.shape != h.shape or raw_update.dtype != h.dtype or raw_update.device != h.device:
        raise ValueError("raw MLP update violates the residual contract")
    update = raw_update if enabled else torch.zeros_like(raw_update)
    if update.shape != h.shape or update.dtype != h.dtype or update.device != h.device:
        raise ValueError("MLP residual contract failed")
    return h + update, {
        "normalized": normalized, "gate": gate, "up": up,
        "product": product, "raw_update": raw_update, "update": update,
    }

B, S, D, I = 2, 3, 4, 6
h = torch.randn(B, S, D)
norm_weight = torch.tensor([0.8, 1.0, 1.2, 0.9])
gate_weight = torch.randn(D, I) * 0.2
up_weight = torch.randn(D, I) * 0.2
down_weight = torch.randn(I, D) * 0.2

output, debug = swiglu_residual(h, norm_weight, gate_weight, up_weight, down_weight, True)
disabled, disabled_debug = swiglu_residual(h, norm_weight, gate_weight, up_weight, down_weight, False)

assert debug["gate"].shape == debug["up"].shape == debug["product"].shape == (2, 3, 6)
assert debug["update"].shape == output.shape == (2, 3, 4)
torch.testing.assert_close(output, h + debug["update"])
assert torch.equal(disabled, h)
assert torch.count_nonzero(disabled_debug["update"]) == 0
print("MLP residual checks passed")
```

开关只把已验证的更新变成零；identity skip 始终保留。正式模型通常不需要这个开关，它只是帮助观察每条 residual update 的贡献。

<a id="m3-e1-question"></a>
**M3-E1：** 若 `h=[2,4,8]`、`I=24`，写出 post-attention norm、gate、up、product、MLP update 和层输出 shape。

<a id="m3-e2-question"></a>
**M3-E2：** 错误公式 `y=x+MLP(norm(h))` 丢失了什么？为什么 shape 检查抓不到？

<a id="module-4"></a>
## 模块 4：完整 Dense Decoder Layer

### 4.1 两个 norm、两个更新、两个基准

单层必须拥有独立的 input norm 和 post-attention norm。下面用小型线性 attention stand-in 隔离组合顺序；模块 2 已验证真实 attention 内部，综合任务会把完整 causal GQA 接回来。

### 4.2 Predict-Run-Explain：四种支路组合

```python
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)

class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d, intermediate):
        super().__init__()
        self.gate = nn.Linear(d, intermediate, bias=False)
        self.up = nn.Linear(d, intermediate, bias=False)
        self.down = nn.Linear(intermediate, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class DenseDecoderLayer(nn.Module):
    def __init__(self, d, intermediate):
        super().__init__()
        self.input_norm = RMSNorm(d)
        self.attention = nn.Linear(d, d, bias=False)  # composition-only stand-in
        self.post_attention_norm = RMSNorm(d)
        self.mlp = SwiGLU(d, intermediate)

    def forward(self, x, attention_enabled=True, mlp_enabled=True):
        attention_norm = self.input_norm(x)
        raw_attention_update = self.attention(attention_norm)
        if (raw_attention_update.shape != x.shape or raw_attention_update.dtype != x.dtype
                or raw_attention_update.device != x.device):
            raise ValueError("raw attention update violates the residual contract")
        attention_update = raw_attention_update if attention_enabled else torch.zeros_like(x)
        h = x + attention_update
        mlp_norm = self.post_attention_norm(h)
        raw_mlp_update = self.mlp(mlp_norm)
        if (raw_mlp_update.shape != h.shape or raw_mlp_update.dtype != h.dtype
                or raw_mlp_update.device != h.device):
            raise ValueError("raw MLP update violates the residual contract")
        mlp_update = raw_mlp_update if mlp_enabled else torch.zeros_like(h)
        y = h + mlp_update
        return y, {
            "attention_norm": attention_norm, "attention_update": attention_update,
            "after_attention": h, "mlp_norm": mlp_norm,
            "mlp_update": mlp_update, "layer_output": y,
        }

layer = DenseDecoderLayer(d=4, intermediate=6)
x = torch.randn(2, 3, 4)

full, full_trace = layer(x, True, True)
attention_only, attention_trace = layer(x, True, False)
mlp_only, mlp_trace = layer(x, False, True)
identity, identity_trace = layer(x, False, False)

# 显式参考按同一组模块和正确 residual 基准重算。
ref_attention_norm = layer.input_norm(x)
ref_attention_update = layer.attention(ref_attention_norm)
ref_h = x + ref_attention_update
ref_mlp_norm = layer.post_attention_norm(ref_h)
ref_mlp_update = layer.mlp(ref_mlp_norm)
reference = ref_h + ref_mlp_update

for name, expected in {
    "attention_norm": ref_attention_norm, "attention_update": ref_attention_update,
    "after_attention": ref_h, "mlp_norm": ref_mlp_norm,
    "mlp_update": ref_mlp_update, "layer_output": reference,
}.items():
    torch.testing.assert_close(full_trace[name], expected)

assert layer.input_norm is not layer.post_attention_norm
assert torch.equal(identity, x)
torch.testing.assert_close(attention_only, x + attention_trace["attention_update"])
torch.testing.assert_close(mlp_only, x + mlp_trace["mlp_update"])
torch.testing.assert_close(full, reference)

for label, value in {
    "input": x, "full": full, "attention_only": attention_only,
    "mlp_only": mlp_only, "identity": identity,
}.items():
    print(label, "norm=", torch.linalg.vector_norm(value).item())
print("dense layer composition checks passed")
```

MLP-only 路径先得到 `h=x`，再从这个 `h` 计算 post-attention norm 和 MLP update。不能先用 full 路径的 `h` 算好 MLP，再事后把 attention update 删掉。

<a id="m4-e1-question"></a>
**M4-E1：** 写出 full 路径的六个稳定边界名称，并指出哪两个边界分别是 residual 相加后的结果。

<a id="m4-e2-question"></a>
**M4-E2：** 为什么 `ModuleList([layer] * 3)` 不适合构造三层 Decoder？

<a id="module-5"></a>
## 模块 5：Decoder stack 与首次偏差

### 5.1 Trace 的职责

最终 logits 不一致只说明“某处不同”。稳定 trace 应按执行顺序记录克隆后的边界值：

```text
layers.0.attention_norm -> layers.0.attention_update -> layers.0.after_attention
-> layers.0.mlp_norm -> layers.0.mlp_update -> layers.0.output -> layers.1...
```

先比较 key 顺序和 metadata，再比较 values。第一个不同边界比后续全部不同更有诊断价值。

### 5.2 Predict-Run-Explain：扰动第 2 层

```python
import copy
import torch
from torch import nn

torch.manual_seed(7)

def snapshot(trace, key, value):
    trace[key] = value.detach().clone()

class TraceLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = nn.Linear(width, width, bias=False)
        self.mlp = nn.Linear(width, width, bias=False)

    def forward(self, x, trace, prefix):
        attention_norm = x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        attention_update = self.attention(attention_norm)
        h = x + attention_update
        mlp_norm = h / torch.sqrt(h.square().mean(-1, keepdim=True) + 1e-6)
        mlp_update = self.mlp(torch.tanh(mlp_norm))
        y = h + mlp_update
        for name, value in (
            ("attention_norm", attention_norm), ("attention_update", attention_update),
            ("after_attention", h), ("mlp_norm", mlp_norm),
            ("mlp_update", mlp_update), ("output", y),
        ):
            snapshot(trace, f"{prefix}.{name}", value)
        return y

class Stack(nn.Module):
    def __init__(self, layers, width):
        super().__init__()
        self.layers = nn.ModuleList([TraceLayer(width) for _ in range(layers)])

    def forward(self, x):
        trace = {"embedding": x.detach().clone()}
        for index, layer in enumerate(self.layers):
            x = layer(x, trace, f"layers.{index}")
        return x, trace

def first_difference(left, right, atol=1e-6, rtol=1e-6):
    if list(left) != list(right):
        raise ValueError("trace key order differs")
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
            return key, float("inf")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"non-finite trace at {key}")
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            return key, (a - b).abs().max().item()
    return None, 0.0

model = Stack(layers=2, width=4)
changed = copy.deepcopy(model)
x = torch.randn(1, 3, 4)
base_output, base_trace = model(x)
copy_output, copy_trace = changed(x)
assert first_difference(base_trace, copy_trace) == (None, 0.0)
torch.testing.assert_close(base_output, copy_output, atol=0, rtol=0)

with torch.no_grad():
    changed.layers[1].mlp.weight[0, 0] += 0.25
changed_output, changed_trace = changed(x)
key, max_abs = first_difference(base_trace, changed_trace)
assert key == "layers.1.mlp_update"
assert not torch.allclose(base_output, changed_output)
print("first difference:", key, "max_abs=", max_abs)
```

第 2 层 MLP 之前的输入、attention 和 `mlp_norm` 都没变；downstream 的 `layers.1.output` 也会不同，但它不是首因。

<a id="m5-e1-question"></a>
**M5-E1：** 若 `layers.2.attention_update` 首次不同，哪些更早的层级边界必须一致？

<a id="m5-e2-question"></a>
**M5-E2：** 为什么 trace 要 `detach().clone()`，而不能只保存正在继续参与计算的 Tensor 引用？

<a id="module-6"></a>
## 模块 6：Embedding、final norm 与 LM Head

### 6.1 Decoder 外壳

Decoder stack 前后还需要三个接口：整数 token IDs 查 embedding；所有层后做 final RMSNorm；LM Head 把 `D` 投影到词表宽度 `V`。

### 6.2 Predict-Run-Explain：`[B,S] -> [B,S,V]`

```python
import torch
from torch import nn

torch.manual_seed(7)

class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight

class LMShell(nn.Module):
    def __init__(self, vocab, width, layers):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab, width)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(layers)])
        self.final_norm = RMSNorm(width)
        self.lm_head = nn.Linear(width, vocab, bias=False)

    def forward(self, token_ids):
        if token_ids.ndim != 2 or token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must be rank-2 integers")
        if (token_ids < 0).any() or (token_ids >= self.vocab).any():
            raise ValueError("token ID out of vocabulary range")
        hidden = self.embedding(token_ids)
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        normalized = self.final_norm(hidden)
        logits = self.lm_head(normalized)
        return logits, hidden, normalized

model = LMShell(vocab=11, width=8, layers=2)
token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
logits, hidden, normalized = model(token_ids)

assert token_ids.shape == (2, 4)
assert hidden.shape == normalized.shape == (2, 4, 8)
assert logits.shape == (2, 4, 11)
assert token_ids.dtype == torch.long
assert hidden.dtype == logits.dtype == torch.float32
assert model.embedding.weight is not model.lm_head.weight
print("LM shell shape checks passed")
```

这里的层只是外壳演示；综合任务会换成完整 Dense Decoder Layer。Embedding 与 LM Head 参数独立，意味着两者 shape 分别为 `[V,D]` 与 `[V,D]`，但不是同一个 Parameter。

<a id="m6-e1-question"></a>
**M6-E1：** 对 `B=3,S=6,V=100,D=16,L=4`，写出 embedding、每层输出、final norm 和 logits shape。

<a id="m6-e2-question"></a>
**M6-E2：** 为什么 token IDs 应为整数，而 hidden states 和 logits 应为浮点数？

<a id="capstone"></a>
## 综合任务：两层微型 Dense Causal LM

固定配置：

```text
B=2, S=4, V=11
D=8, I=12
Hq=2, Hkv=1, Dh=4, G=2
L=2, dtype=float32, device=cpu, bias=False
```

运行前先预测以下首个偏差实验：基础模型与副本完全一致；只修改副本第 2 层 MLP down projection 后，`layers.1.mlp_norm` 是否改变？第一个改变的稳定边界应该是什么？

```python
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)

def snapshot(mapping, key, value):
    mapping[key] = value.detach().clone()

def require_finite(name, value):
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")

def require_residual(name, residual, raw_update):
    if residual.shape != raw_update.shape:
        raise ValueError(f"{name} shape mismatch: {tuple(residual.shape)} vs {tuple(raw_update.shape)}")
    if residual.dtype != raw_update.dtype or residual.device != raw_update.device:
        raise ValueError(f"{name} dtype/device mismatch")
    require_finite(f"{name} raw update", raw_update)

class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        if width <= 0 or eps <= 0:
            raise ValueError("RMSNorm width and eps must be positive")
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        if not x.is_floating_point() or x.shape[-1] != self.weight.numel():
            raise ValueError(f"RMSNorm expected floating [...,{self.weight.numel()}], got {tuple(x.shape)}")
        if x.dtype != torch.float32:
            raise ValueError("this CPU teaching path requires float32 hidden states")
        if x.dtype != self.weight.dtype or x.device != self.weight.device:
            raise ValueError("RMSNorm input and weight must share dtype/device")
        output = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight
        require_finite("RMSNorm output", output)
        return output

def rotate_half(x):
    if x.shape[-1] % 2:
        raise ValueError("split-half RoPE requires even Dh")
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

class CausalGQA(nn.Module):
    def __init__(self, d, hq, hkv, dh):
        super().__init__()
        if min(d, hq, hkv, dh) <= 0 or d != hq * dh or hq % hkv or dh % 2:
            raise ValueError("invalid D/Hq/Hkv/Dh configuration")
        self.d, self.hq, self.hkv, self.dh = d, hq, hkv, dh
        self.group_size = hq // hkv
        self.q_proj = nn.Linear(d, hq * dh, bias=False)
        self.k_proj = nn.Linear(d, hkv * dh, bias=False)
        self.v_proj = nn.Linear(d, hkv * dh, bias=False)
        self.q_norm = RMSNorm(dh)
        self.k_norm = RMSNorm(dh)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, positions, mask):
        b, s, d = x.shape
        if d != self.d:
            raise ValueError(f"attention expected D={self.d}, got {d}")
        if positions.shape != (b, s) or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"positions must be integer [{b},{s}]")
        if positions.device != x.device or (positions < 0).any():
            raise ValueError("positions must be non-negative and on the hidden device")
        if mask.shape != (1, 1, s, s) or mask.dtype != torch.bool or mask.device != x.device:
            raise ValueError(f"mask must be bool [1,1,{s},{s}] on the hidden device")

        q_projected = self.q_proj(x).view(b, s, self.hq, self.dh)
        k_projected = self.k_proj(x).view(b, s, self.hkv, self.dh)
        v = self.v_proj(x).view(b, s, self.hkv, self.dh)
        q_pre_rope = self.q_norm(q_projected).transpose(1, 2)
        k_pre_rope = self.k_norm(k_projected).transpose(1, 2)
        v = v.transpose(1, 2)

        inv_freq = 1.0 / (
            torch.tensor(100.0, device=x.device, dtype=torch.float32)
            ** (torch.arange(0, self.dh, 2, device=x.device, dtype=torch.float32) / self.dh)
        )
        angles = positions.float().unsqueeze(-1) * inv_freq
        full_angles = torch.cat((angles, angles), dim=-1)
        cos, sin = full_angles.cos().unsqueeze(1), full_angles.sin().unsqueeze(1)
        q = q_pre_rope * cos + rotate_half(q_pre_rope) * sin
        k = k_pre_rope * cos + rotate_half(k_pre_rope) * sin

        repeated_k = k.repeat_interleave(self.group_size, dim=1)
        repeated_v = v.repeat_interleave(self.group_size, dim=1)
        scores = torch.einsum("bhsd,bhtd->bhst", q, repeated_k) / math.sqrt(self.dh)
        expanded_mask = mask.expand_as(scores)
        if expanded_mask.all(-1).any():
            raise ValueError("causal mask contains a fully blocked row")
        probabilities = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        context = torch.einsum("bhst,bhtd->bhsd", probabilities, repeated_v)
        merged = context.transpose(1, 2).contiguous().view(b, s, d)
        update = self.out_proj(merged)

        for name, value in {
            "q": q, "k": k, "v": v, "scores": scores,
            "probabilities": probabilities, "update": update,
        }.items():
            require_finite(f"attention {name}", value)
        return update, {
            "q_projected": q_projected, "k_projected": k_projected,
            "q_pre_rope": q_pre_rope, "k_pre_rope": k_pre_rope,
            "q": q, "k": k, "v": v, "repeated_k": repeated_k, "repeated_v": repeated_v,
            "scores": scores, "probabilities": probabilities,
        }

class SwiGLU(nn.Module):
    def __init__(self, d, intermediate):
        super().__init__()
        if d <= 0 or intermediate <= 0:
            raise ValueError("SwiGLU widths must be positive")
        self.d = d
        self.gate_proj = nn.Linear(d, intermediate, bias=False)
        self.up_proj = nn.Linear(d, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d, bias=False)

    def forward(self, x):
        if x.shape[-1] != self.d:
            raise ValueError(f"SwiGLU expected D={self.d}, got {x.shape[-1]}")
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if gate.shape != up.shape:
            raise ValueError("gate/up outputs must match exactly")
        product = F.silu(gate) * up
        update = self.down_proj(product)
        require_finite("SwiGLU product", product)
        require_finite("SwiGLU update", update)
        return update, {"gate": gate, "up": up, "product": product}

class DenseDecoderLayer(nn.Module):
    def __init__(self, d, intermediate, hq, hkv, dh):
        super().__init__()
        self.input_norm = RMSNorm(d)
        self.self_attn = CausalGQA(d, hq, hkv, dh)
        self.post_attention_norm = RMSNorm(d)
        self.mlp = SwiGLU(d, intermediate)

    def forward(self, x, positions, mask, trace, diagnostics, prefix,
                attention_enabled=True, mlp_enabled=True):
        attention_norm = self.input_norm(x)
        raw_attention_update, attention_debug = self.self_attn(attention_norm, positions, mask)
        require_residual(f"{prefix} attention residual", x, raw_attention_update)
        attention_update = raw_attention_update if attention_enabled else torch.zeros_like(x)
        h = x + attention_update

        mlp_norm = self.post_attention_norm(h)
        raw_mlp_update, mlp_debug = self.mlp(mlp_norm)
        require_residual(f"{prefix} MLP residual", h, raw_mlp_update)
        mlp_update = raw_mlp_update if mlp_enabled else torch.zeros_like(h)
        output = h + mlp_update

        for name, value in (
            ("attention_norm", attention_norm), ("attention_update", attention_update),
            ("after_attention", h), ("mlp_norm", mlp_norm),
            ("mlp_update", mlp_update), ("output", output),
        ):
            require_finite(f"{prefix}.{name}", value)
            snapshot(trace, f"{prefix}.{name}", value)
        for name, value in attention_debug.items():
            snapshot(diagnostics, f"{prefix}.attention.{name}", value)
        for name, value in mlp_debug.items():
            snapshot(diagnostics, f"{prefix}.mlp.{name}", value)
        return output

class TinyDenseCausalLM(nn.Module):
    def __init__(self, vocab, d, intermediate, hq, hkv, dh, layers):
        super().__init__()
        if min(vocab, d, intermediate, layers) <= 0:
            raise ValueError("model dimensions must be positive")
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            DenseDecoderLayer(d, intermediate, hq, hkv, dh) for _ in range(layers)
        ])
        self.final_norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, token_ids, num_layers=None):
        if token_ids.ndim != 2 or token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must be rank-2 integers")
        if (token_ids < 0).any() or (token_ids >= self.vocab).any():
            raise ValueError("token ID out of vocabulary range")
        if token_ids.shape[0] <= 0 or token_ids.shape[1] <= 0:
            raise ValueError("batch and sequence dimensions must be positive")
        if num_layers is not None and (not isinstance(num_layers, int) or isinstance(num_layers, bool)):
            raise ValueError("num_layers must be an integer")
        layer_count = len(self.layers) if num_layers is None else num_layers
        if not 1 <= layer_count <= len(self.layers):
            raise ValueError("num_layers is outside the configured stack")

        b, s = token_ids.shape
        positions = torch.arange(s, device=token_ids.device).view(1, s).expand(b, s)
        mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=token_ids.device), diagonal=1)
        mask = mask.view(1, 1, s, s)
        trace, diagnostics = {}, {}
        hidden = self.embedding(token_ids)
        snapshot(trace, "embedding", hidden)
        for index, layer in enumerate(self.layers[:layer_count]):
            hidden = layer(hidden, positions, mask, trace, diagnostics, f"layers.{index}")
        normalized = self.final_norm(hidden)
        logits = self.lm_head(normalized)
        require_finite("logits", logits)
        snapshot(trace, "final_norm", normalized)
        snapshot(trace, "logits", logits)
        return logits, trace, diagnostics

def first_difference(left, right, atol=1e-6, rtol=1e-6):
    if list(left) != list(right):
        raise ValueError("trace key set or execution order differs")
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
            return key, float("inf"), tuple(a.shape), tuple(b.shape)
        require_finite(f"left {key}", a)
        require_finite(f"right {key}", b)
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            return key, (a - b).abs().max().item(), tuple(a.shape), tuple(b.shape)
    return None, 0.0, None, None

B, S, V = 2, 4, 11
D, I, Hq, Hkv, Dh, L = 8, 12, 2, 1, 4, 2
model = TinyDenseCausalLM(V, D, I, Hq, Hkv, Dh, L).eval()
token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)

# 非均匀 Q/K Norm weight 让“先 RoPE 再 QK Norm”的错误顺序可被观察。
with torch.no_grad():
    for layer in model.layers:
        layer.self_attn.q_norm.weight.copy_(torch.tensor([0.7, 1.2, 1.5, 0.9]))
        layer.self_attn.k_norm.weight.copy_(torch.tensor([1.1, 0.8, 1.4, 0.6]))

with torch.inference_mode():
    logits, trace, diagnostics = model(token_ids)

assert token_ids.shape == (2, 4)
assert trace["embedding"].shape == (2, 4, 8)
assert trace["layers.0.output"].shape == trace["layers.1.output"].shape == (2, 4, 8)
assert trace["final_norm"].shape == (2, 4, 8)
assert logits.shape == (2, 4, 11)
for key, value in trace.items():
    assert value.dtype == torch.float32 and value.device.type == "cpu", key
    assert torch.isfinite(value).all(), key

mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1).view(1, 1, S, S)
for layer_index in range(L):
    probabilities = diagnostics[f"layers.{layer_index}.attention.probabilities"]
    assert diagnostics[f"layers.{layer_index}.attention.q"].shape == (2, 2, 4, 4)
    assert diagnostics[f"layers.{layer_index}.attention.k"].shape == (2, 1, 4, 4)
    assert diagnostics[f"layers.{layer_index}.attention.v"].shape == (2, 1, 4, 4)
    assert diagnostics[f"layers.{layer_index}.mlp.gate"].shape == (2, 4, 12)
    assert diagnostics[f"layers.{layer_index}.mlp.up"].shape == (2, 4, 12)
    assert diagnostics[f"layers.{layer_index}.mlp.product"].shape == (2, 4, 12)
    assert probabilities.shape == (2, 2, 4, 4)
    future = probabilities.masked_select(mask.expand_as(probabilities))
    assert torch.equal(future, torch.zeros_like(future))
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(B, Hq, S), atol=1e-6, rtol=0)
for key, value in diagnostics.items():
    assert value.dtype == torch.float32 and value.device.type == "cpu", key
    assert torch.isfinite(value).all(), key

# 显式证明目标顺序 QK Norm -> RoPE 与错误顺序 RoPE -> QK Norm 不同。
positions = torch.arange(S).view(1, S).expand(B, S)
inv_freq = 1.0 / (100.0 ** (torch.arange(0, Dh, 2).float() / Dh))
angles = positions.float().unsqueeze(-1) * inv_freq
full_angles = torch.cat((angles, angles), dim=-1)
cos, sin = full_angles.cos().unsqueeze(1), full_angles.sin().unsqueeze(1)
q_projected = diagnostics["layers.0.attention.q_projected"].transpose(1, 2)
q_rope_first = q_projected * cos + rotate_half(q_projected) * sin
wrong_q = model.layers[0].self_attn.q_norm(q_rope_first)
correct_q = diagnostics["layers.0.attention.q"]
assert (correct_q - wrong_q).abs().max() > 1e-4
k_projected = diagnostics["layers.0.attention.k_projected"].transpose(1, 2)
k_rope_first = k_projected * cos + rotate_half(k_projected) * sin
wrong_k = model.layers[0].self_attn.k_norm(k_rope_first)
correct_k = diagnostics["layers.0.attention.k"]
assert (correct_k - wrong_k).abs().max() > 1e-4

# 用真实 CausalGQA 的 Hkv=2 辅助路径验证连续 query-head 分组。
aux_attention = CausalGQA(d=8, hq=4, hkv=2, dh=2)
aux_x = torch.randn(1, 3, 8)
aux_positions = torch.arange(3).view(1, 3)
aux_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1).view(1, 1, 3, 3)
_, aux_debug = aux_attention(aux_x, aux_positions, aux_mask)
torch.testing.assert_close(aux_debug["repeated_k"][:, 0], aux_debug["k"][:, 0])
torch.testing.assert_close(aux_debug["repeated_k"][:, 1], aux_debug["k"][:, 0])
torch.testing.assert_close(aux_debug["repeated_k"][:, 2], aux_debug["k"][:, 1])
torch.testing.assert_close(aux_debug["repeated_k"][:, 3], aux_debug["k"][:, 1])
torch.testing.assert_close(aux_debug["repeated_v"][:, 0], aux_debug["v"][:, 0])
torch.testing.assert_close(aux_debug["repeated_v"][:, 3], aux_debug["v"][:, 1])

for layer_index in range(L):
    layer_input = trace["embedding"] if layer_index == 0 else trace[f"layers.{layer_index - 1}.output"]
    torch.testing.assert_close(
        trace[f"layers.{layer_index}.after_attention"],
        layer_input + trace[f"layers.{layer_index}.attention_update"],
    )
    torch.testing.assert_close(
        trace[f"layers.{layer_index}.output"],
        trace[f"layers.{layer_index}.after_attention"] + trace[f"layers.{layer_index}.mlp_update"],
    )

# 四种 residual-update 开关；identity skip 永不关闭。
embedding = model.embedding(token_ids)
branch_outputs = {}
for attention_enabled, mlp_enabled, label in (
    (True, True, "full"), (True, False, "attention_only"),
    (False, True, "mlp_only"), (False, False, "identity"),
):
    branch_trace, branch_diagnostics = {}, {}
    branch_outputs[label] = model.layers[0](
        embedding, positions, mask, branch_trace, branch_diagnostics, "layer",
        attention_enabled, mlp_enabled,
    )
    if label == "identity":
        assert torch.equal(branch_outputs[label], embedding)
    if label == "attention_only":
        torch.testing.assert_close(branch_outputs[label], embedding + branch_trace["layer.attention_update"])
    if label == "mlp_only":
        assert torch.equal(branch_trace["layer.after_attention"], embedding)

print("branch output norms:", {
    key: round(torch.linalg.vector_norm(value).item(), 6)
    for key, value in branch_outputs.items()
})

# 同一个模型只运行 1 层与完整 2 层。
with torch.inference_mode():
    one_layer_logits, one_layer_trace, _ = model(token_ids, num_layers=1)
torch.testing.assert_close(one_layer_trace["layers.0.output"], trace["layers.0.output"], atol=0, rtol=0)
assert one_layer_logits.shape == logits.shape
assert not torch.allclose(trace["layers.0.output"], trace["layers.1.output"])

# 两条等长序列只改变最后 token；前 3 个位置 logits 必须保持。
future_changed = token_ids.clone()
future_changed[:, -1] = torch.tensor([9, 10])
assert not torch.equal(model.embedding(token_ids)[:, -1], model.embedding(future_changed)[:, -1])
with torch.inference_mode():
    changed_future_logits, _, _ = model(future_changed)
torch.testing.assert_close(logits[:, :3], changed_future_logits[:, :3], atol=1e-6, rtol=1e-6)

# 层和 norm 不能意外共享实例或参数存储。
assert model.layers[0] is not model.layers[1]
assert model.layers[0].input_norm is not model.layers[0].post_attention_norm
assert model.layers[0].mlp.down_proj.weight.data_ptr() != model.layers[1].mlp.down_proj.weight.data_ptr()
assert model.embedding.weight is not model.lm_head.weight

# 完全复制先严格一致，再扰动第 2 层实际活跃的 SwiGLU 通道。
copied = copy.deepcopy(model)
with torch.inference_mode():
    copied_logits, copied_trace, _ = copied(token_ids)
assert first_difference(trace, copied_trace) == (None, 0.0, None, None)
torch.testing.assert_close(logits, copied_logits, atol=0, rtol=0)

product = diagnostics["layers.1.mlp.product"]
activity = product.abs().amax(dim=(0, 1))
channel = int(activity.argmax())
assert activity[channel] > 0
expected_delta = 0.25 * product[..., channel]
assert expected_delta.abs().max() > 1e-4
with torch.no_grad():
    copied.layers[1].mlp.down_proj.weight[0, channel] += 0.25
with torch.inference_mode():
    perturbed_logits, perturbed_trace, _ = copied(token_ids)

key, max_abs, left_shape, right_shape = first_difference(trace, perturbed_trace)
assert key == "layers.1.mlp_update"
assert not torch.allclose(logits, perturbed_logits)
print("first difference:", key)
print("max abs difference:", max_abs, "shapes:", left_shape, right_shape)

# 受控错误：覆盖入口、配置、attention、residual 和 finite 边界。
def expect_value_error(fn, expected_text):
    try:
        fn()
    except ValueError as error:
        assert expected_text in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {expected_text!r}")

expect_value_error(lambda: model(torch.tensor([[0, V]], dtype=torch.long)), "range")
expect_value_error(lambda: model(torch.zeros(1, 2)), "integers")
expect_value_error(lambda: model(torch.empty(0, 2, dtype=torch.long)), "positive")
expect_value_error(lambda: model(token_ids, num_layers=1.5), "integer")
expect_value_error(lambda: TinyDenseCausalLM(V, D, I, 3, 1, Dh, L), "configuration")
attention_input = model.layers[0].input_norm(embedding)
expect_value_error(
    lambda: model.layers[0].self_attn(torch.randn(B, S, D + 1), positions, mask), "expected D"
)
expect_value_error(
    lambda: model.layers[0].self_attn(attention_input, positions, torch.zeros(S, S, dtype=torch.bool)),
    "mask must",
)
expect_value_error(lambda: require_residual("test residual", embedding, embedding[..., :1]), "shape mismatch")
expect_value_error(lambda: require_finite("test", torch.tensor([float("nan")])), "NaN or Inf")

print("all dense causal LM capstone checks passed")
```

### 综合任务结果解释

- 两层输入输出都保持 `[B,S,D]`，但第二层继续更新 values；shape 相同不代表层没有作用。
- 四种开关实验只控制 update。两条 update 都关闭时，单层严格成为 identity；各范数只记录观察结果，不要求固定大小关系。
- 每层都对自己的 scores 应用同一个形状兼容的 causal mask，因此未来概率严格为 0。
- 只改变最后 token 时，前 3 个 query 位置不能读取它，所以前缀 logits 保持。
- 扰动发生在第 2 层 MLP down projection。该层 `mlp_norm` 仍一致，首个稳定差异是 `layers.1.mlp_update`；layer output、final norm 和 logits 的差异都是传播结果。

### 综合任务验收

1. 所有断言通过，并能说明每个断言保护哪个边界。
2. 不看代码写出 `[2,4] -> [2,4,11]` 的完整 shape ledger。
3. 能解释为什么 residual update 开关不等于删除 residual connection。
4. 能从 trace 报告定位到层号、子层和第一个不一致边界。
5. 能说明 equal-prefix 实验验证了什么，以及它没有验证真实模型质量或生成能力。

<a id="acceptance"></a>
## 最终验收

### 模块练习

完成 `M1-E1` 至 `M6-E2`。每题先独立作答，再查看提示和答案。

### 概念题：至少答对 8/10

<a id="c1-question"></a>
**C1：** Pre-norm attention block 的三步是什么？它与 post-norm 的顺序有何不同？

<a id="c2-question"></a>
**C2：** 为什么第二条 residual 必须加回 `h` 而不是原始 `x`？

<a id="c3-question"></a>
**C3：** “关闭 attention residual 支路”在本教程中精确定义为什么？

<a id="c4-question"></a>
**C4：** 为什么 hidden-state 范数在 residual 相加后不保证增大？

<a id="c5-question"></a>
**C5：** 每层 attention 为什么都必须应用 causal mask？

<a id="c6-question"></a>
**C6：** 为什么 Decoder stack 的层应参数独立，但 shape contract 相同？

<a id="c7-question"></a>
**C7：** final RMSNorm 和每层两个 RMSNorm 分别位于哪里？

<a id="c8-question"></a>
**C8：** equal-prefix logits 实验怎样暴露未来信息泄漏？

<a id="c9-question"></a>
**C9：** 为什么首次偏差比最终 logits 最大误差更适合定位根因？

<a id="c10-question"></a>
**C10：** 第八周引入 MoE 时，Dense Decoder 的哪些部分保持不变，哪个边界开始改变？

### Shape 与调试题：至少答对 4/5

<a id="s1-question"></a>
**S1：** 对 `B=2,S=5,D=12,Hq=3,Hkv=1,Dh=4`，写出单层 attention 的 Q、K/V、scores、context merge、attention update 和 after-attention shape。

<a id="s2-question"></a>
**S2：** 对 `B=2,S=5,D=12,I=32`，写出 MLP norm、gate/up/product、MLP update 和 layer output shape。

<a id="s3-question"></a>
**S3：** 对 `B=3,S=6,V=100,D=16,L=4`，写出端到端 shape ledger。

<a id="s4-question"></a>
**S4：** Trace 首次差异为 `layers.2.attention_update`。列出它之前最后三个应一致的稳定边界，并写出之后最先受影响的边界。

<a id="s5-question"></a>
**S5：** 两个模型 logits 不同，但从 embedding 到 `layers.1.mlp_norm` 全部一致，第一个不同是 `layers.1.mlp_update`。最优先检查哪组参数和哪个中间张量？

评分规则：概念题至少 `8/10`，Shape 与调试题至少 `4/5`，综合任务全部断言通过。若未达标，按首个错误边界回到对应模块，不要直接重写整个模型。

<a id="hints"></a>
## 提示

<a id="m1-e1-hint"></a>**M1-E1：** 广播只判断尺寸能否扩展，不理解每个 hidden 通道应一一对应。

<a id="m1-e2-hint"></a>**M1-E2：** attention update 为零时 `h=x`；第二个 norm 仍作用于 `h`。

<a id="m2-e1-hint"></a>**M2-E1：** `Dh=D/Hq=4`；K/V 只保留一个 head。

<a id="m2-e2-hint"></a>**M2-E2：** mask Tensor 可以复用，但每层都有新的 scores 和 softmax。

<a id="m3-e1-hint"></a>**M3-E1：** norm 保持 D，三种中间门控张量使用 I，down 回到 D。

<a id="m3-e2-hint"></a>**M3-E2：** 展开 `h=x+attention_update` 后观察少了哪一项。

<a id="m4-e1-hint"></a>**M4-E1：** 按两次 norm、update、residual 的执行顺序列出。

<a id="m4-e2-hint"></a>**M4-E2：** 列表乘法复制的是 Python 引用，不是重新构造模块。

<a id="m5-e1-hint"></a>**M5-E1：** 第 0、1 层全部边界，以及第 2 层 attention norm 都在它之前。

<a id="m5-e2-hint"></a>**M5-E2：** Trace 是某一执行时刻的证据，不应继续保留 autograd 图或受后续原地修改影响。

<a id="m6-e1-hint"></a>**M6-E1：** 只有入口没有 D 轴，只有出口把 D 换成 V。

<a id="m6-e2-hint"></a>**M6-E2：** 一个是离散表索引，另两个参与线性代数和归一化。

<a id="c1-hint"></a>**C1：** 比较 `x+F(norm(x))` 与 `norm(x+F(x))`。

<a id="c2-hint"></a>**C2：** `h` 已包含第一条 update。

<a id="c3-hint"></a>**C3：** identity skip 保留，只把 applied update 置零。

<a id="c4-hint"></a>**C4：** 向量相加的范数取决于夹角。

<a id="c5-hint"></a>**C5：** 每层都会产生全新的 Q/K scores。

<a id="c6-hint"></a>**C6：** 同构不等于共享权重。

<a id="c7-hint"></a>**C7：** 单层两次，stack 结束后再一次。

<a id="c8-hint"></a>**C8：** 构造等长序列，只改变待验证前缀之后的 token。

<a id="c9-hint"></a>**C9：** 后续层会传播并变换早期误差。

<a id="c10-hint"></a>**C10：** 先替换每层的前馈子层，不动 attention 外壳。

<a id="s1-hint"></a>**S1：** Q head 数为 3，K/V head 数为 1，scores key/query 长度均为 5。

<a id="s2-hint"></a>**S2：** `[B,S,D] -> [B,S,I] -> [B,S,D]`。

<a id="s3-hint"></a>**S3：** 每层都保持 `[3,6,16]`，最后投影到 100。

<a id="s4-hint"></a>**S4：** 在同一层，attention update 前是 attention norm，后是 after-attention。

<a id="s5-hint"></a>**S5：** 首差发生在 SwiGLU 输出从 I 收缩回 D 的边界。

<a id="answers"></a>
## 参考答案

<a id="m1-e1-answer"></a>**M1-E1：** `[B,S,1]` 会沿 D 广播，让每个 token 的一个标量重复加到所有 hidden 通道；真实 attention/MLP update 应为逐通道 `[B,S,D]`。可执行不等于语义正确，因此 residual 两侧必须 shape 完全相同。

<a id="m1-e2-answer"></a>**M1-E2：** attention update 为零时 `h=x+0=x`；MLP 的 residual 基准是 `h`，post-attention norm 的输入也是 `h`。代码仍应按正常顺序计算，而不是改成另一种架构。

<a id="m2-e1-answer"></a>**M2-E1：** `Dh=4`；Q `[2,3,5,4]`，K/V `[2,1,5,4]`，scores `[2,3,5,5]`，attention update `[2,5,12]`。

<a id="m2-e2-answer"></a>**M2-E2：** 同一个 `[1,1,S,S]` mask 可以广播复用；但每层根据自己的 hidden 产生新的 scores，必须在该层 softmax 前再次 mask。第 0 层的 masked probabilities 不会自动约束后续层。

<a id="m3-e1-answer"></a>**M3-E1：** post-attention norm `[2,4,8]`；gate/up/product 都是 `[2,4,24]`；MLP update 与层输出都是 `[2,4,8]`。

<a id="m3-e2-answer"></a>**M3-E2：** 正确结果为 `x+attention_update+mlp_update`；错误公式只得到 `x+mlp_update`，丢失 attention update 的 identity 累积。两者外部 shape 都是 `[B,S,D]`，所以必须靠边界数值对齐发现。

<a id="m4-e1-answer"></a>**M4-E1：** `attention_norm`、`attention_update`、`after_attention`、`mlp_norm`、`mlp_update`、`layer_output`；其中 `after_attention=x+attention_update`，`layer_output=after_attention+mlp_update`。

<a id="m4-e2-answer"></a>**M4-E2：** `[layer]*3` 重复同一个对象引用，三次调用共享全部 Parameter；应分别构造三次，例如列表推导 `ModuleList([Layer(...) for _ in range(3)])`。

<a id="m5-e1-answer"></a>**M5-E1：** embedding、第 0 层全部六个边界、第 1 层全部六个边界和 `layers.2.attention_norm` 必须一致；首差才是 `layers.2.attention_update`。

<a id="m5-e2-answer"></a>**M5-E2：** `detach()` 让 trace 不保留 autograd 图，`clone()` 固化当时 values，避免后续原地修改或复用存储改变证据。Trace 用于诊断而不是继续计算。

<a id="m6-e1-answer"></a>**M6-E1：** token IDs `[3,6]`；embedding `[3,6,16]`；4 层每层输出都为 `[3,6,16]`；final norm `[3,6,16]`；logits `[3,6,100]`。

<a id="m6-e2-answer"></a>**M6-E2：** token IDs 是 embedding 表的离散行索引，必须为整数且位于词表范围；hidden/logits 要进行乘法、加法、归一化和 softmax 等连续数值运算，因此使用浮点 dtype。

<a id="c1-answer"></a>**C1：** Pre-norm 是 `normalized=norm(x)`、`update=F(normalized)`、`output=x+update`；post-norm 是先形成 `x+F(x)` 再 norm。两者运算顺序和数值都不同。

<a id="c2-answer"></a>**C2：** `h=x+attention_update` 已包含 attention 的结果；第二条 residual 加回 `h` 才能保留它并叠加 MLP update。加回原始 `x` 会丢失第一条 update。

<a id="c3-answer"></a>**C3：** 保留 identity skip 和正常子层输入，只把 applied attention update 设为全零。于是 attention block 输出等于输入；这不是删除 residual connection。

<a id="c4-answer"></a>**C4：** `||x+u||` 取决于 `x`、`u` 的大小和夹角；方向相反时可能减小，方向相近时可能增大。Residual 架构不提供单调范数保证。

<a id="c5-answer"></a>**C5：** 每层都从新的 hidden states 计算新的 Q/K 和 scores。只有在当前层 softmax 前屏蔽未来 key，当前层输出才保持因果性。

<a id="c6-answer"></a>**C6：** 所有层接收和输出相同 `[B,S,D]`，才能顺序堆叠；但不同层学习不同变换，必须拥有独立参数。意外共享会把 stack 变成重复应用同一组权重。

<a id="c7-answer"></a>**C7：** 每层 attention 前有 input RMSNorm，attention residual 后、MLP 前有 post-attention RMSNorm；所有 L 层结束后还有 final RMSNorm，再进入 LM Head。

<a id="c8-answer"></a>**C8：** 构造等长序列，保持前缀相同、只改变未来 token；若前缀位置 logits 改变，说明某层让 query 读取了未来信息，常见原因是 mask 缺失、方向错误或应用时机错误。

<a id="c9-answer"></a>**C9：** 早期微小错误会被后续 norm、attention、MLP 和 LM Head 传播、旋转和放大；最终误差无法指出起点。首差直接给出最早错误层与子层边界。

<a id="c10-answer"></a>**C10：** Attention、两条 residual、Decoder stack、final norm 和 LM Head 的外部接口保持；第八周先在当前 Dense SwiGLU 的输入位置计算 router logits 与 Top-K，为后续用多个 expert 替换单一 MLP update 做准备。

<a id="s1-answer"></a>**S1：** Q `[2,3,5,4]`；K/V `[2,1,5,4]`；scores `[2,3,5,5]`；context merge `[2,5,12]`；attention update `[2,5,12]`；after-attention `[2,5,12]`。

<a id="s2-answer"></a>**S2：** MLP norm `[2,5,12]`；gate/up/product `[2,5,32]`；MLP update `[2,5,12]`；layer output `[2,5,12]`。

<a id="s3-answer"></a>**S3：** token IDs `[3,6] -> embedding [3,6,16] -> 4 x layer output [3,6,16] -> final norm [3,6,16] -> logits [3,6,100]`。

<a id="s4-answer"></a>**S4：** 紧邻之前三个稳定边界是 `layers.1.output`、`layers.2.attention_norm`，再向前是 `layers.1.mlp_update`；之后最先受影响的是 `layers.2.after_attention`。更完整比较中，所有更早 key 都必须一致。

<a id="s5-answer"></a>**S5：** 最优先检查第 2 层 SwiGLU 的 `down_proj.weight [D,I]`（以及是否加载到正确层），再检查该层 `product [B,S,I]` 与 down projection 的矩阵方向；gate/up 和 `mlp_norm` 已由更早一致边界缩小了嫌疑范围。

<a id="glossary"></a>
## 数据流、形状与调试速查

### 单层顺序

| 顺序 | 边界 | Shape |
| --- | --- | --- |
| 1 | layer input | `[B,S,D]` |
| 2 | input RMSNorm | `[B,S,D]` |
| 3 | causal GQA update | `[B,S,D]` |
| 4 | after attention residual | `[B,S,D]` |
| 5 | post-attention RMSNorm | `[B,S,D]` |
| 6 | SwiGLU product | `[B,S,I]` |
| 7 | MLP update | `[B,S,D]` |
| 8 | layer output | `[B,S,D]` |

### Attention 形状

| Tensor | Shape |
| --- | --- |
| Q | `[B,Hq,S,Dh]` |
| K/V | `[B,Hkv,S,Dh]` |
| repeated K/V | `[B,Hq,S,Dh]` |
| scores/probabilities | `[B,Hq,S,S]` |
| merged context/update | `[B,S,D]` |
| causal mask | `[1,1,S,S]`，`True` 表示屏蔽 |

### 端到端形状

```text
token IDs [B,S]
embedding [B,S,D]
L x Decoder Layer [B,S,D]
final norm [B,S,D]
LM Head logits [B,S,V]
```

### 首次偏差流程

1. 比较 trace key 集合与执行顺序。
2. 逐项比较 shape、dtype 和 device。
3. 拒绝 NaN/Inf。
4. 用明确 `atol/rtol` 比较 values。
5. 报告第一个不同 key、最大绝对差和两侧 shape。
6. 进入该子层检查更细的 Q/K/V、probabilities 或 gate/up/product。

### 常见错误

| 现象 | 优先检查 |
| --- | --- |
| shape 正确但从第一层 MLP 后偏离 | 第二条 residual 是否加回 `h` |
| 前缀 logits 随未来 token 改变 | 每层 causal mask 的方向和应用时机 |
| 多层输出异常相似 | 是否重复引用同一个 layer 实例 |
| `layers.k.attention_update` 首差 | input norm、Q/K/V、QK Norm、RoPE、GQA、mask |
| `layers.k.mlp_update` 首差 | post-attention norm、gate/up/product、down projection |
| final norm 前一致、logits 不同 | LM Head weight、dtype/device、权重绑定配置 |
| residual 相加意外广播 | 两侧 shape 是否完全相同 |

### 权重绑定可选扩展

Embedding 和无 bias LM Head 的参数 shape 都可为 `[V,D]`。若目标配置要求 tying，可让 LM Head 使用 embedding 的同一个 Parameter，从而减少一份 `V*D` 参数；但这会改变状态管理和权重加载契约。本周主线保持独立，接入真实模型时再以目标配置为准。

<a id="next-week"></a>
## Week 8 预告

第七周得到的 Dense Causal LM 已经能端到端 forward，并能把偏差定位到具体层和子层。第八周先不立即重写整个 Decoder，而是在当前 MLP 输入边界把 `[B,S,D]` 展平为 `[N,D]`，计算 router logits `[N,E]` 和 Top-K indices/weights `[N,K]`。Attention、两条 residual、stack、final norm 和 LM Head 接口暂时保持不变。
