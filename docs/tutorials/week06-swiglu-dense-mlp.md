# 第六周教程：SwiGLU 与 Dense MLP

> 本周目标：解决 attention 只负责 token 间信息混合、却缺少逐 token 特征变换的问题；从普通 Dense MLP 出发，逐步实现 Qwen3-style 无 bias SwiGLU，并核算参数与中间激活。

## 目录

1. [本周在完整推理中的位置](#inference-position)
2. [学习目标](#goals)
3. [学习方式](#method)
4. [环境、记号与边界](#environment)
5. [模块 1：Dense MLP 基线](#module-1)
6. [模块 2：SiLU 与激活函数](#module-2)
7. [模块 3：SwiGLU 门控路径](#module-3)
8. [模块 4：函数版与 nn.Module 版 SwiGLU](#module-4)
9. [模块 5：参数量与理论内存](#module-5)
10. [模块 6：中间激活、输出分布与 I 扫描](#module-6)
11. [综合任务](#capstone)
12. [最终验收](#acceptance)
13. [提示](#hints)
14. [参考答案](#answers)
15. [公式、形状、内存与错误速查](#glossary)
16. [Week 7 预告](#next-week)

<a id="inference-position"></a>
## 本周在完整推理中的位置

第三周建立了 token 到 logits 的接口，第四、五周完成了带 causal GQA、RMSNorm、RoPE 和 QK Norm 的 attention 路径。Attention 能让 token 读取其他位置，但还没有回答：**每个 token 读完上下文后，怎样独立地扩展、筛选并重组自己的特征？**

Decoder 中通常由前馈子层完成这项工作：

```text
hidden [B,S,D]
-> attention 子层：沿 token 轴混合信息
-> MLP 子层：每个 token 独立变换最后一维特征
-> hidden [B,S,D]
```

本周只实现独立 MLP 子层。虽然输入输出都是 `[B,S,D]`，但这不等于已经实现残差连接；`x + mlp(x)`、pre-norm 和完整 Decoder Layer 留到第七周。

<a id="goals"></a>
## 学习目标

完成后，你应该能：

1. 解释 attention 的 token mixing 与 MLP 的逐 token feature transformation 有何不同。
2. 写出普通两层 Dense MLP 和 SwiGLU 的公式与 shape 流。
3. 写出 gate、up、down 三个无 bias 投影的矩阵方向。
4. 明确区分显式 `x @ W` 权重布局与 `nn.Linear.weight` 存储布局。
5. 解释 `silu(gate(x)) * up(x)` 是逐元素乘法，且两分支 shape 必须完全一致。
6. 实现函数版和 `nn.Module` 版 SwiGLU，并比较所有关键中间值。
7. 区分 Parameter、普通属性、输入输出激活和 forward 中间张量。
8. 推导无 bias SwiGLU 参数量 `3DI` 与理论字节数。
9. 建立 forward shape ledger，并说明同时存活张量为什么影响峰值内存。
10. 扫描 `I`，观察参数量和主要中间激活的线性变化。
11. 在受控条件下比较 ReLU MLP、GELU MLP 与 SwiGLU 的输出统计，而不把小实验推广为模型质量结论。

<a id="method"></a>
## 学习方式：Predict-Run-Explain

每个实验都按同一闭环进行：

1. **Predict**：先写公式、shape、权重布局、预期 values 或预期错误。
2. **Run**：独立运行当前 Python 围栏，观察打印结果与断言。
3. **Explain**：脱离代码说明收缩轴、保留轴、参数归属和内存类别。

推荐调试顺序：`D/I 合法性 -> 输入最后一维 -> 权重方向 -> gate/up shape 完全相等 -> dtype/device -> finite -> 数值对齐 -> 内存分类`。

<a id="environment"></a>
## 环境、记号与边界

仓库锁定环境可用时，从根目录运行：

```bash
uv sync --locked --python 3.11.15
uv run python scripts/check_environment.py
uv run pytest
```

本周所有 Python 围栏都自行 import、定义函数和构造固定数据，可以单独保存并运行；只使用 CPU FP32、固定种子和小张量，不下载模型、Tokenizer、checkpoint 或数据集。

统一记号：

```text
B    batch size
S    sequence length
D    hidden width
I    intermediate width
```

本周主线采用 Qwen3-style 无 bias SwiGLU：

```text
x [B,S,D]
-> gate/up [B,S,I]
-> SiLU(gate) [B,S,I]
-> elementwise product [B,S,I]
-> down [B,S,D]
```

两种等价但不能混写的权重布局：

| 投影 | 显式右乘 `x @ W` | `nn.Linear(in,out).weight` | 模块等价计算 |
| --- | --- | --- | --- |
| gate | `W_gate [D,I]` | `[I,D]` | `x @ gate.weight.T` |
| up | `W_up [D,I]` | `[I,D]` | `x @ up.weight.T` |
| down | `W_down [I,D]` | `[D,I]` | `hidden @ down.weight.T` |

教学 Tensor 函数版统一使用右乘布局；模块版统一使用 PyTorch 的 `[out_features,in_features]` 存储布局。复制同一组权重时必须转置。

范围边界：不加入 attention、residual、pre-norm、dropout、训练、反向传播、MoE router、专家分发、KV Cache、fused kernel 或性能 benchmark。真实模型的 `D/I`、bias、激活函数和权重名必须来自目标 `config.json` 与参考实现，不能把本周教学值当作官方配置。

<a id="module-1"></a>
## 模块 1：Dense MLP 基线

### 1.1 为什么 attention 之外还需要 MLP

Attention 主要沿位置轴读取和汇总信息；MLP 对每个 token 使用同一组权重，只改变最后一维特征。普通无 bias 两层 MLP 可以写成：

```text
h = activation(x @ W_up)    W_up [D,I]
y = h @ W_down              W_down [I,D]
```

批量矩阵乘法只收缩最后一维，`B/S` 原样保留。若改变一个 token，其他 token 的输出不应变化，因为公式中没有沿 `S` 求和。

### 1.2 Predict-Run-Explain：逐 token 与批量路径

运行前预测：`x [2,3,4] @ W_up [4,5]` 和第二次投影的 shape；再判断循环路径与批量路径是否应完全一致。

```python
import torch

torch.manual_seed(6)

def dense_mlp(x, w_up, w_down):
    if x.ndim != 3:
        raise ValueError(f"expected x [B,S,D], got {tuple(x.shape)}")
    if w_up.ndim != 2 or w_down.ndim != 2:
        raise ValueError("weights must be rank-2 tensors")
    d, i = w_up.shape
    if d <= 0 or i <= 0:
        raise ValueError(f"D and I must be positive, got D={d}, I={i}")
    if x.shape[-1] != d:
        raise ValueError(f"input width must be {d}, got shape {tuple(x.shape)}")
    if tuple(w_down.shape) != (i, d):
        raise ValueError(f"w_down must be {(i, d)}, got {tuple(w_down.shape)}")
    if x.dtype != w_up.dtype or x.dtype != w_down.dtype:
        raise TypeError("x and weights must use the same dtype")
    if x.device != w_up.device or x.device != w_down.device:
        raise ValueError("x and weights must use the same device")
    hidden = torch.relu(x @ w_up)
    output = hidden @ w_down
    if not torch.isfinite(hidden).all() or not torch.isfinite(output).all():
        raise ValueError("Dense MLP produced NaN or Inf")
    return hidden, output

B, S, D, I = 2, 3, 4, 5
x = torch.arange(B * S * D, dtype=torch.float32).reshape(B, S, D) / 10
w_up = torch.arange(D * I, dtype=torch.float32).reshape(D, I) / 20 - 0.4
w_down = torch.arange(I * D, dtype=torch.float32).reshape(I, D) / 25 - 0.3

hidden, batched = dense_mlp(x, w_up, w_down)
loop = torch.empty_like(batched)
for b in range(B):
    for s in range(S):
        _, loop[b, s] = dense_mlp(x[b:b + 1, s:s + 1], w_up, w_down)

print("shape flow:", tuple(x.shape), tuple(hidden.shape), tuple(batched.shape))
print("max loop difference:", (batched - loop).abs().max().item())
assert hidden.shape == (B, S, I)
assert batched.shape == (B, S, D)
torch.testing.assert_close(batched, loop, atol=0, rtol=0)

changed = x.clone()
changed[0, 1] += 10
_, changed_output = dense_mlp(changed, w_up, w_down)
unchanged_positions = torch.ones(B, S, dtype=torch.bool)
unchanged_positions[0, 1] = False
torch.testing.assert_close(
    changed_output[unchanged_positions], batched[unchanged_positions], atol=0, rtol=0
)

for bad_x, bad_down in (
    (torch.ones(2, 3, 6), w_down),
    (x, torch.ones(I + 1, D)),
):
    try:
        dense_mlp(bad_x, w_up, bad_down)
    except (ValueError, TypeError) as error:
        print("controlled error:", error)
```

### 1.3 解释与常见误区

`[B,S,D] @ [D,I]` 对每个 `[D]` token 向量独立执行同一个投影。循环对照证明批量写法没有混合 token 轴；token 独立不表示不同 token 输出必然不同，因为相同输入经过相同权重会得到相同输出。

- 把 `S` 当作矩阵乘法收缩轴，会错误混合位置。
- 只看输入输出同为 `[B,S,D]`，忽略中间扩展为 `I`。
- 第二层误用 `[D,I]`，无法把 `I` 收回 `D`。
- 将“共享权重”误解成“token 互相读取”。

### 1.4 练习

<a id="m1-e1-question"></a>
**M1-E1**（[提示](#m1-e1-hint) · [答案](#m1-e1-answer)）：`x [3,5,8]`、`W_up [8,20]`、`W_down [20,8]` 时，写出两个中间 shape，并指出哪一维被收缩。

<a id="m1-e2-question"></a>
**M1-E2**（[提示](#m1-e2-hint) · [答案](#m1-e2-answer)）：为什么改变 `x[0,2]` 不应改变其他 token 的 Dense MLP 输出？什么操作一旦加入就可能破坏这个性质？

### 模块 1 验收

1. 能写出 `[B,S,D] -> [B,S,I] -> [B,S,D]`。
2. 能用逐 token 循环证明不混合 `S`。
3. 能让错误信息同时包含实际和期望 shape。

<a id="module-2"></a>
## 模块 2：SiLU 与激活函数

### 2.1 SiLU 的公式

SiLU（Sigmoid Linear Unit）定义为：

```text
SiLU(x) = x * sigmoid(x)
sigmoid(x) = 1 / (1 + exp(-x))
```

负数不会像 ReLU 一样全部截成 0，而是被平滑抑制；0 映射为 0；较大正数的 sigmoid 接近 1，因此输出接近输入。SiLU 本身没有 Parameter。

### 2.2 Predict-Run-Explain：手写与原语对齐

```python
import torch
import torch.nn.functional as F

torch.manual_seed(6)

def handwritten_silu(x):
    if not x.is_floating_point():
        raise TypeError(f"SiLU expects floating tensor, got {x.dtype}")
    output = x * torch.sigmoid(x)
    if output.shape != x.shape or output.dtype != x.dtype or output.device != x.device:
        raise RuntimeError("SiLU changed tensor metadata")
    if not torch.isfinite(output).all():
        raise ValueError("SiLU produced NaN or Inf")
    return output

x = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0], dtype=torch.float32)
manual = handwritten_silu(x)
reference = F.silu(x)
relu = F.relu(x)
gelu = F.gelu(x)

print("x:   ", x)
print("ReLU:", relu)
print("GELU:", gelu)
print("SiLU:", manual)
torch.testing.assert_close(manual, reference, atol=1e-7, rtol=1e-6)
assert manual.shape == x.shape
assert manual.dtype == x.dtype and manual.device == x.device
assert torch.isfinite(relu).all() and torch.isfinite(gelu).all()

try:
    handwritten_silu(torch.tensor([1, 2, 3]))
except TypeError as error:
    print("controlled error:", error)
```

### 2.3 怎样解释比较结果

对 `x=-1`，SiLU 为 `-1*sigmoid(-1)`，约为 `-0.269`；ReLU 为 0。GELU 与 SiLU 都是平滑激活，但公式不同。这个五点实验只说明局部函数形状，不证明某种激活在所有模型、训练任务或硬件上更优。

- 激活不改变 shape，不等于不改变数值分布。
- SiLU 中的 `*` 是逐元素乘法。
- 激活函数无可学习参数；围绕它的线性投影才有参数。
- 不要用单次随机输出判断训练后模型质量。

### 2.4 练习

<a id="m2-e1-question"></a>
**M2-E1**（[提示](#m2-e1-hint) · [答案](#m2-e1-answer)）：手算 `SiLU(0)`；说明 `x` 很大且为正时输出为什么接近 `x`。

<a id="m2-e2-question"></a>
**M2-E2**（[提示](#m2-e2-hint) · [答案](#m2-e2-answer)）：比较负输入经过 ReLU 与 SiLU 的差别，并说明为什么这不足以推出模型质量结论。

### 模块 2 验收

1. 能写出 `x*sigmoid(x)` 并与 `F.silu` 对齐。
2. 能验证 shape、dtype、device 和 finite。
3. 能解释激活函数没有 Parameter。

<a id="module-3"></a>
## 模块 3：SwiGLU 门控路径

### 3.1 从单分支扩展到 gate/up 双分支

普通 MLP 只有一条扩展投影。SwiGLU 使用两个独立投影：

```text
gate = x @ W_gate                [B,S,I]
up   = x @ W_up                  [B,S,I]
activated_gate = SiLU(gate)      [B,S,I]
hidden = activated_gate * up     [B,S,I]
```

最后一个 `*` 必须逐位置、逐中间通道相乘。gate 和 up 必须具有完全相同的 shape；“可以广播”不是充分条件，因为 `[B,S,1]` 广播到 `[B,S,I]` 会把一个 gate 值复制到全部中间通道，改变门控语义。

### 3.2 Predict-Run-Explain：可手算门控

```python
import torch
import torch.nn.functional as F

def gated_hidden(x, w_gate, w_up):
    if x.ndim != 3:
        raise ValueError(f"expected x [B,S,D], got {tuple(x.shape)}")
    if w_gate.ndim != 2 or w_up.ndim != 2:
        raise ValueError("gate/up weights must be rank 2")
    if w_gate.shape != w_up.shape:
        raise ValueError(
            f"gate/up weights must match exactly, got {tuple(w_gate.shape)} and {tuple(w_up.shape)}"
        )
    if x.shape[-1] != w_gate.shape[0]:
        raise ValueError(
            f"input width {x.shape[-1]} does not match weight input width {w_gate.shape[0]}"
        )
    if x.dtype != w_gate.dtype or x.dtype != w_up.dtype:
        raise TypeError("x, gate weight and up weight must share dtype")
    if x.device != w_gate.device or x.device != w_up.device:
        raise ValueError("x, gate weight and up weight must share device")
    gate = x @ w_gate
    up = x @ w_up
    if gate.shape != up.shape:
        raise ValueError(f"gate/up outputs must match, got {tuple(gate.shape)} and {tuple(up.shape)}")
    activated_gate = F.silu(gate)
    hidden = activated_gate * up
    if not all(torch.isfinite(t).all() for t in (gate, up, activated_gate, hidden)):
        raise ValueError("gated path produced NaN or Inf")
    return gate, up, activated_gate, hidden

x = torch.tensor([[[1.0, 2.0], [-1.0, 1.0]]], dtype=torch.float32)
w_gate = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
w_up = torch.tensor([[2.0, 1.0, 0.5], [1.0, -1.0, 2.0]])
gate, up, activated_gate, hidden = gated_hidden(x, w_gate, w_up)

print("gate:\n", gate)
print("up:\n", up)
print("activated gate:\n", activated_gate)
print("hidden:\n", hidden)
assert gate.shape == up.shape == activated_gate.shape == hidden.shape == (1, 2, 3)
torch.testing.assert_close(hidden, F.silu(gate) * up, atol=0, rtol=0)

def strict_elementwise_gate(gate_branch, up_branch):
    if gate_branch.shape != up_branch.shape:
        raise ValueError(
            f"refuse broadcasting: gate {tuple(gate_branch.shape)} != up {tuple(up_branch.shape)}"
        )
    return gate_branch * up_branch

try:
    strict_elementwise_gate(torch.ones(1, 2, 1), torch.ones(1, 2, 3))
except ValueError as error:
    print("controlled broadcast error:", error)
```

### 3.3 gate 怎样控制 up

若某个 gate pre-activation 很负，SiLU 后的数值通常较小且为负，该位置的 up 通道被抑制并可能改变符号；若 gate 很正，SiLU 接近线性，该 up 通道通过得更多。门控发生在每个 token、每个 `I` 通道，不是给整个 token 只分配一个标量。

- `gate @ up` 是错误的矩阵乘法，不是门控。
- gate/up 权重 shape 相同，但它们是独立参数，不要求 values 相同。
- 不能用 `expand` 掩盖分支宽度错误。
- shape 相等后仍要检查 dtype、device 和 finite。

### 3.4 练习

<a id="m3-e1-question"></a>
**M3-E1**（[提示](#m3-e1-hint) · [答案](#m3-e1-answer)）：`x [2,4,8]` 与两个 `[8,16]` 显式权重产生哪些 shape？逐元素乘法的第 `[b,s,i]` 项表示什么？

<a id="m3-e2-question"></a>
**M3-E2**（[提示](#m3-e2-hint) · [答案](#m3-e2-answer)）：为什么 gate `[2,4,1]` 与 up `[2,4,16]` 虽可广播，主线实现仍必须拒绝？

### 模块 3 验收

1. 能画出 gate/up/SiLU/逐元素乘法。
2. 能证明四个中间张量都是 `[B,S,I]`。
3. 能在乘法前拒绝广播。

<a id="module-4"></a>
## 模块 4：函数版与 `nn.Module` 版 SwiGLU

### 4.1 完整公式与权重转置

函数版右乘布局：

```text
gate = x @ W_gate       W_gate [D,I]
up   = x @ W_up         W_up   [D,I]
hidden = silu(gate) * up
out  = hidden @ W_down  W_down [I,D]
```

模块版 `Linear(D,I)` 保存 `[I,D]`，内部等价于 `x @ weight.T`；`Linear(I,D)` 保存 `[D,I]`。因此从函数版复制到模块版时使用 `.copy_(W.T)`。

### 4.2 Predict-Run-Explain：两条路径与模块状态

```python
import torch
from torch import nn
import torch.nn.functional as F

def validate_dims(d, i):
    if not isinstance(d, int) or not isinstance(i, int) or d <= 0 or i <= 0:
        raise ValueError(f"D and I must be positive integers, got D={d}, I={i}")

def functional_swiglu(x, w_gate, w_up, w_down):
    if x.ndim != 3:
        raise ValueError(f"expected x [B,S,D], got {tuple(x.shape)}")
    d, i = w_gate.shape
    validate_dims(d, i)
    expected = {"w_gate": (d, i), "w_up": (d, i), "w_down": (i, d)}
    actual = {"w_gate": tuple(w_gate.shape), "w_up": tuple(w_up.shape), "w_down": tuple(w_down.shape)}
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(f"{name} must be {expected[name]}, got {actual[name]}")
    if x.shape[-1] != d:
        raise ValueError(f"expected input width {d}, got {tuple(x.shape)}")
    tensors = (w_gate, w_up, w_down)
    if any(t.dtype != x.dtype for t in tensors):
        raise TypeError("input and all weights must share dtype")
    if any(t.device != x.device for t in tensors):
        raise ValueError("input and all weights must share device")
    gate = x @ w_gate
    up = x @ w_up
    if gate.shape != up.shape:
        raise ValueError("gate/up outputs must have exactly equal shapes")
    activated_gate = F.silu(gate)
    hidden = activated_gate * up
    output = hidden @ w_down
    if not all(torch.isfinite(t).all() for t in (gate, up, activated_gate, hidden, output)):
        raise ValueError("SwiGLU produced NaN or Inf")
    return gate, up, activated_gate, hidden, output

class TinySwiGLU(nn.Module):
    def __init__(self, d, i):
        super().__init__()
        validate_dims(d, i)
        self.d = d
        self.i = i
        self.description = "three no-bias projections"  # 普通属性，不进入 state_dict。
        self.gate_proj = nn.Linear(d, i, bias=False)
        self.up_proj = nn.Linear(d, i, bias=False)
        self.down_proj = nn.Linear(i, d, bias=False)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.d:
            raise ValueError(f"expected x [B,S,{self.d}], got {tuple(x.shape)}")
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if gate.shape != up.shape:
            raise ValueError(f"gate/up shapes differ: {tuple(gate.shape)} vs {tuple(up.shape)}")
        activated_gate = F.silu(gate)
        hidden = activated_gate * up
        output = self.down_proj(hidden)
        if not all(torch.isfinite(t).all() for t in (gate, up, activated_gate, hidden, output)):
            raise ValueError("module SwiGLU produced NaN or Inf")
        return gate, up, activated_gate, hidden, output

torch.manual_seed(6)
B, S, D, I = 2, 3, 4, 6
x = torch.randn(B, S, D, dtype=torch.float32)
w_gate = torch.randn(D, I, dtype=torch.float32) * 0.2
w_up = torch.randn(D, I, dtype=torch.float32) * 0.2
w_down = torch.randn(I, D, dtype=torch.float32) * 0.2

model = TinySwiGLU(D, I).eval()
with torch.no_grad():
    model.gate_proj.weight.copy_(w_gate.T)
    model.up_proj.weight.copy_(w_up.T)
    model.down_proj.weight.copy_(w_down.T)

with torch.inference_mode():
    functional = functional_swiglu(x, w_gate, w_up, w_down)
    module = model(x)
    train_mode_output = model.train()(x)[-1]
    eval_mode_output = model.eval()(x)[-1]

for left, right in zip(functional, module):
    torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
torch.testing.assert_close(train_mode_output, eval_mode_output, atol=0, rtol=0)

expected_keys = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
assert list(model.state_dict().keys()) == expected_keys
assert tuple(model.gate_proj.weight.shape) == (I, D)
assert tuple(model.up_proj.weight.shape) == (I, D)
assert tuple(model.down_proj.weight.shape) == (D, I)
assert "description" not in model.state_dict()
print("state keys:", list(model.state_dict().keys()))
print("functional/module output shape:", tuple(functional[-1].shape))

for action in (
    lambda: TinySwiGLU(0, I),
    lambda: model(torch.ones(B, S, D + 1)),
    lambda: functional_swiglu(x, w_gate, w_up, torch.ones(D, I)),
):
    try:
        action()
    except (ValueError, TypeError) as error:
        print("controlled error:", error)
```

### 4.3 Parameter、属性与推理模式

三个 `nn.Linear.weight` 是 Parameter，会出现在 `parameters()` 和 `state_dict()` 中；`d/i/description` 是普通 Python 属性，不会自动保存为模型权重。`.eval()` 切换模块行为，`torch.inference_mode()` 关闭 autograd 跟踪；本周没有 dropout 或 batch-dependent training behavior，所以 train/eval 输出相同，但两者职责仍不同。

- `.eval()` 不等于关闭梯度。
- values 相同不等于共享同一个 Parameter。
- 复制显式权重时忘记 `.T` 会造成方向错误。
- 只比较最终输出可能掩盖 gate/up 分支交换或中间错误。

### 4.4 练习

<a id="m4-e1-question"></a>
**M4-E1**（[提示](#m4-e1-hint) · [答案](#m4-e1-answer)）：`D=8,I=24` 时，写出函数版三权重与模块版三个 `.weight` shape，并说明转置关系。

<a id="m4-e2-question"></a>
**M4-E2**（[提示](#m4-e2-hint) · [答案](#m4-e2-answer)）：为什么普通字符串属性不进入 `state_dict`？`.eval()` 与 `inference_mode()` 分别解决什么问题？

### 模块 4 验收

1. 能让函数版与模块版五类结果对齐。
2. 能列出恰好三个 state key。
3. 能拒绝非法维度、输入宽度和错误权重方向。

<a id="module-5"></a>
## 模块 5：参数量与理论内存

### 5.1 参数量公式

无 bias SwiGLU 有三个矩阵：

```text
gate: D*I
up:   D*I
down: I*D
total = 3DI
```

普通无 bias 两层 MLP 只有 up/down，因此是 `2DI`。SwiGLU 多出的 `DI` 来自独立 gate 投影。若讨论带 bias 变体，额外 bias 参数为 `I + I + D`，但它不是本周主线。

理论参数字节数为 `parameter_count * bytes_per_element`。FP32 为 4 bytes，FP16/BF16 为 2 bytes，INT8 为 1 byte；这里的 INT8 只做体积练习，不代表量化格式没有 scale、zero point、padding 或其他元数据。

### 5.2 Predict-Run-Explain：公式与模块核对

```python
import torch
from torch import nn

def swiglu_parameter_count(d, i, bias=False):
    if not isinstance(d, int) or not isinstance(i, int) or d <= 0 or i <= 0:
        raise ValueError(f"D and I must be positive integers, got D={d}, I={i}")
    count = 3 * d * i
    if bias:
        count += 2 * i + d
    return count

def theoretical_bytes(count, bits_per_element):
    if count < 0 or bits_per_element <= 0 or bits_per_element % 8 != 0:
        raise ValueError("count must be non-negative and bits must be positive whole bytes")
    return count * (bits_per_element // 8)

class TinySwiGLU(nn.Module):
    def __init__(self, d, i):
        super().__init__()
        if d <= 0 or i <= 0:
            raise ValueError("D and I must be positive")
        self.gate = nn.Linear(d, i, bias=False)
        self.up = nn.Linear(d, i, bias=False)
        self.down = nn.Linear(i, d, bias=False)

D, I = 4, 6
model = TinySwiGLU(D, I)
formula_count = swiglu_parameter_count(D, I)
module_count = sum(parameter.numel() for parameter in model.parameters())

print("Dense MLP count:", 2 * D * I)
print("SwiGLU count:", formula_count)
for name, bits in (("FP32", 32), ("FP16/BF16", 16), ("INT8 volume exercise", 8)):
    print(name, theoretical_bytes(formula_count, bits), "bytes")

assert formula_count == module_count == 3 * D * I == 72
assert theoretical_bytes(formula_count, 32) == 288
assert swiglu_parameter_count(D, I, bias=True) == 72 + 6 + 6 + 4

for bad in ((0, I), (D, -1)):
    try:
        swiglu_parameter_count(*bad)
    except ValueError as error:
        print("controlled error:", error)
```

### 5.3 理论体积不等于进程峰值

`72*4=288 bytes` 只包含 FP32 参数数据。它不包含输入、输出、gate/up/activated gate/product、梯度、优化器状态、allocator 缓存、Python 对象、kernel workspace 或框架开销。本周是 inference-only，因此不把梯度和优化器混入参数体积，但仍要知道真实进程内存大于单一理论类别。

- 参数量与 `B/S` 无关；激活量与 `B/S` 有关。
- bias 只能在明确采用 bias 时计入。
- INT8 理论字节不等于任意真实量化 checkpoint 的最终文件大小。
- `element_size()` 反映 Tensor dtype，不自动代表 packed 量化格式。

### 5.4 练习

<a id="m5-e1-question"></a>
**M5-E1**（[提示](#m5-e1-hint) · [答案](#m5-e1-answer)）：无 bias SwiGLU 取 `D=8,I=32`，求参数量与 FP32/BF16 理论字节数。

<a id="m5-e2-question"></a>
**M5-E2**（[提示](#m5-e2-hint) · [答案](#m5-e2-answer)）：列出至少四类不包含在“参数理论体积”中的内存，并解释为什么不能把参数 bytes 当作峰值内存。

### 模块 5 验收

1. 能推导 `3DI` 并对比 `2DI`。
2. 能核对 `D=4,I=6` 得 72 参数、FP32 288 bytes。
3. 能区分参数、激活与框架/运行时开销。

<a id="module-6"></a>
## 模块 6：中间激活、输出分布与 I 扫描

### 6.1 Forward shape ledger

对固定 `B/S/D/I`，教学版显式中间张量为：

| 名称 | shape | 元素数 | FP32 理论字节 |
| --- | --- | ---: | ---: |
| input | `[B,S,D]` | `BSD` | `4BSD` |
| gate | `[B,S,I]` | `BSI` | `4BSI` |
| up | `[B,S,I]` | `BSI` | `4BSI` |
| activated gate | `[B,S,I]` | `BSI` | `4BSI` |
| product | `[B,S,I]` | `BSI` | `4BSI` |
| output | `[B,S,D]` | `BSD` | `4BSD` |

把表中每行相加只能得到“这些逻辑 Tensor 的理论体积和”，不一定等于峰值：实际峰值取决于哪些张量同时存活、autograd 是否开启、算子是否融合、内存复用与 allocator。教学代码为了观察中间值会故意同时保留更多 Tensor。

### 6.2 Predict-Run-Explain：I 扫描与分布统计

```python
import torch
import torch.nn.functional as F

def stats(name, tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return {
        "name": name,
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "finite": True,
    }

def run_controlled(width, intermediate, seed=6):
    if width <= 0 or intermediate <= 0:
        raise ValueError("D and I must be positive")
    torch.manual_seed(seed)
    B, S = 2, 3
    x = torch.randn(B, S, width, dtype=torch.float32)
    scale = 0.2
    w1 = torch.randn(width, intermediate) * scale
    w2 = torch.randn(intermediate, width) * scale
    w_gate = torch.randn(width, intermediate) * scale
    w_up = torch.randn(width, intermediate) * scale
    w_down = torch.randn(intermediate, width) * scale
    relu_output = F.relu(x @ w1) @ w2
    gelu_output = F.gelu(x @ w1) @ w2
    gate = x @ w_gate
    up = x @ w_up
    if gate.shape != up.shape:
        raise ValueError("gate/up output shapes must match exactly")
    activated_gate = F.silu(gate)
    product = activated_gate * up
    swiglu_output = product @ w_down
    tensors = (x, gate, up, activated_gate, product, swiglu_output)
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("forward contains NaN or Inf")
    ledger = {
        "input": (tuple(x.shape), x.numel(), x.numel() * x.element_size()),
        "gate": (tuple(gate.shape), gate.numel(), gate.numel() * gate.element_size()),
        "up": (tuple(up.shape), up.numel(), up.numel() * up.element_size()),
        "activated_gate": (tuple(activated_gate.shape), activated_gate.numel(), activated_gate.numel() * activated_gate.element_size()),
        "product": (tuple(product.shape), product.numel(), product.numel() * product.element_size()),
        "output": (tuple(swiglu_output.shape), swiglu_output.numel(), swiglu_output.numel() * swiglu_output.element_size()),
    }
    return ledger, [stats("ReLU MLP", relu_output), stats("GELU MLP", gelu_output), stats("SwiGLU", swiglu_output)]

base_ledger, summaries = run_controlled(width=4, intermediate=6)
double_ledger, _ = run_controlled(width=4, intermediate=12)

for row in summaries:
    print(row)
print("I=6 ledger:", base_ledger)
print("I=12 gate ledger:", double_ledger["gate"])

assert base_ledger["gate"] == ((2, 3, 6), 36, 144)
assert base_ledger["output"] == ((2, 3, 4), 24, 96)
assert double_ledger["gate"][1] == 2 * base_ledger["gate"][1]
assert double_ledger["gate"][2] == 2 * base_ledger["gate"][2]
assert 3 * 4 * 12 == 2 * (3 * 4 * 6)
```

### 6.3 怎样解释扫描结果

固定 `B/S/D` 时，每个 `[B,S,I]` 张量的元素数与 `I` 成正比，`3DI` 参数量也与 `I` 成正比。输出 `[B,S,D]` 不随 `I` 改变。ReLU、GELU 与 SwiGLU 的均值、标准差、最小值和最大值不同，是因为激活和分支结构不同；这些固定随机权重没有训练，统计只用于观察结构，不用于评价语言建模质量，也不是硬件性能 benchmark。

- 扫描 `I` 时要保持其他条件和随机种子受控。
- “四个 `[B,S,I]` 张量”不表示生产 fused 实现必然全部物化并同时存活。
- 参数增长和激活增长都线性，不表示运行时间必然严格线性。
- 比较分布时必须先检查 finite。

### 6.4 练习

<a id="m6-e1-question"></a>
**M6-E1**（[提示](#m6-e1-hint) · [答案](#m6-e1-answer)）：`B=2,S=5,D=8,I=32` 时，求单个 gate Tensor 的元素数与 FP32 字节数，以及 output 的对应数值。

<a id="m6-e2-question"></a>
**M6-E2**（[提示](#m6-e2-hint) · [答案](#m6-e2-answer)）：`I` 翻倍时，哪些参数/激活量翻倍，哪些 shape 不变？为什么不能据此断言峰值内存或耗时恰好翻倍？

### 模块 6 验收

1. 能建立 input 到 output 的完整 ledger。
2. 能验证参数和单个主要中间 Tensor 随 `I` 线性变化。
3. 能解释分布实验和峰值内存结论的边界。

<a id="capstone"></a>
## 综合任务：三条路径走通微型 SwiGLU

固定配置：

```text
B=2, S=3, D=4, I=6
dtype=float32, device=cpu, bias=False
```

运行前先写出逐 token、函数和模块三条路径的全部 shape；再预测参数量、FP32 参数 bytes、每个 ledger 条目，以及三类受控错误在哪一步被拒绝。本实验不包含 input RMSNorm、post-attention RMSNorm、residual connection 或 attention 输出。

```python
import torch
from torch import nn
import torch.nn.functional as F

ATOL, RTOL = 1e-6, 1e-6

def check_positive_dims(d, i):
    if not isinstance(d, int) or not isinstance(i, int) or d <= 0 or i <= 0:
        raise ValueError(f"D and I must be positive integers, got D={d}, I={i}")

def checked_silu(x):
    manual = x * torch.sigmoid(x)
    reference = F.silu(x)
    torch.testing.assert_close(manual, reference, atol=ATOL, rtol=RTOL)
    if not torch.isfinite(manual).all():
        raise ValueError("SiLU produced NaN or Inf")
    return manual

def checked_product(activated_gate, up):
    if activated_gate.shape != up.shape:
        raise ValueError(
            f"refuse broadcasting: activated gate {tuple(activated_gate.shape)} "
            f"!= up {tuple(up.shape)}"
        )
    return activated_gate * up

def validate_explicit_inputs(x, w_gate, w_up, w_down):
    if x.ndim != 3:
        raise ValueError(f"expected x [B,S,D], got {tuple(x.shape)}")
    if w_gate.ndim != 2:
        raise ValueError(f"w_gate must be rank 2, got {tuple(w_gate.shape)}")
    d, i = w_gate.shape
    check_positive_dims(d, i)
    expected = {"w_gate": (d, i), "w_up": (d, i), "w_down": (i, d)}
    actual = {"w_gate": tuple(w_gate.shape), "w_up": tuple(w_up.shape), "w_down": tuple(w_down.shape)}
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} must be {shape}, got {actual[name]}")
    if x.shape[-1] != d:
        raise ValueError(f"expected input width {d}, got x shape {tuple(x.shape)}")
    weights = (w_gate, w_up, w_down)
    if any(weight.dtype != x.dtype for weight in weights):
        raise TypeError(f"input dtype {x.dtype} must match every weight dtype")
    if any(weight.device != x.device for weight in weights):
        raise ValueError(f"input device {x.device} must match every weight device")

def functional_path(x, w_gate, w_up, w_down):
    validate_explicit_inputs(x, w_gate, w_up, w_down)
    gate = x @ w_gate
    up = x @ w_up
    activated_gate = checked_silu(gate)
    product = checked_product(activated_gate, up)
    output = product @ w_down
    result = (gate, up, activated_gate, product, output)
    if not all(torch.isfinite(tensor).all() for tensor in result):
        raise ValueError("functional path produced NaN or Inf")
    return result

def token_loop_path(x, w_gate, w_up, w_down):
    validate_explicit_inputs(x, w_gate, w_up, w_down)
    B, S, D = x.shape
    I = w_gate.shape[1]
    gate = torch.empty(B, S, I, dtype=x.dtype, device=x.device)
    up = torch.empty_like(gate)
    activated_gate = torch.empty_like(gate)
    product = torch.empty_like(gate)
    output = torch.empty(B, S, D, dtype=x.dtype, device=x.device)
    for b in range(B):
        for s in range(S):
            token = x[b, s]
            gate[b, s] = token @ w_gate
            up[b, s] = token @ w_up
            activated_gate[b, s] = checked_silu(gate[b, s])
            product[b, s] = checked_product(activated_gate[b, s], up[b, s])
            output[b, s] = product[b, s] @ w_down
    result = (gate, up, activated_gate, product, output)
    if not all(torch.isfinite(tensor).all() for tensor in result):
        raise ValueError("token loop path produced NaN or Inf")
    return result

class TinySwiGLU(nn.Module):
    def __init__(self, d, i):
        super().__init__()
        check_positive_dims(d, i)
        self.d = d
        self.i = i
        self.gate_proj = nn.Linear(d, i, bias=False)
        self.up_proj = nn.Linear(d, i, bias=False)
        self.down_proj = nn.Linear(i, d, bias=False)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.d:
            raise ValueError(f"expected x [B,S,{self.d}], got {tuple(x.shape)}")
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated_gate = checked_silu(gate)
        product = checked_product(activated_gate, up)
        output = self.down_proj(product)
        result = (gate, up, activated_gate, product, output)
        if not all(torch.isfinite(tensor).all() for tensor in result):
            raise ValueError("module path produced NaN or Inf")
        return result

def ledger(tensors):
    names = ("gate", "up", "activated_gate", "product", "output")
    return {
        name: {
            "shape": tuple(tensor.shape),
            "elements": tensor.numel(),
            "bytes": tensor.numel() * tensor.element_size(),
        }
        for name, tensor in zip(names, tensors)
    }

B, S, D, I = 2, 3, 4, 6
torch.manual_seed(6)
x = torch.tensor([
    [[1.0, 0.0, -1.0, 2.0], [0.5, 1.0, 0.0, -0.5], [2.0, -1.0, 1.0, 0.0]],
    [[-1.0, 2.0, 0.5, 1.0], [0.0, -0.5, 1.5, 2.0], [1.0, 1.0, 1.0, 1.0]],
], dtype=torch.float32)
w_gate = torch.arange(D * I, dtype=torch.float32).reshape(D, I) / 20 - 0.5
w_up = torch.arange(D * I, dtype=torch.float32).reshape(D, I).flip(1) / 25 - 0.3
w_down = torch.arange(I * D, dtype=torch.float32).reshape(I, D) / 30 - 0.2

loop_result = token_loop_path(x, w_gate, w_up, w_down)
function_result = functional_path(x, w_gate, w_up, w_down)
model = TinySwiGLU(D, I).eval()
with torch.no_grad():
    model.gate_proj.weight.copy_(w_gate.T)
    model.up_proj.weight.copy_(w_up.T)
    model.down_proj.weight.copy_(w_down.T)
with torch.inference_mode():
    module_result = model(x)

expected_shapes = ((B, S, I), (B, S, I), (B, S, I), (B, S, I), (B, S, D))
for path_name, result in (("loop", loop_result), ("function", function_result), ("module", module_result)):
    assert tuple(tuple(tensor.shape) for tensor in result) == expected_shapes, path_name
    assert all(tensor.dtype == torch.float32 and tensor.device.type == "cpu" for tensor in result)
    assert all(torch.isfinite(tensor).all() for tensor in result)

for index, name in enumerate(("gate", "up", "activated_gate", "product", "output")):
    torch.testing.assert_close(loop_result[index], function_result[index], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(function_result[index], module_result[index], atol=ATOL, rtol=RTOL)
    print(name, tuple(function_result[index].shape))

assert function_result[0].shape == function_result[1].shape
torch.testing.assert_close(checked_silu(function_result[0]), F.silu(function_result[0]), atol=ATOL, rtol=RTOL)

changed_x = x.clone()
changed_x[1, 1] += torch.tensor([3.0, -2.0, 1.0, 0.5])
changed_result = functional_path(changed_x, w_gate, w_up, w_down)
other_tokens = torch.ones(B, S, dtype=torch.bool)
other_tokens[1, 1] = False
for before, after in zip(function_result, changed_result):
    torch.testing.assert_close(before[other_tokens], after[other_tokens], atol=0, rtol=0)

expected_keys = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
assert list(model.state_dict().keys()) == expected_keys
assert tuple(model.gate_proj.weight.shape) == (I, D)
assert tuple(model.up_proj.weight.shape) == (I, D)
assert tuple(model.down_proj.weight.shape) == (D, I)

parameter_count = sum(parameter.numel() for parameter in model.parameters())
parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
assert parameter_count == 3 * D * I == 72
assert parameter_bytes == 72 * 4 == 288

shape_ledger = ledger(function_result)
expected_ledger = {
    "gate": {"shape": (2, 3, 6), "elements": 36, "bytes": 144},
    "up": {"shape": (2, 3, 6), "elements": 36, "bytes": 144},
    "activated_gate": {"shape": (2, 3, 6), "elements": 36, "bytes": 144},
    "product": {"shape": (2, 3, 6), "elements": 36, "bytes": 144},
    "output": {"shape": (2, 3, 4), "elements": 24, "bytes": 96},
}
assert shape_ledger == expected_ledger

double_i = 12
assert 3 * D * double_i == 2 * parameter_count
assert B * S * double_i == 2 * shape_ledger["gate"]["elements"]

controlled_failures = (
    lambda: functional_path(torch.ones(B, S, D + 1), w_gate, w_up, w_down),
    lambda: functional_path(x, w_gate, w_up, torch.ones(D, I)),
    lambda: checked_product(torch.ones(B, S, 1), torch.ones(B, S, I)),
)
for action in controlled_failures:
    try:
        action()
    except (ValueError, TypeError) as error:
        print("controlled error:", error)
    else:
        raise AssertionError("expected controlled failure")

print("state keys:", expected_keys)
print("parameter count/FP32 bytes:", parameter_count, parameter_bytes)
print("shape ledger:", shape_ledger)
print("all capstone checks passed")
```

### 综合任务结果解释

- 三条路径的 gate、up、activated gate、product 和 output 全部在 `atol=rtol=1e-6` 内一致。
- 显式权重使用 `[D,I]/[I,D]`；模块存储使用 `[I,D]/[D,I]`，复制时转置。
- gate/up 乘法前检查 shape 完全相等，明确拒绝 `[2,3,1]` 到 `[2,3,6]` 的广播。
- 改变一个 token 只改变同位置的五类结果，证明独立 MLP 不混合 token 轴。
- 参数量 `3*4*6=72`，FP32 参数理论体积 `72*4=288 bytes`。
- `I=6 -> 12` 时参数量和单个 `[B,S,I]` Tensor 元素数都翻倍；外部 output shape 仍为 `[2,3,4]`。

<a id="acceptance"></a>
## 最终验收

先独立作答，再查看提示和答案。

### 概念题 C1-C10

<a id="c1-question"></a>
**C1**（[提示](#c1-hint) · [答案](#c1-answer)）：attention 与 MLP 分别主要混合哪个维度的信息？

<a id="c2-question"></a>
**C2**（[提示](#c2-hint) · [答案](#c2-answer)）：普通两层 MLP 与 SwiGLU 的数据流有何差别？

<a id="c3-question"></a>
**C3**（[提示](#c3-hint) · [答案](#c3-answer)）：SiLU 的公式是什么？它是否包含可学习参数？

<a id="c4-question"></a>
**C4**（[提示](#c4-hint) · [答案](#c4-answer)）：为什么 gate/up 是独立投影，却必须输出完全相同的 shape？

<a id="c5-question"></a>
**C5**（[提示](#c5-hint) · [答案](#c5-answer)）：为什么 `silu(gate) * up` 不能写成矩阵乘法，也不能依赖隐式广播？

<a id="c6-question"></a>
**C6**（[提示](#c6-hint) · [答案](#c6-answer)）：显式 `x @ W` 与 `nn.Linear.weight` 为什么具有转置布局？

<a id="c7-question"></a>
**C7**（[提示](#c7-hint) · [答案](#c7-answer)）：Parameter 与普通模块属性在 `state_dict` 中有何区别？

<a id="c8-question"></a>
**C8**（[提示](#c8-hint) · [答案](#c8-answer)）：`.eval()` 与 `torch.inference_mode()` 的职责有什么不同？

<a id="c9-question"></a>
**C9**（[提示](#c9-hint) · [答案](#c9-answer)）：参数理论体积为什么不等于实际 forward 峰值内存？

<a id="c10-question"></a>
**C10**（[提示](#c10-hint) · [答案](#c10-answer)）：Dense MLP、SwiGLU 与未来 MoE expert MLP 的关系和边界是什么？

### 推导题 S1-S5

<a id="s1-question"></a>
**S1**（[提示](#s1-hint) · [答案](#s1-answer)）：`x [2,5,8]`、`I=24` 时，写出函数版 gate/up/down 权重和全部关键激活 shape。

<a id="s2-question"></a>
**S2**（[提示](#s2-hint) · [答案](#s2-answer)）：同上配置，写出三个 `nn.Linear.weight` shape，并说明函数权重怎样复制进去。

<a id="s3-question"></a>
**S3**（[提示](#s3-hint) · [答案](#s3-answer)）：无 bias SwiGLU 取 `D=8,I=24`，求参数量以及 FP32/BF16 理论字节数。

<a id="s4-question"></a>
**S4**（[提示](#s4-hint) · [答案](#s4-answer)）：`B=2,S=5,D=8,I=24` 时，求单个 `[B,S,I]` 中间张量与 output 的元素数和 FP32 字节数。

<a id="s5-question"></a>
**S5**（[提示](#s5-hint) · [答案](#s5-answer)）：固定 `B/S/D`，将 `I=24` 改成 48，参数量、单个中间 Tensor、输出 Tensor 分别怎样变化？

### 评分建议

- C1-C10 每题 1 分，至少答对 **8/10**。
- S1-S5 每题 2 分，至少答对 **4/5**。
- 综合任务所有断言通过。
- 能脱离代码画出 `gate -> SiLU` 与 `up` 汇合、逐元素乘法、down projection。
- 能分别写出显式矩阵布局和 `nn.Linear.weight` 布局。
- 能区分参数内存、单个激活 Tensor 体积和实际 forward 峰值内存。
- 若只会背 `3DI`，却不能解释三项来自哪里、权重方向和激活生命周期，不算通过。

<a id="hints"></a>
## 提示

### 模块练习提示

<a id="m1-e1-hint"></a>**M1-E1：** 每次矩阵乘法只替换最后一维，先收缩 8，再收缩 20。

<a id="m1-e2-hint"></a>**M1-E2：** 公式没有沿 `S` 求和；attention 或 sequence pooling 会读取其他位置。

<a id="m2-e1-hint"></a>**M2-E1：** `sigmoid(0)=0.5`；大正数的 sigmoid 接近 1。

<a id="m2-e2-hint"></a>**M2-E2：** ReLU 截断负数，SiLU 平滑缩放；模型质量还依赖训练、架构和数据。

<a id="m3-e1-hint"></a>**M3-E1：** 两个投影都把末轴 8 替换为 16；索引 `[b,s,i]` 不跨 token。

<a id="m3-e2-hint"></a>**M3-E2：** 广播会让一个 gate 通道控制 16 个 up 通道，失去逐通道门控语义。

<a id="m4-e1-hint"></a>**M4-E1：** 函数版按 `[in,out]`，模块版按 `[out,in]`。

<a id="m4-e2-hint"></a>**M4-E2：** `state_dict` 保存注册状态；模式切换与 autograd 控制是两个维度。

<a id="m5-e1-hint"></a>**M5-E1：** 先算 `3*8*32`，再分别乘 4 和 2 bytes。

<a id="m5-e2-hint"></a>**M5-E2：** 想到输入、输出、中间激活、allocator、workspace、梯度和优化器。

<a id="m6-e1-hint"></a>**M6-E1：** gate 用 `B*S*I`，output 用 `B*S*D`，FP32 每元素 4 bytes。

<a id="m6-e2-hint"></a>**M6-E2：** `3DI` 和 `[B,S,I]` 线性增长；输出仍由 `D` 决定，实际运行还受生命周期和实现影响。

### 最终验收提示

<a id="c1-hint"></a>**C1：** 一个读取位置，另一个独立处理每个位置的特征末轴。

<a id="c2-hint"></a>**C2：** 比较单扩展分支与 gate/up 双分支。

<a id="c3-hint"></a>**C3：** 激活函数只由固定数学运算组成。

<a id="c4-hint"></a>**C4：** 参数可独立，逐元素配对仍要求同一索引空间。

<a id="c5-hint"></a>**C5：** 目标是每个 `[b,s,i]` 一一相乘。

<a id="c6-hint"></a>**C6：** `Linear(in,out)` 为每个输出单元保存一行输入权重。

<a id="c7-hint"></a>**C7：** 只有注册的 Parameter/buffer 自动进入状态字典。

<a id="c8-hint"></a>**C8：** 一个切换模块模式，一个关闭梯度记录。

<a id="c9-hint"></a>**C9：** 参数只是运行时内存类别之一。

<a id="c10-hint"></a>**C10：** expert 内部可以复用 SwiGLU，但 router/dispatch 属于额外机制。

<a id="s1-hint"></a>**S1：** 显式 gate/up 为 `[D,I]`，down 为 `[I,D]`。

<a id="s2-hint"></a>**S2：** 对 S1 的每个权重转置。

<a id="s3-hint"></a>**S3：** `3*8*24`，FP32/BF16 分别乘 4/2。

<a id="s4-hint"></a>**S4：** 分别计算 `2*5*24` 和 `2*5*8`。

<a id="s5-hint"></a>**S5：** 检查公式中是否含 `I`。

<a id="answers"></a>
## 参考答案

### 模块练习答案

<a id="m1-e1-answer"></a>**M1-E1：** 第一层 `[3,5,8] @ [8,20] -> [3,5,20]`，收缩 8；第二层 `[3,5,20] @ [20,8] -> [3,5,8]`，收缩 20。`B=3,S=5` 始终保留。

<a id="m1-e2-answer"></a>**M1-E2：** Dense MLP 对每个 `[D]` 切片独立应用同一函数，没有沿 `S` 聚合，所以其他 token 不变。加入 self-attention、卷积或 sequence pooling 等跨位置运算后可能破坏该性质。

<a id="m2-e1-answer"></a>**M2-E1：** `SiLU(0)=0*0.5=0`。当 `x` 很大且为正，`sigmoid(x)` 接近 1，因此乘积接近 `x`。

<a id="m2-e2-answer"></a>**M2-E2：** ReLU 把所有负输入变成 0；SiLU 通常保留小的负输出并平滑变化。局部函数差异不能单独决定训练后质量，结果还依赖权重、数据、优化和整体架构。

<a id="m3-e1-answer"></a>**M3-E1：** gate、up、activated gate 和 product 都是 `[2,4,16]`。第 `[b,s,i]` 项表示第 `b` 个样本、第 `s` 个 token、第 `i` 个中间通道的 gate 值与 up 值一一相乘。

<a id="m3-e2-answer"></a>**M3-E2：** 广播会把每个 token 的单一 gate 值复制给 16 个通道，不能表达逐中间通道门控。shape 可执行不代表符合模型语义，因此必须拒绝。

<a id="m4-e1-answer"></a>**M4-E1：** 函数版为 gate/up `[8,24]`、down `[24,8]`；模块版为 gate/up `.weight [24,8]`、down `.weight [8,24]`。复制时对每个函数权重转置。

<a id="m4-e2-answer"></a>**M4-E2：** 普通字符串没有注册为 Parameter 或 buffer，所以不进入 `state_dict`。`.eval()` 切换 dropout/batch norm 等模块行为；`inference_mode()` 关闭 autograd 跟踪并用于纯推理，两者不能互相替代。

<a id="m5-e1-answer"></a>**M5-E1：** 参数量 `3*8*32=768`；FP32 为 `768*4=3072 bytes`，BF16 为 `768*2=1536 bytes`。

<a id="m5-e2-answer"></a>**M5-E2：** 例如输入、输出、中间激活、allocator 缓存、kernel workspace、框架对象、梯度、优化器状态。峰值取决于这些类别及其生命周期，参数 bytes 只描述权重 Tensor 数据。

<a id="m6-e1-answer"></a>**M6-E1：** gate 元素数 `2*5*32=320`，FP32 为 `1280 bytes`；output 元素数 `2*5*8=80`，FP32 为 `320 bytes`。

<a id="m6-e2-answer"></a>**M6-E2：** gate/up/down 参数总量与每个 `[B,S,I]` 中间 Tensor 都翻倍；输入和输出 `[B,S,D]` shape 不变。峰值还取决于同时存活 Tensor、融合、复用和框架开销，耗时还取决于 kernel 与硬件，所以不能保证恰好翻倍。

### 最终验收答案

<a id="c1-answer"></a>**C1：** Attention 主要沿 token/sequence 轴动态混合允许位置的信息；MLP 对每个 token 独立变换最后一维特征，不沿 `S` 聚合。

<a id="c2-answer"></a>**C2：** 普通两层 MLP 是单扩展投影、激活、down；SwiGLU 使用 gate/up 两个独立扩展投影，以 `SiLU(gate)` 逐元素调制 up，再 down。

<a id="c3-answer"></a>**C3：** `SiLU(x)=x*sigmoid(x)`；它没有可学习参数，参数来自前后的线性层。

<a id="c4-answer"></a>**C4：** 两组权重独立学习不同 values，但逐元素乘法要求每个 token、每个中间通道一一对应，所以输出 shape 必须完全相同。

<a id="c5-answer"></a>**C5：** 门控目标是同索引元素相乘，不收缩任何轴；矩阵乘法会收缩轴，广播会复制缺失维度，两者都改变预期语义。

<a id="c6-answer"></a>**C6：** 显式右乘把每个输出通道放在矩阵列中，所以是 `[in,out]`；`nn.Linear` 为每个输出单元保存一行输入系数，所以 weight 是 `[out,in]`，forward 使用其转置。

<a id="c7-answer"></a>**C7：** 注册 Parameter 会被优化器、设备/dtype 转换和 `state_dict` 管理；普通属性只作为 Python 元数据存在，除非显式保存，否则不在状态字典中。

<a id="c8-answer"></a>**C8：** `.eval()` 改变依赖训练模式的模块行为；`inference_mode()` 禁止 autograd 记录。无 dropout 的本例 train/eval 数值相同，但职责仍不同。

<a id="c9-answer"></a>**C9：** 参数体积不含输入输出、中间张量、临时 workspace、allocator、框架对象以及训练时的梯度/优化器；峰值还取决于生命周期与实现。

<a id="c10-answer"></a>**C10：** Dense MLP 是所有 token 都执行同一前馈模块；SwiGLU 是该前馈模块的一种门控结构；MoE 可让多个 expert 各自采用 SwiGLU，但还需 router、Top-K、dispatch 和聚合，本周均未实现。

<a id="s1-answer"></a>**S1：** 显式权重 gate/up `[8,24]`、down `[24,8]`；gate、up、activated gate、product 都是 `[2,5,24]`，output `[2,5,8]`。

<a id="s2-answer"></a>**S2：** 模块 gate/up weight `[24,8]`、down weight `[8,24]`；使用 `linear.weight.copy_(explicit_weight.T)`。

<a id="s3-answer"></a>**S3：** 参数量 `3*8*24=576`；FP32 `2304 bytes`，BF16 `1152 bytes`。

<a id="s4-answer"></a>**S4：** 单个 `[2,5,24]` 中间 Tensor 有 240 元素、960 FP32 bytes；output `[2,5,8]` 有 80 元素、320 bytes。

<a id="s5-answer"></a>**S5：** `3DI` 参数量翻倍；每个 `[B,S,I]` Tensor 的元素和 bytes 翻倍；input/output `[B,S,D]` 的 shape、元素数和理论 bytes 不变。

<a id="glossary"></a>
## 公式、形状、内存与错误速查

### 核心公式

| 结构 | 公式 | 无 bias 参数量 |
| --- | --- | ---: |
| ReLU MLP | `relu(x @ W_up) @ W_down` | `2DI` |
| GELU MLP | `gelu(x @ W_up) @ W_down` | `2DI` |
| SwiGLU | `(silu(x @ W_gate) * (x @ W_up)) @ W_down` | `3DI` |

### Shape 与权重布局

| 名称 | Shape/布局 | 关键检查 |
| --- | --- | --- |
| input | `[B,S,D]` | rank 3，最后一维为 `D` |
| explicit gate/up | `[D,I]` | 与输入右乘 |
| explicit down | `[I,D]` | 把 `I` 收回 `D` |
| Linear gate/up weight | `[I,D]` | `Linear(D,I,bias=False)` |
| Linear down weight | `[D,I]` | `Linear(I,D,bias=False)` |
| gate/up/activated/product | `[B,S,I]` | gate/up 必须完全相等，禁止广播 |
| output | `[B,S,D]` | shape 保持不等于残差已实现 |

### 理论内存

```text
Tensor bytes = numel * bytes_per_element
FP32=4 bytes, FP16/BF16=2 bytes, INT8=1 byte
SwiGLU parameter bytes = 3DI * bytes_per_element
single major intermediate bytes = BSI * bytes_per_element
```

实际峰值不能只把表格机械相加：先列出同一时刻仍被引用的 Tensor，再考虑 autograd、融合、复用、allocator 和 workspace。

### 错误速查

| 症状 | 首先检查 | 应有错误信息 |
| --- | --- | --- |
| 输入 matmul 失败 | `x.shape[-1] == D` | 实际 x shape 与期望宽度 |
| gate/up 投影失败 | 两权重均为 `[D,I]` | 两个实际 weight shape |
| down 方向错误 | 显式 `[I,D]`，模块 `[D,I]` | 实际与期望 shape |
| 乘法能跑但结果怪 | gate/up 是否仅“可广播” | 明确写 `refuse broadcasting` |
| 模块与函数不一致 | 复制权重是否转置 | 两种布局对照 |
| 出现 NaN/Inf | gate/up/SiLU/product/output 首个异常 | 首个非有限张量名 |
| 参数量多出一项 | 是否错误计入 bias | 当前是否 `bias=False` |
| 内存估算过小/过大 | 是否混淆参数、激活、峰值 | 分类别列账 |

<a id="next-week"></a>
## Week 7 预告

第五周和第六周结束后，我们拥有两个外部接口都保持 `[B,S,D]` 的独立子层：带 normalization 与 position 的 causal GQA，以及本周的 SwiGLU MLP。第七周将加入 pre-norm 与 residual connection，按正确顺序组合：

```text
x
-> x + Attention(RMSNorm(x))
-> h + SwiGLU(RMSNorm(h))
-> Dense Decoder Layer output [B,S,D]
```

然后堆叠微型 Dense Decoder，并接回 Embedding、final norm 和 LM Head。开始前请确保你不会把“MLP 输入输出 shape 相同”误说成“残差已经存在”，并能从函数版与模块版的任一偏差定位到权重布局、门控乘法或 down projection。
