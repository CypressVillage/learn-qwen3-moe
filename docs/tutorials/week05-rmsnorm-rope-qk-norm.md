# 第五周教程：RMSNorm、RoPE 与 QK Norm

> 本周目标：解决第四周 attention 尚未处理的两个问题：控制进入子层及 Q/K 的数值尺度，并让 Q/K 点积感知 token 的相对位置；随后按 Qwen3 attention 的顺序接回 causal GQA。

## 目录

1. [本周在完整推理中的位置](#inference-position)
2. [学习目标](#goals)
3. [学习方式](#method)
4. [环境与边界](#environment)
5. [模块 1：RMSNorm](#module-1)
6. [模块 2：QK Norm](#module-2)
7. [模块 3：二维 split-half RoPE](#module-3)
8. [模块 4：向量化 RoPE 与 position offset](#module-4)
9. [模块 5：dtype 与操作顺序](#module-5)
10. [模块 6：接入 causal GQA](#module-6)
11. [综合任务](#capstone)
12. [最终验收](#acceptance)
13. [提示](#hints)
14. [参考答案](#answers)
15. [术语与速查表](#glossary)
16. [Week 6 预告](#next-week)

<a id="inference-position"></a>
## 本周在完整推理中的位置

第四周的 causal GQA 已经能让每个位置读取允许看到的 token，但仍留下两个结构性缺口：

1. **尺度缺口**：hidden 经多层子层和残差路径传播时，需要明确的归一化边界；Q/K 点积也直接受每颗 head 向量尺度影响。
2. **位置缺口**：causal mask 只告诉 attention “未来不可见”，没有告诉它两个可见 token 相隔多远。若只对同一 token 表示做共享 Q/K 投影，attention 公式本身不会自动获得绝对或相对 position。

本周在第四周数据流中加入三处操作：

```text
hidden [B,S,D]
-> input RMSNorm
-> Q/K/V projection + reshape
-> QK Norm(Q/K only)
-> RoPE(Q/K only)
-> causal GQA
-> output [B,S,D]
```

input RMSNorm 和 QK Norm 都使用 RMSNorm 形式，但职责、输入位置和统计宽度不同；RoPE 则只改变 Q/K 的方向，不改变 shape。完成本周后仍缺少残差连接和前馈子层，因此仍不是完整 Decoder Block。

<a id="goals"></a>
## 学习目标

完成本周后，你应该能：

1. 解释第四周 attention 为什么仍需要归一化和位置信息。
2. 写出 RMSNorm 公式，解释它与 LayerNorm 的差别，并沿最后一维实现 FP32 统计。
3. 区分 hidden-state RMSNorm `[B,S,D]` 与 attention 内部逐 token、逐 head 的 QK Norm `[B,S,H,Dh]`。
4. 比较 learned absolute embedding、固定 sinusoidal、relative bias 与 RoPE 的注入位置和主要取舍。
5. 口述正确顺序：input RMSNorm -> Q/K/V projection -> reshape -> QK Norm -> transpose -> RoPE(Q/K) -> GQA。
6. 从二维旋转推导 split-half RoPE，并解释它为何改变 Q/K 数值但不改变 shape。
7. 构造 `inv_freq`、`angles`、`cos/sin`，说明 position、batch、head 与通道轴如何广播。
8. 验证 position 0 恒等、通道对二范数保持、同位置 Q/K 点积保持和共同 offset 下 scores 保持。
9. 解释 RoPE 频率与角度使用 FP32 教学基准的原因，并区分参考实现与 fused/低精度生产实现。
10. 保持物理复制和逻辑分组两条 causal GQA 路径数值一致。

建议投入 6-10 小时。CPU 可以完成全部必修内容。

<a id="method"></a>
## 学习方式：Predict-Run-Explain

每个实验都按同一闭环进行：

1. **Predict**：先写 shape、归一化轴、旋转通道配对和预期不变量。
2. **Run**：独立运行当前代码块，观察输出和断言，不依赖前一个代码块。
3. **Explain**：脱离代码说明数值为什么变化、哪些 shape 不变、错误顺序为何不等价。

推荐调试顺序：shape -> 最后一维宽度 -> 归一化轴 -> position 合法性 -> RoPE 布局 -> head 映射 -> causal mask -> softmax 轴 -> dtype 与有限性。

<a id="environment"></a>
## 环境与边界

在仓库根目录运行：

```bash
uv sync --locked
uv run python scripts/check_environment.py
uv run pytest
```

本周所有必修代码只依赖 PyTorch，使用确定性 CPU 小张量，不下载模型、Tokenizer、checkpoint 或数据集。代码中的 `eps`、`rope_theta`、head 数和宽度都是明确的微型教学值；接入真实模型时必须读取目标配置，不能把这些数字当作 Qwen3 官方配置。

本周只研究 normalization 与 position 如何进入 attention，不实现完整 Decoder 残差路径、MLP、SwiGLU、MoE、KV Cache、padding mask、dropout、cross-attention 或长上下文 RoPE scaling。示例是可观察中间值的教学参考实现，不是生产级 fused kernel，也不用于性能结论。真实实现可能融合 normalization、旋转或 attention 运算，并在受控位置使用低精度；判断正确性应依据目标配置、操作语义和误差容忍，而不是要求内部 Tensor 与本教程逐 bit 相同。

统一记号：

```text
B    batch size
S    query 序列长度
T    key/value 序列长度（本周无 KV Cache，所以 T=S）
D    hidden width
Hq   query head 数
Hkv  key/value head 数
Dh   每颗 head 的宽度
G    每颗 KV head 服务的 query head 数，G=Hq/Hkv
```

本周固定采用 Qwen3 参考实现使用的 **split-half** 约定：

```text
x = [x1, x2]，其中 x1/x2 各占最后一维的一半
rotate_half(x) = [-x2, x1]
```

adjacent-pair（相邻通道两两配对）也是合法布局。两种布局可通过相应通道排列表达同类二维旋转，但 checkpoint 权重、`rotate_half` 公式和 cos/sin 排列必须从头到尾一致，不能只替换其中一项。

<a id="module-1"></a>
## 模块 1：RMSNorm

### 1.1 为什么 attention 前还要归一化

第四周直接把 hidden 送入 Q/K/V projection。单层小张量可以正常运行，但真实 Decoder 会重复堆叠子层并通过残差路径传递表示；模型需要在明确边界控制送入子层的数值尺度。Qwen3 使用 RMSNorm，而不是 LayerNorm，作为这种归一化操作。

两者都会按特征维计算统计量并带可学习逐通道缩放，主要公式差别是：LayerNorm 先减均值、再按中心化方差缩放；RMSNorm 不减均值，只按平方均值控制整体尺度。RMSNorm 公式更少一步居中运算，但这里不把“更简单”直接等同于在所有硬件和 shape 上必然更快，也不声称两者功能完全等价。

常见 LayerNorm 实现还可包含逐通道 bias；本周的 Qwen3-style RMSNorm 只有逐通道 `weight`，没有 bias。参数细节仍应以目标实现和 checkpoint 为准，不能只凭层名猜测。

目标 Qwen3 选择 RMSNorm。本教程选择显式函数和 FP32 统计，让归一化轴、`eps` 与 weight 广播都可观察，而不先使用 fused norm。完成 input RMSNorm 后，数据仍是 `[B,S,D]`，下一步回到整体流程进入 Q/K/V projection。

### 1.2 公式与轴

对最后一维为 `D` 的向量 `x`：

```text
rms(x)       = sqrt(mean(x^2) + eps)
normalized   = x / rms(x)
RMSNorm(x)   = normalized * weight
```

与 LayerNorm 不同，RMSNorm 不减均值，也不使用“减均值后的方差”重新居中。对 `hidden [B,S,D]`，每个 batch、每个 token 独立沿 `D` 统计。为减少平方、求均值和倒平方根的低精度误差，统计量使用 FP32。

当 `weight=1` 时，`eps` 很小的非零向量在归一化后的 RMS 约为 1。若 `weight` 非均匀，最终输出各通道被不同倍数缩放，输出 RMS 不必等于 1。

### 1.3 运行前预测

先判断：输入 `[1,2,4]` 经 RMSNorm 后 shape 是否变化？第一行均值不为 0 时，RMSNorm 会不会把输出均值强制变成 0？非均匀 weight 会不会改变最终 RMS？

```python
import torch

def rms_norm(x, weight, eps):
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(
            f"weight width {tuple(weight.shape)} does not match input width {x.shape[-1]}"
        )
    x_fp32 = x.float()
    inverse_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * inverse_rms * weight.float()).to(x.dtype)

x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                   [-2.0, 0.0, 2.0, 4.0]]])
ones = torch.ones(4)
nonuniform = torch.tensor([0.5, 1.0, 1.5, 2.0])

y_unit = rms_norm(x, ones, eps=1e-6)
y_weighted = rms_norm(x, nonuniform, eps=1e-6)

print("shape:", tuple(x.shape), "->", tuple(y_unit.shape))
print("input means:", x.mean(dim=-1))
print("unit-weight output means:", y_unit.mean(dim=-1))
print("unit-weight RMS:", y_unit.square().mean(dim=-1).sqrt())
print("weighted output RMS:", y_weighted.square().mean(dim=-1).sqrt())

assert y_unit.shape == x.shape == y_weighted.shape
torch.testing.assert_close(
    y_unit.square().mean(dim=-1), torch.ones(1, 2), atol=1e-6, rtol=0
)
assert not torch.allclose(y_unit.mean(dim=-1), torch.zeros(1, 2))
assert not torch.allclose(
    y_weighted.square().mean(dim=-1), torch.ones(1, 2), atol=1e-3, rtol=0
)

for bad_weight, bad_eps in ((torch.ones(3), 1e-6), (torch.ones(4), 0.0)):
    try:
        rms_norm(x, bad_weight, bad_eps)
    except ValueError as error:
        print("controlled error:", error)
```

### 1.4 解释输出

输入输出都为 `[B,S,D]`。`mean(x^2)` 只消去最后一维，因此保留 batch 与 token 轴。RMSNorm 只控制平方均值对应的整体尺度，不保证输出均值为 0。非均匀 weight 是可学习的逐通道缩放，发生在标准化之后。

### 1.5 常见误区

- 把 RMSNorm 写成 `x - mean(x)`：这混入了 LayerNorm 的居中步骤。
- 沿 batch 或 sequence 轴统计：不同样本或 token 会互相影响。
- 看到非均匀 weight 后输出 RMS 不为 1，就误判实现错误。
- 直接在低精度输入上完成平方与均值，却没有明确精度基准。

### 1.6 练习

<a id="m1-e1-question"></a>
**M1-E1**（[提示](#m1-e1-hint) · [答案](#m1-e1-answer)）：`x [2,3,8]` 做 hidden-state RMSNorm 时，统计量 shape 是什么？沿哪个轴计算？输出 shape 是什么？

<a id="m1-e2-question"></a>
**M1-E2**（[提示](#m1-e2-hint) · [答案](#m1-e2-answer)）：为什么 `weight=[0.5,1.0,1.5,2.0]` 时，标准化中间值 RMS 约为 1，而最终输出 RMS 通常不为 1？

### 模块 1 验收

1. 能写出 `x * rsqrt(mean(x^2)+eps) * weight`，并指出 FP32 统计位置。
2. 能解释 RMSNorm 与 LayerNorm 是否减均值的差别。
3. 能拒绝错误 weight width 和 `eps<=0`。

<a id="module-2"></a>
## 模块 2：QK Norm

### 2.1 hidden norm 与 head norm

input RMSNorm 已经控制 projection 输入的尺度，但不能替代投影后的逐 head 归一化：Q/K 经过不同权重矩阵，随后又被拆成各自的 `[Dh]` 向量。QK Norm 直接作用在参与点积的每个 query/key head 上，为 scores 的输入建立更明确的尺度边界。

Attention 实现也可以只依赖 input norm 与 `1/sqrt(Dh)` 缩放，或采用其他 score 控制方案；QK Norm 不是 scaled dot-product attention 公式的普遍必需项。目标 Qwen3 额外选择逐 head QK Norm，本教程因此按 checkpoint 语义复现它，而不是把这一步推广为所有 Transformer 的规则。

input RMSNorm 作用于投影前的 `hidden [B,S,D]`，沿 `D` 统计。QK Norm 作用于投影并 reshape 后的 `q [B,S,Hq,Dh]` 和 `k [B,S,Hkv,Dh]`，沿 `Dh` 统计。也就是说，每个 token、每颗 head 独立归一化。两者公式相似，但不能互换：它们看到的表示、参数宽度和作用时机都不同。

QK Norm 不改变 Q/K shape。V 不参与 QK 匹配分数，因此目标架构不对 V 使用 QK Norm；它在概率形成后提供被加权汇总的内容。这里是在复现 Qwen3 的具体数据流，不主张所有 attention 变体都必须采用同一选择。

本教程继续复用与 input RMSNorm 相同的显式最后一维公式，但给 Q/K 使用各自宽度为 `Dh` 的 weight。完成后 q/k/v 仍保持 `[B,S,H,Dh]`；下一步 transpose 到 head-first 布局，再只对 Q/K 注入 position。

正确局部顺序是：

```text
projection -> reshape [B,S,H,Dh] -> QK Norm along Dh -> transpose [B,H,S,Dh]
```

### 2.2 运行前预测

对 `q [1,2,2,4]`，正确统计量 shape 是 `[1,2,2,1]`。如果误沿 head 轴归一化，代码可能仍运行；请先预测两种结果为什么 shape 相同但语义不同。

```python
import torch

def rms_norm_last(x, weight, eps=1e-6):
    if eps <= 0:
        raise ValueError("eps must be positive")
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("weight must match the last dimension")
    x_fp32 = x.float()
    normalized = x_fp32 * torch.rsqrt(
        x_fp32.square().mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * weight.float()).to(x.dtype)

q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                    [2.0, 1.0, 0.5, 3.0]],
                   [[-1.0, 1.0, 2.0, 0.5],
                    [4.0, 2.0, 1.0, 0.25]]]])
k = q[:, :, :1].clone() * 0.75
v = torch.tensor([[[[0.1, 0.2, 0.3, 0.4]],
                   [[0.5, 0.6, 0.7, 0.8]]]])
q_weight = torch.tensor([0.7, 1.1, 1.4, 0.9])
k_weight = torch.tensor([1.2, 0.8, 1.5, 0.6])

q_normed = rms_norm_last(q, q_weight)
k_normed = rms_norm_last(k, k_weight)
v_unchanged = v.clone()

# 错误示范：沿 head 轴 dim=2 统计，shape 合法但语义错误。
q_fp32 = q.float()
wrong_axis = q_fp32 * torch.rsqrt(
    q_fp32.square().mean(dim=2, keepdim=True) + 1e-6
) * q_weight

print("q/k/v shapes:", tuple(q_normed.shape), tuple(k_normed.shape), tuple(v.shape))
print("correct vs wrong max diff:", (q_normed - wrong_axis).abs().max().item())

assert q_normed.shape == q.shape
assert k_normed.shape == k.shape
assert torch.equal(v_unchanged, v)
assert (q_normed - wrong_axis).abs().max() > 1e-3
assert torch.isfinite(q_normed).all() and torch.isfinite(k_normed).all()
```

### 2.3 为什么错误轴危险

沿 `Dh` 统计时，每颗 head 得到自己的标量尺度；沿 head 轴统计时，同一个通道上的不同 heads 被混在一起。两者都能广播回原 shape，因此只检查 shape 无法发现错误，必须检查轴语义或与参考结果对齐。

非均匀 Q/K weights 还会逐通道改变方向。它们分别属于 Q 与 K，不应错误复用成 V 的归一化参数。

### 2.4 常见误区

- 在 projection 前对 hidden 做一次 RMSNorm，就认为已替代 QK Norm。
- 对 `[B,H,S,Dh]` 沿 `S` 或 `H` 归一化，shape 仍合法却改变语义。
- 先 transpose 再忘记最后一维仍必须是 `Dh`。
- 对 V 应用 QK Norm，破坏本周规定的数据流。

### 2.5 练习

<a id="m2-e1-question"></a>
**M2-E1**（[提示](#m2-e1-hint) · [答案](#m2-e1-answer)）：对 `q [2,5,6,4]` 和 `k [2,5,2,4]`，写出 QK Norm 统计量 shape、输出 shape，以及每个统计量覆盖哪些元素。

<a id="m2-e2-question"></a>
**M2-E2**（[提示](#m2-e2-hint) · [答案](#m2-e2-answer)）：为什么沿 `H` 轴归一化可能不报 shape 错误，却仍是严重的语义错误？V 应如何处理？

### 模块 2 验收

1. 能区分 `[B,S,D]` 的 input RMSNorm 与 `[B,S,H,Dh]` 的 QK Norm。
2. 能指出 Q/K 沿 `Dh` 独立统计，且 shape 不变。
3. 能说明 V 不做 QK Norm，并识别 shape 合法的错误轴。

<a id="module-3"></a>
## 模块 3：二维 split-half RoPE

### 3.1 为什么 mask 不能代替位置编码

causal mask 对同一行只给出“允许/禁止”关系。例如 query 位置 8 可以读取位置 2 和 7，但 mask 本身不编码它们分别距离 6 和 1。模型需要另一种机制让 attention scores 感知 position。

常见方案的注入位置不同：

| 方案 | position 怎样进入模型 | 主要取舍 |
| --- | --- | --- |
| learned absolute embedding | position 向量加到 token hidden | 简单直接，但有固定训练位置表及绝对位置参数 |
| fixed sinusoidal encoding | 固定正弦向量加到 hidden | 无需学习位置表，但位置与内容在 hidden 中相加 |
| relative position bias | 按相对距离给 attention scores 加 bias | 直接影响匹配分数，需要定义和实现 bias 规则 |
| RoPE | 按 position 旋转 Q/K 通道对 | 不改变 shape，使 QK 点积自然依赖相对角度 |

learned absolute table 通常受已分配位置表范围约束；relative bias 和 RoPE 具有相对位置结构，但这不等于模型可无条件泛化到任意长度。超出训练上下文时仍要考虑训练分布、数值分辨率以及目标模型规定的 RoPE scaling 等边界。

目标 Qwen3 使用 RoPE，因此本教程不在 hidden 上叠加绝对 position embedding，而是在形成 scores 前旋转 Q/K。本周选择显式 split-half `rotate_half` 和可检查的 cos/sin Tensor，与目标权重布局保持一致；不实现长上下文 scaling。完成旋转后 q/k shape 不变，数据流回到第四周的 scaled dot-product causal GQA。

### 3.2 从二维旋转开始

二维向量 `[a,b]` 旋转角度 `theta`：

```text
[a']   [ cos(theta)  -sin(theta) ] [a]
[b'] = [ sin(theta)   cos(theta) ] [b]
```

因此：

```text
[a',b'] = [a,b] * cos(theta) + [-b,a] * sin(theta)
```

split-half 把 `Dh` 分成前后两半。若 `x=[x1,x2]`，每个配对是 `(x1[i],x2[i])`，所以 `rotate_half(x)=[-x2,x1]`。`Dh` 必须为偶数。

旋转矩阵是正交矩阵，因此保持每对通道的二范数。Q 和 K 若在同一 position 使用同一组角度，它们的点积也保持。

### 3.3 运行前预测

position 0 对应角度 0，此时 `cos=1,sin=0`。请先预测：输出是否等于输入？两个 split-half 通道对 `(0,2)`、`(1,3)` 的平方和是否变化？

```python
import torch

def rotate_half(x):
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"head dimension must be even, got {x.shape[-1]}")
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

def apply_angles(x, half_angles):
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    if half_angles.shape[-1] * 2 != x.shape[-1]:
        raise ValueError("angle count must equal Dh/2")
    full_angles = torch.cat((half_angles, half_angles), dim=-1).float()
    return x.float() * full_angles.cos() + rotate_half(x.float()) * full_angles.sin()

def pair_squared_norms(x):
    half = x.shape[-1] // 2
    return x[..., :half].square() + x[..., half:].square()

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
q = torch.tensor([0.5, -1.0, 2.0, 0.25])
k = torch.tensor([1.5, 0.5, -0.5, 2.0])
zero = torch.zeros(2)
angles = torch.tensor([0.4, -0.7])

x_zero = apply_angles(x, zero)
x_rotated = apply_angles(x, angles)
q_rotated = apply_angles(q, angles)
k_rotated = apply_angles(k, angles)

print("rotate_half(x):", rotate_half(x))
print("position 0:", x_zero)
print("pair squared norms before/after:", pair_squared_norms(x), pair_squared_norms(x_rotated))
print("dot before/after:", torch.dot(q, k).item(), torch.dot(q_rotated, k_rotated).item())

torch.testing.assert_close(x_zero, x, atol=0, rtol=0)
torch.testing.assert_close(pair_squared_norms(x_rotated), pair_squared_norms(x), atol=1e-6, rtol=1e-6)
torch.testing.assert_close(torch.dot(q_rotated, k_rotated), torch.dot(q, k), atol=1e-6, rtol=1e-6)
assert x_rotated.shape == x.shape

try:
    rotate_half(torch.ones(5))
except ValueError as error:
    print("controlled error:", error)
```

### 3.4 RoPE 改变什么

RoPE 不添加 position 轴，也不改变 Q/K shape；它依据 position 改变最后一维中的方向。position 0 是恒等变换。单独旋转向量保持配对二范数；同角度同时旋转 Q/K 保持点积；不同位置使用不同角度时，点积会携带相对位置信息。

### 3.5 常见误区

- 把 split-half 的配对误写成 `(0,1)`、`(2,3)`；这里实际是 `(0,2)`、`(1,3)`。
- 只检查整向量范数，不检查布局定义下的每对通道。
- 允许奇数 `Dh`，导致最后一个通道没有旋转伙伴。
- Q 使用 split-half、K 使用 adjacent-pair，造成不可比较的布局。

### 3.6 练习

<a id="m3-e1-question"></a>
**M3-E1**（[提示](#m3-e1-hint) · [答案](#m3-e1-answer)）：对 `x=[1,2,3,4]`，写出 split-half 的两对通道和 `rotate_half(x)`。

<a id="m3-e2-question"></a>
**M3-E2**（[提示](#m3-e2-hint) · [答案](#m3-e2-answer)）：为什么同一角度旋转 Q/K 后点积保持，而不同 position 的 Q/K 点积通常会变化？

### 模块 3 验收

1. 能从二维旋转矩阵推导 `x*cos + rotate_half(x)*sin`。
2. 能验证 position 0、shape、配对范数与同位置点积不变量。
3. 能拒绝奇数 `Dh`，并区分 split-half 与 adjacent-pair。

<a id="module-4"></a>
## 模块 4：向量化 RoPE 与 position offset

### 4.1 频率、角度与广播

教学版 inverse frequencies 使用 FP32 基准：

```text
inv_freq [Dh/2]
positions [B,S]
angles = positions[...,None] * inv_freq       [B,S,Dh/2]
cos/sin = cat(angles, angles, dim=-1)         [B,S,Dh]
```

Q/K 已 transpose 为 `[B,H,S,Dh]`。将 `cos/sin` 增加 head 轴得到 `[B,1,S,Dh]`，即可对所有 heads 广播。position 必须是整数 Tensor、非负，并与序列长度匹配。

RoPE 的 attention 点积只依赖 Q/K position 的角度差。若所有 Q/K positions 同时加相同 offset，相对位置不变，因此未 mask 的完整 scores 应保持。

### 4.2 运行前预测

给定 `B=1,S=3,Dh=4`，写出 `inv_freq`、`angles`、扩展后 `cos/sin` 与 q 的 shape。预测 positions 从 `[0,1,2]` 改成 `[7,8,9]` 后 q 数值是否相同，完整 scores 是否相同。

```python
import math
import torch

def rotate_half(x):
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires even Dh")
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def build_rope(positions, head_dim, sequence, base=100.0):
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even Dh")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be an integer tensor")
    if positions.ndim != 2 or positions.shape[1] != sequence:
        raise ValueError(f"positions must be [B,{sequence}]")
    if (positions < 0).any():
        raise ValueError("positions must be non-negative")
    inv_freq = 1.0 / (
        torch.tensor(base, dtype=torch.float32)
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angles = positions.float().unsqueeze(-1) * inv_freq
    full_angles = torch.cat((angles, angles), dim=-1)
    return inv_freq, angles, full_angles.cos(), full_angles.sin()

def apply_rope(x, cos, sin):
    if x.shape[-1] != cos.shape[-1] or cos.shape != sin.shape:
        raise ValueError("cos/sin last dimension must match Q/K")
    try:
        torch.broadcast_shapes(x.shape, cos.unsqueeze(1).shape)
    except RuntimeError as error:
        raise ValueError("cos/sin cannot broadcast to Q/K") from error
    return x.float() * cos.unsqueeze(1) + rotate_half(x.float()) * sin.unsqueeze(1)

torch.manual_seed(11)
q = torch.randn(1, 2, 3, 4)
k = torch.randn(1, 1, 3, 4)
positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
offset_positions = positions + 7

inv_freq, angles, cos, sin = build_rope(positions, q.shape[-1], q.shape[-2])
_, _, offset_cos, offset_sin = build_rope(offset_positions, q.shape[-1], q.shape[-2])
q_rotated = apply_rope(q, cos, sin)
k_rotated = apply_rope(k, cos, sin)
q_offset = apply_rope(q, offset_cos, offset_sin)
k_offset = apply_rope(k, offset_cos, offset_sin)

scores = torch.einsum("bhsd,bhtd->bhst", q_rotated, k_rotated) / math.sqrt(4)
offset_scores = torch.einsum("bhsd,bhtd->bhst", q_offset, k_offset) / math.sqrt(4)

print("inv_freq/angles/cos:", tuple(inv_freq.shape), tuple(angles.shape), tuple(cos.shape))
print("q/rotated q:", tuple(q.shape), tuple(q_rotated.shape))
print("position changes q max diff:", (q_rotated - q_offset).abs().max().item())
print("common-offset score max diff:", (scores - offset_scores).abs().max().item())

assert tuple(inv_freq.shape) == (2,)
assert tuple(angles.shape) == (1, 3, 2)
assert tuple(cos.shape) == (1, 3, 4)
assert q_rotated.shape == q.shape and k_rotated.shape == k.shape
torch.testing.assert_close(q_rotated[:, :, 0], q[:, :, 0], atol=0, rtol=0)
assert (q_rotated - q_offset).abs().max() > 1e-3
torch.testing.assert_close(scores, offset_scores, atol=2e-6, rtol=2e-6)

bad_positions = (
    torch.tensor([[0.0, 1.0, 2.0]]),
    torch.tensor([[0, -1, 2]]),
    torch.tensor([[0, 1]]),
)
for bad in bad_positions:
    try:
        build_rope(bad, head_dim=4, sequence=3)
    except ValueError as error:
        print("controlled error:", error)
```

### 4.3 offset 不变量的边界

共同 offset 保持的是相对角度以及由此产生的 QK scores，不表示旋转后的 Q/K values 本身保持。若只移动 Q positions、不移动 K positions，或两者 offset 不同，相对位置改变，scores 通常也改变。

当前周没有 KV Cache，所以 positions 长度与 `S=T` 一致。以后 decode 时 Q 与 K 的 position 范围可能不同，但广播轴和“相对位置”原则不变。

### 4.4 常见误区

- positions 使用浮点 dtype，静默接受非整数位置语义。
- 把 `[B,S,Dh]` 直接与 `[B,H,S,Dh]` 相乘，误让 batch 轴对齐到 head 轴。
- 只比较 offset 前后的 Q values，却误以为它们应相同。
- cos/sin 长度与 Q/K sequence 不匹配，依赖偶然广播掩盖错误。

### 4.5 练习

<a id="m4-e1-question"></a>
**M4-E1**（[提示](#m4-e1-hint) · [答案](#m4-e1-answer)）：`positions [2,5]`、`Dh=8` 时，写出 `inv_freq`、`angles`、扩展后 `cos/sin` 以及用于 q `[2,6,5,8]` 广播时的 shape。

<a id="m4-e2-question"></a>
**M4-E2**（[提示](#m4-e2-hint) · [答案](#m4-e2-answer)）：为什么 positions `[0,1,2]` 与 `[10,11,12]` 得到的 Q/K values 不同，但同时应用于 Q/K 后的完整 scores 可以相同？

### 模块 4 验收

1. 能写出 `inv_freq -> angles -> cos/sin` 的 shape 流。
2. 能正确广播到 `[B,H,S,Dh]` 并验证共同 offset 的 scores 不变量。
3. 能拒绝 position dtype、负值、长度、奇数 `Dh` 和广播错误。

<a id="module-5"></a>
## 模块 5：dtype 与操作顺序

### 5.1 FP32 频率基准

position 与 inverse frequencies 相乘后，角度误差会进入 `sin/cos`。教学主线把 FP32 构造的 `inv_freq`、`angles`、`cos/sin` 作为参考，这样便于把公式错误与低精度舍入误差分开。模块 5 会把参考结果转换到低精度后与“直接低精度构造”比较；后续必修微型 GQA 仍整体使用 FP32 oracle，不演示完整的 model-dtype 边界。

这不是“整个 attention 必须始终运行在 FP32”的性能处方。生产实现可能只在统计量或角度形成时临时升精度，再转换回 model dtype，也可能缓存 cos/sin 或把旋转融合进 kernel；只要遵守目标实现的数值契约并通过误差验证即可。

BF16 支持依赖当前 CPU 与 PyTorch 后端。实验应在支持时比较误差，不支持时记录并跳过，而不是强迫通过。无论何种 dtype，关键输出必须有限。

### 5.2 QK Norm 与 RoPE 不可交换

目标 Qwen3 顺序是 `QK Norm -> RoPE`：先在每颗 head 的原始通道基底中完成可学习缩放，再按 position 旋转。若 norm weight 全为 1，RMS 标量缩放与正交旋转可能表现出更多可交换性；但非均匀逐通道 weight 会改变方向，而旋转又混合配对通道，因此 `Norm -> RoPE` 与 `RoPE -> Norm` 通常不等价。实现顺序属于 checkpoint 语义的一部分，不能因为 shape 相同就调整。

### 5.3 运行前预测

先预测低精度直接构造角度与 FP32 后转换是否完全相同。再预测非均匀 weight 下交换 QK Norm 与 RoPE 的最大差异是否为 0。

```python
import torch

def rotate_half(x):
    if x.shape[-1] % 2:
        raise ValueError("RoPE requires even Dh")
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def rms_norm_last(x, weight, eps=1e-6):
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("weight width mismatch")
    if eps <= 0:
        raise ValueError("eps must be positive")
    x32 = x.float()
    normalized = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
    return normalized * weight.float()

def fp32_cos_sin(positions, head_dim, base=100.0):
    inv = 1.0 / (
        torch.tensor(base, dtype=torch.float32)
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angles = positions.float().unsqueeze(-1) * inv
    full = torch.cat((angles, angles), dim=-1)
    return full.cos(), full.sin()

def apply_rope(x, cos, sin):
    return x.float() * cos.unsqueeze(1).float() + rotate_half(x.float()) * sin.unsqueeze(1).float()

positions = torch.tensor([[0, 17, 63]], dtype=torch.long)
q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                    [0.5, -1.0, 2.0, 0.25],
                    [3.0, 1.0, -2.0, 0.75]]]])
weight = torch.tensor([0.6, 1.3, 1.8, 0.9])
cos32, sin32 = fp32_cos_sin(positions, head_dim=4)

correct = apply_rope(rms_norm_last(q, weight), cos32, sin32)
wrong_order = rms_norm_last(apply_rope(q, cos32, sin32), weight)
order_difference = (correct - wrong_order).abs().max()

print("order max difference:", order_difference.item())
assert order_difference > 1e-3
assert torch.isfinite(correct).all() and torch.isfinite(wrong_order).all()

try:
    bf16 = torch.bfloat16
    base_low = torch.tensor(100.0, dtype=bf16)
    exponent_low = torch.arange(0, 4, 2, dtype=bf16) / 4
    inv_low = 1.0 / (base_low ** exponent_low)
    angles_low = positions.to(bf16).unsqueeze(-1) * inv_low
    full_low = torch.cat((angles_low, angles_low), dim=-1)
    cos_low, sin_low = full_low.cos(), full_low.sin()
    baseline_low = torch.cat((cos32, sin32), dim=-1).to(bf16).float()
    direct_low = torch.cat((cos_low, sin_low), dim=-1).float()
    error = (baseline_low - direct_low).abs().max()
    print("BF16 direct vs FP32-then-cast max error:", error.item())
    assert torch.isfinite(direct_low).all()
except RuntimeError as error:
    print("BF16 path unsupported on this CPU/backend; skipped:", error)
```

### 5.4 如何解释误差

FP32 后转换与全程低精度是两条不同路径。误差大小取决于 position、频率、dtype 和后端；本教程不把某个固定误差数字当作普遍结论。关键是明确参考路径、打印最大绝对误差并检查有限性。

操作顺序差异不是浮点噪声造成的偶然现象。非均匀 weight 与旋转矩阵通常不交换，错误顺序改变了几何变换本身。

### 5.5 常见误区

- 把“BF16 能运行”误写成所有 CPU 必然支持所有 BF16 原语。
- 频率先低精度计算，再声称它与 FP32 基准完全等价。
- 用全 1 weight 测顺序，未能暴露逐通道缩放与旋转的冲突。
- 只检查误差，不检查 NaN/Inf。

### 5.6 练习

<a id="m5-e1-question"></a>
**M5-E1**（[提示](#m5-e1-hint) · [答案](#m5-e1-answer)）：为什么本教程把 inverse frequencies、angles、cos/sin 的 FP32 计算作为基准，而不是要求所有步骤都沿输入 dtype？

<a id="m5-e2-question"></a>
**M5-E2**（[提示](#m5-e2-hint) · [答案](#m5-e2-answer)）：用矩阵乘法语言解释非均匀对角 weight `W` 与旋转矩阵 `R` 为什么通常满足 `RW != WR`。

### 模块 5 验收

1. 能建立 FP32 RoPE 参考路径并量化低精度最大绝对误差。
2. 能在 BF16 不受支持时非阻塞跳过，并始终检查有限性。
3. 能用非均匀 weight 证明 `QK Norm -> RoPE` 与反序不等价。

<a id="module-6"></a>
## 模块 6：接入 causal GQA

### 6.1 正确数据流

本模块把前五个模块接到第四周 GQA。这里同时存在三种“保持 shape 的变换”，最容易因 shape 全部合法而掩盖顺序错误，因此必须按目标架构保持：

```text
hidden [B,S,D]
-> input RMSNorm [B,S,D]
-> Q projection [B,S,Hq*Dh]；K/V projection [B,S,Hkv*Dh]
-> reshape q [B,S,Hq,Dh]；k/v [B,S,Hkv,Dh]
-> q_norm/k_norm along Dh；V 不归一化
-> transpose q [B,Hq,S,Dh]；k/v [B,Hkv,T,Dh]
-> split-half RoPE(Q/K)；V 不旋转
-> physical or logical causal GQA
-> context [B,Hq,S,Dh] -> output [B,S,D]
```

### 6.2 运行前预测

固定 `B=1,S=T=3,D=8,Hq=2,Hkv=1,Dh=4,G=2`。先写出 reshape 前后 Q/K/V、scores、grouped q、context 与 output shape，并预测物理/逻辑路径哪些结果应一致。

```python
import math
import torch

def rms_norm(x, weight, eps=1e-6):
    if eps <= 0 or weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("invalid RMSNorm configuration")
    x32 = x.float()
    return x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps) * weight.float()

def rotate_half(x):
    if x.shape[-1] % 2:
        raise ValueError("RoPE requires even Dh")
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def rope_values(positions, head_dim, base=100.0):
    inv = 1.0 / (
        torch.tensor(base, dtype=torch.float32)
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angles = positions.float().unsqueeze(-1) * inv
    full = torch.cat((angles, angles), dim=-1)
    return full.cos().unsqueeze(1), full.sin().unsqueeze(1)

def apply_rope(x, cos, sin):
    return x.float() * cos + rotate_half(x.float()) * sin

B, S, D, Hq, Hkv, Dh = 1, 3, 8, 2, 1, 4
G = Hq // Hkv
hidden = torch.tensor([[[1.0, 0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 1.0],
                        [0.0, 1.0, 2.0, -1.0, 1.0, -0.5, 1.5, 0.5],
                        [1.5, -1.0, 0.5, 1.0, -0.5, 2.0, 1.0, 0.0]]])
input_weight = torch.tensor([0.8, 1.1, 0.9, 1.2, 1.0, 0.7, 1.3, 0.85])
q_weight = torch.tensor([0.7, 1.2, 1.5, 0.9])
k_weight = torch.tensor([1.1, 0.8, 1.4, 0.6])

normalized_hidden = rms_norm(hidden, input_weight)
q_projected = normalized_hidden @ torch.eye(D)
k_projected = normalized_hidden @ torch.tensor([
    [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.5],
])
v_projected = normalized_hidden @ torch.tensor([
    [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.5],
    [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0],
])

q_reshaped = q_projected.view(B, S, Hq, Dh)
k_reshaped = k_projected.view(B, S, Hkv, Dh)
v_reshaped = v_projected.view(B, S, Hkv, Dh)
q = rms_norm(q_reshaped, q_weight).transpose(1, 2)
k = rms_norm(k_reshaped, k_weight).transpose(1, 2)
v = v_reshaped.transpose(1, 2)
positions = torch.tensor([[0, 1, 2]])
cos, sin = rope_values(positions, Dh)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)

mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
repeated_k = k.repeat_interleave(G, dim=1)
repeated_v = v.repeat_interleave(G, dim=1)
physical_scores = torch.einsum("bhsd,bhtd->bhst", q, repeated_k) / math.sqrt(Dh)
physical_probabilities = torch.softmax(
    physical_scores.masked_fill(mask.view(1, 1, S, S), float("-inf")), dim=-1
)
physical_context = torch.einsum("bhst,bhtd->bhsd", physical_probabilities, repeated_v)

grouped_q = q.reshape(B, Hkv, G, S, Dh)
grouped_scores = torch.einsum("bhgsd,bhtd->bhgst", grouped_q, k) / math.sqrt(Dh)
grouped_probabilities = torch.softmax(
    grouped_scores.masked_fill(mask.view(1, 1, 1, S, S), float("-inf")), dim=-1
)
grouped_context = torch.einsum("bhgst,bhtd->bhgsd", grouped_probabilities, v)
logical_scores = grouped_scores.reshape(B, Hq, S, S)
logical_probabilities = grouped_probabilities.reshape(B, Hq, S, S)
logical_context = grouped_context.reshape(B, Hq, S, Dh)

physical_output = physical_context.transpose(1, 2).contiguous().view(B, S, D)
logical_output = logical_context.transpose(1, 2).contiguous().view(B, S, D)

print("q/k/v:", tuple(q.shape), tuple(k.shape), tuple(v.shape))
print("scores/context/output:", tuple(physical_scores.shape), tuple(physical_context.shape), tuple(physical_output.shape))
torch.testing.assert_close(physical_scores, logical_scores, atol=1e-6, rtol=1e-6)
torch.testing.assert_close(physical_probabilities, logical_probabilities, atol=1e-6, rtol=1e-6)
torch.testing.assert_close(physical_context, logical_context, atol=1e-6, rtol=1e-6)
torch.testing.assert_close(physical_output, logical_output, atol=1e-6, rtol=1e-6)
future = physical_probabilities.masked_select(mask.view(1, 1, S, S).expand_as(physical_probabilities))
assert torch.equal(future, torch.zeros_like(future))
torch.testing.assert_close(physical_probabilities.sum(dim=-1), torch.ones(B, Hq, S), atol=1e-6, rtol=0)
assert torch.isfinite(physical_output).all()
```

### 6.3 两条 GQA 路径

物理路径把 K/V 从 `[B,Hkv,T,Dh]` 连续复制到 `[B,Hq,T,Dh]`。逻辑路径把 Q reshape 为 `[B,Hkv,G,S,Dh]`，K/V 保持原 shape。两者只要采用相同的连续 head 映射，就应得到一致的 scores、probabilities、context 和 output。

V 虽然被物理复制或逻辑共享，但它没有经过 QK Norm 或 RoPE。input RMSNorm 则在三个 projection 之前共同作用于 hidden。

### 6.4 常见误区

- 顺序写成 projection -> transpose -> QK Norm，却沿错了最后一维语义。
- 对 V 套用 K 的 norm weight 或 RoPE。
- RoPE 后才 reshape heads，导致 split-half 配对跨越 head 边界。
- 只比较最终 output，不比较两条 GQA 路径的中间 scores 与概率。

### 6.5 练习

<a id="m6-e1-question"></a>
**M6-E1**（[提示](#m6-e1-hint) · [答案](#m6-e1-answer)）：给定 `hidden [2,5,12]`、`Hq=3,Hkv=1,Dh=4`，写出从 input RMSNorm 到 q/k/v transpose 后、RoPE 后、scores 和 output 的关键 shape。

<a id="m6-e2-question"></a>
**M6-E2**（[提示](#m6-e2-hint) · [答案](#m6-e2-answer)）：列出 V 与 Q/K 在本周数据流中的共同操作和不同操作，并说明为什么 V 不参与 RoPE。

### 模块 6 验收

1. 能按 Qwen3 顺序画出完整 attention 子层数据流。
2. 能同时实现物理复制与逻辑分组，并比较四类结果。
3. 能验证 causal future=0、概率和=1、输出有限且为 `[B,S,D]`。

<a id="capstone"></a>
## 综合任务：带 normalization 与 position 的微型 causal GQA

固定配置：

```text
B=1, S=T=3, D=8
Hq=2, Hkv=1, Dh=4, G=2
positions=[0,1,2]
```

完整执行：

```text
hidden -> input RMSNorm -> Q/K/V projection -> reshape [B,S,H,Dh]
       -> QK Norm along Dh -> transpose -> split-half RoPE(Q/K)
       -> physical/logical causal GQA -> merge heads -> output projection
```

这是 attention 子层的数据流实验，不是完整 Decoder Block。运行前先写出 shape ledger，并预测 position 0、配对范数、同位置点积、共同 offset、causal 概率和两条 GQA 路径的不变量。

```python
import math
import torch

def rms_norm(x, weight, eps=1e-6):
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("RMSNorm weight width mismatch")
    x32 = x.float()
    result = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
    result = result * weight.float()
    if not torch.isfinite(result).all():
        raise ValueError("RMSNorm produced NaN or Inf")
    return result

def rotate_half(x):
    if x.shape[-1] % 2:
        raise ValueError("RoPE requires an even head dimension")
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def build_rope(positions, head_dim, sequence, base=100.0):
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be integer")
    if positions.ndim != 2 or positions.shape[-1] != sequence:
        raise ValueError(f"positions must have shape [B,{sequence}]")
    if (positions < 0).any():
        raise ValueError("positions must be non-negative")
    inv_freq = 1.0 / (
        torch.tensor(base, dtype=torch.float32)
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angles = positions.float().unsqueeze(-1) * inv_freq
    full_angles = torch.cat((angles, angles), dim=-1)
    return inv_freq, angles, full_angles.cos(), full_angles.sin()

def apply_rope(x, cos, sin):
    if cos.shape != sin.shape or cos.shape[-1] != x.shape[-1]:
        raise ValueError("cos/sin width must match Q/K")
    try:
        torch.broadcast_shapes(x.shape, cos.unsqueeze(1).shape)
    except RuntimeError as error:
        raise ValueError("cos/sin cannot broadcast to Q/K") from error
    result = x.float() * cos.unsqueeze(1) + rotate_half(x.float()) * sin.unsqueeze(1)
    if not torch.isfinite(result).all():
        raise ValueError("RoPE produced NaN or Inf")
    return result

def pair_squared_norms(x):
    half = x.shape[-1] // 2
    return x[..., :half].square() + x[..., half:].square()

def causal_probabilities(scores, mask):
    try:
        expanded = mask.expand_as(scores)
    except RuntimeError as error:
        raise ValueError("causal mask cannot broadcast to scores") from error
    if expanded.all(dim=-1).any():
        raise ValueError("causal mask contains a fully blocked row")
    probabilities = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("attention probabilities contain NaN or Inf")
    return probabilities

B, S, T, D = 1, 3, 3, 8
Hq, Hkv, Dh, G = 2, 1, 4, 2
if Hq % Hkv != 0 or G != Hq // Hkv:
    raise ValueError("invalid GQA head counts")

hidden = torch.tensor([[[1.0, 0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 1.0],
                        [0.0, 1.0, 2.0, -1.0, 1.0, -0.5, 1.5, 0.5],
                        [1.5, -1.0, 0.5, 1.0, -0.5, 2.0, 1.0, 0.0]]])
positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
input_weight = torch.tensor([0.8, 1.1, 0.9, 1.2, 1.0, 0.7, 1.3, 0.85])
q_norm_weight = torch.tensor([0.7, 1.2, 1.5, 0.9])
k_norm_weight = torch.tensor([1.1, 0.8, 1.4, 0.6])

q_projection = torch.eye(8)
k_projection = torch.tensor([
    [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.5],
])
v_projection = torch.tensor([
    [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.5],
    [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0],
])
output_projection = torch.tensor([
    [1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1],
    [0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
])

# 1. input RMSNorm -> projections -> reshape [B,S,H,Dh]
normalized_hidden = rms_norm(hidden, input_weight)
q_projected = normalized_hidden @ q_projection
k_projected = normalized_hidden @ k_projection
v_projected = normalized_hidden @ v_projection
q_reshaped = q_projected.view(B, S, Hq, Dh)
k_reshaped = k_projected.view(B, T, Hkv, Dh)
v_reshaped = v_projected.view(B, T, Hkv, Dh)

# 2. QK Norm 沿 Dh；V 保持未归一化值，然后 transpose。
q_normed = rms_norm(q_reshaped, q_norm_weight)
k_normed = rms_norm(k_reshaped, k_norm_weight)
q_pre_rope = q_normed.transpose(1, 2)
k_pre_rope = k_normed.transpose(1, 2)
v = v_reshaped.transpose(1, 2)

# 3. split-half RoPE 只作用于 Q/K，频率和角度以 FP32 构造。
inv_freq, angles, cos, sin = build_rope(positions, Dh, S)
q = apply_rope(q_pre_rope, cos, sin)
k = apply_rope(k_pre_rope, cos, sin)

# 4. 物理复制 GQA。
mask = torch.triu(torch.ones(S, T, dtype=torch.bool), diagonal=1).view(1, 1, S, T)
repeated_k = k.repeat_interleave(G, dim=1)
repeated_v = v.repeat_interleave(G, dim=1)
physical_scores = torch.einsum("bhsd,bhtd->bhst", q, repeated_k) / math.sqrt(Dh)
physical_probabilities = causal_probabilities(physical_scores, mask)
physical_context = torch.einsum("bhst,bhtd->bhsd", physical_probabilities, repeated_v)
physical_merged = physical_context.transpose(1, 2).contiguous().view(B, S, D)
physical_output = physical_merged @ output_projection

# 5. 逻辑分组 GQA，不复制 K/V。
grouped_q = q.reshape(B, Hkv, G, S, Dh)
grouped_scores = torch.einsum("bhgsd,bhtd->bhgst", grouped_q, k) / math.sqrt(Dh)
grouped_mask = mask.unsqueeze(2)
grouped_probabilities = causal_probabilities(grouped_scores, grouped_mask)
grouped_context = torch.einsum("bhgst,bhtd->bhgsd", grouped_probabilities, v)
logical_scores = grouped_scores.reshape(B, Hq, S, T)
logical_probabilities = grouped_probabilities.reshape(B, Hq, S, T)
logical_context = grouped_context.reshape(B, Hq, S, Dh)
logical_merged = logical_context.transpose(1, 2).contiguous().view(B, S, D)
logical_output = logical_merged @ output_projection

# 6. 共同 position offset：Q/K values 改变，但完整 scores 保持。
_, _, offset_cos, offset_sin = build_rope(positions + 9, Dh, S)
q_offset = apply_rope(q_pre_rope, offset_cos, offset_sin)
k_offset = apply_rope(k_pre_rope, offset_cos, offset_sin)
offset_scores = torch.einsum(
    "bhsd,bhtd->bhst", q_offset, k_offset.repeat_interleave(G, dim=1)
) / math.sqrt(Dh)

# 7. 错误顺序：先 RoPE，再用非均匀 weight 做 QK Norm。
wrong_q = rms_norm(apply_rope(q_reshaped.transpose(1, 2), cos, sin), q_norm_weight)
wrong_k = rms_norm(apply_rope(k_reshaped.transpose(1, 2), cos, sin), k_norm_weight)
wrong_scores = torch.einsum(
    "bhsd,bhtd->bhst", wrong_q, wrong_k.repeat_interleave(G, dim=1)
) / math.sqrt(Dh)

expected_shapes = {
    "hidden": (1, 3, 8), "normalized_hidden": (1, 3, 8),
    "q_projected": (1, 3, 8), "k_projected": (1, 3, 4), "v_projected": (1, 3, 4),
    "q_reshaped": (1, 3, 2, 4), "k_reshaped": (1, 3, 1, 4), "v_reshaped": (1, 3, 1, 4),
    "q": (1, 2, 3, 4), "k": (1, 1, 3, 4), "v": (1, 1, 3, 4),
    "inv_freq": (2,), "angles": (1, 3, 2), "cos": (1, 3, 4),
    "repeated_k": (1, 2, 3, 4), "physical_scores": (1, 2, 3, 3),
    "physical_context": (1, 2, 3, 4), "grouped_q": (1, 1, 2, 3, 4),
    "grouped_scores": (1, 1, 2, 3, 3), "physical_output": (1, 3, 8),
}
values = locals()
for name, expected in expected_shapes.items():
    assert tuple(values[name].shape) == expected, (name, values[name].shape)

assert normalized_hidden.shape == hidden.shape
assert q_normed.shape == q_reshaped.shape and k_normed.shape == k_reshaped.shape
assert q.shape == q_pre_rope.shape and k.shape == k_pre_rope.shape
torch.testing.assert_close(q[:, :, 0], q_pre_rope[:, :, 0], atol=0, rtol=0)
torch.testing.assert_close(k[:, :, 0], k_pre_rope[:, :, 0], atol=0, rtol=0)
torch.testing.assert_close(pair_squared_norms(q), pair_squared_norms(q_pre_rope), atol=2e-6, rtol=2e-6)
torch.testing.assert_close(pair_squared_norms(k), pair_squared_norms(k_pre_rope), atol=2e-6, rtol=2e-6)
pre_dots = (q_pre_rope * k_pre_rope.repeat_interleave(G, dim=1)).sum(dim=-1)
post_dots = (q * repeated_k).sum(dim=-1)
torch.testing.assert_close(pre_dots, post_dots, atol=2e-6, rtol=2e-6)
torch.testing.assert_close(physical_scores, offset_scores, atol=3e-6, rtol=3e-6)

for probabilities in (physical_probabilities, logical_probabilities):
    future = probabilities.masked_select(mask.expand_as(probabilities))
    assert torch.equal(future, torch.zeros_like(future))
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(B, Hq, S), atol=1e-6, rtol=0)

torch.testing.assert_close(physical_scores, logical_scores, atol=1e-6, rtol=1e-6)
torch.testing.assert_close(physical_probabilities, logical_probabilities, atol=1e-6, rtol=1e-6)
torch.testing.assert_close(physical_context, logical_context, atol=1e-6, rtol=1e-6)
torch.testing.assert_close(physical_output, logical_output, atol=1e-6, rtol=1e-6)
assert physical_output.shape == (1, 3, 8)
assert all(torch.isfinite(tensor).all() for tensor in (
    normalized_hidden, q, k, v, physical_scores, physical_probabilities,
    physical_context, physical_output, logical_output,
))
assert (physical_scores - wrong_scores).abs().max() > 1e-3
assert torch.equal(v, v_reshaped.transpose(1, 2))

print("shape ledger checks passed")
print("common-offset max score difference:", (physical_scores - offset_scores).abs().max().item())
print("wrong-order max score difference:", (physical_scores - wrong_scores).abs().max().item())
print("physical output:\n", physical_output)
print("all capstone checks passed")
```

### 综合任务结果解释

- input RMSNorm、QK Norm 与 RoPE 都保持各自输入 shape，但归一化轴和作用对象不同。
- position 0 的角度为 0，因此 Q/K 第 0 个位置严格保持；其他位置只在每个 split-half 通道平面内旋转。
- Q/K 同位置点积保持；所有 positions 同加 offset 后，任意 query-key 相对角度不变，所以完整 scores 保持。
- causal mask 使未来概率严格为 0，softmax 沿 `T` 后每行和约为 1。
- 物理与逻辑 GQA 保持相同 head 映射，因此四类结果一致。
- 非均匀 QK Norm weight 暴露了错误操作顺序；差异不是 shape 错误，而是几何语义错误。

### 综合任务验收

1. 所有断言通过，并能解释每一组断言对应哪个不变量。
2. 不看代码写出从 `[1,3,8]` 到 `[1,3,8]` 的完整 shape ledger。
3. 能指出 V 从 projection 到 context 经历了什么，以及明确没有经历什么。
4. 能从 offset、物理/逻辑差异或错误顺序差异定位首个错误边界。

<a id="acceptance"></a>
## 最终验收

先独立作答，再点击提示和答案。通过标准：C1-C10 至少 8/10；S1-S5 至少 4/5；综合任务全部断言通过；能脱离代码画出正确数据流并解释每个归一化、旋转和 softmax 轴。

### 概念题 C1-C10

<a id="c1-question"></a>
**C1**（[提示](#c1-hint) · [答案](#c1-answer)）：RMSNorm 与 LayerNorm 在“是否减均值”和统计量上有何区别？

<a id="c2-question"></a>
**C2**（[提示](#c2-hint) · [答案](#c2-answer)）：为什么 input RMSNorm 与 QK Norm 不能互相替代？

<a id="c3-question"></a>
**C3**（[提示](#c3-hint) · [答案](#c3-answer)）：Qwen3 attention 中 QK Norm 应放在 reshape 和 RoPE 的什么位置？沿哪个轴？

<a id="c4-question"></a>
**C4**（[提示](#c4-hint) · [答案](#c4-answer)）：split-half `rotate_half` 如何配对通道？为什么不能与 adjacent-pair 混用？

<a id="c5-question"></a>
**C5**（[提示](#c5-hint) · [答案](#c5-answer)）：RoPE 为什么不改变 Q/K shape，却能让 attention 感知相对位置？

<a id="c6-question"></a>
**C6**（[提示](#c6-hint) · [答案](#c6-answer)）：position 0、配对范数、同位置 Q/K 点积分别具有什么不变量？

<a id="c7-question"></a>
**C7**（[提示](#c7-hint) · [答案](#c7-answer)）：为什么 Q/K positions 同加 offset 后 scores 保持，而旋转后的 Q/K values 通常不保持？

<a id="c8-question"></a>
**C8**（[提示](#c8-hint) · [答案](#c8-answer)）：为什么频率和角度以 FP32 为教学基准？BF16 不支持时应如何处理？

<a id="c9-question"></a>
**C9**（[提示](#c9-hint) · [答案](#c9-answer)）：为什么非均匀 QK Norm weight 下，`Norm -> RoPE` 与 `RoPE -> Norm` 通常不等价？

<a id="c10-question"></a>
**C10**（[提示](#c10-hint) · [答案](#c10-answer)）：V 为什么不做 QK Norm 和 RoPE？物理/逻辑 GQA 为什么仍可共享或复制 V？

### 推导题 S1-S5

<a id="s1-question"></a>
**S1**（[提示](#s1-hint) · [答案](#s1-answer)）：`x [3,5,12]` 做 input RMSNorm，写出统计量与输出 shape；若 reshape 为 q `[3,5,3,4]`，写出 QK Norm 统计量 shape。

<a id="s2-question"></a>
**S2**（[提示](#s2-hint) · [答案](#s2-answer)）：`positions [2,7]`、`Dh=8`，推导 `inv_freq`、`angles`、`cos/sin` 以及广播到 q `[2,6,7,8]` 时的 shape。

<a id="s3-question"></a>
**S3**（[提示](#s3-hint) · [答案](#s3-answer)）：对 `x=[a,b,c,d,e,f]`，写出 split-half 通道对与 `rotate_half(x)`。

<a id="s4-question"></a>
**S4**（[提示](#s4-hint) · [答案](#s4-answer)）：`B=2,S=T=5,Hq=8,Hkv=2,Dh=4`，求 `G`、q/k/v、物理 K、grouped q、scores、context shape。

<a id="s5-question"></a>
**S5**（[提示](#s5-hint) · [答案](#s5-answer)）：写出本周正确数据流，并标注 input RMSNorm、QK Norm、RoPE、softmax 各自作用轴或对象，以及 V 跳过的步骤。

### 评分建议

- C1-C10 每题 1 分，至少 8 分。
- S1-S5 每题 2 分，至少答对 4 题。
- 综合任务所有断言通过，且能解释 shape 合法但语义错误的归一化轴、旋转布局、position 广播与操作顺序。
- 只背公式但不能画出 `hidden -> output` 数据流，不算通过。

<a id="hints"></a>
## 提示

### 模块练习提示

<a id="m1-e1-hint"></a>**M1-E1：** `keepdim=True` 后，被归约的最后一维长度变为 1。

<a id="m1-e2-hint"></a>**M1-E2：** 区分乘 weight 前的 normalized 与乘 weight 后的 output。

<a id="m2-e1-hint"></a>**M2-E1：** 每个 `[Dh]` 向量产生一个标量统计量。

<a id="m2-e2-hint"></a>**M2-E2：** 广播只验证尺寸兼容，不理解 head 与通道语义。

<a id="m3-e1-hint"></a>**M3-E1：** 前半 `[1,2]` 与后半 `[3,4]` 按相同索引配对。

<a id="m3-e2-hint"></a>**M3-E2：** 同一正交矩阵满足 `R^T R=I`；不同位置对应不同旋转矩阵。

<a id="m4-e1-hint"></a>**M4-E1：** `Dh/2=4`，head 广播轴需要显式插入长度 1。

<a id="m4-e2-hint"></a>**M4-E2：** scores 取决于两者角度之差，而不是各自绝对角度。

<a id="m5-e1-hint"></a>**M5-E1：** 低精度舍入会先影响角度，再进入非线性的 sin/cos。

<a id="m5-e2-hint"></a>**M5-E2：** 非均匀 `W` 对不同坐标缩放不同，`R` 会混合这些坐标。

<a id="m6-e1-hint"></a>**M6-E1：** Q 宽度为 `Hq*Dh=12`，K/V 宽度为 4；RoPE 不改 shape。

<a id="m6-e2-hint"></a>**M6-E2：** 三者共享 input RMSNorm 后的输入和 projection/reshape；只有 Q/K 形成 scores。

### 最终验收提示

<a id="c1-hint"></a>**C1：** 比较 `mean(x^2)` 与 `mean((x-mean(x))^2)`。

<a id="c2-hint"></a>**C2：** 两者的输入 shape、时机和统计宽度不同。

<a id="c3-hint"></a>**C3：** projection 后先获得明确的 head 轴和 `Dh`。

<a id="c4-hint"></a>**C4：** 把最后一维切成两个连续半区。

<a id="c5-hint"></a>**C5：** 旋转只在线性空间内部改 values；QK 点积包含角度差。

<a id="c6-hint"></a>**C6：** 使用 `cos(0)=1`、正交旋转和同一个 `R`。

<a id="c7-hint"></a>**C7：** 共同增加的角度在 Q/K 相减时抵消。

<a id="c8-hint"></a>**C8：** 区分参考精度与硬件可选路径。

<a id="c9-hint"></a>**C9：** 比较 `R(Wx)` 与 `W(Rx)`。

<a id="c10-hint"></a>**C10：** V 不参与 score 匹配，但参与概率加权汇总。

<a id="s1-hint"></a>**S1：** 沿最后一维归约并保留维度。

<a id="s2-hint"></a>**S2：** 从 `[B,S]` 乘 `[Dh/2]` 开始。

<a id="s3-hint"></a>**S3：** 前半 `[a,b,c]`，后半 `[d,e,f]`。

<a id="s4-hint"></a>**S4：** `G=Hq/Hkv`，物理路径扩 K/V，逻辑路径拆 Q head 轴。

<a id="s5-hint"></a>**S5：** 从 `[B,S,D]` 开始，按规格顺序逐行写，不要把 transpose 提前。

<a id="answers"></a>
## 参考答案

### 模块练习答案

<a id="m1-e1-answer"></a>**M1-E1：** 沿最后的 `D=8` 计算，保留维度时统计量为 `[2,3,1]`，输出仍为 `[2,3,8]`。每个 batch 的每个 token 独立得到一个 inverse RMS。

<a id="m1-e2-answer"></a>**M1-E2：** 乘 weight 前，统一标量缩放使 `mean(normalized^2)` 约为 1；乘非均匀 weight 后，各通道平方分别乘不同的 `weight^2`，其平均值通常不再为 1。

<a id="m2-e1-answer"></a>**M2-E1：** q 统计量 `[2,5,6,1]`、输出 `[2,5,6,4]`；k 统计量 `[2,5,2,1]`、输出 `[2,5,2,4]`。每个统计量只覆盖固定 batch、token、head 的 4 个通道。

<a id="m2-e2-answer"></a>**M2-E2：** 沿 H 轴归约后仍可用长度 `Dh` 的 weight 广播回原 shape，所以尺寸检查可能通过，但它让不同 heads 在同一通道上共享统计量。V 保持 projection/reshape 的值，不应用 QK Norm。

<a id="m3-e1-answer"></a>**M3-E1：** 两对是 `(x[0],x[2])=(1,3)` 与 `(x[1],x[3])=(2,4)`；`rotate_half(x)=[-3,-4,1,2]`。

<a id="m3-e2-answer"></a>**M3-E2：** 同角度时 `dot(Rq,Rk)=q^T R^T R k=q^T k`。不同 position 使用 `Rq`、`Rk`，中间成为 `Rq^T Rk`，它编码角度差，通常不再是恒等矩阵。

<a id="m4-e1-answer"></a>**M4-E1：** `inv_freq [4]`，`angles [2,5,4]`，扩展后 `cos/sin [2,5,8]`；插入 head 轴后为 `[2,1,5,8]`，广播到 q `[2,6,5,8]`。

<a id="m4-e2-answer"></a>**M4-E2：** offset 改变每个绝对角度，所以 Q/K values 改变；但 Q/K 同加 10 后任意位置对的角度差不变，因此由点积形成的完整 scores 保持。

<a id="m5-e1-answer"></a>**M5-E1：** FP32 减少 inverse frequency 与大 position 相乘时的舍入，并为 sin/cos 提供稳定参考。目标低精度可在参考值形成后转换；若硬件不支持 BF16 原语，应跳过该对照而不是修改正确性基准。

<a id="m5-e2-answer"></a>**M5-E2：** `W` 是不同对角元素的缩放，`R` 混合一对坐标。先缩放再混合为 `RWx`，先混合再缩放为 `WRx`；除非权重在被混合坐标上相同等特殊情况，否则两个矩阵乘积不同。

<a id="m6-e1-answer"></a>**M6-E1：** input RMSNorm `[2,5,12]`；投影 Q `[2,5,12]`、K/V `[2,5,4]`；reshape Q `[2,5,3,4]`、K/V `[2,5,1,4]`；QK Norm shape 不变；transpose/RoPE 后 q `[2,3,5,4]`、k `[2,1,5,4]`、v `[2,1,5,4]`；scores `[2,3,5,5]`；context `[2,3,5,4]`；output `[2,5,12]`。

<a id="m6-e2-answer"></a>**M6-E2：** Q/K/V 都从 input RMSNorm 后的 hidden 独立 projection，并 reshape/transpose。只有 Q/K 做逐 head QK Norm 和 RoPE，因为它们通过点积形成位置相关 scores；V 只在概率确定后被加权汇总。GQA 可以复制或共享 V，是因为多个 query heads 使用同一 KV head 的内容。

### 最终验收答案

<a id="c1-answer"></a>**C1：** RMSNorm 使用 `mean(x^2)`，不减均值；LayerNorm 先减均值，再按中心化方差缩放，因此会重新居中。

<a id="c2-answer"></a>**C2：** input RMSNorm 在 projection 前作用于 `[B,S,D]`，沿 D；QK Norm 在 projection/reshape 后分别作用于 Q/K `[B,S,H,Dh]`，沿 Dh。它们规范不同表示、不同阶段，不能替代。

<a id="c3-answer"></a>**C3：** 顺序是 Q/K projection -> reshape `[B,S,H,Dh]` -> QK Norm along Dh -> transpose `[B,H,S,Dh]` -> RoPE -> attention。

<a id="c4-answer"></a>**C4：** split-half 将前半与后半同索引配对，`[x1,x2] -> [-x2,x1]`。adjacent-pair 使用不同通道排列；混用会让 cos/sin 与权重解释对应到错误通道。

<a id="c5-answer"></a>**C5：** RoPE 只对最后一维做等宽旋转，因此 shape 不变。不同 position 使用不同角度，Q/K 点积包含角度差，从而携带相对位置信息。

<a id="c6-answer"></a>**C6：** position 0 是恒等；每个 split-half 通道对的二范数保持；同 position 使用同一旋转时 Q/K 点积保持。

<a id="c7-answer"></a>**C7：** Q/K values 各自使用新的绝对角度，所以会改变；score 依赖角度差，共同 offset 在差值中抵消，因此保持。

<a id="c8-answer"></a>**C8：** 频率、角度和三角函数对舍入敏感，FP32 提供稳定参考。BF16 是可选误差实验；不支持时记录并跳过，同时保留 FP32 必修路径与有限性检查。

<a id="c9-answer"></a>**C9：** 非均匀 weight 是对角缩放，RoPE 会混合配对通道；一般 `R(Wx) != W(Rx)`，所以必须保持目标实现规定的顺序。

<a id="c10-answer"></a>**C10：** 在目标 Qwen3 数据流中，Q/K 决定匹配 scores，因此对它们应用逐 head QK Norm 和 RoPE；这不是所有 attention 架构的普遍要求。V 不参与匹配，只被 probabilities 汇总。GQA 的多个 query heads 共享同一 KV head，所以 V 可被逻辑共享或物理复制，但目标路径不对它做 QK Norm 或旋转。

<a id="s1-answer"></a>**S1：** input RMSNorm 统计量 `[3,5,1]`、输出 `[3,5,12]`；q QK Norm 统计量 `[3,5,3,1]`、输出 `[3,5,3,4]`。

<a id="s2-answer"></a>**S2：** `inv_freq [4]`；`angles [2,7,4]`；扩展后 `cos/sin [2,7,8]`；插入 head 轴为 `[2,1,7,8]`，广播到 `[2,6,7,8]`。

<a id="s3-answer"></a>**S3：** 前半 `[a,b,c]`、后半 `[d,e,f]`，通道对为 `(a,d)`、`(b,e)`、`(c,f)`；`rotate_half(x)=[-d,-e,-f,a,b,c]`。

<a id="s4-answer"></a>**S4：** `G=8/2=4`；q `[2,8,5,4]`，k/v `[2,2,5,4]`；物理 K `[2,8,5,4]`；grouped q `[2,2,4,5,4]`；恢复后的 scores `[2,8,5,5]`，context `[2,8,5,4]`。

<a id="s5-answer"></a>**S5：** `hidden [B,S,D] -> input RMSNorm along D -> Q/K/V projection -> reshape [B,S,H,Dh] -> QK Norm(Q/K) along Dh，V 跳过 -> transpose [B,H,S,Dh] -> split-half RoPE(Q/K)，V 跳过 -> scores [B,Hq,S,T] -> causal mask -> softmax along T -> context -> merge -> output [B,S,D]`。

<a id="glossary"></a>
## 术语与速查表

### 公式速查

| 运算 | 公式 | 核心不变量/检查 |
| --- | --- | --- |
| RMSNorm | `x * rsqrt(mean(x^2)+eps) * weight` | FP32 统计；沿最后一维；shape 不变 |
| QK Norm | 对 q/k 的每个 `[Dh]` 向量做 RMSNorm | 每 token、每 head 独立；V 跳过 |
| rotate_half | `[x1,x2] -> [-x2,x1]` | split-half；`Dh` 必须为偶数 |
| RoPE | `x*cos + rotate_half(x)*sin` | Q/K shape 不变；V 跳过 |
| scores | `Q @ K^T / sqrt(Dh)` | `[B,Hq,S,T]` |
| causal softmax | `softmax(masked_scores, dim=T)` | future=0；行和约 1；无全屏蔽行 |

### Shape ledger

| 阶段 | Q | K | V |
| --- | --- | --- | --- |
| projection | `[B,S,Hq*Dh]` | `[B,S,Hkv*Dh]` | `[B,S,Hkv*Dh]` |
| reshape | `[B,S,Hq,Dh]` | `[B,S,Hkv,Dh]` | `[B,S,Hkv,Dh]` |
| QK Norm | shape 不变 | shape 不变 | 不执行 |
| transpose | `[B,Hq,S,Dh]` | `[B,Hkv,T,Dh]` | `[B,Hkv,T,Dh]` |
| RoPE | shape 不变 | shape 不变 | 不执行 |
| GQA context | `[B,Hq,S,Dh]` | 共享/复制 | 共享/复制 |

### RoPE 广播

| 张量 | Shape |
| --- | --- |
| positions | `[B,S]` |
| inv_freq | `[Dh/2]` |
| angles | `[B,S,Dh/2]` |
| cos/sin | `[B,S,Dh]` |
| 广播视图 | `[B,1,S,Dh]` |
| q/k | `[B,H,S,Dh]` |

### 必查错误

| 风险 | 必须检查 |
| --- | --- |
| RMSNorm | weight width 匹配；`eps>0`；结果有限 |
| Q/K | 最后一维相同；沿 `Dh` 归一化 |
| RoPE | `Dh` 偶数；split-half 布局一致；cos/sin 可广播 |
| positions | 整数 Tensor；非负；长度匹配 |
| GQA | `Hq % Hkv == 0`；连续 head 映射一致 |
| causal mask | softmax 前应用；不存在全屏蔽行 |
| dtype | FP32 频率/角度基准；低精度结果有限 |

方案边界速记：causal mask 规定可见性，RoPE 提供位置相关的 QK 几何；input RMSNorm 控制子层输入，QK Norm 控制投影后每颗 Q/K head。它们解决的问题不同，不能互相替代。

正确顺序速记：

```text
input RMSNorm
-> Q/K/V projection
-> reshape [B,S,H,Dh]
-> q_norm/k_norm along Dh（V 跳过）
-> transpose [B,H,S,Dh]
-> split-half RoPE(Q/K，V 跳过)
-> causal GQA
-> merge heads
-> output projection
```

<a id="next-week"></a>
## Week 6 预告

第六周将学习 SwiGLU 与 Dense MLP：用 gate/up/down 三个投影和 SiLU 构造另一个保持外部 `[B,S,D] -> [B,S,D]` 接口的子层。本周得到的是正确加入 normalization 与 position 的 attention 输出；第六周先独立完成前馈子层，第七周再把 attention、MLP、RMSNorm 与残差组合成完整 Dense Decoder。
