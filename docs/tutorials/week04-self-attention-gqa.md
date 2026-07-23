# 第四周教程：Self-Attention 与 GQA

> 本周目标：从可手算的 Q/K/V 出发，走通 scaled dot-product attention、causal mask、多头拆合、完整 MHA，以及物理复制和逻辑分组两种 GQA。

## 目录

1. [学习目标](#goals)
2. [学习方式](#method)
3. [环境与边界](#environment)
4. [模块 1：单头 scaled dot-product attention](#module-1)
5. [模块 2：causal mask](#module-2)
6. [模块 3：多头拆分与合并](#module-3)
7. [模块 4：完整 Multi-Head Attention](#module-4)
8. [模块 5：GQA 物理复制 K/V](#module-5)
9. [模块 6：GQA 逻辑分组](#module-6)
10. [综合任务](#capstone)
11. [最终验收](#acceptance)
12. [提示](#hints)
13. [参考答案](#answers)
14. [术语与速查表](#glossary)
15. [下一周预告](#next-week)

<a id="goals"></a>
## 学习目标

完成本周后，你应该能：

1. 解释 Q、K、V，以及 scores 最后两维 `[S,T]` 的含义。
2. 写出 `Q @ K.transpose(-2,-1) / sqrt(Dh)` 的 shape 流。
3. 在 softmax 前应用 causal mask，并验证未来位置概率为 0。
4. 在 `[B,S,D]` 与 `[B,H,S,Dh]` 之间正确拆头和合头。
5. 从 hidden 经 Q/K/V 投影、attention 和 output projection 得到 `[B,S,D]`。
6. 由 `Hq/Hkv` 算出 GQA 组大小，并拒绝不能整除的配置。
7. 实现并对照物理复制与逻辑分组两种 GQA。

建议投入 6-10 小时。CPU 可以完成全部必修内容。

<a id="method"></a>
## 学习方式：Predict-Run-Explain

每个实验都按同一顺序学习：

1. **Predict**：先写出 shape、缩放轴、mask values 或预期错误。
2. **Run**：运行代码，不要只阅读输出。
3. **Explain**：脱离代码解释每个轴、收缩维和不变量。

遇到错误时按顺序检查：shape → head 数整除关系 → transpose 轴 → mask 广播 → softmax 维度 → 数值有限性。

<a id="environment"></a>
## 环境与边界

在仓库根目录运行：

```bash
uv sync --locked
uv run python scripts/check_environment.py
uv run pytest
```

本周所有代码只依赖 PyTorch，固定使用 CPU 小张量，不下载模型、Tokenizer、checkpoint 或数据集。

本周只实现 causal self-attention，不加入 padding mask、dropout、RoPE、KV Cache、cross-attention、训练循环或完整 Decoder Block。没有 KV Cache，所以必修示例中 `T=S`；仍保留两个字母，因为 scores 的 `S` 轴枚举 query 位置，`T` 轴枚举 key 位置。

统一记号：

```text
B    batch size
S    query 长度
T    key/value 长度（本周 T=S）
D    hidden width
Hq   query head 数
Hkv  key/value head 数
Dh   每个 head 的宽度
G    每个 KV head 服务的 query head 数，G=Hq/Hkv
```

<a id="module-1"></a>
## 模块 1：单头 scaled dot-product attention

### 1.1 为什么先只看一颗 head

attention 的核心只有三步：

```text
scores        = Q @ K.T / sqrt(Dh)
probabilities = softmax(scores, dim=key_axis)
context       = probabilities @ V
```

Q 表示“当前位置要寻找什么”，K 表示“每个位置提供什么索引特征”，V 表示“被选中后实际汇总什么内容”。Q 与 K 产生权重，权重再混合 V。

### 1.2 运行前预测

给定 `Q [2,2]`、`K [3,2]`、`V [3,2]`：

- `scores` 是什么 shape？
- softmax 沿哪个轴？
- `context` 是什么 shape？
- 缩放因子是 `sqrt(2)`、`sqrt(3)` 还是其他值？

```python
import math
import torch

q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])          # [S=2,Dh=2]
k = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])  # [T=3,Dh=2]
v = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])  # [T=3,Dh=2]

if q.shape[-1] != k.shape[-1]:
    raise ValueError(f"Q/K head dimensions differ: {q.shape[-1]} vs {k.shape[-1]}")
if k.shape[-2] != v.shape[-2]:
    raise ValueError(f"K/V lengths differ: {k.shape[-2]} vs {v.shape[-2]}")

raw_scores = q @ k.T
scores = raw_scores / math.sqrt(q.shape[-1])
probabilities = torch.softmax(scores, dim=-1)
context = probabilities @ v

print("raw scores:\n", raw_scores)
print("scaled scores:\n", scores)
print("probabilities:\n", probabilities)
print("context:\n", context)
print("shape flow:", tuple(q.shape), tuple(scores.shape), tuple(context.shape))

assert tuple(scores.shape) == (2, 3)
assert tuple(context.shape) == (2, 2)
torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))

try:
    bad_k = torch.ones(3, 4)
    if q.shape[-1] != bad_k.shape[-1]:
        raise ValueError(f"Q/K head dimensions differ: {q.shape[-1]} vs {bad_k.shape[-1]}")
except ValueError as error:
    print("attention error:", error)
```

### 1.3 解释输出

`q @ k.T` 收缩 `Dh`，留下 query 位置 `S` 与 key 位置 `T`，所以 scores 是 `[S,T]`。softmax 必须沿 `T`：同一个 query 位置在所有 key 候选之间分配权重。

除以 `sqrt(Dh)` 能减缓 head dimension 增大时点积方差增长，避免 softmax 过早进入极尖区域。缩放依据是参与点积求和的宽度 `Dh`，不是 `D`、`S` 或 `T`。

### 1.4 常见误区

- 把 `K.T @ Q` 当成同一操作：它留下的轴和语义不同。
- 沿 query 轴 softmax：这会让不同 query 位置彼此归一化。
- 用 `sqrt(D)` 缩放：多头拆分后每次点积只收缩 `Dh`。
- 认为 V 参与 scores：V 只在概率确定后被加权汇总。

### 1.5 练习

<a id="m1-e1-question"></a>
**M1-E1**（[提示](#m1-e1-hint) · [答案](#m1-e1-answer)）：`Q [4,3]`、`K [6,3]`、`V [6,5]` 时，写出 scores、probabilities 和 context shape，并指出 softmax 轴。

<a id="m1-e2-question"></a>
**M1-E2**（[提示](#m1-e2-hint) · [答案](#m1-e2-answer)）：若 `Dh=4`，raw score 为 8，scaled score 是多少？为什么不能除以 `sqrt(S)`？

### 模块 1 验收

1. 能否口述 `[S,Dh] @ [Dh,T] -> [S,T]`？
2. 能否解释 scores 的行和列分别代表什么？
3. 能否说明缩放为什么使用 `sqrt(Dh)`？

<a id="module-2"></a>
## 模块 2：causal mask

### 2.1 mask 解决什么问题

在 next-token 建模中，query 位置 `s` 只能读取 key 位置 `t <= s`。causal mask 把 `t > s` 的 scores 在 softmax 前改为 `-inf`：这些项指数为 0，因此概率严格为 0。

### 2.2 运行前预测

对长度 3 的序列，先写出 `[3,3]` 的布尔上三角 mask。第 0、1、2 行分别允许几个 key 位置？

```python
import torch

def causal_softmax(scores, mask):
    try:
        torch.broadcast_shapes(scores.shape, mask.shape)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot broadcast to scores {tuple(scores.shape)}"
        ) from error

    blocked = mask.expand_as(scores)
    if blocked.all(dim=-1).any():
        raise ValueError("at least one query row masks every key position")

    masked_scores = scores.masked_fill(mask, float("-inf"))
    return masked_scores, torch.softmax(masked_scores, dim=-1)

scores = torch.tensor([[[[2.0, 1.0, 3.0],
                         [1.0, 4.0, 2.0],
                         [0.0, 1.0, 2.0]]]])  # [B=1,H=1,S=3,T=3]
s, t = scores.shape[-2:]
mask = torch.triu(torch.ones(s, t, dtype=torch.bool), diagonal=1).view(1, 1, s, t)
masked_scores, probabilities = causal_softmax(scores, mask)

print("mask:\n", mask[0, 0])
print("masked scores:\n", masked_scores[0, 0])
print("probabilities:\n", probabilities[0, 0])
print("row sums:", probabilities.sum(dim=-1))

assert torch.equal(probabilities.masked_select(mask), torch.zeros(mask.sum()))
torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1, 1, 3))
assert torch.isfinite(probabilities).all()

for bad_mask in (torch.zeros(2, 2, dtype=torch.bool), torch.ones_like(scores, dtype=torch.bool)):
    try:
        causal_softmax(scores, bad_mask)
    except ValueError as error:
        print("mask error:", error)
```

### 2.3 解释输出

上三角 `t>s` 为 `True`，表示禁止读取未来。第 0 行只保留位置 0；第 1 行保留 0 和 1；第 2 行保留全部三个位置。

mask 可以从 `[S,T]` 或 `[1,1,S,T]` 广播到 `[B,H,S,T]`，但不能破坏最后两维的位置语义。若整行全部变成 `-inf`，softmax 会得到 NaN，因此主线代码在计算前拒绝这种输入。

不能先 softmax 再简单把未来概率设为 0：剩余项之和会小于 1。除非再次归一化，否则它不再是合法概率分布。

### 2.4 常见误区

- 把允许位置设为 `-inf`：mask 的布尔方向反了。
- 在 softmax 后 mask，却忘记重新归一化。
- mask 对角线：当前位置通常允许读取自己。
- 只检查概率和，不检查未来概率：错误轴也可能产生看似正常的和。

### 2.5 练习

<a id="m2-e1-question"></a>
**M2-E1**（[提示](#m2-e1-hint) · [答案](#m2-e1-answer)）：写出 `S=T=4` 时的 causal mask，并说明第 2 个 query 位置允许读取哪些 key 位置。

<a id="m2-e2-question"></a>
**M2-E2**（[提示](#m2-e2-hint) · [答案](#m2-e2-answer)）：为什么全 `-inf` 的一行会产生 NaN？主线实现应在 softmax 前还是后拒绝它？

### 模块 2 验收

1. 能否不运行代码写出长度 4 的 causal mask？
2. 能否解释为什么未来概率严格为 0？
3. 能否同时验证未来 0、行和 1、全部有限？

<a id="module-3"></a>
## 模块 3：多头拆分与合并

### 3.1 shape 变换

若 `D=H*Dh`，拆头不是改变元素，只是重新解释和排列轴：

```text
[B,S,D] -> [B,S,H,Dh] -> [B,H,S,Dh]
```

合头执行逆过程：

```text
[B,H,S,Dh] -> [B,S,H,Dh] -> [B,S,D]
```

### 3.2 运行前预测

`hidden [2,3,8]`、`H=4` 时，`Dh` 是多少？transpose 后 Tensor 是否通常 contiguous？

```python
import torch

def split_heads(hidden, num_heads):
    batch, sequence, width = hidden.shape
    if width % num_heads != 0:
        raise ValueError(f"width {width} is not divisible by heads {num_heads}")
    head_dim = width // num_heads
    return hidden.view(batch, sequence, num_heads, head_dim).transpose(1, 2)

def merge_heads(heads):
    batch, num_heads, sequence, head_dim = heads.shape
    return heads.transpose(1, 2).contiguous().view(batch, sequence, num_heads * head_dim)

hidden = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
heads = split_heads(hidden, num_heads=4)
restored = merge_heads(heads)

print("hidden/heads/restored:", tuple(hidden.shape), tuple(heads.shape), tuple(restored.shape))
print("heads contiguous:", heads.is_contiguous())
print("values restored:", torch.equal(hidden, restored))

assert tuple(heads.shape) == (2, 4, 3, 2)
assert torch.equal(hidden, restored)

try:
    split_heads(hidden, num_heads=3)
except ValueError as error:
    print("split error:", error)
```

### 3.3 解释 `transpose` 与 `contiguous`

`view` 先把 `D` 拆成 `H,Dh`，`transpose(1,2)` 再把 head 轴移到序列轴前面，使 attention 能按 head 并行计算。transpose 通常只改变 stride，因此结果可能 non-contiguous。

合头时先 transpose 回 `[B,S,H,Dh]`，再调用 `contiguous()` 建立符合当前顺序的连续布局，最后安全地 view 为 `[B,S,D]`。

### 3.4 常见误区

- `D/H` 得到浮点数后直接当 shape：应先验证整除并用整数除法。
- 忘记 transpose：得到 `[B,S,H,Dh]`，scores 的收缩轴可能仍对，但 head/位置语义错位。
- 在 non-contiguous Tensor 上盲目 `view`。
- 合头时只 reshape，不先恢复 `[B,S,H,Dh]` 轴顺序。

### 3.5 练习

<a id="m3-e1-question"></a>
**M3-E1**（[提示](#m3-e1-hint) · [答案](#m3-e1-answer)）：`hidden [3,5,12]` 拆成 6 heads，写出 `Dh` 和拆头后的 shape。

<a id="m3-e2-question"></a>
**M3-E2**（[提示](#m3-e2-hint) · [答案](#m3-e2-answer)）：解释为什么 `[B,H,S,Dh]` 不能直接按内存顺序 view 成 `[B,S,D]`。

### 模块 3 验收

1. 能否写出拆头和合头的完整轴顺序？
2. 能否解释 `D % H == 0`？
3. 能否用 values 而不只是 shape 验证变换可逆？

<a id="module-4"></a>
## 模块 4：完整 Multi-Head Attention

### 4.1 从 hidden 到 output

MHA 先用三个线性层把同一个 hidden 投影为 Q/K/V，再按 heads 拆分：

```text
hidden [B,S,D]
-> q/k/v projected [B,S,D]
-> q/k/v [B,H,S,Dh]
-> scores/probabilities [B,H,S,T]
-> context [B,H,S,Dh]
-> merged [B,S,D]
-> output [B,S,D]
```

本模块使用 `Hq=Hkv=H`，每颗 query head 与同编号的 K/V head 一一对应。

### 4.2 运行前预测

固定 `B=1,S=T=3,D=4,H=2,Dh=2`。写出 Q、scores、context、output 的 shape，并预测 causal mask 在每颗 head 上是否相同。

```python
import math
import torch
from torch import nn

def split_heads(x, num_heads):
    batch, sequence, width = x.shape
    if width % num_heads != 0:
        raise ValueError(f"width {width} is not divisible by heads {num_heads}")
    return x.view(batch, sequence, num_heads, width // num_heads).transpose(1, 2)

def merge_heads(x):
    batch, heads, sequence, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, sequence, heads * head_dim)

class TinyMHA(nn.Module):
    def __init__(self, width=4, num_heads=2):
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width {width} is not divisible by heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden):
        q = split_heads(self.q_proj(hidden), self.num_heads)
        k = split_heads(self.k_proj(hidden), self.num_heads)
        v = split_heads(self.v_proj(hidden), self.num_heads)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        sequence = hidden.shape[1]
        mask = torch.triu(
            torch.ones(sequence, sequence, dtype=torch.bool, device=hidden.device), diagonal=1
        ).view(1, 1, sequence, sequence)
        probabilities = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        context = probabilities @ v
        output = self.o_proj(merge_heads(context))
        return output, probabilities, (q, k, v, scores, context, mask)

model = TinyMHA().eval()
with torch.no_grad():
    identity = torch.eye(4)
    for projection in (model.q_proj, model.k_proj, model.v_proj, model.o_proj):
        projection.weight.copy_(identity)

hidden = torch.tensor([[[1.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 0.0, 1.0]]])

with torch.inference_mode():
    output, probabilities, parts = model(hidden)
    q, k, v, scores, context, mask = parts
    assert tuple(q.shape) == (1, 2, 3, 2)
    assert tuple(scores.shape) == (1, 2, 3, 3)
    assert tuple(context.shape) == (1, 2, 3, 2)
    assert tuple(output.shape) == (1, 3, 4)
    assert torch.equal(probabilities.masked_select(mask.expand_as(probabilities)), torch.zeros(6))
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1, 2, 3), atol=1e-6, rtol=0)
    assert torch.isfinite(output).all()

print("q/scores/context/output:", tuple(q.shape), tuple(scores.shape),
      tuple(context.shape), tuple(output.shape))
print("head 0 probabilities:\n", probabilities[0, 0])
```

### 4.3 解释投影和输出

Q/K/V 投影可以学习不同的特征子空间。即使三个输入都是 hidden，它们也承担不同职责。每颗 head 独立生成 `[S,T]` 概率矩阵与 `[S,Dh]` context，合头后重新得到宽度 `D`，output projection 再混合各 head 的结果。

`eval()` 切换模块模式；`inference_mode()` 关闭 autograd 并施加更强的纯推理限制。本例所有依赖 inference tensor 的校验都留在同一上下文中。

### 4.4 常见误区

- 认为 Q/K/V 必须具有相同 values：它们只需要可兼容的 shape。
- 忘记 output projection：教学上可暂时省略，但完整 MHA 数据流包含它。
- 为每颗 head 构造不同 causal mask：序列位置限制通常相同，可广播复用。
- 把 `[B,H,S,T]` 的 head 轴与 query 位置轴交换。

### 4.5 练习

<a id="m4-e1-question"></a>
**M4-E1**（[提示](#m4-e1-hint) · [答案](#m4-e1-answer)）：`hidden [2,5,12]`、`H=3` 时，写出 q、scores、context、merged 和 output shape。

<a id="m4-e2-question"></a>
**M4-E2**（[提示](#m4-e2-hint) · [答案](#m4-e2-answer)）：为什么 Q/K/V 都来自同一个 hidden，却需要三个独立投影？

### 模块 4 验收

1. 能否从 `[B,S,D]` 逐步推导回 `[B,S,D]`？
2. 能否指出 scores 和 probabilities 的 softmax 轴？
3. 能否解释 MHA 中 query heads 与 KV heads 的一一对应？

<a id="module-5"></a>
## 模块 5：GQA 物理复制 K/V

### 5.1 从 MHA 到 GQA

MHA 使用 `Hq=Hkv`。Grouped Query Attention 允许 `Hq>Hkv`，多个 query heads 共享同一组 K/V。必须满足：

```text
Hq % Hkv == 0
G = Hq // Hkv
```

最直观的参考实现是沿 head 轴物理复制 K/V，使其临时变成 `Hq` 颗 heads，再复用 MHA 公式。

### 5.2 运行前预测

`Hq=4,Hkv=2` 时 `G` 是多少？query heads 0-3 分别使用哪个 KV head？K/V 的 `numel` 会变成原来的几倍？

```python
import math
import torch

def repeat_kv(k, v, num_query_heads):
    num_kv_heads = k.shape[1]
    if v.shape[1] != num_kv_heads:
        raise ValueError(f"K/V head counts differ: {num_kv_heads} vs {v.shape[1]}")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"query heads {num_query_heads} are not divisible by KV heads {num_kv_heads}"
        )
    group_size = num_query_heads // num_kv_heads
    return (
        k.repeat_interleave(group_size, dim=1),
        v.repeat_interleave(group_size, dim=1),
        group_size,
    )

q = torch.arange(1 * 4 * 2 * 2, dtype=torch.float32).reshape(1, 4, 2, 2) / 10
k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]],
                   [[1.0, 1.0], [1.0, -1.0]]]])
v = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]],
                   [[3.0, 1.0], [1.0, 3.0]]]])

repeated_k, repeated_v, group_size = repeat_kv(k, v, q.shape[1])
scores = q @ repeated_k.transpose(-2, -1) / math.sqrt(q.shape[-1])
probabilities = torch.softmax(scores, dim=-1)
context = probabilities @ repeated_v

print("group size:", group_size)
print("head mapping:", [head // group_size for head in range(q.shape[1])])
print("K shapes:", tuple(k.shape), "->", tuple(repeated_k.shape))
print("K numel:", k.numel(), "->", repeated_k.numel())
print("context shape:", tuple(context.shape))

assert group_size == 2
assert torch.equal(repeated_k[:, 0], k[:, 0])
assert torch.equal(repeated_k[:, 1], k[:, 0])
assert torch.equal(repeated_k[:, 2], k[:, 1])
assert repeated_k.numel() == group_size * k.numel()
assert tuple(context.shape) == (1, 4, 2, 2)

try:
    repeat_kv(k, v, num_query_heads=3)
except ValueError as error:
    print("GQA error:", error)
```

### 5.3 head 映射

`repeat_interleave(G, dim=1)` 连续复制每个 KV head。`Hq=4,Hkv=2,G=2` 时：

```text
query head 0,1 -> KV head 0
query head 2,3 -> KV head 1
```

这个实现语义清晰，适合作为正确性基线，但复制后的 K/V 逻辑元素数是原来的 `G` 倍。真实高性能实现通常不会真的创建这份完整复制。

### 5.4 常见误区

- 用普通 `repeat` 得到交错顺序，导致 head 映射错误。
- 只检查 `Hq>=Hkv`，却不检查整除。
- 复制 query 而不是 K/V：GQA 保留更多 query heads。
- 把教学 Tensor 的 `numel` 对比直接等同于真实 kernel 显存表现。

### 5.5 练习

<a id="m5-e1-question"></a>
**M5-E1**（[提示](#m5-e1-hint) · [答案](#m5-e1-answer)）：`Hq=12,Hkv=3` 时求 `G`，并写出 query heads 8-11 使用哪个 KV head。

<a id="m5-e2-question"></a>
**M5-E2**（[提示](#m5-e2-hint) · [答案](#m5-e2-answer)）：原始 K 为 `[2,2,7,4]`，扩展到 `Hq=8` 后 shape 和 `numel` 倍率是多少？

### 模块 5 验收

1. 能否由 `Hq/Hkv` 推导 `G` 与 head 映射？
2. 能否解释 `repeat_interleave` 的复制顺序？
3. 能否拒绝 `Hq % Hkv != 0`？

<a id="module-6"></a>
## 模块 6：GQA 逻辑分组

### 6.1 不复制 K/V 的计算

物理路径把 K/V 扩到 `[B,Hq,T,Dh]`。逻辑路径改为把 query heads 分组：

```text
q [B,Hq,S,Dh] -> grouped q [B,Hkv,G,S,Dh]
k/v            保持 [B,Hkv,T,Dh]
scores          [B,Hkv,G,S,T]
context         [B,Hkv,G,S,Dh]
```

由于每个 KV head 轴与它服务的 `G` 个 query heads 放在一起，可以用 `einsum` 直接计算，不创建复制 K/V。

### 6.2 运行前预测

给定 `B=1,Hq=4,Hkv=2,G=2,S=T=3,Dh=2`，写出 grouped q、grouped scores 和恢复后 context shape。

```python
import math
import torch

torch.manual_seed(7)
batch, query_heads, kv_heads, sequence, head_dim = 1, 4, 2, 3, 2
group_size = query_heads // kv_heads
q = torch.randn(batch, query_heads, sequence, head_dim)
k = torch.randn(batch, kv_heads, sequence, head_dim)
v = torch.randn(batch, kv_heads, sequence, head_dim)
mask = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool), diagonal=1)

# 物理复制参考路径
physical_k = k.repeat_interleave(group_size, dim=1)
physical_v = v.repeat_interleave(group_size, dim=1)
physical_scores = torch.einsum("bhsd,bhtd->bhst", q, physical_k) / math.sqrt(head_dim)
physical_probabilities = torch.softmax(
    physical_scores.masked_fill(mask.view(1, 1, sequence, sequence), float("-inf")), dim=-1
)
physical_context = torch.einsum("bhst,bhtd->bhsd", physical_probabilities, physical_v)

# 逻辑分组路径
grouped_q = q.reshape(batch, kv_heads, group_size, sequence, head_dim)
grouped_scores = torch.einsum("bhgsd,bhtd->bhgst", grouped_q, k) / math.sqrt(head_dim)
grouped_probabilities = torch.softmax(
    grouped_scores.masked_fill(mask.view(1, 1, 1, sequence, sequence), float("-inf")), dim=-1
)
grouped_context = torch.einsum("bhgst,bhtd->bhgsd", grouped_probabilities, v)
logical_scores = grouped_scores.reshape(batch, query_heads, sequence, sequence)
logical_probabilities = grouped_probabilities.reshape(batch, query_heads, sequence, sequence)
logical_context = grouped_context.reshape(batch, query_heads, sequence, head_dim)

print("grouped q/scores/context:", tuple(grouped_q.shape), tuple(grouped_scores.shape),
      tuple(grouped_context.shape))
print("physical/logical K numel:", physical_k.numel(), k.numel())

torch.testing.assert_close(logical_scores, physical_scores, atol=1e-6, rtol=1e-6)
torch.testing.assert_close(logical_probabilities, physical_probabilities, atol=1e-6, rtol=1e-6)
torch.testing.assert_close(logical_context, physical_context, atol=1e-6, rtol=1e-6)
assert physical_k.numel() == group_size * k.numel()
assert tuple(logical_context.shape) == (1, 4, 3, 2)
```

### 6.3 为什么两条路径等价

两条路径的 head 对应关系相同。物理版本把每个 K/V head 复制 `G` 次；逻辑版本保留一份 K/V，并在额外的 group 轴上让 `G` 个 query heads 与它计算。只要 reshape 的 head 顺序与 `repeat_interleave` 一致，数学结果相同。

逻辑版本展示了 GQA 节省 K/V 表示规模的直觉，但这个 Python `einsum` 示例不是生产级高性能 kernel，也不用于性能结论。

### 6.4 常见误区

- reshape 为 `[B,G,Hkv,S,Dh]` 后忘记调整映射顺序。
- 将 grouped scores 的 `G` 轴与 `S` 轴混淆。
- 只比较最终 output，不比较中间 scores、概率和 context。
- 宣称没有复制逻辑 Tensor 就一定没有任何底层临时存储。

### 6.5 练习

<a id="m6-e1-question"></a>
**M6-E1**（[提示](#m6-e1-hint) · [答案](#m6-e1-answer)）：`q [2,8,5,4]`、`k [2,2,5,4]` 时写出 `G`、grouped q 和 grouped scores shape。

<a id="m6-e2-question"></a>
**M6-E2**（[提示](#m6-e2-hint) · [答案](#m6-e2-answer)）：为什么逻辑分组版本必须与物理复制版本使用相同的连续 head 分组顺序？

### 模块 6 验收

1. 能否写出 grouped q 与 grouped scores shape？
2. 能否解释逻辑分组如何避免 K/V 的 `G` 倍复制？
3. 能否设计中间值与最终值的等价性断言？

<a id="capstone"></a>
## 综合任务：走通微型 causal GQA

这次把六个模块连接成一个完整的 attention 子层。它从 `hidden [B,S,D]` 开始，执行 Q/K/V 投影、拆头、causal attention、合头和 output projection，并同时实现物理复制与逻辑分组两条 GQA 路径。

固定配置：

```text
B=1, S=T=3, D=4
Hq=2, Hkv=1, Dh=2, G=2
```

它仍然不是完整 Decoder Block：没有 RMSNorm、RoPE、残差、MLP、MoE 或 KV Cache。

### 运行前预测

先在纸上完成：

1. Q/K/V 投影后的 shape，以及拆头后的 shape。
2. 物理复制后 K/V 的 shape 与 `numel` 倍率。
3. 逻辑分组后 Q、scores、context 的 shape。
4. causal mask 的三行分别允许读取哪些位置。
5. `physical_scores[0,0,0,0]` 的值。
6. 两条路径最终 output 的 shape，以及哪些中间张量应该数值一致。

```python
import math
import torch
from torch import nn

class TinyGQA(nn.Module):
    def __init__(self, width=4, query_heads=2, kv_heads=1):
        super().__init__()
        if width % query_heads != 0:
            raise ValueError(f"width {width} is not divisible by query heads {query_heads}")
        if query_heads % kv_heads != 0:
            raise ValueError(
                f"query heads {query_heads} are not divisible by KV heads {kv_heads}"
            )

        self.width = width
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = width // query_heads
        self.group_size = query_heads // kv_heads
        kv_width = kv_heads * self.head_dim

        self.q_proj = nn.Linear(width, query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(width, kv_width, bias=False)
        self.v_proj = nn.Linear(width, kv_width, bias=False)
        self.o_proj = nn.Linear(query_heads * self.head_dim, width, bias=False)

    def split(self, projected, heads):
        batch, sequence, projected_width = projected.shape
        expected_width = heads * self.head_dim
        if projected_width != expected_width:
            raise ValueError(
                f"projected width {projected_width} does not equal heads*Dh {expected_width}"
            )
        return projected.view(batch, sequence, heads, self.head_dim).transpose(1, 2)

    def merge(self, context):
        batch, heads, sequence, head_dim = context.shape
        if heads != self.query_heads or head_dim != self.head_dim:
            raise ValueError(f"unexpected context shape {tuple(context.shape)}")
        return context.transpose(1, 2).contiguous().view(batch, sequence, self.width)

    def forward(self, hidden):
        if hidden.ndim != 3 or hidden.shape[-1] != self.width:
            raise ValueError(f"expected hidden [B,S,{self.width}], got {tuple(hidden.shape)}")

        q_projected = self.q_proj(hidden)
        k_projected = self.k_proj(hidden)
        v_projected = self.v_proj(hidden)
        q = self.split(q_projected, self.query_heads)
        k = self.split(k_projected, self.kv_heads)
        v = self.split(v_projected, self.kv_heads)

        batch, _, sequence, head_dim = q.shape
        mask = torch.triu(
            torch.ones(sequence, sequence, dtype=torch.bool, device=hidden.device), diagonal=1
        )

        # 路径 1：物理复制 K/V 到 Hq 颗 heads。
        repeated_k = k.repeat_interleave(self.group_size, dim=1)
        repeated_v = v.repeat_interleave(self.group_size, dim=1)
        physical_scores = torch.einsum("bhsd,bhtd->bhst", q, repeated_k) / math.sqrt(head_dim)
        physical_probabilities = torch.softmax(
            physical_scores.masked_fill(
                mask.view(1, 1, sequence, sequence), float("-inf")
            ),
            dim=-1,
        )
        physical_context = torch.einsum(
            "bhst,bhtd->bhsd", physical_probabilities, repeated_v
        )
        physical_output = self.o_proj(self.merge(physical_context))

        # 路径 2：Q 增加 group 轴，K/V 保持 Hkv 颗 heads。
        grouped_q = q.reshape(
            batch, self.kv_heads, self.group_size, sequence, self.head_dim
        )
        grouped_scores = torch.einsum("bhgsd,bhtd->bhgst", grouped_q, k) / math.sqrt(head_dim)
        grouped_probabilities = torch.softmax(
            grouped_scores.masked_fill(
                mask.view(1, 1, 1, sequence, sequence), float("-inf")
            ),
            dim=-1,
        )
        grouped_context = torch.einsum(
            "bhgst,bhtd->bhgsd", grouped_probabilities, v
        )
        logical_scores = grouped_scores.reshape(
            batch, self.query_heads, sequence, sequence
        )
        logical_probabilities = grouped_probabilities.reshape(
            batch, self.query_heads, sequence, sequence
        )
        logical_context = grouped_context.reshape(
            batch, self.query_heads, sequence, self.head_dim
        )
        logical_output = self.o_proj(self.merge(logical_context))

        return {
            "q_projected": q_projected,
            "k_projected": k_projected,
            "v_projected": v_projected,
            "q": q,
            "k": k,
            "v": v,
            "mask": mask,
            "repeated_k": repeated_k,
            "repeated_v": repeated_v,
            "physical_scores": physical_scores,
            "physical_probabilities": physical_probabilities,
            "physical_context": physical_context,
            "physical_output": physical_output,
            "grouped_q": grouped_q,
            "grouped_scores": grouped_scores,
            "grouped_probabilities": grouped_probabilities,
            "grouped_context": grouped_context,
            "logical_scores": logical_scores,
            "logical_probabilities": logical_probabilities,
            "logical_context": logical_context,
            "logical_output": logical_output,
        }

model = TinyGQA().eval()
with torch.no_grad():
    model.q_proj.weight.copy_(torch.eye(4))
    model.k_proj.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0]]))
    model.v_proj.weight.copy_(torch.tensor([[0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]]))
    model.o_proj.weight.copy_(torch.eye(4))

hidden = torch.tensor([[[1.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 0.0, 1.0]]])

with torch.inference_mode():
    result = model(hidden)

    expected_shapes = {
        "q_projected": (1, 3, 4),
        "k_projected": (1, 3, 2),
        "v_projected": (1, 3, 2),
        "q": (1, 2, 3, 2),
        "k": (1, 1, 3, 2),
        "v": (1, 1, 3, 2),
        "repeated_k": (1, 2, 3, 2),
        "physical_scores": (1, 2, 3, 3),
        "physical_context": (1, 2, 3, 2),
        "physical_output": (1, 3, 4),
        "grouped_q": (1, 1, 2, 3, 2),
        "grouped_scores": (1, 1, 2, 3, 3),
        "grouped_context": (1, 1, 2, 3, 2),
        "logical_output": (1, 3, 4),
    }
    for name, expected in expected_shapes.items():
        assert tuple(result[name].shape) == expected, (name, result[name].shape)

    mask = result["mask"].view(1, 1, 3, 3)
    for name in ("physical_probabilities", "logical_probabilities"):
        probabilities = result[name]
        future = probabilities.masked_select(mask.expand_as(probabilities))
        assert torch.equal(future, torch.zeros_like(future))
        torch.testing.assert_close(
            probabilities.sum(dim=-1), torch.ones(1, 2, 3), atol=1e-6, rtol=0
        )
        assert torch.isfinite(probabilities).all()

    torch.testing.assert_close(
        result["physical_scores"], result["logical_scores"], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        result["physical_probabilities"],
        result["logical_probabilities"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        result["physical_context"], result["logical_context"], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        result["physical_output"], result["logical_output"], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        result["physical_scores"][0, 0, 0, 0],
        torch.tensor(1 / math.sqrt(2)),
        atol=1e-6,
        rtol=0,
    )
    assert result["repeated_k"].numel() == model.group_size * result["k"].numel()
    assert result["repeated_v"].numel() == model.group_size * result["v"].numel()
    assert torch.isfinite(result["physical_output"]).all()
    assert torch.isfinite(result["logical_output"]).all()

    print("q/k/v:", tuple(result["q"].shape), tuple(result["k"].shape),
          tuple(result["v"].shape))
    print("grouped q/scores:", tuple(result["grouped_q"].shape),
          tuple(result["grouped_scores"].shape))
    print("head 0 probabilities:\n", result["physical_probabilities"][0, 0])
    print("output:\n", result["physical_output"])
    print("all capstone checks passed")
```

### 结果解释

- Q 保留 `Hq=2`，K/V 只保留 `Hkv=1`；两条路径都产生两颗 query heads 的 context。
- `physical_scores[0,0,0,0]=1/sqrt(2)`，因为第 0 颗 query head 的第 0 个向量与第 0 个 key 都是 `[1,0]`。
- 第 0 个 query 位置的概率行为 `[1,0,0]`；第 1 个位置只能在 key 0-1 中归一化；第 2 个位置可以读取全部 key。
- 物理路径的 K/V 元素数变为原来的 `G=2` 倍；逻辑路径增加 Q 的 group 轴，不复制 K/V 表示。
- 两条路径的 scores、probabilities、context 和 output 都一致，说明 head 映射与计算语义相同。

### 综合任务验收

1. 不看代码写出从 `hidden [1,3,4]` 到 `output [1,3,4]` 的全部关键 shape。
2. 解释为什么 K/V 投影宽度是 2，而 Q 投影宽度是 4。
3. 指出物理路径中新增的 `G` 倍张量，以及逻辑路径用什么轴代替复制。
4. 能从任一结果偏差回溯到 projection、head mapping、mask、softmax 或 merge 中的首个错误。

<a id="acceptance"></a>
## 最终验收

先独立作答，再查看提示和答案。

### 概念题 C1-C10

<a id="c1-question"></a>
**C1**（[提示](#c1-hint) · [答案](#c1-answer)）：Q、K、V 分别在 attention 中承担什么职责？

<a id="c2-question"></a>
**C2**（[提示](#c2-hint) · [答案](#c2-answer)）：为什么 `q [B,Hq,S,Dh] @ k.transpose(-2,-1)` 的最后两维是 `[S,T]`？

<a id="c3-question"></a>
**C3**（[提示](#c3-hint) · [答案](#c3-answer)）：scaled dot-product attention 为什么除以 `sqrt(Dh)`，而不是 `sqrt(D)` 或 `sqrt(S)`？

<a id="c4-question"></a>
**C4**（[提示](#c4-hint) · [答案](#c4-answer)）：softmax 为什么沿 key 轴 `T`，而不是 query 轴 `S`？

<a id="c5-question"></a>
**C5**（[提示](#c5-hint) · [答案](#c5-answer)）：causal mask 为什么应在 softmax 前应用？全被 mask 的一行为什么必须提前拒绝？

<a id="c6-question"></a>
**C6**（[提示](#c6-hint) · [答案](#c6-answer)）：拆头后为什么通常需要 transpose？合头前为什么常见 `contiguous()`？

<a id="c7-question"></a>
**C7**（[提示](#c7-hint) · [答案](#c7-answer)）：MHA 与 GQA 的 `Hq/Hkv` 关系有何不同？GQA 共享了什么，没有共享什么？

<a id="c8-question"></a>
**C8**（[提示](#c8-hint) · [答案](#c8-answer)）：为什么 GQA 必须检查 `Hq % Hkv == 0`？

<a id="c9-question"></a>
**C9**（[提示](#c9-hint) · [答案](#c9-answer)）：物理复制和逻辑分组为什么可以数值等价？它们的表示规模有何差别？

<a id="c10-question"></a>
**C10**（[提示](#c10-hint) · [答案](#c10-answer)）：本周明明 `T=S`，为什么仍要分别保留 query 长度 `S` 和 key/value 长度 `T`？

### 推导题 S1-S5

<a id="s1-question"></a>
**S1**（[提示](#s1-hint) · [答案](#s1-answer)）：给定 `q [2,6,5,4]`、`k/v [2,2,7,4]`，求 `G`、grouped q、grouped scores 和恢复后的 scores shape。

<a id="s2-question"></a>
**S2**（[提示](#s2-hint) · [答案](#s2-answer)）：`hidden [3,4,12]`、`Hq=3`、`Hkv=1` 时，求 `Dh`、Q/K/V 投影宽度、拆头后 q/k/v 和最终 output shape。

<a id="s3-question"></a>
**S3**（[提示](#s3-hint) · [答案](#s3-answer)）：写出 `S=T=4` 的 causal mask，并列出 query 位置 2 允许读取的 key 位置。

<a id="s4-question"></a>
**S4**（[提示](#s4-hint) · [答案](#s4-answer)）：`Dh=9`、raw score 为 12 时，scaled score 是多少？

<a id="s5-question"></a>
**S5**（[提示](#s5-hint) · [答案](#s5-answer)）：原始 K 为 `[2,2,7,4]`，扩展到 `Hq=8`，求 `G`、复制后的 shape、复制前后 `numel`。

### 评分建议

- C1-C10 每题 1 分，至少 8 分。
- S1-S5 每题 2 分，至少答对 4 题。
- 综合任务所有断言通过，并能脱离代码口述 `[B,S,D] -> Q/K/V -> scores -> probabilities -> context -> [B,S,D]`。
- 若只会背 shape，不能解释每一维语义、mask 位置和 head 映射，不算通过。

<a id="hints"></a>
## 提示

### 模块练习提示

<a id="m1-e1-hint"></a>**M1-E1：** 矩阵乘法收缩 Q/K 共有的最后一维；context 的最后一维来自 V。

<a id="m1-e2-hint"></a>**M1-E2：** 缩放因子是 `sqrt(Dh)`；点积实际累加的是 `Dh` 项。

<a id="m2-e1-hint"></a>**M2-E1：** `True` 放在主对角线上方；位置编号从 0 开始。

<a id="m2-e2-hint"></a>**M2-E2：** 考虑 `exp(-inf)` 后分子和分母会发生什么。

<a id="m3-e1-hint"></a>**M3-E1：** `Dh=D/H`，拆头后把 head 轴移到序列轴前。

<a id="m3-e2-hint"></a>**M3-E2：** transpose 改变轴的逻辑顺序和 stride，但通常没有重排底层元素。

<a id="m4-e1-hint"></a>**M4-E1：** 本题 MHA 中 `Hq=Hkv=H=3`，`Dh=12/3`。

<a id="m4-e2-hint"></a>**M4-E2：** 输入来源相同不代表要学习的特征子空间和职责相同。

<a id="m5-e1-hint"></a>**M5-E1：** 每 `G` 个连续 query heads 共用一个 KV head。

<a id="m5-e2-hint"></a>**M5-E2：** 先求 `G=Hq/Hkv`，只替换 head 轴的大小。

<a id="m6-e1-hint"></a>**M6-E1：** 从 q 的第二维得到 `Hq=8`，然后把它拆成 `Hkv,G` 两轴。

<a id="m6-e2-hint"></a>**M6-E2：** 比较 `repeat_interleave` 产生的 head 顺序与 reshape 解释的连续内存顺序。

### 最终验收提示

<a id="c1-hint"></a>**C1：** scores 只由 Q/K 产生，V 在概率确定后才被混合。

<a id="c2-hint"></a>**C2：** 写成 `[S,Dh] @ [Dh,T]`，观察哪个轴被收缩。

<a id="c3-hint"></a>**C3：** 缩放要对应点积求和项的数量。

<a id="c4-hint"></a>**C4：** 每一个 query 都要在全部 key 候选之间分配一组权重。

<a id="c5-hint"></a>**C5：** mask 后概率是否仍应和为 1？全 `-inf` 行的 softmax 分母是多少？

<a id="c6-hint"></a>**C6：** attention 希望 head 轴位于序列轴前；`view` 依赖可兼容的 stride。

<a id="c7-hint"></a>**C7：** MHA 是一一对应；GQA 是多个 query heads 对一个 KV head。

<a id="c8-hint"></a>**C8：** 所有 KV heads 是否必须服务相同数量的 query heads？

<a id="c9-hint"></a>**C9：** 两条路径要保持完全相同的 query-to-KV head 映射。

<a id="c10-hint"></a>**C10：** 想一想未来加入历史 KV Cache 后，当前 query 长度和可读历史总长度是否仍相等。

<a id="s1-hint"></a>**S1：** `G=6/2`，将 query head 轴重解释为 `[Hkv,G]`。

<a id="s2-hint"></a>**S2：** `D=Hq*Dh`；K/V 宽度是 `Hkv*Dh`。

<a id="s3-hint"></a>**S3：** mask 中 `t>s` 的位置为 `True`。

<a id="s4-hint"></a>**S4：** `sqrt(9)=3`。

<a id="s5-hint"></a>**S5：** 原始 `numel` 是四个维度之积，复制后乘 `G`。

<a id="answers"></a>
## 参考答案

### 模块练习答案

<a id="m1-e1-answer"></a>**M1-E1：** scores 和 probabilities 均为 `[4,6]`；softmax 沿长度 6 的 key 轴；context 为 `[4,5]`，最后一维来自 `V [6,5]`。

<a id="m1-e2-answer"></a>**M1-E2：** `8/sqrt(4)=4`。不能除以 `sqrt(S)`，因为方差增长来自点积沿 `Dh` 累加，而不是 query 位置数量。

<a id="m2-e1-answer"></a>**M2-E1：** mask 为 `[[F,T,T,T],[F,F,T,T],[F,F,F,T],[F,F,F,F]]`；query 位置 2 允许读取 key 位置 0、1、2。

<a id="m2-e2-answer"></a>**M2-E2：** 全 `-inf` 行的指数全部为 0，归一化成为 `0/0`，产生 NaN；应在 softmax 前拒绝。

<a id="m3-e1-answer"></a>**M3-E1：** `Dh=12/6=2`，拆头后为 `[3,6,5,2]`。

<a id="m3-e2-answer"></a>**M3-E2：** `[B,H,S,Dh]` 的逻辑相邻轴顺序不是 `[B,S,H,Dh]`。直接按当前内存顺序 view 会混淆 head 与序列元素；应先 transpose 回去，再 contiguous/reshape。

<a id="m4-e1-answer"></a>**M4-E1：** `Dh=4`；q `[2,3,5,4]`，scores `[2,3,5,5]`，context `[2,3,5,4]`，merged 和 output 均为 `[2,5,12]`。

<a id="m4-e2-answer"></a>**M4-E2：** 三个独立投影分别学习“查询特征”“索引特征”和“被汇总内容”。共享输入不意味着职责或权重应相同。

<a id="m5-e1-answer"></a>**M5-E1：** `G=12/3=4`；query heads 8-11 都使用 KV head 2。

<a id="m5-e2-answer"></a>**M5-E2：** 原始 `Hkv=2`，`G=8/2=4`；扩展后 K 为 `[2,8,7,4]`，`numel` 是原来的 4 倍。

<a id="m6-e1-answer"></a>**M6-E1：** `Hq=8,Hkv=2,G=4`；grouped q `[2,2,4,5,4]`，grouped scores `[2,2,4,5,5]`。

<a id="m6-e2-answer"></a>**M6-E2：** 物理路径用 `repeat_interleave` 让连续的 `G` 个 query heads 对应同一 KV head。逻辑 reshape 也必须把原 query head 顺序解释为连续的 `[Hkv,G]`，否则两条路径会建立不同映射。

### 最终验收答案

<a id="c1-answer"></a>**C1：** Q 描述当前位置寻找什么，K 描述每个候选位置提供的匹配特征，V 描述该位置被选中后实际汇总的内容。

<a id="c2-answer"></a>**C2：** 每颗 head 上执行 `[S,Dh] @ [Dh,T]`，`Dh` 被收缩，留下 query 位置 `S` 和 key 位置 `T`。

<a id="c3-answer"></a>**C3：** 一个 raw score 沿 `Dh` 累加点积项，方差随 `Dh` 增长；除以 `sqrt(Dh)` 用于控制尺度。`D` 是全部 heads 的总宽度，`S` 是位置数，都不是当前点积的收缩宽度。

<a id="c4-answer"></a>**C4：** 对每个固定 query 位置，需要在 `T` 个 key 候选间形成和为 1 的权重，所以沿最后的 key 轴归一化。

<a id="c5-answer"></a>**C5：** softmax 前把禁止位置设为 `-inf`，可让其概率为 0 且允许位置自动重新归一化。全被 mask 时分母为 0，会产生 NaN，因此应提前拒绝。

<a id="c6-answer"></a>**C6：** 拆头先得到 `[B,S,H,Dh]`，transpose 为 `[B,H,S,Dh]` 才便于每颗 head 独立计算。transpose 通常产生 non-contiguous Tensor；合头时恢复轴顺序并 contiguous 后才能安全 view。

<a id="c7-answer"></a>**C7：** MHA 中 `Hq=Hkv`，query 与 KV heads 一一对应。GQA 中 `Hq>Hkv`，多个 query heads 共享同一 K/V head；query heads 自身及其 Q 表示没有被合并。

<a id="c8-answer"></a>**C8：** 整除保证每个 KV head 服务相同的整数个 query heads，才能定义固定 `G=Hq/Hkv` 和无歧义的连续分组。

<a id="c9-answer"></a>**C9：** 物理路径复制每个 KV head `G` 次，逻辑路径把相同的 `G` 个 query heads 放入对应 KV head 的 group 轴；映射相同，所以计算等价。物理 K/V 的逻辑元素数增大 `G` 倍，逻辑路径保留原 K/V shape。

<a id="c10-answer"></a>**C10：** 本周无 cache，因此数值上 `T=S`；但两个轴语义不同。未来 decode 时当前 query 可有 `S=1`，而历史加当前的 K/V 总长度为 `T=P+S`。

<a id="s1-answer"></a>**S1：** `G=6/2=3`；grouped q `[2,2,3,5,4]`，grouped scores `[2,2,3,5,7]`，恢复后的 scores `[2,6,5,7]`。

<a id="s2-answer"></a>**S2：** `Dh=12/3=4`；Q 投影宽度 12，K/V 投影宽度均为 4；q `[3,3,4,4]`，k/v `[3,1,4,4]`，最终 output `[3,4,12]`。

<a id="s3-answer"></a>**S3：** mask 为 `[[F,T,T,T],[F,F,T,T],[F,F,F,T],[F,F,F,F]]`；query 位置 2 允许读取 key 0、1、2。

<a id="s4-answer"></a>**S4：** `12/sqrt(9)=12/3=4`。

<a id="s5-answer"></a>**S5：** `G=8/2=4`；复制后 `[2,8,7,4]`；原始 `numel=2*2*7*4=112`，复制后 `2*8*7*4=448`。

<a id="glossary"></a>
## 术语与速查表

| 运算/张量 | 输入 | 输出 | 关键检查 |
| --- | --- | --- | --- |
| Q 投影 | hidden `[B,S,D]` | `[B,S,Hq*Dh]` | `D` 与投影宽度来自配置 |
| K/V 投影 | hidden `[B,S,D]` | `[B,S,Hkv*Dh]` | GQA 中可小于 Q 宽度 |
| 拆头 | `[B,S,H*Dh]` | `[B,H,S,Dh]` | 宽度可整除，轴顺序正确 |
| scores | q 与 k | `[B,Hq,S,T]` | 收缩 `Dh`，除以 `sqrt(Dh)` |
| causal mask | scores `[B,Hq,S,T]` | 同 shape | softmax 前屏蔽 `t>s` |
| softmax | masked scores | probabilities `[B,Hq,S,T]` | 沿 `T`，每行和约为 1 |
| context | probabilities 与 v | `[B,Hq,S,Dh]` | 收缩 `T` |
| 合头 | `[B,Hq,S,Dh]` | `[B,S,Hq*Dh]` | transpose 后安全 reshape |
| MHA | `Hq=Hkv` | 每颗 Q head 对一颗 KV head | 一一对应 |
| GQA 分组 | `Hq>Hkv` | `G=Hq/Hkv` | `Hq % Hkv == 0` |
| 物理复制 | k/v `[B,Hkv,T,Dh]` | `[B,Hq,T,Dh]` | 连续复制，元素数乘 `G` |
| 逻辑分组 | q `[B,Hq,S,Dh]` | `[B,Hkv,G,S,Dh]` | K/V 不复制，映射顺序一致 |

调试顺序：先核对 Q/K/V 投影宽度和 `Dh`；再检查拆头轴顺序与 `Hq/Hkv` 整除；然后检查 scores 的 `[S,T]`、mask 布尔方向和 softmax 轴；最后比较 context、合头和 output projection。两条 GQA 路径不一致时，优先打印 query head 到 KV head 的映射。

<a id="next-week"></a>
## 下一周预告

第五周将学习 RMSNorm、RoPE 与 QK Norm。RoPE 会在 attention 计算 scores 前旋转 Q/K 的成对通道，改变与位置相关的数值，但不改变外部 `q/k [B,H,S,Dh]` shape；V 不参与该旋转。

开始前请确保你能画出：Q projection -> 拆头 -> QK Norm（若目标配置启用）-> RoPE -> scaled dot-product attention，并能解释为什么 position 改变后 Q/K 的方向可以变化，而每对旋转分量的二范数应保持不变。
