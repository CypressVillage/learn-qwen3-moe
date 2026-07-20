# 第二周教程：矩阵乘法与 `nn.Module`

**目录**

- [这周要学会什么](#goals)
- [学习方式](#method)
- [开始前的环境检查](#environment-check)
- [模块 1：广播复习与形状语义](#module-1)
- [模块 2：矩阵乘法、批量矩阵乘法与 einsum](#module-2)
- [模块 3：第一个 nn.Module](#module-3)
- [模块 4：parameter、buffer 与普通属性](#module-4)
- [模块 5：手写教学版线性层](#module-5)
- [模块 6：状态、模式、梯度与可复现性](#module-6)
- [综合任务：构建可保存的微型线性模块](#capstone)
- [最终验收](#acceptance)
- [提示](#hints)
- [参考答案](#answers)
- [术语与速查表](#glossary)
- [下一周预告](#next-week)

<a id="goals"></a>
## 这周要学会什么

第一周建立了 values、shape、stride、dtype 和内存直觉。本周把这些知识装进可复用的 PyTorch 模块。完成后，你应该能够：

1. 逐维判断广播、`matmul`、`bmm` 和常用 `einsum` 的输出 shape。
2. 解释 `[B,S,Din] @ [Din,Dout] -> [B,S,Dout]` 中保留、收缩和产生的维度。
3. 区分 `nn.Module`、`nn.Parameter`、buffer 与普通 Python 属性。
4. 不依赖 `nn.Linear` 实现教学版线性层，并与参考层做数值对照。
5. 预测 `parameters()`、`buffers()` 和 `state_dict()` 中有哪些张量。
6. 保存、修改并恢复微型模块状态。
7. 区分 `train()`/`eval()` 与 `no_grad()`/`inference_mode()`。
8. 固定随机种子，并让模块与输入的 dtype/device 保持一致。

本周不下载模型、Tokenizer、checkpoint 或数据集。CPU 是必修路径；CUDA 和低精度实验是可选观察。

<a id="method"></a>
## 学习方式

建议投入 **6-10 小时**：模块 1-2 用 2-3 小时，模块 3-4 用 2-3 小时，模块 5-6 与综合任务用 2-4 小时。

每段核心代码继续遵循 **Predict → Run → Explain**：

1. **Predict：** 写出输入、输出、中间张量 shape 和每个轴的语义。
2. **Run：** 保留输出或完整错误，不先改参数碰运气。
3. **Explain：** 指出哪些维度被广播、保留或收缩，以及状态为何被注册或遗漏。

本周统一使用：`B` 表示 batch，`S` 表示序列长度，`Din` 表示输入宽度，`Dout` 表示输出宽度。

<a id="environment-check"></a>
## 开始前的环境检查

从仓库根目录同步锁定环境并运行权威检查：

```bash
uv --version
uv sync --locked --python 3.11.15
uv run python scripts/check_environment.py
uv run pytest
```

`uv --version` 应以 `uv 0.11.28` 开头，环境检查应总体 `[PASS]`。CPU-only 机器上的 CUDA `[SKIPPED]` 不阻塞学习。把教程代码保存为文件后，使用 `uv run python <文件路径>`，不要调用系统 Python。

<a id="module-1"></a>
## 模块 1：广播复习与形状语义

广播按 shape 从右向左比较：对应维度必须相等、其中一个为 1，或较短 shape 在左侧缺少该维。**PyTorch 只看长度，不认识 `B/S/D` 这些轴名。**

**对 Qwen3-MoE 推理的意义：** 线性层 bias、位置缩放和后续 mask 都依赖广播；shape 能运行但轴语义错位时，模型可能静默地产生错误数值。

### 1.1 三种常见广播

设 `hidden [B,S,D] = [2,3,4]`：

```text
feature_bias [D]      [      4] -> [2,3,4]
position_scale [S,1]  [    3,1] -> [2,3,4]
batch_offset [B,1,1]  [  2,1,1] -> [2,3,4]
```

`torch.arange` 创建确定性序列；逐元素加法和乘法会先应用广播规则。

**Predict：** 运行前分别把三个较短 shape 左侧补 1，写出它们沿哪些轴重复，并手算三个打印结果。

```python
import torch

hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
feature_bias = torch.tensor([10.0, 20.0, 30.0, 40.0])
position_scale = torch.tensor([[1.0], [10.0], [100.0]])
batch_offset = torch.tensor([[[0.0]], [[1000.0]]])

biased = hidden + feature_bias
scaled = hidden * position_scale
shifted = hidden + batch_offset

print("biased shape:", tuple(biased.shape), "first:", biased[0, 0])
print("scaled shape:", tuple(scaled.shape), "positions:", scaled[0, :, 0])
print("shifted shape:", tuple(shifted.shape), "batch starts:", shifted[:, 0, 0])
```

预期观察：三个结果都是 `(2,3,4)`；第一行分别打印 `[10,21,32,43]`、`[0,40,800]` 和 `[0,1012]`。

### 1.2 shape 合法不代表语义正确

若 `B` 和 `S` 恰好都等于 3，shape `[3,1]` 既可能被作者当作位置量，也可能误当作 batch 量。它与 `[3,3,4]` 相加时总会右对齐成 `[1,3,1]`，因此沿 **序列轴** 广播。代码不报错，但如果业务意图是沿 batch 轴应用，就错了；正确 batch shape 应为 `[3,1,1]`。

### 1.3 观察真实广播错误

```python
bad = torch.ones(2, 2)
try:
    hidden + bad
except RuntimeError as error:
    print("hidden shape:", tuple(hidden.shape))
    print("bad shape:", tuple(bad.shape))
    print("broadcast error:", str(error).splitlines()[-1])
```

右侧对齐为 `[2,3,4]` 与 `[1,2,2]`，最后一维 `4 != 2` 且都不是 1，所以失败。

### 常见误区

广播不按轴名匹配，也不按元素总数判断；为消除错误随意插入长度为 1 的轴，可能把代码从“明确报错”变成“静默算错”。

### 练习

<a id="m1-e1-question"></a>
**M1-E1**（[提示](#m1-e1-hint) · [答案](#m1-e1-answer)）：判断 `[2,4,3]` 分别与 `[3]`、`[4,1]`、`[2,1,3]`、`[2,4]` 能否广播，并逐维说明原因。

<a id="m1-e2-question"></a>
**M1-E2**（[提示](#m1-e2-hint) · [答案](#m1-e2-answer)）：`x [B,S,D]=[3,3,2]` 与 `[3,1]` 运算虽然合法，但它默认对应哪个轴？若要表达每个 batch 一个偏移，shape 应怎样写？

### 模块 1 验收

1. 能否从右向左逐维判断广播，而不是只看元素数？
2. 能否解释为什么 shape 合法仍可能语义错误？
3. 能否根据冲突维度解释真实错误？

<a id="module-2"></a>
## 模块 2：矩阵乘法、批量矩阵乘法与 `einsum`

**对 Qwen3-MoE 推理的意义：** Q/K/V 投影、LM Head、Router 和专家层都以矩阵乘法为核心；先找收缩维，才能在更高阶 Tensor 中保持 batch 和 token 轴。

### 2.1 `@` 与 `torch.matmul`

对 `hidden [B,S,Din]` 和 `weight [Din,Dout]`：

```text
[B,S,Din] @ [Din,Dout] -> [B,S,Dout]
```

`Din` 相等并被收缩，`B/S` 保留，右矩阵的 `Dout` 成为输出最后一维。`@` 在这里等价于 `torch.matmul`。

**Predict：** 先把每个 `[a,b]` 的三个输出写成公式，并预测完整 output values 与 shape。

```python
import torch

hidden = torch.tensor(
    [[[1.0, 2.0], [3.0, 4.0]],
     [[5.0, 6.0], [7.0, 8.0]]]
)
weight = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

output = hidden @ weight
reference = torch.matmul(hidden, weight)
print(output)
print("shape:", tuple(output.shape))
print("equal:", torch.equal(output, reference))
```

每个 `[a,b]` 变成 `[a,b,a+b]`，输出 shape 为 `(2,2,3)`。

`matmul` 对本周会遇到的 rank 组合按以下规则处理：

| 左输入 | 右输入 | 结果 | 临时理解 |
|---|---|---|---|
| `[K]` | `[K]` | 标量 `[]` | 点积，两个向量维都被收缩 |
| `[K]` | `[K,N]` | `[N]` | 左向量临时视为 `[1,K]`，结果移除开头的 1 |
| `[M,K]` | `[K]` | `[M]` | 右向量临时视为 `[K,1]`，结果移除末尾的 1 |
| `[...,M,K]` | `[...,K,N]` | `[...,M,N]` | 先广播前导 batch 维，再对最后两维做矩阵乘法 |

因此向量参与时，输出 rank 可能减少；高阶 Tensor 的矩阵维始终是最后两维。可以用以下断言快速确认：

```python
vector = torch.tensor([1.0, 2.0])
matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
assert (vector @ vector).ndim == 0
assert tuple((vector @ matrix).shape) == (3,)
assert tuple((matrix @ torch.ones(3)).shape) == (2,)
```

### 2.2 `torch.bmm`：严格的三维批量矩阵乘法

`torch.bmm(left, right)` 要求两个输入都是 rank 3：`[B,M,K]` 与 `[B,K,N]`，返回 `[B,M,N]`。它不自动广播 batch 维。

```python
left = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
right = torch.ones(2, 3, 4)
batched = torch.bmm(left, right)
print("bmm shape:", tuple(batched.shape))
print(batched)

try:
    rank_two = torch.ones(3, 4)
    print("bmm input shapes:", tuple(left.shape), tuple(rank_two.shape))
    torch.bmm(left, rank_two)
except RuntimeError as error:
    print("bmm rank error:", str(error).splitlines()[-1])
```

输出 shape 是 `(2,2,4)`；每一行的四个结果相同，因为右矩阵全为 1。第二次调用失败，因为右输入只有两个维度。

### 2.3 `torch.einsum`：显式写出轴规则

`torch.einsum("bsd,do->bso", hidden, weight)` 中，输入下标 `d` 不在显式输出中，因此会被求和；同名 `d` 还规定两个输入沿该轴对齐相乘。`b/s` 保留，`o` 来自权重输出轴。

```python
einsum_output = torch.einsum("bsd,do->bso", hidden, weight)
torch.testing.assert_close(einsum_output, output)
print("einsum shape:", tuple(einsum_output.shape))
```

`einsum` 能清楚表达轴关系，但不保证比 `matmul` 更快；普通线性投影优先写更直接的 `@`。

### 2.4 收缩维不匹配

```python
bad_weight = torch.ones(4, 3)
try:
    hidden @ bad_weight
except RuntimeError as error:
    print("hidden:", tuple(hidden.shape), "weight:", tuple(bad_weight.shape))
    print("matmul error:", str(error).splitlines()[-1])
```

根因是 `hidden` 最后一维 2 与权重倒数第二维 4 不相等。

### 常见误区

`*` 仍是逐元素乘法；`bmm` 不会广播 batch；`einsum` 的下标更显式，但不代表它天然更快或更适合所有矩阵乘法。

### 练习

<a id="m2-e1-question"></a>
**M2-E1**（[提示](#m2-e1-hint) · [答案](#m2-e1-answer)）：推导 `[2,4,3] @ [3,5]` 的输出，并指出保留、收缩、新产生的轴。再写出等价 `einsum`。

<a id="m2-e2-question"></a>
**M2-E2**（[提示](#m2-e2-hint) · [答案](#m2-e2-answer)）：判断 `bmm([2,3,4], [2,4,5])`、`bmm([2,3,4], [1,4,5])` 和 `bmm([3,4], [4,5])` 是否合法。

### 模块 2 验收

1. 能否解释 `[B,S,Din] @ [Din,Dout]` 的每个轴？
2. 能否说明 `bmm` 的 rank 与 batch 限制？
3. 能否从 `bsd,do->bso` 找到收缩下标？

<a id="module-3"></a>
## 模块 3：第一个 `nn.Module`

`nn.Module` 是 PyTorch 组织计算和状态的基类。自定义模块通常在 `__init__` 中声明子模块与状态，在 `forward` 中描述一次前向计算。

**对 Qwen3-MoE 推理的意义：** 后续 Embedding、Attention、专家层和完整 Decoder 都会作为模块递归组合；理解调用路径是理解 hooks、状态加载和模式切换的基础。

**Predict：** 运行前判断该模块是否有 parameter、buffer 或状态 key，并预测 `training` 与输出。

```python
import torch
from torch import nn

class AddConstant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value

module = AddConstant(3.0)
x = torch.tensor([1.0, 2.0])
print(module)
print("training:", module.training)
print("output:", module(x))
```

`super().__init__()` 初始化 `nn.Module` 的注册机制；`forward` 返回本模块的计算结果。应使用 `module(x)`，它会进入 `nn.Module.__call__` 的完整调用路径，再调用 `forward`，从而保留 hooks 等框架行为；不要把 `module.forward(x)` 当作常规入口。

`value` 只是普通 Python 属性，不会出现在参数、buffer 或 `state_dict` 中。新模块默认 `training == True`。

### 常见误区

- `nn.Module` 不等于“必须有可训练参数”；无参数模块仍能封装计算。
- 定义 `forward` 不意味着应直接调用它。
- `training` 是模块模式标志，不是“当前一定在计算梯度”的证明。

### 练习

<a id="m3-e1-question"></a>
**M3-E1**（[提示](#m3-e1-hint) · [答案](#m3-e1-answer)）：实现 `MultiplyConstant`，让 `module(x)` 返回 `x * factor`。说明 `factor` 为何仍是普通属性。

<a id="m3-e2-question"></a>
**M3-E2**（[提示](#m3-e2-hint) · [答案](#m3-e2-answer)）：预测 `AddConstant(2.0).parameters()`、`.buffers()` 和 `.state_dict()` 的内容，并运行验证。

### 模块 3 验收

1. 能否说明 `__init__` 与 `forward` 的职责？
2. 能否解释为什么优先使用 `module(x)`？
3. 能否识别一个无参数但合法的模块？

<a id="module-4"></a>
## 模块 4：parameter、buffer 与普通属性

模块状态不止一种。下面把三类对象放在一起观察：

**对 Qwen3-MoE 推理的意义：** 模型权重必须作为 parameter 加载，固定 Tensor 可以作为 buffer 管理，而配置文字不应伪装成 Tensor 状态。

**Predict：** 先把 `weight`、`running_scale` 和 `label` 分别放入 parameter、buffer、普通属性三栏，再预测状态 key 与 FP64 转换结果。

```python
import torch
from torch import nn

class StateDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("running_scale", torch.tensor([0.5]))
        self.label = "demo"

    def forward(self, x):
        return x * self.weight * self.running_scale

module = StateDemo()
print("parameters:", [(n, tuple(v.shape)) for n, v in module.named_parameters()])
print("buffers:", [(n, tuple(v.shape)) for n, v in module.named_buffers()])
print("state keys:", list(module.state_dict().keys()))
print("label:", module.label)
```

预期：parameter 是 `weight`，buffer 是 `running_scale`，二者都进入默认 persistent 的 `state_dict`；字符串 `label` 是普通属性，不进入这些集合。

### 4.1 三类对象的职责

| 类别 | 创建方式 | 默认进入 `state_dict` | 随 `.to(...)` 转换 | 默认需要梯度 |
| --- | --- | --- | --- | --- |
| parameter | `nn.Parameter(tensor)` | 是 | 是 | 是 |
| persistent buffer | `register_buffer(name, tensor)` | 是 | 是 | 否 |
| 普通属性 | 普通赋值 | 否 | 否 | 取决于对象本身，不受模块注册管理 |

buffer 适合保存属于模块状态但通常不由优化器更新的 Tensor，例如统计量或固定常量。普通属性适合标签、配置数字等一般 Python 信息。

### 4.2 dtype 转换与递归注册

`module.to(dtype=...)` 会递归转换已注册的浮点 parameter 和 buffer，不会改写字符串或普通 Python 数字。

```python
module = module.to(dtype=torch.float64)
print(module.weight.dtype)
print(module.running_scale.dtype)
print(module.label)

class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = StateDemo()

parent = Parent()
print(list(parent.state_dict().keys()))
```

预期 dtype 都是 `torch.float64`，label 仍为 `demo`；父模块状态 key 为 `child.weight` 和 `child.running_scale`。把子模块赋给父模块属性后，PyTorch 会递归注册它。

### 常见误区

- buffer 不是“不会保存的临时量”；默认 persistent buffer 会进入 `state_dict`。
- `requires_grad=False` 的普通 Tensor 不会自动变成 buffer，仍需 `register_buffer`。
- `state_dict` 保存已注册状态，不保存模块对象本身或任意普通属性。

### 练习

<a id="m4-e1-question"></a>
**M4-E1**（[提示](#m4-e1-hint) · [答案](#m4-e1-answer)）：一个模块含 parameter `weight`、buffer `mask`、子模块 `proj`（含 parameter `bias`）和普通属性 `name`。预测 `state_dict` key。

<a id="m4-e2-question"></a>
**M4-E2**（[提示](#m4-e2-hint) · [答案](#m4-e2-answer)）：模块从 FP32 转为 FP64 时，parameter、浮点 buffer、字符串属性和普通 Python 浮点数分别怎样变化？

### 模块 4 验收

1. 能否按职责区分 parameter、buffer 和普通属性？
2. 能否预测递归 `state_dict` key？
3. 能否说明 `.to()` 管理哪些已注册对象？

<a id="module-5"></a>
## 模块 5：手写教学版线性层

本教程让权重直接采用公式方向 `[Din,Dout]`：

**对 Qwen3-MoE 推理的意义：** 几乎所有投影层都要把最后一个隐藏维映射到新宽度；手写一次能建立权重方向、bias 广播和参考实现对齐的可靠直觉。

```text
input  [B,S,Din]
weight [Din,Dout]
bias   [Dout]
output [B,S,Dout]
```

### 5.1 实现 `ManualLinear`

```python
import torch
from torch import nn

class ManualLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dimension {self.in_features}, "
                f"got {x.shape[-1]} for shape {tuple(x.shape)}"
            )
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output
```

`nn.init.uniform_` 原地用均匀分布初始化 parameter；实际项目需要根据模型设计选择初始化方案，本周只要求固定随机种子后可复现。`register_parameter("bias", None)` 明确注册“当前没有 bias”，因此它不会成为 `state_dict` Tensor。

**Predict：** 在赋固定值后，先用 `[a,b,c] -> [a+c,b+c] + [10,20]` 手算全部四个 token 的输出。

### 5.2 用手算值验证

```python
torch.manual_seed(7)
manual = ManualLinear(3, 2)

with torch.no_grad():
    manual.weight.copy_(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    )
    manual.bias.copy_(torch.tensor([10.0, 20.0]))

x = torch.tensor(
    [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
     [[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]]]
)
y = manual(x)
print(y)
print("shape:", tuple(y.shape))
assert tuple(y.shape) == (2, 2, 2)
torch.testing.assert_close(
    y[0, 0], torch.tensor([14.0, 25.0]), atol=0, rtol=0
)
torch.testing.assert_close(
    y[1, 1], torch.tensor([13.0, 21.0]), atol=0, rtol=0
)
```

向量 `[a,b,c]` 先变成 `[a+c,b+c]`，再加 `[10,20]`。预期输出：

```text
tensor([[[14., 25.],
         [20., 31.]],
        [[12., 23.],
         [13., 21.]]], grad_fn=<AddBackward0>)
shape: (2, 2, 2)
```

`with torch.no_grad()` 暂时关闭其中操作的 autograd 记录，适合这里给 parameter 指定教学值。

### 5.3 与 `nn.Linear` 对齐

PyTorch `nn.Linear(Din,Dout)` 的 `weight` shape 是 `[Dout,Din]`，其计算等价于 `x @ weight.T + bias`。因此复制时必须转置：

```python
reference = nn.Linear(3, 2, bias=True)
with torch.no_grad():
    reference.weight.copy_(manual.weight.T)
    reference.bias.copy_(manual.bias)

reference_y = reference(x)
torch.testing.assert_close(y, reference_y, atol=1e-6, rtol=1e-6)
print("manual and nn.Linear match")
```

`torch.testing.assert_close` 使用明确绝对和相对容差比较浮点 Tensor。

### 5.4 shape assertion 先于猜测修复

```python
try:
    manual(torch.ones(2, 2, 4))
except ValueError as error:
    print(error)
```

错误会同时报告期望的 3、实际的 4 和完整输入 shape。修复应检查上游为何产生错误 `Din`，而不是任意 reshape 成能运行的尺寸。

### 常见误区

教学版权重方向与 `nn.Linear` 不同；输出能对齐不代表权重可不经转置直接复制。shape assertion 应暴露上游契约错误，而不是替上游猜测修复。

### 练习

<a id="m5-e1-question"></a>
**M5-E1**（[提示](#m5-e1-hint) · [答案](#m5-e1-answer)）：`ManualLinear(4,6,bias=True)` 的 parameter shape、元素总数和 FP32 数据字节数分别是多少？

<a id="m5-e2-question"></a>
**M5-E2**（[提示](#m5-e2-hint) · [答案](#m5-e2-answer)）：把教学权重 `[4,6]` 复制到 `nn.Linear(4,6)` 时，参考权重 shape 是什么，复制表达式是什么？

### 模块 5 验收

1. 能否独立推导线性层输入、权重、bias 和输出 shape？
2. 能否解释 bias 的广播与 `nn.Linear` 权重方向？
3. 能否写出包含明确错误信息的最后一维检查？

<a id="module-6"></a>
## 模块 6：状态、模式、梯度与可复现性

**对 Qwen3-MoE 推理的意义：** 真实权重需要可靠加载，纯推理需要正确模式和梯度上下文，设备或 dtype 错位则会在第一次数值运算处失败。

**Predict：** 先列出保存的 key，预测清零与恢复后的输出关系，并分别判断三种 forward 的 `requires_grad`。

### 6.1 保存、修改与恢复 `state_dict`

`state_dict()` 返回参数和 persistent buffer 的映射。`torch.save` 可序列化它；`torch.load(..., weights_only=True)` 以权重用途加载，`map_location="cpu"` 明确把张量放到 CPU。

```python
import tempfile
from pathlib import Path

baseline = manual(x).detach().clone()

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "manual_linear.pt"
    torch.save(manual.state_dict(), path)

    with torch.no_grad():
        manual.weight.zero_()
        manual.bias.zero_()
    changed = manual(x)
    print("changed:", not torch.equal(baseline, changed))

    state = torch.load(path, weights_only=True, map_location="cpu")
    manual.load_state_dict(state)
    restored = manual(x)
    torch.testing.assert_close(restored, baseline, atol=0, rtol=0)
    print("restored state keys:", list(state.keys()))
```

临时目录退出后自动清理，不在仓库留下 checkpoint。预期 key 是 `weight`、`bias`，恢复后逐值相等。

shape 不匹配不能靠 `strict=False` 修复：

```python
wrong = ManualLinear(4, 2)
try:
    print("source state shapes:", {k: tuple(v.shape) for k, v in manual.state_dict().items()})
    print("target state shapes:", {k: tuple(v.shape) for k, v in wrong.state_dict().items()})
    wrong.load_state_dict(manual.state_dict())
except RuntimeError as error:
    print("load error:", str(error).splitlines()[-1])
```

### 6.2 `train/eval` 不等于梯度开关

`train()` 把模块及子模块切到训练模式；`eval()` 等价于 `train(False)`。它们会影响 Dropout、BatchNorm 等依赖模式的层，但不会自动关闭 autograd。

```python
manual.eval()
normal = manual(x)
print("training:", manual.training, "normal requires_grad:", normal.requires_grad)

with torch.no_grad():
    no_grad_output = manual(x)
print("no_grad requires_grad:", no_grad_output.requires_grad)

with torch.inference_mode():
    inference_output = manual(x)
print("inference requires_grad:", inference_output.requires_grad)
```

预期：`training` 为 `False`，普通输出仍 `requires_grad=True`；两个上下文中的输出为 `False`。`inference_mode` 比 `no_grad` 施加更强的推理限制，适合纯推理，但不要假定它能替代所有需要离开上下文后继续参与 autograd 的代码。

### 6.3 随机种子与 dtype/device

```python
torch.manual_seed(123)
first = ManualLinear(3, 2)
torch.manual_seed(123)
second = ManualLinear(3, 2)
print("same initialization:", torch.equal(first.weight, second.weight))

first = first.to(dtype=torch.float64, device="cpu")
x64 = x.to(dtype=torch.float64, device="cpu")
print("dtype/device:", first.weight.dtype, first.weight.device, x64.dtype, x64.device)
print("output dtype:", first(x64).dtype)
```

固定种子只能在相同代码路径、算子和环境条件下帮助复现。模块与输入必须使用兼容 dtype/device。对 `nn.Module` 而言，`.to()` 会原地修改模块并返回自身；重新赋值不是必须的，只是让目标 dtype/device 在代码中更醒目。`Tensor.to()` 则通常需要接收返回的 Tensor。

下面故意制造 dtype mismatch，并先打印双方 dtype：

```python
dtype_module = ManualLinear(3, 2).to(dtype=torch.float64)
dtype_input = x.to(dtype=torch.float32)
try:
    print("input/weight dtype:", dtype_input.dtype, dtype_module.weight.dtype)
    dtype_module(dtype_input)
except RuntimeError as error:
    print("dtype mismatch:", str(error).splitlines()[-1])
```

可选 device mismatch 使用可用性守卫，CPU-only 环境明确跳过：

```python
if not torch.cuda.is_available():
    print("CUDA 不可用：跳过 device mismatch 可选演示。")
else:
    cuda_module = ManualLinear(3, 2).to("cuda")
    cpu_input = x.to("cpu")
    try:
        print("input/weight device:", cpu_input.device, cuda_module.weight.device)
        cuda_module(cpu_input)
    except RuntimeError as error:
        print("device mismatch:", str(error).splitlines()[-1])
```

最小修复都是根据计算意图把模块和输入转换到同一 dtype/device。

### 6.4 可选：比较 FP32、FP16 与 BF16 误差

低精度支持依赖设备和算子。下面以 FP32 为参考；某 dtype 不支持时记录错误而不是强行通过：

```python
torch.manual_seed(9)
base = ManualLinear(3, 2).eval()
sample = torch.tensor([[[0.1, -0.2, 0.3]]])
tolerances = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}
with torch.inference_mode():
    fp32_output = base(sample)
    for dtype in (torch.float16, torch.bfloat16):
        try:
            low_module = ManualLinear(3, 2).eval()
            low_module.load_state_dict(base.state_dict())
            low_module = low_module.to(dtype=dtype)
            low_output = low_module(sample.to(dtype=dtype)).float()
            max_error = (fp32_output - low_output).abs().max().item()
            tolerance = tolerances[dtype]
            print(dtype, "max abs error:", max_error, tolerance)
            torch.testing.assert_close(
                low_output, fp32_output,
                atol=tolerance["atol"], rtol=tolerance["rtol"],
            )
        except RuntimeError as error:
            print(dtype, "unsupported here:", str(error).splitlines()[-1])
```

误差大小取决于 values、初始化和后端；这里的容差只针对这个小实验，用于把“足够接近”写成可执行标准，不是所有模型的通用阈值。本实验要求记录实际数字，不预设 FP16 与 BF16 谁必然更小。

可选 CUDA：

```python
if torch.cuda.is_available():
    try:
        cuda_module = ManualLinear(3, 2).to(device="cuda", dtype=torch.float16)
        cuda_x = x.to(device="cuda", dtype=torch.float16)
        print(cuda_module(cuda_x).shape, cuda_module(cuda_x).dtype)
    except RuntimeError as error:
        print("CUDA FP16 unsupported here:", str(error).splitlines()[-1])
else:
    print("CUDA 不可用：跳过可选 FP16 实验。")
```

不要用可选低精度结果替代 CPU FP32/FP64 主路径。

### 常见误区

`eval()` 不关闭梯度，`no_grad()` 不切换模块模式；固定种子不等于跨平台逐位复现；只转换模块或只转换输入都会留下 dtype/device mismatch。

### 练习

<a id="m6-e1-question"></a>
**M6-E1**（[提示](#m6-e1-hint) · [答案](#m6-e1-answer)）：解释 `eval()`、`no_grad()` 和 `inference_mode()` 分别改变什么；哪一个负责切换模块模式？

<a id="m6-e2-question"></a>
**M6-E2**（[提示](#m6-e2-hint) · [答案](#m6-e2-answer)）：为什么只把模块转为 FP64、输入仍为 FP32 会失败？写出最小正确转换，并说明固定种子不保证跨所有平台逐位一致。

### 模块 6 验收

1. 能否预测和恢复 `state_dict`？
2. 能否区分模块模式与梯度记录？
3. 能否让模块和输入的 dtype/device 一致并解释种子边界？

<a id="capstone"></a>
## 综合任务：构建可保存的微型线性模块

先不要运行代码。设 `B=2`、`S=2`、`Din=3`、`Dout=2`，模块包含教学方向权重 `[Din,Dout]`、bias `[Dout]`、buffer `output_scale` 和普通属性 `description`。

### 第一步：书面预测

1. 写出 input、weight、bias、matmul 结果和最终 output shape，并解释每一维。
2. 写出收缩维和 bias、scale 的广播过程。
3. 预测 `named_parameters()`、`named_buffers()` 和 `state_dict()` key。
4. 分别计算 parameter 元素数、parameter FP32 字节数，以及包含 buffer 的完整状态字节数。
5. 写出与 `nn.Linear` 对齐时的权重复制表达式。
6. 说明保存、清零、加载后输出应发生什么变化。

### 第二步：运行验证

```python
import tempfile
from pathlib import Path
import torch
from torch import nn

class ScaledManualLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 3
        self.out_features = 2
        self.weight = nn.Parameter(torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ))
        self.bias = nn.Parameter(torch.tensor([10.0, 20.0]))
        self.register_buffer("output_scale", torch.tensor(0.5))
        self.description = "week02 capstone"

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError("input feature size mismatch")
        return (x @ self.weight + self.bias) * self.output_scale

module = ScaledManualLinear()
x = torch.tensor(
    [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
     [[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]]]
)
output = module(x)

assert tuple(output.shape) == (2, 2, 2)
assert list(dict(module.named_parameters())) == ["weight", "bias"]
assert list(dict(module.named_buffers())) == ["output_scale"]
assert list(module.state_dict()) == ["weight", "bias", "output_scale"]
assert module.description == "week02 capstone"

parameter_numel = sum(parameter.numel() for parameter in module.parameters())
parameter_bytes = sum(
    parameter.numel() * parameter.element_size()
    for parameter in module.parameters()
)
state_bytes = sum(t.numel() * t.element_size() for t in module.state_dict().values())
assert parameter_numel == 8
assert parameter_bytes == 32
assert state_bytes == (6 + 2 + 1) * 4

reference = nn.Linear(3, 2)
with torch.no_grad():
    reference.weight.copy_(module.weight.T)
    reference.bias.copy_(module.bias)
reference_output = reference(x) * module.output_scale
torch.testing.assert_close(output, reference_output, atol=1e-6, rtol=1e-6)

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "capstone.pt"
    torch.save(module.state_dict(), path)
    baseline = output.detach().clone()
    with torch.no_grad():
        module.weight.zero_()
    assert not torch.equal(module(x), baseline)
    module.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    torch.testing.assert_close(module(x), baseline, atol=0, rtol=0)

print("output:")
print(output)
print("parameter numel/bytes:", parameter_numel, parameter_bytes)
print("state bytes:", state_bytes)
print("all capstone checks passed")
```

### 第三步：脱离代码口述

不看代码说明 `[B,S,Din] @ [Din,Dout] + [Dout] -> [B,S,Dout]`，再说明 parameter、buffer、普通属性、参考层转置和状态恢复。能完整讲清才算完成。

<a id="acceptance"></a>
## 最终验收

先独立作答，需要时按同编号查看提示，最后再核对答案。

### A. 概念题

1. <a id="c1-question"></a>**C1**（[提示](#c1-hint) · [答案](#c1-answer)）：广播为什么必须从右侧对齐？shape 合法为何仍可能语义错误？
2. <a id="c2-question"></a>**C2**（[提示](#c2-hint) · [答案](#c2-answer)）：解释矩阵乘法中的保留维、收缩维和输出维。
3. <a id="c3-question"></a>**C3**（[提示](#c3-hint) · [答案](#c3-answer)）：`matmul` 与 `bmm` 在本周示例中的主要区别是什么？
4. <a id="c4-question"></a>**C4**（[提示](#c4-hint) · [答案](#c4-answer)）：如何从 `bsd,do->bso` 找到收缩维？为什么不能断言 `einsum` 更快？
5. <a id="c5-question"></a>**C5**（[提示](#c5-hint) · [答案](#c5-answer)）：`nn.Module.__init__`、`forward` 和 `module(x)` 各自承担什么职责？
6. <a id="c6-question"></a>**C6**（[提示](#c6-hint) · [答案](#c6-answer)）：parameter、persistent buffer 和普通属性在状态、转换和梯度方面有何区别？
7. <a id="c7-question"></a>**C7**（[提示](#c7-hint) · [答案](#c7-answer)）：父模块为何能得到 `child.weight` 这样的 `state_dict` key？
8. <a id="c8-question"></a>**C8**（[提示](#c8-hint) · [答案](#c8-answer)）：教学版权重 `[Din,Dout]` 与 `nn.Linear.weight` 有何区别？
9. <a id="c9-question"></a>**C9**（[提示](#c9-hint) · [答案](#c9-answer)）：为什么 `eval()` 不等于关闭梯度，`no_grad()` 也不等于评估模式？
10. <a id="c10-question"></a>**C10**（[提示](#c10-hint) · [答案](#c10-answer)）：`state_dict` 保存什么、不保存什么？加载时 shape 不匹配为什么不能用 `strict=False` 掩盖？

### B. 形状与状态推导题

1. <a id="s1-question"></a>**S1**（[提示](#s1-hint) · [答案](#s1-answer)）：推导 `[2,5,4] + [5,1]` 与 `[2,5,4] + [2,1]` 是否可广播。
2. <a id="s2-question"></a>**S2**（[提示](#s2-hint) · [答案](#s2-answer)）：推导 `[3,7,4] @ [4,6]`，并写等价 `einsum`。
3. <a id="s3-question"></a>**S3**（[提示](#s3-hint) · [答案](#s3-answer)）：推导 `bmm([4,2,3], [4,3,5])`；若右输入 batch 为 1 会怎样？
4. <a id="s4-question"></a>**S4**（[提示](#s4-hint) · [答案](#s4-answer)）：模块含 parameter `w [3,2]`、parameter `b [2]`、buffer `scale []` 和字符串 `name`。列出状态 key、FP32 状态字节数。
5. <a id="s5-question"></a>**S5**（[提示](#s5-hint) · [答案](#s5-answer)）：`ManualLinear(3,2)` 接收 `[2,4,3]`。写出输出 shape、parameter 数量，以及复制到 `nn.Linear` 的表达式。

### 通过标准

- 概念题至少 **8/10**。
- 推导题至少 **4/5**。
- 综合任务全部断言通过，并能脱离答案口述完整数据流与状态流。
- CUDA 与低精度可选实验不计入通过条件。

<a id="hints"></a>
## 提示

### 模块练习提示

<a id="m1-e1-hint"></a>**M1-E1：** 把较短 shape 左侧补 1，再从最后一维逐项比较。

<a id="m1-e2-hint"></a>**M1-E2：** `[3,1]` 补齐后是 `[1,3,1]`；要命中第 0 维，保留三个轴。

<a id="m2-e1-hint"></a>**M2-E1：** 最后一个输入轴与权重第一个轴都标成同一下标。

<a id="m2-e2-hint"></a>**M2-E2：** `bmm` 要求双方 rank 3、batch 相等、内维相等。

<a id="m3-e1-hint"></a>**M3-E1：** 结构与 `AddConstant` 相同，只替换运算；普通赋值不会注册 Tensor 状态。

<a id="m3-e2-hint"></a>**M3-E2：** 检查模块是否创建过 `nn.Parameter` 或调用过 `register_buffer`。

<a id="m4-e1-hint"></a>**M4-E1：** 普通属性排除；子模块 key 使用属性名作为前缀。

<a id="m4-e2-hint"></a>**M4-E2：** `.to()` 递归处理已注册 Tensor，不重写一般 Python 对象。

<a id="m5-e1-hint"></a>**M5-E1：** 权重元素数是 `Din*Dout`，bias 是 `Dout`，FP32 每元素 4 byte。

<a id="m5-e2-hint"></a>**M5-E2：** `nn.Linear.weight` 把输出维放在前面。

<a id="m6-e1-hint"></a>**M6-E1：** 分开回答模块的 `training` 标志与 autograd 记录。

<a id="m6-e2-hint"></a>**M6-E2：** 同一次 matmul 的输入和权重需兼容 dtype/device；种子不是跨平台数值规范。

### 最终验收提示

<a id="c1-hint"></a>**C1：** 广播规则基于尾部轴；轴名只存在于人的解释中。

<a id="c2-hint"></a>**C2：** 找到两个相等且相邻参与点积的内维。

<a id="c3-hint"></a>**C3：** 比较支持的 rank、batch 广播和输入约束。

<a id="c4-hint"></a>**C4：** 输入中未出现在显式输出里的下标会被求和；同名下标还规定输入间怎样对齐。

<a id="c5-hint"></a>**C5：** 初始化注册系统、描述计算、进入框架调用路径。

<a id="c6-hint"></a>**C6：** 分别检查 `state_dict`、`.to()` 和 `requires_grad` 默认值。

<a id="c7-hint"></a>**C7：** 子模块赋值会被父模块递归注册。

<a id="c8-hint"></a>**C8：** 对照 `[Din,Dout]` 与 `[Dout,Din]`。

<a id="c9-hint"></a>**C9：** 模块模式与 autograd 是两套独立机制。

<a id="c10-hint"></a>**C10：** `strict` 主要控制 key 是否匹配，不会让错误 shape 变正确。

<a id="s1-hint"></a>**S1：** 分别补齐为 `[1,5,1]` 和 `[1,2,1]`。

<a id="s2-hint"></a>**S2：** 收缩 4，保留 3、7，引入 6。

<a id="s3-hint"></a>**S3：** `bmm` 不广播 batch。

<a id="s4-hint"></a>**S4：** 标量 Tensor 的 `numel` 是 1。

<a id="s5-hint"></a>**S5：** 参数元素数包含 weight 和 bias；参考层复制权重时转置。

<a id="answers"></a>
## 参考答案

### 模块练习答案

<a id="m1-e1-answer"></a>**M1-E1：** `[3]` 合法；补成 `[1,1,3]`。`[4,1]` 合法；补成 `[1,4,1]`。`[2,1,3]` 合法。`[2,4]` 不合法；补成 `[1,2,4]` 后与 `[2,4,3]` 在最后两维都冲突。

<a id="m1-e2-answer"></a>**M1-E2：** `[3,1]` 被解释为 `[1,S,1]`，沿序列轴变化。每 batch 一个偏移应写 `[B,1,1]=[3,1,1]`。

<a id="m2-e1-answer"></a>**M2-E1：** 输出 `[2,4,5]`；`B=2`、`S=4` 保留，`Din=3` 收缩，`Dout=5` 产生。等价式为 `torch.einsum("bsd,do->bso", x, weight)`。

<a id="m2-e2-answer"></a>**M2-E2：** 第一组合法，输出 `[2,3,5]`。第二组 batch `2 != 1`，`bmm` 不广播，失败。第三组输入不是 rank 3，失败。

<a id="m3-e1-answer"></a>**M3-E1：** `forward` 返回 `x * self.factor`。若 `factor` 通过普通赋值保存 Python 数字，它不是 parameter 或 buffer，因此不进入 `state_dict` 或随 `.to()` 转换。

<a id="m3-e2-answer"></a>**M3-E2：** 三者都是空的：该模块没有 parameter、buffer 或子模块状态；普通 `value` 不进入 `state_dict`。

<a id="m4-e1-answer"></a>**M4-E1：** key 为 `weight`、`mask`、`proj.bias`；`name` 是普通属性，不进入状态。

<a id="m4-e2-answer"></a>**M4-E2：** 浮点 parameter 和浮点 buffer 转为 FP64；字符串不变；普通 Python 浮点数仍是 Python `float`，不会由模块 `.to()` 改写。

<a id="m5-e1-answer"></a>**M5-E1：** weight `[4,6]` 有 24 个元素，bias `[6]` 有 6 个，共 30 个 parameter 元素；FP32 数据为 `30*4=120` byte。

<a id="m5-e2-answer"></a>**M5-E2：** `nn.Linear(4,6).weight` 是 `[6,4]`；使用 `reference.weight.copy_(manual.weight.T)`，bias 直接复制。

<a id="m6-e1-answer"></a>**M6-E1：** `eval()` 设置模块及子模块为评估模式；`no_grad()` 暂停记录梯度；`inference_mode()` 是限制更强的纯推理上下文。只有 `eval()` 负责模块模式，后两者不改变 `training`。

<a id="m6-e2-answer"></a>**M6-E2：** FP64 权重与 FP32 输入不能直接完成预期 matmul。最小修复是同时执行 `module.to(torch.float64)` 与 `x.to(torch.float64)`，并确保 device 一致。固定种子控制随机序列起点，但算子、硬件和实现差异仍可能影响跨平台结果。

### 最终验收答案

<a id="c1-answer"></a>**C1：** 广播从右侧对齐，因为规则按尾部维定义，缺失维只在左侧视为 1。PyTorch 不保存 `B/S/D` 轴名，所以相同长度碰巧兼容时也可能沿错误业务轴运算。

<a id="c2-answer"></a>**C2：** `[B,S,Din] @ [Din,Dout]` 保留 `B/S`，两个 `Din` 参与点积并消失，右侧 `Dout` 成为输出轴，结果为 `[B,S,Dout]`。

<a id="c3-answer"></a>**C3：** `matmul` 支持多种 rank，并可处理高阶 batch 广播；`bmm` 专门要求两个 rank-3 输入 `[B,M,K]`、`[B,K,N]`，且 batch 必须相等。

<a id="c4-answer"></a>**C4：** `d` 没有出现在输出 `bso` 中，因此沿 `d` 求和；同名 `d` 还规定两个输入沿该轴对齐相乘。`einsum` 是表达方式，实际性能取决于表达式、shape、后端与优化，不能仅凭语法断言更快。

<a id="c5-answer"></a>**C5：** `super().__init__()` 建立模块注册机制；`forward` 定义计算；`module(x)` 经过框架调用路径并触发 `forward` 及 hooks 等行为。

<a id="c6-answer"></a>**C6：** parameter 默认需要梯度、进入状态并随 `.to()` 转换；persistent buffer 默认不需要梯度，但也进入状态并转换；普通属性不自动进入状态或随模块转换。

<a id="c7-answer"></a>**C7：** 把一个 `nn.Module` 赋给父模块属性会注册为子模块。状态遍历递归进入子模块，并用属性名加点号作为 key 前缀。

<a id="c8-answer"></a>**C8：** 教学版直接保存 `[Din,Dout]` 并计算 `x @ weight`；`nn.Linear` 保存 `[Dout,Din]` 并在内部使用转置方向，所以复制时用 `manual.weight.T`。

<a id="c9-answer"></a>**C9：** `eval()` 只切换依赖模式的模块行为，不关闭 autograd；`no_grad()` 只关闭上下文中的梯度记录，不把模块切到评估模式。推理通常需要 `eval()` 加合适的梯度上下文。

<a id="c10-answer"></a>**C10：** `state_dict` 保存递归注册的 parameter 和 persistent buffer Tensor，不保存普通属性、Python 代码或完整模块对象。`strict=False` 可放宽缺失/多余 key，但尺寸错误仍代表结构不兼容，不应被掩盖。

<a id="s1-answer"></a>**S1：** `[5,1] -> [1,5,1]`，与 `[2,5,4]` 合法，结果 `[2,5,4]`。`[2,1] -> [1,2,1]`，中间维 `5 != 2`，失败。

<a id="s2-answer"></a>**S2：** 输出 `[3,7,6]`，4 被收缩。等价式为 `torch.einsum("bsd,do->bso", x, w)`，其中具体 `b=3,s=7,d=4,o=6`。

<a id="s3-answer"></a>**S3：** 合法输出 `[4,2,5]`。右输入 batch 为 1 时失败，因为 `bmm` 不把 1 广播到 4。

<a id="s4-answer"></a>**S4：** key 是 `w`、`b`、`scale`，字符串不进入状态。元素数为 `6+2+1=9`，FP32 状态数据为 36 byte。

<a id="s5-answer"></a>**S5：** 输出 `[2,4,2]`。parameter 元素数为 `3*2+2=8`。复制为 `reference.weight.copy_(manual.weight.T)` 与 `reference.bias.copy_(manual.bias)`。

<a id="glossary"></a>
## 术语与速查表

### 运算规则

| 运算 | 输入 | 输出 | 关键检查 |
| --- | --- | --- | --- |
| 广播 | `[B,S,D]` 与 `[D]` | `[B,S,D]` | 从右对齐；相等、为 1 或左侧缺失 |
| `matmul` | `[B,S,Din]` 与 `[Din,Dout]` | `[B,S,Dout]` | `Din` 相等并收缩 |
| `bmm` | `[B,M,K]` 与 `[B,K,N]` | `[B,M,N]` | 双方 rank 3，batch 与 `K` 相等 |
| `einsum` | `bsd,do->bso` | `[B,S,Dout]` | 未出现在输出中的 `d` 被求和，同名 `d` 负责对齐 |

### 模块状态

| 对象 | `state_dict` | `.to()` | 典型用途 |
| --- | --- | --- | --- |
| parameter | 是 | 是 | 学习或加载的权重 |
| persistent buffer | 是 | 是 | 统计量、固定 Tensor 状态 |
| 普通属性 | 否 | 否 | 名称、配置、一般 Python 数据 |

### 模式与梯度

| API | 改变模块 `training` | 关闭梯度记录 |
| --- | --- | --- |
| `train()` | 设为 `True` | 否 |
| `eval()` | 设为 `False` | 否 |
| `torch.no_grad()` | 否 | 是 |
| `torch.inference_mode()` | 否 | 是，并施加更强推理限制 |

### 调试清单

1. 写出每个输入 shape 和轴语义。
2. 广播先右对齐，matmul 先找收缩维。
3. 检查模块与输入 dtype/device。
4. 检查参数方向是 `[Din,Dout]` 还是 `[Dout,Din]`。
5. 检查状态 key 与每个 Tensor shape，不用 `strict=False` 隐藏结构错误。
6. 浮点对照写明 `atol/rtol`，推理同时考虑模块模式和梯度上下文。

<a id="next-week"></a>
## 下一周预告

第三周将把整数 `token_ids [B,S]` 送入 Embedding，得到 `hidden [B,S,D]`，再通过 LM Head 产生 `logits [B,S,V]`。你会学习稳定 softmax、temperature、argmax 和 next-token 选择，但仍先使用微型词表，不下载真实模型。

开始第三周前，请确保你能脱离代码解释本周综合任务中的形状流、状态流和参考层权重转置。
