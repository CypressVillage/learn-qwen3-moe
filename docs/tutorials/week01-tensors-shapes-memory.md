# 第一周教程：张量、形状与内存

**目录**

- [这周要学会什么](#goals)
- [学习方式](#method)
- [开始前的环境检查](#environment-check)
- [模块 1：第一个 Tensor](#module-1)
- [模块 2：读懂 Tensor 的基本属性](#module-2)
- [模块 3：索引、切片与形状变换](#module-3)
- [模块 4：stride、transpose 与 contiguous](#module-4)
- [模块 5：广播与矩阵乘法](#module-5)
- [模块 6：dtype 与内存估算](#module-6)
- [综合任务：走过一次微型语言模型数据流](#capstone)
- [最终验收](#acceptance)
- [提示](#hints)
- [参考答案](#answers)
- [术语与速查表](#glossary)
- [下一周预告](#next-week)

<a id="goals"></a>
## 这周要学会什么

这一周的目标不是背完 PyTorch API，而是建立阅读张量代码时最重要的三种直觉：**数据是什么、形状怎样变化、这些数据大约占多少内存**。这些直觉会贯穿后续的注意力、前馈网络、MoE 路由和模型推理。

完成本周学习后，你应该能够：

1. 用 `torch.tensor`、`torch.arange`、`torch.zeros` 和 `torch.randn` 创建小型 Tensor，并说明其中每个维度代表什么。
2. 看到 `shape`、`ndim`、`numel`、`dtype` 和 `device` 时，能用自己的话解释它们，而不是只复述输出。
3. 在运行代码前，预测索引、切片、`reshape`、`view`、`unsqueeze`、`squeeze` 和 `transpose` 之后的形状。
4. 初步理解 `stride` 描述的是元素在内存中的步进方式，并知道为什么转置后的 Tensor 可能不是 contiguous。
5. 判断两个形状能否广播，并区分逐元素乘法与矩阵乘法。
6. 根据元素数量和 dtype 的每元素字节数，手算 Tensor 的理论内存占用。
7. 沿着一条微型语言模型数据流，跟踪 token、embedding、隐藏状态和 logits 的形状。
8. 遇到形状错误时，先打印和推理关键信息，再修改代码，而不是反复试参数。

本周使用的张量都很小。**CPU 是必须完成的主路径**，没有 NVIDIA GPU 也不会影响学习成果；CUDA 只用于可选观察。全周不下载模型，也不需要大显存。

<a id="method"></a>
## 学习方式

### 时间安排

建议为本周预留 **6-10 小时**，分成 3-5 次完成。一次学习 60-120 分钟通常比连续赶完更容易形成形状直觉。

可以按下面的节奏分配：

| 内容 | 建议时间 |
| --- | ---: |
| 环境检查与模块 1-2 | 1.5-2 小时 |
| 模块 3-4 | 2-3 小时 |
| 模块 5-6 | 1.5-2.5 小时 |
| 综合任务、验收与复盘 | 1-2.5 小时 |

这只是参考。某个概念没有说清楚时，宁可多做两个小例子，也不要为了赶进度跳过。

### Predict-Run-Explain 循环

本教程中的每段核心代码都应按 **Predict（预测）→ Run（运行）→ Explain（解释）** 的顺序学习：

1. **Predict：先预测。** 在纸上、注释里或学习笔记中写下输出，至少写出形状、dtype，以及你认为会发生的关键变化。
2. **Run：再运行。** 执行代码，完整观察输出和报错；不要只看最后一行。
3. **Explain：后解释。** 对照预测与实际结果，用一句完整的话解释一致或不一致的原因。

例如，看到下面的代码时，不要立刻运行。`import torch` 导入 PyTorch；`torch.arange(12)` 创建包含整数 0 到 11 的一维 Tensor；`reshape(3, 4)` 在元素总数不变的前提下把它组织成 3 行 4 列，并返回 shape 为 `torch.Size([3, 4])` 的 Tensor。`.shape` 属性返回每个维度的长度。`x[:, 1:3]` 是切片：`:` 保留第 0 维的所有行，`1:3` 取得第 1 维中下标 1 和 2 两列，因此结果 shape 为 `torch.Size([3, 2])`。

```python
import torch

x = torch.arange(12).reshape(3, 4)
print(x.shape)
print(x[:, 1:3].shape)
```

先写下类似这样的预测：

```text
x 的形状：我预测是 ______
x[:, 1:3] 的形状：我预测是 ______
理由：第 0 维 ______，第 1 维 ______
```

然后运行并解释。预测错误不是失败；没有留下预测就直接运行，才会失去最重要的练习机会。

### 本周学习约定

- **每次都先写预测。** 即使你很确定，也先留下可对照的答案。
- **优先使用小数字。** 形状尽量控制在可以手算、可以完整打印的范围。
- **先走 CPU 主路径。** 教程中的必做代码应在 CPU-only PyTorch 环境运行。
- **CUDA 只是可选项。** 有可用 GPU 时可以比较 `device`，但不能用 CUDA 结果替代 CPU 练习。
- **不下载任何模型。** 本周只使用手工构造或随机生成的小 Tensor，不需要 checkpoint、tokenizer 或数据集。
- **保留错误现场。** 遇到报错时，先记录代码、输入形状和完整错误，再尝试修复。
- **解释要包含维度含义。** 不只写“形状是 `(2, 3)`”，还要说明两个维度分别表示什么。
- **不要追求背 API。** 重点是能通过输入形状推导输出形状，并用实验验证。

建议为每次实验保留这样的记录：

```text
代码位置：
运行前预测：
实际输出：
差异原因：
我能否不运行就解释形状变化：能 / 还不能
```

<a id="environment-check"></a>
## 开始前的环境检查

本节只检查环境，不重复安装步骤。如果 Python、虚拟环境或 PyTorch 尚未配置，请按照 [环境配置与双机工作流](../environment.md) 完成设置后再回来。

### 1. 激活项目环境

在仓库根目录打开终端。如果项目使用 `.venv`，先激活它：

```bash
source .venv/bin/activate
```

如果你的环境采用其他名称或由管理员提供，请激活实际使用的环境。不要为了本周内容修改系统 Python。

### 2. 打印版本与 CUDA 状态

运行下面的检查。它会打印 Python 版本、PyTorch 版本和 CUDA 可用性；CPU-only PyTorch 或当前没有可用 CUDA 时，脚本会正常结束，不会因为访问 GPU 名称而失败。

```bash
python - <<'PY'
import platform

print("Python:", platform.python_version())

try:
    import torch
except ModuleNotFoundError:
    print("PyTorch: 未安装")
    print("请按 docs/environment.md 配置环境后重新检查。")
else:
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        print("CUDA device:", torch.cuda.get_device_name(0))
PY
```

继续学习前，至少应看到：

- `Python:` 后应为 `3.11.x`；如果不是，请返回 [环境配置与双机工作流](../environment.md) 配置推荐版本。
- `PyTorch:` 后有版本号，而不是“未安装”。
- `CUDA available:` 为 `True` 或 `False`；两者都可以完成本周教程。

`PyTorch CUDA runtime: None` 或 `CUDA available: False` 通常表示当前走 CPU 路径，这不是本周学习的阻塞项。不要为了让它变成 `True` 而跳过环境文档中的驱动与安装检查。

### 3. 运行 CPU 必做冒烟测试

先明确预测：下面的 Tensor 形状是什么，元素总数是多少，运行设备是什么？写下答案后再执行。`numel()` 方法计算并返回 Tensor 的元素总数；`.device` 属性返回 Tensor 当前所在的计算设备。这里没有指定其他设备，因此预期是 CPU。

```bash
python - <<'PY'
import torch

x = torch.arange(6).reshape(2, 3)
print(x)
print("shape:", x.shape)
print("numel:", x.numel())
print("device:", x.device)
PY
```

这段代码必须能在 CPU 上运行。预期设备是 `cpu`；如果失败，请回到 [环境配置与双机工作流](../environment.md) 排查 Python 和 PyTorch 安装。

### 4. 可选 CUDA 冒烟测试

只有上面的 CPU 测试通过，并且版本检查显示 `CUDA available: True` 时，才运行这一段。传给 `torch.arange` 的 `device="cuda"` 参数要求 PyTorch 直接在默认 CUDA 设备上创建结果，而不是先创建 CPU Tensor；因此成功分支中的 `x.device` 应显示 CUDA 设备。

```bash
python - <<'PY'
import torch

if torch.cuda.is_available():
    x = torch.arange(6, device="cuda").reshape(2, 3)
    print(x)
    print("shape:", x.shape)
    print("device:", x.device)
else:
    print("CUDA 不可用：跳过可选测试，继续 CPU 主路径。")
PY
```

这段可选测试在 CUDA 不可用时也会正常退出。后续所有必做内容仍以 CPU 为准，并且不需要下载模型。

<a id="module-1"></a>
## 模块 1：第一个 Tensor

Tensor 是 PyTorch 保存和计算数值的基本对象。它可以只放一个数，也可以把许多数按多个维度排列起来。本模块先用一个很小的 token ID 表格认识 Tensor；所有数值都是手工指定的，因此每次运行结果都相同。

### 1.1 从 Python 列表到 Tensor

`import torch` 会把 PyTorch 包导入当前 Python 程序，并让名字 `torch` 指向这个包。之后才能使用 `torch.tensor` 等 PyTorch API。

Python 列表适合保存一般 Python 对象；Tensor 则专门表示规则排列的数值，并记录形状、dtype 和设备等计算所需信息。本周只需记住：列表可以作为创建 Tensor 的原始数据，但进入 PyTorch 计算后，我们主要观察和操作 Tensor。

`torch.tensor(data, dtype=...)` 的用途是根据给定数据新建一个 Tensor；结果包含数据本身，并带有指定或推断出的 dtype。这里显式指定 `torch.int64`，因为 token ID 是整数编号，不是连续小数。

**Predict：先不要运行。** 下面有两行 token ID。你预测 `token_ids` 会包含哪些值？形状是 `(2, 3)` 还是 `(3, 2)`？它的 Python 类型、dtype 和设备分别是什么？

```python
import torch

python_rows = [
    [101, 102, 0],
    [201, 202, 203],
]
token_ids = torch.tensor(python_rows, dtype=torch.int64)

print("values:")
print(token_ids)
print("python type:", type(python_rows))
print("tensor type:", type(token_ids))
print("shape:", token_ids.shape)
print("dtype:", token_ids.dtype)
print("device:", token_ids.device)
```

预期输出如下；`device: cpu` 是本教程 CPU 主路径的结果：

```text
values:
tensor([[101, 102,   0],
        [201, 202, 203]])
python type: <class 'list'>
tensor type: <class 'torch.Tensor'>
shape: torch.Size([2, 3])
dtype: torch.int64
device: cpu
```

逐项解释这些观察：

- `values` 是 Tensor 中实际保存的六个整数；打印时，二维 Tensor 按行显示。
- Python 内置函数 `type(obj)` 用于查看对象的 Python 类型。列表的结果是 `list`，Tensor 的结果是 `torch.Tensor`。
- `shape` 属性给出每个维度的长度。结果 `torch.Size([2, 3])` 表示第 0 维长度为 2，第 1 维长度为 3。
- `dtype` 属性给出每个元素的数据类型。`torch.int64` 表示每个 token ID 是 64 位整数。
- `device` 属性给出 Tensor 当前所在的计算设备。这里没有请求 CUDA，所以结果是 `cpu`。

### 1.2 把 `[B, S]` 读成一句话

语言模型通常把一批 token ID 写成形状 `[B, S]`：

- `B` 是 batch size，一次处理多少条序列。这里 `B = 2`。
- `S` 是 sequence length，每条序列放多少个 token 位置。这里 `S = 3`。

因此，`token_ids.shape == torch.Size([2, 3])` 应读成：“这一批有 2 条序列，每条有 3 个 token 位置。”数值 `101`、`102` 等只是为了演示而手工写出的编号，不对应真实 tokenizer。后续做语言模型推理时，输入通常正是这样的整数 Tensor；模型会根据每个 token ID 查找对应表示。本周不下载 tokenizer 或模型。

### 常见误区

> **误区：** “看到两层方括号，所以 shape 一定是 `[2, 2]`。”
>
> 外层列表长度决定行数，内层每行的元素个数决定列数。本例有 2 行、每行 3 个元素，所以 shape 是 `[2, 3]`。方括号的嵌套层数更接近“有几个维度”，不是每个维度的长度。

### 练习

先独立完成，再到文末“提示”和“参考答案”按稳定编号核对。

**M1-E1**：不运行代码，写出 `torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int64)` 的全部 values、Python 类型、shape、dtype 和默认 device。然后运行验证，并解释 shape 中两个数字分别来自哪里。

**M1-E2**：一批输入含 3 条序列，每条序列有 4 个 token 位置。写出 `token_ids` 的符号形状和具体形状，再手工设计一个满足该形状的整数 Tensor。不要使用随机数。

### 模块 1 验收

1. 你能否用一句话说明 `torch.tensor` 的用途，以及它返回什么对象？
2. 给定 `token_ids.shape == torch.Size([2, 3])`，你能否解释 `B`、`S` 和两个具体数字的含义？
3. 你能否在不运行代码的情况下，区分 values、Python `type`、`shape`、`dtype` 和 `device` 分别回答什么问题？

<a id="module-2"></a>
## 模块 2：读懂 Tensor 的基本属性

上一模块已经见过二维 `token_ids [B, S]`。现在把观察方法扩展到标量、向量、矩阵和更高阶 Tensor，并把每个维度和语言模型数据流中的含义对应起来。

### 2.1 rank 不是 shape

Tensor 的 **rank（阶数）** 是维度的数量，在 PyTorch 中用 `ndim` 属性读取。Tensor 的 **shape（形状）** 是每个维度的长度，在 PyTorch 中用 `shape` 属性读取。

例如，shape 为 `[2, 3]` 的 Tensor 是 rank 2，因为 shape 里有两个维度；shape 为 `[2, 3, 4]` 的 Tensor 是 rank 3。rank 只回答“有几个维度”，shape 还回答“每个维度有多长”。不要因为某个维度长度是 3，就说它是 rank 3。

常用名称如下：

| 名称 | rank | shape 示例 | 直观理解 |
| --- | ---: | --- | --- |
| 标量 scalar | 0 | `[]` | 一个数，没有可遍历的轴 |
| 向量 vector | 1 | `[3]` | 一条长度为 3 的数列 |
| 矩阵 matrix | 2 | `[2, 3]` | 2 行、3 列的数表 |
| 更高阶 Tensor | 3 或更多 | `[2, 3, 2]` | 多个有明确业务含义的维度 |

### 2.2 先手算，再询问 API

下面沿用三个符号：

- `token_ids [B, S]`：`B` 条序列，每条有 `S` 个 token 位置。
- `embedding_weight [V, D]`：词表有 `V` 行，每个 token 的表示宽度是 `D`。
- `hidden [B, S, D]`：每条序列的每个 token 位置都有一个长度为 `D` 的隐藏表示。

这里取 `B = 2`、`S = 3`、`V = 4`、`D = 2`。运行前先手算：

- `token_ids [2, 3]` 是 rank 2，共有 `2 * 3 = 6` 个元素。
- `embedding_weight [4, 2]` 是 rank 2，共有 `4 * 2 = 8` 个元素。
- `hidden [2, 3, 2]` 是 rank 3，共有 `2 * 3 * 2 = 12` 个元素。

`ndim` 的用途是读取 Tensor 的维度数量，结果是 Python 整数。`numel()` 的用途是计算 Tensor 的元素总数，结果也是 Python 整数；它应等于 shape 中各维度长度的乘积。`shape`、`dtype` 和 `device` 的含义与模块 1 相同。

```python
import torch

scalar = torch.tensor(7, dtype=torch.int64)
vector = torch.tensor([4, 5, 6], dtype=torch.int64)
token_ids = torch.tensor(
    [[101, 102, 0], [201, 202, 203]],
    dtype=torch.int64,
)
embedding_weight = torch.tensor(
    [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]],
    dtype=torch.float32,
)
hidden = torch.tensor(
    [
        [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]],
        [[3.0, 3.1], [4.0, 4.1], [5.0, 5.1]],
    ],
    dtype=torch.float32,
)

for name, x in [
    ("scalar", scalar),
    ("vector", vector),
    ("token_ids", token_ids),
    ("embedding_weight", embedding_weight),
    ("hidden", hidden),
]:
    print(
        name,
        "ndim=", x.ndim,
        "shape=", tuple(x.shape),
        "dtype=", x.dtype,
        "device=", x.device,
        "numel=", x.numel(),
    )
```

为避免 `torch.Size` 的显示形式干扰比较，代码用 Python 内置的 `tuple(...)` 把 shape 转成普通元组；它不改变 Tensor。预期输出为：

```text
scalar ndim= 0 shape= () dtype= torch.int64 device= cpu numel= 1
vector ndim= 1 shape= (3,) dtype= torch.int64 device= cpu numel= 3
token_ids ndim= 2 shape= (2, 3) dtype= torch.int64 device= cpu numel= 6
embedding_weight ndim= 2 shape= (4, 2) dtype= torch.float32 device= cpu numel= 8
hidden ndim= 3 shape= (2, 3, 2) dtype= torch.float32 device= cpu numel= 12
```

### 2.3 维度顺序决定语义

`hidden [B, S, D] = [2, 3, 2]` 不能只读成“一个三维 Tensor”。应该从左到右说明：

1. 第 0 维 `B = 2`：两条序列。
2. 第 1 维 `S = 3`：每条序列三个 token 位置。
3. 第 2 维 `D = 2`：每个位置用两个数表示。

同样，`embedding_weight [V, D] = [4, 2]` 表示 4 个词表条目，每个条目有 2 个表示数。虽然它和某个 `[B, S] = [4, 2]` 的 token ID 表可能拥有相同具体 shape，但两个维度的业务含义完全不同。shape 只记录长度，不自动记录 `B`、`S`、`V`、`D` 这些语义；读代码的人必须根据变量用途补上它们。

device 说明数据在哪里参与计算，本模块的 Tensor 仍全部位于 CPU。

### 2.4 dtype 会影响数值和内存

dtype 不只是打印信息，它决定三件事：**精度**表示能多细致地区分相近数值，**可表示范围**表示能保存多大或多小的数值，**内存**表示每个元素需要多少字节。通常，位数更多能换来更高精度或更大范围，但每个元素也占更多内存。

| dtype | 代表用途 | 精度与范围 | 每元素内存 |
| --- | --- | --- | ---: |
| `torch.bool` | 真/假掩码 | 只表示 `True` 和 `False`，不表示普通整数或小数 | 1 字节 |
| `torch.int32` | 范围适中的整数 | 精确表示整数，但范围小于 `int64` | 4 字节 |
| `torch.int64` | token ID、较大整数 | 精确表示整数，范围大于 `int32`；不能表示小数 | 8 字节 |
| `torch.float16` | 低精度浮点数 | 精度和范围都小于 `float32`，较容易舍入或溢出 | 2 字节 |
| `torch.bfloat16` | 低精度浮点数 | 精度低于 `float32`，但可表示范围接近 `float32` | 2 字节 |
| `torch.float32` | 常见权重和隐藏状态 | 在精度、范围和内存之间取得常用平衡 | 4 字节 |
| `torch.float64` | 需要更高浮点精度的计算 | 精度高于 `float32`，范围也更大，但内存更多 | 8 字节 |

本教程中的 `token_ids` 使用 `torch.int64`，因为它保存必须精确匹配的整数编号；`embedding_weight` 和 `hidden` 使用 `torch.float32`，因为它们保存连续数值表示。这里先建立“dtype 会改变精度、范围和每元素内存”的直觉；模块 6 再用 `numel` 和每元素字节数详细计算整个 Tensor 的理论内存。

### 常见误区

- **把 rank 和 shape 混为一谈：** `[2, 3, 2]` 的 rank 是 3，shape 是 `[2, 3, 2]`，元素数是 12；这三个答案解决不同问题。
- **只背具体数字，不写维度含义：** `hidden.shape == torch.Size([2, 3, 2])` 还不完整；应继续说明它是 `[B, S, D]`，即序列数、token 位置数、表示宽度。
- **认为相同 shape 就代表相同数据：** `[4, 2]` 既可能是 `[V, D]`，也可能是其他语义。shape 相同不等于用途相同。

### 练习

**M2-E1**：给定 `hidden` 的 shape 为 `[3, 4, 5]`，先在纸上写出它的 rank、`B`、`S`、`D` 和 `numel`，再说明第 1 维长度 4 的含义。最后用一个不要求打印全部 values 的确定性 Tensor 验证属性。

**M2-E2**：给定 `embedding_weight [V, D]`，其中 `V = 6`、`D = 3`。写出 rank、shape 和元素总数。如果另一个 `token_ids [B, S]` 恰好也是 `[6, 3]`，解释两者为什么仍不能被视为同一种数据。

### 模块 2 验收

1. 你能否分别定义 rank、shape 和 `numel`，并对 `[2, 3, 4]` 给出三个答案？
2. 你能否把 `token_ids [B, S]`、`embedding_weight [V, D]`、`hidden [B, S, D]` 的每个维度逐一说清？
3. 你能否先用 shape 手算元素总数，再用 `numel()` 验证，而不是反过来猜？

<a id="module-3"></a>
## 模块 3：索引、切片与形状变换

本模块只改变“取哪些值”或“怎样组织维度”，不引入新的模型计算。继续使用小 Tensor，并在每次运行前写出结果 values、shape 和元素总数。

### 3.1 索引与切片

索引使用某个位置的整数下标取得数据，例如 `x[0]` 或 `x[0, 1]`。一个维度如果被单个整数选中，该维度通常会从结果 shape 中消失。切片使用 `start:stop` 选择一个范围，例如 `x[0:1]`；即使范围里只有一项，被切片的维度通常仍会保留。

在下面的 `[B, S]` token ID 中，先手算每个表达式取到的值和 shape：

- `token_ids[0, 1]` 同时用整数选定 batch 和 token 位置，结果应是 rank 0 标量。
- `token_ids[0]` 选定第 0 条序列，结果应保留长度为 `S` 的一维数据。
- `token_ids[0:1, 1:3]` 在两个维度都使用切片，结果应保留 `[B, S]` 两个维度。
- `token_ids[:, 1]` 中的 `:` 表示该维全部保留，而整数 `1` 选定一个 token 位置，所以 `S` 维消失。

```python
import torch

token_ids = torch.tensor(
    [[101, 102, 0], [201, 202, 203]],
    dtype=torch.int64,
)

print("one token:", token_ids[0, 1], "shape:", tuple(token_ids[0, 1].shape))
print("one sequence:", token_ids[0], "shape:", tuple(token_ids[0].shape))
print("kept slices:", token_ids[0:1, 1:3], "shape:", tuple(token_ids[0:1, 1:3].shape))
print("position 1:", token_ids[:, 1], "shape:", tuple(token_ids[:, 1].shape))
```

预期输出为：

```text
one token: tensor(102) shape: ()
one sequence: tensor([101, 102,   0]) shape: (3,)
kept slices: tensor([[102,   0]]) shape: (1, 2)
position 1: tensor([102, 202]) shape: (2,)
```

### 3.2 `reshape` 与 `view`：元素数量必须兼容

形状变换不会凭空增加或删除元素。旧 shape 的维度乘积必须等于新 shape 的维度乘积。例如，12 个元素可以组织成 `[3, 4]`、`[2, 6]` 或 `[2, 3, 2]`，但不能组织成 `[5, 3]`，因为 15 不等于 12。

`reshape(new_shape)` 的用途是把 Tensor 变成元素数兼容的新形状。结果可能与原 Tensor 共享底层存储，也可能在需要时创建副本，因此不要笼统地说它“总是 view”或“总是 copy”。

`view(new_shape)` 也用于返回元素数兼容的新形状，但它要求目标 shape 与当前 size/stride 兼容，能够在不重排数据的情况下共享存储；后续模块 4 会看到不满足该条件的例子。在本节连续存储的简单 Tensor 上，`view` 可以成功，但“非连续”本身并不等于所有 `view` 都必然失败。

```python
import torch

flat = torch.tensor(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    dtype=torch.int64,
)

as_matrix = flat.reshape(3, 4)
as_batches = flat.view(2, 3, 2)

print("flat shape:", tuple(flat.shape), "numel:", flat.numel())
print("reshape(3, 4):")
print(as_matrix)
print("shape:", tuple(as_matrix.shape), "numel:", as_matrix.numel())
print("view(2, 3, 2) shape:", tuple(as_batches.shape), "numel:", as_batches.numel())

try:
    flat.reshape(5, 3)
except RuntimeError as error:
    print("reshape error:")
    print(str(error).splitlines()[-1])
```

预期输出的前几项固定；异常文本可能随 PyTorch 版本略有差异，但最后一行应同时指出目标 shape `[5, 3]` 和输入元素数 12。错误被 `try/except` 捕获，所以整个脚本仍正常结束，退出码为 0：

```text
flat shape: (12,) numel: 12
reshape(3, 4):
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
shape: (3, 4) numel: 12
view(2, 3, 2) shape: (2, 3, 2) numel: 12
reshape error:
shape '[5, 3]' is invalid for input of size 12
```

`try/except` 是 Python 的错误处理结构：`try` 中故意执行不兼容的 `reshape`，PyTorch 抛出 `RuntimeError` 后进入 `except`。`as error` 把异常对象保存到变量 `error`，`str(error).splitlines()[-1]` 再显示实际异常文本的最后一行，而不是用教程自写的固定消息替代它。

阅读这行错误时，先找目标 shape `[5, 3]`，它需要 `5 * 3 = 15` 个元素；再找输入的 size 12，表示原 Tensor 只有 12 个元素。15 与 12 不相等就是失败原因。捕获异常只为了让后续教程继续运行，并不表示这次 `reshape` 成功。

### 3.3 `unsqueeze` 与 `squeeze`：增加或删除长度为 1 的维度

`unsqueeze(dim)` 的用途是在指定位置插入一个长度为 1 的维度，返回的 Tensor 元素数量不变。例如 `[B, S]` 在末尾插入一维后得到 `[B, S, 1]`。

`squeeze(dim)` 的用途是删除指定位置上长度为 1 的维度；如果该维度长度不是 1，指定维度不会被删除。无参数的 `squeeze()` 会删除所有长度为 1 的维度，初学时更容易误删 batch 维，因此本教程优先显式写出 `dim`。

```python
import torch

token_ids = torch.tensor(
    [[101, 102, 0], [201, 202, 203]],
    dtype=torch.int64,
)

with_feature_axis = token_ids.unsqueeze(2)
restored = with_feature_axis.squeeze(2)

print("original:", tuple(token_ids.shape), "numel:", token_ids.numel())
print("unsqueeze(2):", tuple(with_feature_axis.shape), "numel:", with_feature_axis.numel())
print("squeeze(2):", tuple(restored.shape), "numel:", restored.numel())
print("same values:")
print(restored)
```

预期输出为：

```text
original: (2, 3) numel: 6
unsqueeze(2): (2, 3, 1) numel: 6
squeeze(2): (2, 3) numel: 6
same values:
tensor([[101, 102,   0],
        [201, 202, 203]])
```

注意：`unsqueeze` 和 `squeeze` 在这里改变 rank 和 shape，但不改变 values 的顺序，也不改变 `numel`。

### 常见误区

- **认为索引只取值、不影响 shape：** 整数索引通常会消去被选中的维度，而切片通常保留维度。
- **只比较新旧 rank，不比较元素数：** `reshape` 和 `view` 首先要求 `numel` 相同；维度数量可以变化，但元素总数不能变化。
- **认为 `reshape` 一定复制或一定共享：** 两种说法都不准确。它可能返回共享存储的 view，也可能返回副本，取决于输入布局和目标形状。
- **随意使用无参数 `squeeze()`：** 如果 batch size 恰好是 1，它也可能删除 batch 维。明确知道要删除哪一维时，应传入 `dim`。

### 练习

**M3-E1**：给定 `hidden` 的 shape 为 `[2, 3, 4]`，分别预测 `hidden[0].shape`、`hidden[0:1].shape`、`hidden[:, 1].shape` 和 `hidden[:, 1:2].shape`。逐项说明哪个整数索引消去了哪个维度，哪个切片保留了哪个维度。

**M3-E2**：一个 shape 为 `[2, 3, 4]` 的 Tensor 有多少元素？判断它能否 `reshape` 为 `[6, 4]`、`[4, 6]`、`[2, 2, 6]` 和 `[5, 5]`。然后写出两步形状变换：先用 `unsqueeze` 把 `[2, 3, 4]` 变成 `[2, 1, 3, 4]`，再用带明确 `dim` 的 `squeeze` 恢复原 shape。

### 模块 3 验收

1. 你能否在运行前判断整数索引和切片分别会不会保留被选择的维度？
2. 你能否只靠新旧 shape 的维度乘积，解释一次 `reshape` 为什么成功或失败？
3. 你能否准确说明 `reshape` 的 view/copy 行为，并安全地使用 `unsqueeze(dim)` 与 `squeeze(dim)`？

<a id="module-4"></a>
## 模块 4：stride、transpose 与 contiguous

前面主要观察了 Tensor 的 values 和 shape。本模块再加一层：**逻辑形状**说明下标怎样组织数据，**存储布局**说明沿某个维度移动一个位置时，要在底层存储中跨过多少个元素。两个 Tensor 可以包含相同元素，却有不同的 shape、stride 或连续性。

### 4.1 用二维小矩阵读懂 `stride()`

`stride()` 方法返回每个维度的步长。某一维的 stride 是 `n`，表示该维下标增加 1、其他下标不变时，在底层存储中要跨过 `n` 个元素。stride 的单位是“元素个数”，不是字节。

下面的 `matrix` 有 3 行 4 列。它按行连续存放，所以同一行向右移动一列只跨 1 个元素，向下移动一行要跨 4 个元素。

**Predict：** 先完整画出 values，再预测 `matrix.shape`、`matrix.stride()` 和 `matrix.is_contiguous()`。`is_contiguous()` 方法检查 Tensor 是否采用 PyTorch 当前要求的连续布局，并返回 Python 布尔值。

```python
import torch

matrix = torch.arange(12).reshape(3, 4)

print(matrix)
print("shape:", tuple(matrix.shape))
print("stride:", matrix.stride())
print("contiguous:", matrix.is_contiguous())
```

预期输出为：

```text
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
shape: (3, 4)
stride: (4, 1)
contiguous: True
```

`matrix[1, 2]` 是 6。按照 stride，从 `matrix[0, 0]` 走到它，需要沿第 0 维走 1 步、沿第 1 维走 2 步，因此底层位置偏移量是 `1 * 4 + 2 * 1 = 6`。这个小计算把逻辑下标 `[1, 2]` 和存储步长 `(4, 1)` 连接了起来。

### 4.2 `transpose` 改变逻辑视图，不自动重排存储

`transpose(dim0, dim1)` 的用途是交换两个维度，并返回交换后的 Tensor。下面交换矩阵的行维和列维。values 的二维显示会变化，但 `transposed[0, 1]` 仍对应原来的 `matrix[1, 0]`。`.item()` 从只含一个元素的 Tensor 中取出对应的 Python 标量值；这里两次调用都应得到 Python 整数 `4`，所以最后一行预期打印 `same logical value: 4 4`。

**Predict：** 先把原矩阵的列写成新矩阵的行。新 shape 是什么？原 stride `(4, 1)` 中的两个数字会怎样交换？

```python
transposed = matrix.transpose(0, 1)

print(transposed)
print("shape:", tuple(transposed.shape))
print("stride:", transposed.stride())
print("contiguous:", transposed.is_contiguous())
print("same logical value:", transposed[0, 1].item(), matrix[1, 0].item())
```

预期输出为：

```text
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
shape: (4, 3)
stride: (1, 4)
contiguous: False
same logical value: 4 4
```

这里的 shape `(4, 3)` 是逻辑视图：现在看起来有 4 行 3 列。stride `(1, 4)` 是存储布局：沿新第 0 维走一步，只跨底层 1 个元素；沿新第 1 维走一步，要跨 4 个元素。`transpose` 在这个例子中返回共享原存储的非连续 view，没有把 values 复制成新的逐行顺序。

### 4.3 亲眼看到非连续 `view` 失败

`view(12)` 想在不重排底层数据的前提下，把 Tensor 解释成一维。`view` 是否成功取决于目标 shape 与当前 size/stride 是否兼容；连续布局通常容易满足条件，但非连续 Tensor 并非一律不能 `view`。对上面的 `transposed`，逻辑遍历顺序是 `0, 4, 8, 1, 5, 9, ...`，当前 stride 无法把两个维度直接合并成这一维；若要得到该顺序就必须重排数据，因此这一次 `view(12)` 会抛出 `RuntimeError`。

下面必须保留 `try/except`。代码打印异常对象中**实际错误文本的最后一行**；不同 PyTorch 版本的措辞可能不同，因此不要把教程中的某一句英文当成要背的固定答案。

`reshape(12)` 会在可能时返回 view，布局不允许时也可以创建副本。`contiguous()` 的用途是返回采用连续布局的 Tensor；输入已经连续时可能直接返回原 Tensor，输入不连续时会生成连续数据。随后 `view(12)` 就能成功。

**Predict：** 两种成功结果的 values 应该是什么？它们为什么都不等于原 `matrix` 逐行展平后的 `0, 1, 2, ...`？

```python
try:
    transposed.view(12)
except RuntimeError as error:
    print("view error type:", type(error).__name__)
    print("view error last line:")
    print(str(error).splitlines()[-1])

reshaped = transposed.reshape(12)
contiguous_view = transposed.contiguous().view(12)

print("reshape values:", reshaped)
print("reshape contiguous:", reshaped.is_contiguous())
print("contiguous().view values:", contiguous_view)
print("contiguous().view contiguous:", contiguous_view.is_contiguous())
```

应观察到：

```text
view error type: RuntimeError
view error last line:
reshape values: tensor([ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
reshape contiguous: True
contiguous().view values: tensor([ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
contiguous().view contiguous: True
```

`view error last line:` 后面还会多一行由当前 PyTorch 版本产生的实际英文错误；上面的固定输出没有伪造该版本相关文本。两条成功路径都保持 `transposed` 的**逻辑元素顺序**，所以本例打印相同 values。但是不能由此声称二者总有相同复制行为：`reshape` 自己决定能否共享存储；`contiguous().view(...)` 则先明确请求连续布局，再创建 view。选择 API 时应描述这个语义差异，而不是只比较一次运行的 values。

### 4.4 `permute` 一次重排多个维度

`permute(dims)` 的用途是按给定顺序重新排列所有维度。参数必须列出每个原维度一次。下面的 `cube` 是 `[B, S, D] = [2, 2, 3]`；`permute(1, 2, 0)` 把轴顺序从 `[B, S, D]` 改成 `[S, D, B]`。

**Predict：** 先在纸上写出新 shape `[2, 3, 2]`。再检查新 Tensor 的最后一维：它应把两个 batch 中相同 `S`、`D` 位置的 values 放在一起。

```python
cube = torch.arange(12).reshape(2, 2, 3)
reordered = cube.permute(1, 2, 0)

print("cube:")
print(cube)
print("cube shape/stride:", tuple(cube.shape), cube.stride())
print("reordered [S, D, B]:")
print(reordered)
print("reordered shape/stride:", tuple(reordered.shape), reordered.stride())
print("reordered contiguous:", reordered.is_contiguous())
```

预期输出为：

```text
cube:
tensor([[[ 0,  1,  2],
         [ 3,  4,  5]],

        [[ 6,  7,  8],
         [ 9, 10, 11]]])
cube shape/stride: (2, 2, 3) (6, 3, 1)
reordered [S, D, B]:
tensor([[[ 0,  6],
         [ 1,  7],
         [ 2,  8]],

        [[ 3,  9],
         [ 4, 10],
         [ 5, 11]]])
reordered shape/stride: (2, 3, 2) (3, 1, 6)
reordered contiguous: False
```

### 常见误区

- **把 shape 当成存储顺序：** shape 只说明每个逻辑轴的长度；还要看 stride 和连续性，才能描述下标如何映射到底层存储。
- **认为 `transpose` 或 `permute` 会自动复制并重排数据：** 它们通常返回共享存储、stride 改变的 view。需要连续布局时应显式检查或调用 `contiguous()`。
- **认为非连续就一定不能 `view`：** `view` 检查的是目标 shape 与当前 size/stride 是否兼容。某些非连续 Tensor 仍能做部分 `view`；本例只是不能把转置后的两个维度直接合并成一维。
- **认为 `reshape` 和 `contiguous().view` 总是同样复制：** 本例 values 相同不代表内部路径永远相同。`reshape` 可能共享也可能复制，`contiguous()` 则明确保证返回连续布局。

### 练习

**M4-E1**：不运行代码，分析 `x = torch.arange(6).reshape(2, 3)`。写出 `x` 和 `x.transpose(0, 1)` 的 values、shape、stride 与 `is_contiguous()`，再用一个下标等式说明转置前后的 value 对应关系。

**M4-E2**：给定 `x = torch.arange(24).reshape(2, 3, 4)`，预测 `y = x.permute(2, 0, 1)` 的 shape 和 stride。说明新三个轴分别来自原来的哪个轴，并判断 `y.is_contiguous()`。最后比较 `y.reshape(24)` 与 `y.contiguous().view(24)`：应比较哪些可观察结果，又不能仅凭一次结果断言什么？

### 模块 4 验收

1. 你能否分别用一句话解释 shape、stride 和 `is_contiguous()` 回答什么问题？
2. 你能否根据 `(3, 4)` 的连续矩阵推导 stride `(4, 1)`，再推导转置后的 `(1, 4)`？
3. 你能否用 shape/stride 不兼容解释这次转置 Tensor 的 `view(12)` 报错，同时说明非连续 Tensor 并非所有 `view` 都必然失败，再正确使用 `reshape` 或 `contiguous().view(...)`？
4. 你能否解释为什么两种成功路径 values 相同，也不能证明它们总有相同复制行为？

<a id="module-5"></a>
## 模块 5：广播与矩阵乘法

广播和矩阵乘法都会让不同 shape 的 Tensor 一起计算，但规则完全不同。广播比较对应维度能否做逐元素运算；矩阵乘法则收缩一对相等的内维度。先在纸上对齐 shape，能避免把两套规则混在一起。

### 5.1 广播从尾部维度开始比较

两个 shape 做逐元素运算时，从最右侧维度开始逐项对齐。每一对维度必须满足以下条件之一：

1. 两个长度相等。
2. 其中一个长度是 1，可以沿该维扩展使用。
3. 某个 shape 左侧没有该维，把它视为长度 1。

如果任意一对都不满足，广播失败。所谓“扩展”是逻辑上的重复使用，不要求先手工创建一份更大的 Tensor。

先取 `hidden [B, S, D] = [2, 3, 2]`：

```text
hidden:       [2, 3, 2]
feature_bias: [      2]
补齐后:       [1, 1, 2]
结果:         [2, 3, 2]
```

从右向左解释每一维：`D` 维是 `2` 对 `2`，长度相等；`S` 维是 `3` 对补齐的 `1`，bias 沿三个位置使用；`B` 维是 `2` 对补齐的 `1`，bias 沿两个 batch 使用。因此 `[B, S, D] + [D] -> [B, S, D]`。

**Predict：** 下面的 `[10, 100]` 会加到每个 token 的哪两个数上？先写出 12 个结果 values。

```python
import torch

hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
feature_bias = torch.tensor([10.0, 100.0])
biased = hidden + feature_bias

print("hidden:")
print(hidden)
print("feature_bias shape:", tuple(feature_bias.shape))
print("biased:")
print(biased)
print("biased shape:", tuple(biased.shape))
```

预期输出为：

```text
hidden:
tensor([[[ 0.,  1.],
         [ 2.,  3.],
         [ 4.,  5.]],

        [[ 6.,  7.],
         [ 8.,  9.],
         [10., 11.]]])
feature_bias shape: (2,)
biased:
tensor([[[ 10., 101.],
         [ 12., 103.],
         [ 14., 105.]],

        [[ 16., 107.],
         [ 18., 109.],
         [ 20., 111.]]])
biased shape: (2, 3, 2)
```

再对齐 `[B, S, D] + [S, 1]`：

```text
hidden:          [2, 3, 2]
position_offset: [   3, 1]
补齐后:          [1, 3, 1]
结果:            [2, 3, 2]
```

右侧 `D` 维是 `2` 对 `1`，每个位置的 offset 加到两个特征上；中间 `S` 维是 `3` 对 `3`，三个 offset 分别对应三个 token 位置；左侧 `B` 维是 `2` 对补齐的 `1`，同一组位置 offset 用于两个 batch。因此 `[B, S, D] + [S, 1] -> [B, S, D]`。

**Predict：** 三行 offset 分别只对应 `S = 0, 1, 2`。先预测它们怎样跨两个 batch、两个特征重复使用。

```python
position_offset = torch.tensor([[100.0], [200.0], [300.0]])
shifted = hidden + position_offset

print("position_offset shape:", tuple(position_offset.shape))
print(shifted)
print("shifted shape:", tuple(shifted.shape))
```

预期输出为：

```text
position_offset shape: (3, 1)
tensor([[[100., 101.],
         [202., 203.],
         [304., 305.]],

        [[106., 107.],
         [208., 209.],
         [310., 311.]]])
shifted shape: (2, 3, 2)
```

形状 `[2, 2]` 则不能与 `[2, 3, 2]` 广播：从右侧看 `2` 对 `2` 合法，但下一维是 `3` 对 `2`，既不相等也没有 1。`torch.ones(shape)` 创建给定 shape、values 全为 1 的 Tensor；这里用它提供确定性的失败输入。下面打印当前 PyTorch 版本返回的真实错误文本：

```python
try:
    hidden + torch.ones(2, 2)
except RuntimeError as error:
    print("broadcast error type:", type(error).__name__)
    print(str(error).splitlines()[-1])
```

不要从报错中猜一个新 shape 反复尝试。先在纸上右对齐 `[2, 3, 2]` 和 `[2, 2]`，定位冲突的 `3` 与 `2`，再决定数据本来应该按 batch、位置还是特征对齐。

### 5.2 `*` 是逐元素乘法，不是矩阵乘法

对 Tensor 使用 `*` 时，PyTorch 按广播规则做逐元素乘法。下面的 `feature_scale [D]` 依次缩放每个隐藏向量的两个特征，输出仍是 `[B, S, D]`。

**Predict：** 右对齐 `[2, 3, 2]` 与 `[2]`，再预测每个隐藏向量的第一个数和第二个数分别乘什么。

```python
feature_scale = torch.tensor([10.0, 0.1])
elementwise = hidden * feature_scale

print(elementwise)
print("elementwise shape:", tuple(elementwise.shape))
```

应观察第一个特征乘 10、第二个特征乘 0.1，shape 仍为 `(2, 3, 2)`。这里没有求行列点积，也没有消去任何维度。

### 5.3 `@` 收缩 `D`，保留 `B`、`S` 并产生 `H`

`@` 对 Tensor 执行矩阵乘法，与这里使用 `torch.matmul` 的含义相同。令 `hidden [B, S, D] = [2, 3, 2]`，`projection [D, H] = [2, 3]`：

```text
hidden:     [B, S, D] = [2, 3, 2]
projection:       [D, H] =    [2, 3]
                         相等的 D 收缩
output:     [B, S, H] = [2, 3, 3]
```

从矩阵乘法角度看，`B` 是 matmul 的 batch 维度；对每个 batch，左侧都可看成一个 `[S, D]` 矩阵，其中 `S` 是矩阵的行维，也对应序列位置。`D` 是收缩的内维度：每个长度为 2 的行向量与 `projection` 的每一列做点积，因而不再出现在输出 shape 中。右侧 `[D, H]` 的 `H = 3` 是列数，并成为输出矩阵 `[S, H]` 的列维。因此整体输出是 `[B, S, H]`：保留 batch 维 `B` 和行/序列维 `S`，产生输出列维 `H`。

**Predict：** 下面的 projection 会把向量 `[a, b]` 变成 `[a, b, a + b]`。据此手算全部输出，再运行。

```python
projection = torch.tensor(
    [[1.0, 0.0, 1.0],
     [0.0, 1.0, 1.0]]
)
output = hidden @ projection
matmul_output = torch.matmul(hidden, projection)

print("projection shape:", tuple(projection.shape))
print(output)
print("output shape:", tuple(output.shape))
print("same size and values:", torch.equal(output, matmul_output))
print("same dtype:", output.dtype == matmul_output.dtype)
```

`torch.equal(a, b)` 在两个 Tensor 的 size 相同且对应元素值相同时返回 `True`；它不要求 dtype 相同，并且对应位置都是 `NaN` 时仍视为不相等。如果还需要验证 dtype，应像代码最后一行那样单独比较 `.dtype`。预期输出为：

```text
projection shape: (2, 3)
tensor([[[ 0.,  1.,  1.],
         [ 2.,  3.,  5.],
         [ 4.,  5.,  9.]],

        [[ 6.,  7., 13.],
         [ 8.,  9., 17.],
         [10., 11., 21.]]])
output shape: (2, 3, 3)
same size and values: True
same dtype: True
```

矩阵乘法要求左侧最后一维与右侧倒数第二维相等。下面故意使用 `[4, 3]` 的矩阵，使 `hidden` 的 `D = 2` 与矩阵的输入维 `4` 冲突，并打印实际 `RuntimeError`：

```python
bad_projection = torch.ones(4, 3)

try:
    hidden @ bad_projection
except RuntimeError as error:
    print("matmul error type:", type(error).__name__)
    print("hidden shape:", tuple(hidden.shape))
    print("bad projection shape:", tuple(bad_projection.shape))
    print("matmul error last line:")
    print(str(error).splitlines()[-1])
```

运行时最后一行由当前 PyTorch 版本产生。无论英文措辞如何，调试重点都是自己打印的两个 shape：`[2, 3, 2] @ [4, 3]` 中，应该相等的收缩维度是 `2` 和 `4`，但它们不同。

### 常见误区

- **从左侧开始比较广播：** 广播必须从尾部右对齐；左侧缺失的维度才视为 1。
- **看到长度 1 就随意对齐：** 维度位置仍然重要。`[S, 1]` 右对齐到 `[B, S, D]`，不是自动寻找名字为 `S` 的轴。
- **把 `*` 当成线性层：** `*` 做逐元素乘法并遵循广播；`@`/`torch.matmul` 做矩阵乘法并收缩内维度。
- **只背 `[B,S,D] @ [D,H]` 的结果：** 真正需要检查的是左侧最后一维 `D` 是否等于右侧倒数第二维 `D`，并明确哪些维度保留、收缩和新产生。

### 练习

**M5-E1**：对 `x [B, S, D] = [2, 4, 3]`，先右对齐并判断以下 shape 能否与它相加：`[3]`、`[4, 1]`、`[2, 1, 3]`、`[2, 4]`。对每个维度逐项说明相等、长度为 1，或发生冲突；不要只写“能/不能”。

**M5-E2**：给定 `x [B, S, D] = [2, 4, 3]` 和 `weight [D, H] = [3, 5]`，推导 `x @ weight` 的输出 shape，并分别指出 matmul batch 维、左矩阵行/序列维、收缩维和输出列维。再判断 `x * weight` 是否可广播，并解释它为什么不是同一个计算。

### 模块 5 验收

1. 你能否把两个 shape 右对齐，并逐维应用“相等、其中一个为 1、左侧缺失视为 1”的规则？
2. 你能否解释 `[B,S,D] + [D]` 与 `[B,S,D] + [S,1]` 中每个维度为什么合法？
3. 你能否区分 `*` 和 `@`，推导 `[B,S,D] @ [D,H] -> [B,S,H]`，并把 `B`、`S`、`D`、`H` 分别解释为 matmul batch、左矩阵行/序列、收缩和输出列维？
4. 面对真实 matmul 报错时，你能否用输入 shape 找出冲突的两个收缩维度？

<a id="module-6"></a>
## 模块 6：dtype 与内存估算

Tensor 的理论数据量由两个因素决定：有多少元素，以及每个元素用多少位。先做与设备无关的纸笔估算，再用 PyTorch 的真实 dtype 验证。理论数据量不包含 Python 对象、Tensor 元数据、分配器对齐、缓存和计算库工作区等额外开销。

### 6.1 从位数计算理论字节数

设 Tensor 有 `N = numel` 个元素，每个元素占 `b` 位：

```text
理论位数 = N * b
理论字节数 = N * b / 8
```

对于每元素位数能整除 8 的常见格式，也可以直接使用“元素数乘每元素字节数”。下面取一个故意为奇数的 `N = 13`：

| 格式 | 每元素位数 | 13 个元素的计算 | 理论字节数 |
| --- | ---: | --- | ---: |
| FP32 | 32 | `13 * 32 / 8` | 52 |
| FP16 | 16 | `13 * 16 / 8` | 26 |
| BF16 | 16 | `13 * 16 / 8` | 26 |
| INT8 | 8 | `13 * 8 / 8` | 13 |
| packed INT4 | 4 | `ceil(13 * 4 / 8)` | 7 |

FP16 和 BF16 都是 16 位，所以理论数据量相同；它们的数值编码、精度和范围并不相同。INT8 每元素一字节。packed INT4 把两个 4 位值打包进一个字节；13 个值需要 52 位，即 6.5 字节，存储必须使用完整字节，所以向上取整为 7 字节。整数计算可以写成：

```python
def packed_int4_bytes(numel):
    return (numel * 4 + 7) // 8

print(packed_int4_bytes(12))
print(packed_int4_bytes(13))
```

预期输出是 `6` 和 `7`。`//` 是整数向下除法；在分子先加 `7`，实现了除以 8 时的向上取整。

普通 PyTorch Tensor API 没有供本教程直接创建和逐元素观察的常规 instructional INT4 dtype。实际 INT4 量化通常使用打包存储、量化参数和专用 kernel。因此表中的 INT4 是**仅针对打包数据本体的理论估算**，不能写成 `torch.zeros(..., dtype=torch.int4)` 来验证，也不能自动代表某个量化文件或运行时的全部内存。

### 6.2 用 `numel() * element_size()` 验证真实 dtype

`element_size()` 方法返回 Tensor 中单个元素占用的字节数。对普通未打包 PyTorch Tensor，`numel() * element_size()` 给出其数据本体的理论字节数。`torch.zeros(shape, dtype=...)` 创建给定 shape、values 全为 0 的 Tensor；这里 values 不重要，选择 0 是为了让构造确定且容易检查。

**Predict：** 先根据 13 个元素手算下面四行的每元素字节数和总字节数，再运行验证。

```python
import torch

for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int8]:
    tensor = torch.zeros(13, dtype=dtype)
    print(
        dtype,
        "numel=", tensor.numel(),
        "element_size=", tensor.element_size(),
        "bytes=", tensor.numel() * tensor.element_size(),
    )
```

预期输出为：

```text
torch.float32 numel= 13 element_size= 4 bytes= 52
torch.float16 numel= 13 element_size= 2 bytes= 26
torch.bfloat16 numel= 13 element_size= 2 bytes= 26
torch.int8 numel= 13 element_size= 1 bytes= 13
```

这段验证回答的是“这个 Tensor 的元素数据理论上占多少字节”，不是“整个 Python 进程增加了多少内存”。同理，把结果除以 `1024` 得到 KiB，除以 `1024 ** 2` 得到 MiB；不要把十进制 MB 与二进制 MiB 混写。

### 6.3 可选 CUDA：比较理论数据量与分配器指标

本小节只用于有可用 CUDA 的环境。`torch.cuda.is_available()` 守卫保证 CPU-only 学习者看到清晰跳过消息并正常结束。实验不下载模型，只分配一个 `1024 x 1024` 的 FP32 Tensor，其理论数据量是 `4,194,304` 字节，即 4 MiB。

`torch.device("cuda")` 创建一个表示默认 CUDA 设备的设备对象，便于把同一设备明确传给后续 API。`empty_cache()` 释放缓存分配器中当前未被活跃 Tensor 使用的缓存块，但不会释放仍被 Tensor 引用的内存。`torch.cuda.synchronize()` 等待当前设备上已提交的 CUDA 工作完成，使观察点更明确。`reset_peak_memory_stats()` 把峰值统计重置到调用时的当前 allocator 基线；调用后若尚未增加分配，`max_memory_allocated()` 的基线就是当时的 `memory_allocated()`。`memory_allocated()` 统计当前由活跃 PyTorch Tensor 占用、并由 PyTorch CUDA allocator 计数的内存；`memory_reserved()` 统计该 allocator 已向 CUDA 保留的内存；`max_memory_allocated()` 返回重置以来前者达到的峰值。`torch.empty(shape, dtype=..., device=...)` 在指定设备上分配 Tensor 而不初始化 values；本实验只观察内存，不读取其中的任意值。

```python
import torch

if not torch.cuda.is_available():
    print("CUDA 不可用：跳过模块 6 可选显存实验；CPU 主路径已足够完成本模块。")
else:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    before_allocated = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    sample = torch.empty((1024, 1024), dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)

    theoretical_bytes = sample.numel() * sample.element_size()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)

    print("theoretical bytes:", theoretical_bytes)
    print("allocated bytes:", allocated)
    print("reserved bytes:", reserved)
    print("peak allocated bytes:", peak_allocated)
    print("allocated increase:", allocated - before_allocated)
    print("reserved increase:", reserved - before_reserved)
    print("peak allocated increase:", peak_allocated - before_allocated)

    del sample
    torch.cuda.synchronize(device)
```

有 CUDA 时，必须记录实际数字而不是预先抄一个固定输出。在这个隔离实验中，没有同时创建其他 PyTorch Tensor 时，`allocated increase` 通常应当**精确等于** `theoretical bytes`，因为两者都描述新 Tensor 的数据存储。`peak allocated increase` 也从同一个 `before_allocated` 基线计算；峰值在 reset 时被设为当前基线，分配 `sample` 后它记录相对该基线达到的最大增量。`reserved increase` 则可能与 Tensor 字节数不同，因为缓存分配器按内存块申请、保留和复用内存。

这三个值都是 **PyTorch CUDA allocator 计数器**，不是 CUDA 进程的总 GPU 内存。它们不计入 CUDA context、非 PyTorch 分配以及某些外部库直接进行的分配；`nvidia-smi` 显示的进程 GPU 内存口径更广，因此不能拿它与 `memory_allocated()` 或 `memory_reserved()` 当作同一个指标逐字节比较。

因此要区分：

- **理论字节数：** 由 shape、numel 和 dtype 位数决定，只计算目标数据本体。
- **allocated：** PyTorch 当前活跃 Tensor 由缓存分配器计入的字节数。
- **reserved：** PyTorch 已向 CUDA 保留、其中可能有暂未被活跃 Tensor 使用的内存。
- **peak allocated：** 峰值重置到当前 allocated 基线后，allocated 曾达到的最大值；减去 `before_allocated` 才得到本实验的峰值增量。
- **进程总 GPU 内存：** `nvidia-smi` 等工具按更广口径观察的进程占用，可能包含 allocator 计数器之外的 CUDA context 和其他分配。

### 常见误区

- **把 bit 当 byte：** 32 bit 是 4 byte，因为 1 byte = 8 bit。
- **对奇数个 INT4 元素直接舍去半字节：** 实际存储必须占完整字节，所以应向上取整；13 个 packed INT4 是 7 字节，不是 6 字节。
- **假设有普通 `torch.int4` 可直接验证：** INT4 通常依赖打包表示和专用实现，本教程只做数据本体理论估算。
- **混淆 allocator 与进程总显存：** allocated/reserved 只反映 PyTorch CUDA allocator；reserved 还受 allocator 内存块影响。CUDA context 等更广开销可能出现在 `nvidia-smi` 的进程总 GPU 内存中，但不应归因于 allocated/reserved。
- **认为 reserved 都被当前 Tensor 使用：** reserved 包含缓存分配器保留但当前可能空闲的块，通常不等于 allocated。

### 练习

**M6-E1**：一个 shape 为 `[3, 5, 7]` 的 Tensor 有多少元素？分别计算 FP32、FP16、BF16、INT8 和 packed INT4 的理论字节数。INT4 必须写出向上取整步骤。

**M6-E2**：创建 shape 为 `[2, 3, 4]` 的 `torch.float32`、`torch.float16`、`torch.bfloat16` 和 `torch.int8` Tensor，用 `numel()`、`element_size()` 和乘积验证理论数据量。解释为什么 FP16 与 BF16 字节数相同，却不能据此说它们的数值性质相同。

**M6-E3（可选 CUDA）**：在 CUDA 可用性守卫内，把示例 shape 改成 `[512, 512]`，先手算 FP32 理论字节数，再记录 allocated increase、reserved increase 和 `peak allocated increase = peak allocated - before_allocated`。检查隔离实验中的 allocated increase 是否精确等于 Tensor 理论字节数，解释 reserved increase 为什么可能不同，并说明这些 allocator 计数器为什么不等于 `nvidia-smi` 的进程总 GPU 内存。CUDA 不可用时，写下跳过原因即可通过本题。

### 模块 6 验收

1. 你能否从 `numel * bits / 8` 推导 FP32、FP16、BF16、INT8 的理论字节数？
2. 你能否对奇数个 packed INT4 元素正确向上取整，并说明为什么普通 PyTorch dtype 示例不直接包含 INT4？
3. 你能否使用 `numel() * element_size()` 验证真实 PyTorch dtype 的数据本体字节数？
4. 你能否从当前基线计算 peak allocated increase，解释隔离分配时 allocated increase 与理论 Tensor 字节数的关系，并区分 allocator 的 allocated/reserved 与 `nvidia-smi` 的进程总 GPU 内存？

<a id="capstone"></a>
## 综合任务：走过一次微型语言模型数据流

综合任务将串联 token ID、embedding、隐藏状态和 logits，要求逐步记录每个 Tensor 的形状、dtype、维度含义与内存估算。

<a id="acceptance"></a>
## 最终验收

验收将检查你是否能脱离运行结果预测形状、解释关键属性、判断广播与矩阵乘法，并完成小 Tensor 的内存手算。

<a id="hints"></a>
## 提示

提示只给思考方向，不直接替代练习。先回到题目写下自己的预测，再按需展开对应编号。

### 模块 1

**M1-E1 提示：** 先数外层有几行，再数每行有几个元素。`dtype` 已由参数明确指定；没有显式设备参数时，沿用本周 CPU 主路径。

**M1-E2 提示：** 把符号 `[B, S]` 中的 `B` 和 `S` 分别替换成题目给出的序列数与每条序列的位置数。构造数据时检查每一行长度是否一致。

### 模块 2

**M2-E1 提示：** rank 等于 shape 中数字的个数，`numel` 等于这些数字的乘积。沿 `[B, S, D]` 的顺序定位第 1 维，不要按自然语言中的“第一个”猜。

**M2-E2 提示：** 先分别把 `[V, D]` 和 `[B, S]` 的字母写在两个具体数字上方。比较的不只是长度，还要比较每个轴和每个 value 表示什么。

### 模块 3

**M3-E1 提示：** 对每个表达式从左到右检查三个轴。整数下标会选定并消去一个轴，`start:stop` 或 `:` 会保留对应轴。

**M3-E2 提示：** 先算 `2 * 3 * 4`，再逐个计算候选 shape 的乘积。`unsqueeze` 的目标新维度位于原第 0 维和第 1 维之间。

### 模块 4

**M4-E1 提示：** 连续二维 Tensor 的最后一维 stride 为 1；向下一行要跨过一整行的元素数。转置会交换 shape 和 stride 中对应的两个轴。可以用 `xt[i, j] == x[j, i]` 表达 value 对应关系。

**M4-E2 提示：** 原 shape `[2, 3, 4]` 的连续 stride 是 `(12, 4, 1)`。`permute(2, 0, 1)` 同时按顺序挑选原 shape 和原 stride。比较结果时观察 values、shape 和连续性，但复制或共享不能仅由 values 相等推出。

### 模块 5

**M5-E1 提示：** 给每个较短 shape 在左侧补 1，再从最右侧逐项写出比较。`[2, 4]` 补齐后是 `[1, 2, 4]`，不要把其中的数字自动按业务字母重新排序。

**M5-E2 提示：** 把每个 batch 内的左侧看成 `[S, D] = [4, 3]` 矩阵：`B = 2` 是 matmul batch 维，`S = 4` 是左矩阵行/序列维，`D = 3` 与右矩阵倒数第二维收缩，`H = 5` 是输出列维。判断 `*` 时改用广播规则，把 `[2, 4, 3]` 与 `[3, 5]` 右对齐。

### 模块 6

**M6-E1 提示：** 先算 `3 * 5 * 7 = 105`。packed INT4 的字节数可用 `(105 * 4 + 7) // 8`，不要把 52.5 直接向下取整。

**M6-E2 提示：** 四个 Tensor 的 `numel` 都是 `2 * 3 * 4`，差异只来自 `element_size()`。内存位数相同只说明数据本体大小相同，不说明指数位、尾数位、精度或范围相同。

**M6-E3 提示：** `[512, 512]` 有 262,144 个元素，FP32 每元素 4 字节。先在 `reset_peak_memory_stats()` 前记录 `before_allocated`；reset 把峰值设到这个当前基线，所以峰值增量应计算为 `peak_allocated - before_allocated`。隔离分配时 allocated increase 通常精确匹配 Tensor 字节数，reserved increase 可能因 allocator 内存块而不同；CUDA context 属于更广的进程总显存口径，不应归因于 allocated/reserved。

<a id="answers"></a>
## 参考答案

只有在完成预测和运行验证后再核对答案。重点不是抄写 shape，而是确认推理路径。

### 模块 1

**M1-E1 答案：** values 是三行 `[[7, 8], [9, 10], [11, 12]]`；对象类型是 `torch.Tensor`；shape 是 `torch.Size([3, 2])`；dtype 是 `torch.int64`；默认 device 是 `cpu`。外层有 3 个子列表，所以第 0 维长度为 3；每个子列表有 2 个整数，所以第 1 维长度为 2。

**M1-E2 答案：** 符号形状是 `[B, S]`，代入 `B = 3`、`S = 4` 后是 `[3, 4]`。一种确定性构造是 `torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int64)`。它有 3 行，对应 3 条序列；每行 4 个整数，对应每条序列的 4 个 token 位置，因此满足题意。具体编号可以不同，但必须是规则的 3 行 4 列整数数据。

### 模块 2

**M2-E1 答案：** `[3, 4, 5]` 含三个维度，所以 rank 是 3。按 `[B, S, D]` 对应，`B = 3`、`S = 4`、`D = 5`；元素总数是 `3 * 4 * 5 = 60`。第 1 维是 `S`，长度 4 表示每条序列有 4 个 token 位置。可以用 `torch.arange(60).reshape(3, 4, 5)` 验证；`torch.arange(60)` 生成从 0 到 59 的 60 个连续整数，`reshape` 再把它们组织成目标 shape。应观察 `ndim == 3`、`shape == torch.Size([3, 4, 5])`、`numel() == 60`，无需打印全部 values。

**M2-E2 答案：** `embedding_weight [V, D] = [6, 3]` 是 rank 2，shape 为 `[6, 3]`，元素总数为 `6 * 3 = 18`。即使 `token_ids [B, S]` 也是 `[6, 3]`，前者的第 0 维是 6 个词表条目、第 1 维是宽度为 3 的表示，而且 values 通常是浮点数；后者的两个维度是 6 条序列和每条 3 个 token 位置，values 是整数编号。具体 shape 相同不代表维度语义、dtype 或用途相同。

### 模块 3

**M3-E1 答案：** `hidden[0].shape` 是 `torch.Size([3, 4])`，因为整数 `0` 选定并消去 `B` 维；`hidden[0:1].shape` 是 `torch.Size([1, 3, 4])`，因为切片保留长度变为 1 的 `B` 维；`hidden[:, 1].shape` 是 `torch.Size([2, 4])`，因为 `:` 保留 `B` 维，而整数 `1` 消去 `S` 维；`hidden[:, 1:2].shape` 是 `torch.Size([2, 1, 4])`，因为两个切片都保留轴，只把 `S` 维长度变为 1。

**M3-E2 答案：** 原 Tensor 有 `2 * 3 * 4 = 24` 个元素。`[6, 4]`、`[4, 6]` 和 `[2, 2, 6]` 的乘积都是 24，因此都兼容；`[5, 5]` 的乘积是 25，因此不兼容。若 `x.shape == torch.Size([2, 3, 4])`，`y = x.unsqueeze(1)` 会在第 1 维插入长度 1，得到 `[2, 1, 3, 4]`；`z = y.squeeze(1)` 只删除该长度为 1 的维度，恢复 `[2, 3, 4]`。两步都不改变元素总数。

### 模块 4

**M4-E1 答案：** `x` 的 values 是 `[[0, 1, 2], [3, 4, 5]]`，shape 是 `[2, 3]`，连续 stride 是 `(3, 1)`，`is_contiguous()` 为 `True`。转置后 values 是 `[[0, 3], [1, 4], [2, 5]]`，shape 是 `[3, 2]`，stride 是 `(1, 3)`，`is_contiguous()` 为 `False`。例如 `x.transpose(0, 1)[2, 1] == x[1, 2] == 5`，因为新第 0、1 维分别来自原第 1、0 维。

**M4-E2 答案：** 原连续 stride 是 `(12, 4, 1)`。`permute(2, 0, 1)` 按“原第 2 维、原第 0 维、原第 1 维”重排，所以 `y.shape == torch.Size([4, 2, 3])`，`y.stride() == (1, 12, 4)`，通常 `y.is_contiguous()` 为 `False`。`y.reshape(24)` 和 `y.contiguous().view(24)` 应得到相同逻辑 values、shape `[24]` 和连续结果；但一次输出不能证明复制路径相同。`reshape` 可能返回 view 或副本，后者明确先请求连续布局再 `view`。

### 模块 5

**M5-E1 答案：** `[3]` 补齐为 `[1, 1, 3]`：与 `[2, 4, 3]` 比较时依次是 `3=3`、`1` 可扩展到 `4`、`1` 可扩展到 `2`，所以合法。`[4, 1]` 补齐为 `[1, 4, 1]`：`1` 扩展到 `3`、`4=4`、`1` 扩展到 `2`，合法。`[2, 1, 3]`：`3=3`、`1` 扩展到 `4`、`2=2`，合法。`[2, 4]` 补齐为 `[1, 2, 4]`：最右侧先遇到 `3` 对 `4`，既不相等也没有 1，所以不合法；无需继续寻找别的对齐方式。

**M5-E2 答案：** `[2, 4, 3] @ [3, 5] -> [2, 4, 5]`。`B = 2` 是 matmul batch 维；在每个 batch 内，左侧 `[S, D] = [4, 3]` 的 `S = 4` 是矩阵行维，也对应序列位置；左侧最后一维 `D = 3` 与右侧倒数第二维 `D = 3` 收缩；右侧 `H = 5` 是列数，并成为输出列维。对 `x * weight`，右对齐 `[2, 4, 3]` 与 `[1, 3, 5]` 后，最右侧是 `3` 对 `5`，不能广播，所以该具体表达式失败。即使换成可广播 shape，`*` 也只会逐元素相乘，不会执行点积或把 `D` 替换为 `H`。

### 模块 6

**M6-E1 答案：** 元素总数是 `3 * 5 * 7 = 105`。FP32 为 `105 * 32 / 8 = 420` 字节；FP16 为 `105 * 16 / 8 = 210` 字节；BF16 同样为 210 字节；INT8 为 `105 * 8 / 8 = 105` 字节；packed INT4 为 `ceil(105 * 4 / 8) = ceil(52.5) = 53` 字节，也可算 `(105 * 4 + 7) // 8 = 53`。

**M6-E2 答案：** `[2, 3, 4]` 有 24 个元素。`torch.float32` 的 `element_size()` 为 4，总计 96 字节；`torch.float16` 为 2，总计 48 字节；`torch.bfloat16` 为 2，总计 48 字节；`torch.int8` 为 1，总计 24 字节。FP16 与 BF16 都使用 16 位，所以数据本体字节数相同；但两者对指数和有效精度的编码取舍不同，数值范围与舍入特性不能由字节数推出。

**M6-E3 答案：** `[512, 512]` 有 `512 * 512 = 262,144` 个元素，FP32 理论数据量是 `262,144 * 4 = 1,048,576` 字节，即 1 MiB。`reset_peak_memory_stats()` 把峰值重置到调用时的 `before_allocated` 基线，所以本实验应报告 `peak_allocated - before_allocated`。在没有并发 PyTorch 分配的隔离实验中，allocated increase 通常精确等于 `1,048,576` 字节，peak allocated increase 也通常达到该值；reserved increase 可能因 allocator 申请或复用内存块而不同。这些数字只属于 PyTorch CUDA allocator，不包含 CUDA context 等更广进程开销；`nvidia-smi` 的进程总 GPU 内存因此可能更大，不能与 allocated/reserved 视为同一指标。CUDA 不可用时，记录守卫打印的跳过消息和 CPU 理论计算即可。

<a id="glossary"></a>
## 术语与速查表

本节将汇总 Tensor、axis、shape、stride、contiguous、broadcast、dtype 等术语和常用形状规则。

<a id="next-week"></a>
## 下一周预告

下一周将在本周形状与内存直觉之上进入神经网络基本计算，继续坚持先预测、再运行、最后解释。
