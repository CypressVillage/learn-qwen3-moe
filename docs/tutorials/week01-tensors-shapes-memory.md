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

例如，看到下面的代码时，不要立刻运行：

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

先明确预测：下面的 Tensor 形状是什么，元素总数是多少，运行设备是什么？写下答案后再执行。

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

只有上面的 CPU 测试通过，并且版本检查显示 `CUDA available: True` 时，才运行这一段：

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

因此，`token_ids.shape == [2, 3]` 应读成：“这一批有 2 条序列，每条有 3 个 token 位置。”数值 `101`、`102` 等只是为了演示而手工写出的编号，不对应真实 tokenizer。后续做语言模型推理时，输入通常正是这样的整数 Tensor；模型会根据每个 token ID 查找对应表示。本周不下载 tokenizer 或模型。

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

dtype 也与语义有关：`token_ids` 使用整数 `torch.int64`，因为它保存编号；`embedding_weight` 和 `hidden` 使用浮点数 `torch.float32`，因为它们保存连续数值表示。device 则说明这些数据在哪里参与计算，本模块仍全部位于 CPU。

### 常见误区

- **把 rank 和 shape 混为一谈：** `[2, 3, 2]` 的 rank 是 3，shape 是 `[2, 3, 2]`，元素数是 12；这三个答案解决不同问题。
- **只背具体数字，不写维度含义：** `hidden.shape == [2, 3, 2]` 还不完整；应继续说明它是 `[B, S, D]`，即序列数、token 位置数、表示宽度。
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

`view(new_shape)` 也用于返回元素数兼容的新形状，但它要求当前内存布局能直接按该形状解释；后续模块 4 会学习 stride 和 contiguous，并看到不满足布局条件时的例子。在本节连续存储的简单 Tensor 上，`view` 可以成功。

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
except RuntimeError:
    print("reshape failed: 12 elements cannot become shape (5, 3)")
```

预期输出为；最后一项错误被 `try/except` 捕获，所以整个脚本仍正常结束，退出码为 0：

```text
flat shape: (12,) numel: 12
reshape(3, 4):
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
shape: (3, 4) numel: 12
view(2, 3, 2) shape: (2, 3, 2) numel: 12
reshape failed: 12 elements cannot become shape (5, 3)
```

`try/except` 是 Python 的错误处理结构：`try` 中故意执行不兼容的 `reshape`，PyTorch 抛出 `RuntimeError` 后进入 `except`，打印稳定的学习提示，而不是让脚本中断。这里要学的不是报错文本，而是报错原因：新旧形状的元素数量不同。

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

本模块将用小矩阵观察 stride、转置后的内存布局，以及何时需要把 Tensor 转为 contiguous。

<a id="module-5"></a>
## 模块 5：广播与矩阵乘法

本模块将用逐维比较的方法判断广播是否合法，并区分逐元素运算、向量点积和矩阵乘法的形状规则。

<a id="module-6"></a>
## 模块 6：dtype 与内存估算

本模块将连接 dtype、每元素字节数、元素总数和理论内存占用，为后续估算模型权重与激活打基础。

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

**M3-E1 答案：** `hidden[0].shape` 是 `[3, 4]`，因为整数 `0` 选定并消去 `B` 维；`hidden[0:1].shape` 是 `[1, 3, 4]`，因为切片保留长度变为 1 的 `B` 维；`hidden[:, 1].shape` 是 `[2, 4]`，因为 `:` 保留 `B` 维，而整数 `1` 消去 `S` 维；`hidden[:, 1:2].shape` 是 `[2, 1, 4]`，因为两个切片都保留轴，只把 `S` 维长度变为 1。

**M3-E2 答案：** 原 Tensor 有 `2 * 3 * 4 = 24` 个元素。`[6, 4]`、`[4, 6]` 和 `[2, 2, 6]` 的乘积都是 24，因此都兼容；`[5, 5]` 的乘积是 25，因此不兼容。若 `x.shape == [2, 3, 4]`，`y = x.unsqueeze(1)` 会在第 1 维插入长度 1，得到 `[2, 1, 3, 4]`；`z = y.squeeze(1)` 只删除该长度为 1 的维度，恢复 `[2, 3, 4]`。两步都不改变元素总数。

<a id="glossary"></a>
## 术语与速查表

本节将汇总 Tensor、axis、shape、stride、contiguous、broadcast、dtype 等术语和常用形状规则。

<a id="next-week"></a>
## 下一周预告

下一周将在本周形状与内存直觉之上进入神经网络基本计算，继续坚持先预测、再运行、最后解释。
