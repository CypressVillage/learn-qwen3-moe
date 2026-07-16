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

- `Python:` 后有版本号。
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

本模块将从 Python 列表出发创建第一个 Tensor，对照标量、向量和矩阵，并练习在运行前预测内容与形状。

<a id="module-2"></a>
## 模块 2：读懂 Tensor 的基本属性

本模块将集中阅读 `shape`、`ndim`、`numel`、`dtype` 和 `device`，建立“属性输出对应什么事实”的基本语言。

<a id="module-3"></a>
## 模块 3：索引、切片与形状变换

本模块将比较索引与切片对维度的影响，并覆盖常见的 `reshape`、`view`、`unsqueeze` 和 `squeeze` 形状变化。

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

本节将按模块提供由弱到强的思考线索，优先帮助定位维度和元素数量，而不是直接给出结果。

<a id="answers"></a>
## 参考答案

本节将给出核心练习的预测结果与解释路径，用于完成练习后的核对和复盘。

<a id="glossary"></a>
## 术语与速查表

本节将汇总 Tensor、axis、shape、stride、contiguous、broadcast、dtype 等术语和常用形状规则。

<a id="next-week"></a>
## 下一周预告

下一周将在本周形状与内存直觉之上进入神经网络基本计算，继续坚持先预测、再运行、最后解释。
