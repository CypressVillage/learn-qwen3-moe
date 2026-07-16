# 第一周张量教程设计规格

## 目标

为具备 Python 基础、尚未系统学习 PyTorch 的学习者编写一份第一周自包含教程。学习者每周可投入 6-10 小时，使用 RTX 4060 Laptop 8GB 和单张 A10 24GB，但所有必修内容必须能够仅使用 CPU 完成。

教程完成后，学习者应能：

- 解释 Tensor、rank、shape、dtype、device、stride 和 `numel()`。
- 在运行代码前预测索引、切片、形状变换、广播和矩阵乘法的输出 shape。
- 区分逻辑形状、底层存储布局和是否 contiguous。
- 根据 shape 和 dtype 估算张量理论存储大小。
- 区分理论张量字节数、CUDA allocated memory、reserved memory 和进程峰值显存。
- 从 `token_ids [B,S]` 和 `embedding_weight [V,D]` 推导 `hidden [B,S,D]`，再推导线性投影输出。

## 教程形式

采用模块化的“预测－实验－解释”闭环，不绑定具体日期。学习者可以按模块独立暂停和继续，但应按顺序首次学习。

每个模块统一包含：

1. 本节目标。
2. 该知识在 Qwen3-MoE 推理中的用途。
3. 运行前预测。
4. 可直接运行的最小 PyTorch 实验。
5. 代码和输出的逐步解释。
6. 常见误区。
7. 动手练习。
8. 思路提示。
9. 与题目保持距离的参考答案。
10. 模块验收问题。

练习采用两级帮助。学习者应先独立作答，卡住时查看“提示”，仍无法完成时再查看“参考答案”。答案不紧跟题目，避免无意中直接看到。

## 文件与链接

本次教程正文阶段只改动教学文档：

- 新增 `docs/tutorials/week01-tensors-shapes-memory.md`。
- 修改 `README.md`，增加“开始第一周”的直接入口。
- 修改 `docs/roadmap.md`，在第 1 周加入完整教程链接。

不创建 `src/`、`tests/`、练习脚本、Notebook 或模型实现。教程中的代码使用内联代码块，作为可复制运行的教学实验。

## 模块安排

### 模块 1：环境检查与第一个 Tensor

- 验证 Python、PyTorch 和可选 CUDA。
- 导入 PyTorch，创建第一个小 Tensor。
- 查看值、类型、shape、dtype 和 device。
- 解释 Python list 与 Tensor 的用途差异，但不展开性能实现细节。

### 模块 2：rank、shape、dtype、device 与元素数

- 区分标量、向量、矩阵和更高阶 Tensor。
- 解释 rank 与 shape 的区别。
- 使用 `ndim`、`shape`、`dtype`、`device` 和 `numel()`。
- 介绍常见 dtype，并明确 dtype 会影响精度、可表示范围和内存。

### 模块 3：索引、切片与形状变换

- 练习单维和多维索引、切片。
- 使用 `reshape`、`view`、`unsqueeze` 和 `squeeze`。
- 强调元素总数必须兼容，以及形状变化不会改变元素的逻辑总数。
- 解释 `reshape` 可能返回 view，也可能复制，不承诺固定存储行为。

### 模块 4：transpose、permute、stride 与 contiguous

- 用二维和三维小张量解释维度交换。
- 观察形状相同或元素相同不代表存储布局相同。
- 读取 stride 和 `is_contiguous()`。
- 安排非 contiguous Tensor 使用 `view` 的故意失败实验，并用 `reshape` 或 `contiguous().view(...)` 正确处理。

### 模块 5：广播与矩阵乘法

- 从尾部维度开始解释广播规则。
- 区分逐元素乘法和矩阵乘法。
- 逐步推导 `[B,S,D] @ [D,H] -> [B,S,H]`。
- 安排 shape 不兼容的故意失败实验，训练阅读报错信息。

### 模块 6：dtype 与理论内存估算

- 使用“元素数乘每元素位数/字节数”进行估算。
- 对比 FP32、FP16、BF16、INT8 和 INT4。
- INT4 按位打包，奇数元素的总位数换算成完整字节时向上取整。
- 明确普通 PyTorch Tensor 不提供用于本教程直接创建的常规 INT4 dtype；INT4 体积是量化打包格式的理论估算。
- 可选 CUDA 实验观察 `memory_allocated`、`memory_reserved` 和峰值统计。

## 综合任务

综合任务使用微型语言模型数据流：

```text
token_ids:        [B,S]
embedding_weight: [V,D]
hidden:           [B,S,D]
projection:       [D,H]
output:           [B,S,H]
```

学习者必须先完成以下书面预测，再运行 PyTorch 验证：

- 每个张量的 shape 和每一维语义。
- 每个张量的元素数。
- 在指定 dtype 下的理论字节数。
- Embedding 查表后为什么多出隐藏维 `D`。
- 线性投影中哪个维度被收缩，哪些维度被保留。
- 理论总字节数为什么不等于 GPU 进程峰值显存。

综合任务不下载 Tokenizer、模型配置或模型权重。

## 错误与调试教学

教程不隐藏所有错误，而是包含可控的故意失败实验：

- 元素数不兼容的 reshape。
- 对非 contiguous Tensor 使用不兼容的 `view`。
- 广播或矩阵乘法 shape 不匹配。
- 可选 CUDA 环境中混用 CPU 和 CUDA Tensor。

每个失败实验要求先阅读报错最后一行，再向上寻找触发代码和相关 shape。教程随后解释根因和最小修正，不鼓励通过随意 reshape 或移动 device 掩盖问题。

## 验收标准

最终验收包括：

- 10 道概念题，通过线为 8/10。
- 5 道形状推导题，通过线为 4/5。
- 1 个综合实验，必须能脱离答案口述完整数据流。

综合实验通过条件：

- shape 推导正确。
- 元素数与理论内存计算正确。
- 代码运行结果与预测一致，或能解释不一致的原因。
- 能解释 rank 与 shape、view 与 copy、shape 与 stride、理论内存与峰值显存的区别。

CUDA 可选实验不计入通过条件。

## 写作与技术约束

- 正文预计 8,000-12,000 中文字，知识块保持短小。
- 变量统一使用 `B/S/V/D/H`，与 `docs/roadmap.md` 一致。
- 每个 PyTorch API 第一次出现时解释用途、关键参数和返回结果。
- 每段代码说明预期输出或明确要求观察哪些属性。
- 小张量应可手算，避免随机结果；需要随机时固定种子。
- 不提前教授 Attention、MoE 数学或训练，只指出本周知识未来出现的位置。
- 不要求 Hugging Face 登录，不下载模型。
- CUDA 内容明确标为可选，并在不可用时给出跳过方式。
- 文末提供术语表、shape 推导速查表、内存估算速查表、验收清单和第 2 周预告。

## 文档验证

提交教程正文前必须检查：

- Markdown 围栏代码块完整。
- `README.md`、`docs/roadmap.md` 与教程之间的相对链接有效。
- 章节标题、术语和形状符号前后一致。
- 不存在 `TODO`、`TBD`、未填写占位符或声称已经提供但实际缺失的内容。
- 必修代码不依赖 CUDA，也不触发模型下载。
- 提示和参考答案覆盖所有练习，且不与题目直接相邻。
