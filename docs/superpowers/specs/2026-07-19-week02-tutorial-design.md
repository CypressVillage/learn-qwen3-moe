# 第二周 PyTorch 模块教程设计规格

## 目标

为完成第一周张量教程的初学者编写一份第二周自包含教程。教程继续采用 Predict-Run-Explain 学习闭环，用可手算的小张量连接广播、矩阵乘法与 PyTorch 模块系统，并为第三周的 Embedding、LM Head 和 logits 数据流做好准备。

教程完成后，学习者应能：

- 逐维判断广播、矩阵乘法、批量矩阵乘法和常用 `einsum` 表达式的输出形状。
- 区分逐元素乘法、矩阵乘法以及显式下标收缩。
- 解释 `nn.Module`、`nn.Parameter`、buffer 和普通 Python 属性的职责。
- 不依赖 `nn.Linear` 实现教学版线性层，并与 PyTorch 参考层做数值对照。
- 读懂 `parameters()`、`named_parameters()`、`buffers()`、`state_dict()` 及递归注册行为。
- 保存并加载微型 `state_dict`，理解它保存张量状态而不是任意 Python 属性。
- 区分 `train()`/`eval()` 与梯度开关，并正确使用 `torch.no_grad()` 和 `torch.inference_mode()`。
- 固定随机种子，显式管理模块和输入的 dtype/device，并对低精度误差使用明确容差。

## 教程形式

沿用第一周的单篇自包含长教程形式：

- 新增 `docs/tutorials/week02-matmul-nn-module.md`。
- 更新 `README.md` 和 `docs/roadmap.md`，提供稳定入口。
- 示例、练习和验证代码全部放在 Markdown 内联代码块中。
- 不新增正式 `src/` 模块、独立练习脚本、Notebook 或测试文件。
- CPU 是必修主路径；CUDA 和 CPU 不稳定支持的低精度运算均为可选观察。
- 不下载模型、Tokenizer、checkpoint 或数据集。

教程维持第一周的统一结构：学习目标、学习方式、环境检查、多个教学模块、综合任务、最终验收、提示、参考答案、术语与速查表、下一周预告。每个模块包含用途说明、运行前预测、确定性实验、输出解释、常见误区、稳定编号练习和模块验收。

## 模块安排

### 模块 1：广播复习与显式形状推理

- 复习从尾部对齐的广播规则，但不重复第一周全部内容。
- 重点分析 bias、缩放量和 batch/sequence 维上的广播语义。
- 强调广播只看维度长度，不理解 `B/S/D` 轴名。
- 包含一个 shape 合法但语义错误的例子，以及一个真实广播失败例子。

### 模块 2：矩阵乘法、批量矩阵乘法与 `einsum`

- 复习 `[B,S,Din] @ [Din,Dout] -> [B,S,Dout]`。
- 解释向量、矩阵和高阶 Tensor 的 `matmul` 规则，只覆盖本周需要的组合。
- 使用 `torch.bmm` 展示严格三维批量矩阵乘法。
- 使用 `torch.einsum` 显式写出 `bsd,do->bso`，解释保留和收缩下标。
- 比较 `@`、`torch.matmul`、`torch.bmm` 和 `einsum` 的适用边界，不把 `einsum` 宣传为默认更快。

### 模块 3：第一个 `nn.Module`

- 解释继承 `nn.Module`、调用 `super().__init__()`、定义 `forward()` 和通过实例调用模块。
- 说明 `module(x)` 应优先于直接调用 `forward(x)`，因为模块调用路径还负责 hooks 等框架行为。
- 用无参数的小模块观察 `training` 状态和模块表示。

### 模块 4：parameter、buffer 与普通属性

- 通过一个微型模块同时注册 `nn.Parameter`、`register_buffer` 和普通属性。
- 对比 `named_parameters()`、`named_buffers()`、`state_dict()` 和 dtype/device 转换。
- 说明 parameter 默认参与梯度跟踪，buffer 属于模块状态但通常不参与优化，普通属性不会自动进入 `state_dict` 或随 `.to()` 转换。
- 介绍 persistent buffer 的默认行为，不展开复杂 checkpoint 兼容策略。

### 模块 5：手写教学版线性层

- 实现 `ManualLinear(Din, Dout, bias=True)`，权重形状固定为 `[Din,Dout]`，与教学公式直接一致。
- 使用固定随机种子初始化小权重和 bias。
- 在 `forward()` 中加入清晰的最后一维 shape assertion。
- 验证 `[B,S,Din] -> [B,S,Dout]`，并解释 bias `[Dout]` 的广播。
- 与 `nn.Linear` 对照时显式处理 PyTorch 权重 `[Dout,Din]` 的转置方向。

### 模块 6：状态、模式、梯度与可复现性

- 保存和加载微型 `state_dict`，先篡改参数再恢复，验证输出恢复。
- 解释 `train()`/`eval()` 改变模块模式，但不会自动关闭 autograd。
- 比较普通 forward、`torch.no_grad()` 和 `torch.inference_mode()` 下输出的 `requires_grad`。
- 用固定随机种子说明初始化可复现的条件和限制。
- 演示模块与输入一起迁移 dtype/device；不支持的低精度组合应明确跳过或记录错误。

## 综合任务

综合任务构建一个微型 `ManualLinear`，输入为 `hidden [B,S,Din]`，输出为 `projected [B,S,Dout]`。学习者必须先书面完成：

- 输入、权重、bias 和输出的 shape 及轴语义。
- matmul 收缩维与 bias 广播过程。
- parameter、buffer、普通属性和 `state_dict` key 清单预测。
- 参数元素数与 FP32 理论字节数。
- 与 `nn.Linear` 对齐时的权重转置关系。
- 保存、修改、恢复状态前后的预期输出关系。

验证代码使用固定小数值和断言，在 CPU 上独立运行，不依赖文件下载。临时 checkpoint 使用标准库临时目录，避免在仓库留下生成文件。

## 错误与调试教学

教程包含可控失败实验：

- 广播 shape 不兼容。
- 矩阵乘法收缩维不匹配。
- `torch.bmm` 输入 rank 错误。
- 手写线性层最后一维不符合 `Din`。
- 模块与输入 dtype 或 device 不一致。
- 加载 shape 不匹配的 `state_dict`。

每个错误先打印输入 shape、dtype 或 state key，再阅读真实异常。修复必须针对根因，不通过任意 reshape、强制转换或 `strict=False` 掩盖结构问题。

## 验收标准

最终验收沿用第一周形式：

- 10 道概念题，至少答对 8/10。
- 5 道形状与状态推导题，至少答对 4/5。
- 1 个综合任务，能够脱离答案说明 `[B,S,Din] @ [Din,Dout] + [Dout] -> [B,S,Dout]`，并解释模块注册、状态保存和参考层对齐。
- CUDA 与低精度可选实验不计入通过条件。

## 写作与技术约束

- 面向具备 Python 基础、刚完成第一周教程的学习者。
- 使用 `B/S/Din/Dout`，并与路线中的 `B/S/D` 语义保持一致。
- 每个 PyTorch API 第一次出现时解释用途、关键参数和返回结果。
- 必修示例使用确定性小张量；涉及随机初始化时先固定种子。
- 每段代码说明预期输出或明确观察项。
- 浮点对照使用明确的 `atol` 和 `rtol`，不要求逐位相等。
- 不提前教授训练循环、优化器、反向传播细节、Embedding、softmax、Attention 或 MoE。
- 不声称 `eval()` 等于关闭梯度，也不声称 `no_grad()` 等于切换评估模式。
- 不声称 `einsum` 天生更快，或 `inference_mode()` 在所有代码中都可替代 `no_grad()`。

## 文档验证

提交前检查：

- Markdown 围栏完整，目录和锚点可用。
- README、路线与教程的相对链接存在。
- 所有练习编号均有题目、提示和答案。
- 不存在 `TODO`、`TBD` 或未填写占位符。
- 必修代码可在锁定的 CPU PyTorch 环境运行，无模型下载或仓库生成物。
- 综合代码、手写线性层和参考层对照通过明确断言。
- 概念题、形状题和综合任务覆盖路线中的第二周验收标准。
