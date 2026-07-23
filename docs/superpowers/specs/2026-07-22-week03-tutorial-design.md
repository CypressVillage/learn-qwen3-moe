# 第三周 Token、Logits 与因果语言建模教程设计规格

## 目标

为完成前两周教程的初学者编写一份第三周自包含教程。教程继续采用 Predict-Run-Explain 学习闭环，以一条可手算的数据流连接手工微型词表、Embedding、LM Head、稳定 softmax、temperature、greedy next-token 选择与因果语言建模。

教程完成后，学习者应能：

- 解释文本、Tokenizer 边界、整数 token ID 与神经网络 `forward` 的职责分界。
- 从 `token_ids [B,S]` 推导 Embedding 输出 `hidden [B,S,D]`，并说明查表不是普通矩阵乘法。
- 从 `hidden [B,S,D]` 推导 LM Head 输出 `logits [B,S,V]`，识别词表轴。
- 解释 logits、概率与 token 选择的区别，不把 logits 当成概率。
- 沿词表维实现减最大值的稳定 softmax，并验证概率和约为 1。
- 解释 temperature 对分布尖锐程度的影响，并正确处理非正 temperature。
- 从最后一个序列位置选择 `last_logits [B,V]`，再用 argmax 得到 `next_token_ids [B,1]`。
- 构造因果语言建模的输入和目标错位，解释 teacher forcing 不改变 next-token 目标，也不等于允许当前位置读取未来信息。

## 教程形式

沿用前两周的单篇自包含长教程形式：

- 新增 `docs/tutorials/week03-token-logits-causal-lm.md`。
- 更新 `README.md` 和 `docs/roadmap.md`，提供稳定入口并标记第三周教程可用。
- 示例、练习和验证代码全部放在 Markdown 内联代码块中。
- 不新增正式 `src/` 模块、独立练习脚本、Notebook 或测试文件。
- CPU 是完整必修路径；本周计算规模很小，不设置必需 CUDA 实验。
- 不下载模型、Tokenizer、checkpoint 或数据集，不依赖 Transformers 等高层库。

教程维持统一结构：学习目标、学习方式、环境检查、六个教学模块、综合任务、最终验收、提示、参考答案、术语与速查表、下一周预告。每个模块包含用途说明、运行前预测、确定性实验、输出解释、常见误区、两个稳定编号练习和模块验收。

## 统一形状与微型配置

主线统一使用：

- `B`：batch size。
- `S`：当前输入序列长度。
- `V`：微型词表大小。
- `D`：隐藏表示宽度。

综合任务使用足够小且可手算的固定配置，例如 `B=2`、`S=3`、`V=6`、`D=3`。所有 token ID 必须位于 `[0,V)`。教程应在每个边界重复标明：

```text
token_ids      [B,S]
hidden         [B,S,D]
lm_head_weight [V,D]
logits         [B,S,V]
last_logits    [B,V]
next_token_ids [B,1]
```

LM Head 使用 PyTorch `nn.Linear(D,V,bias=False)`，其权重为 `[V,D]`，计算等价于 `hidden @ weight.T`。这会复用第二周关于 `nn.Linear` 权重方向的知识，不引入教学方向的新权重约定。

## 模块安排

### 模块 1：手工微型词表与 Tokenizer 边界

- 使用固定 Python 字典建立微型 `token_to_id` 和 `id_to_token`。
- 用简单的空格切分函数演示文本到 token 字符串、token ID，以及 token ID 到文本片段的边界。
- 明确这只是教学映射，不代表真实 Qwen tokenizer 的分词算法、特殊 token 或 chat template。
- 明确神经网络 `forward` 接收整数 `token_ids`，不负责原始文本切分或字符串解码。
- 包含未知 token 的受控失败，错误应报告具体 token，而不是静默映射到任意 ID。

### 模块 2：Embedding 查表

- 介绍 `nn.Embedding(V,D)` 的权重形状 `[V,D]` 和输入 dtype 要求。
- 使用固定权重展示每个 token ID 如何选择对应行。
- 推导 `[B,S] -> [B,S,D]`，解释输入轴保留、末尾新增表示宽度 `D`。
- 用直接索引 `embedding.weight[token_ids]` 与模块输出做数值对照。
- 包含越界 token ID 和错误浮点 dtype 的受控失败；修复必须针对词表范围或 ID dtype，而不是任意 clamp 或转换掩盖上游错误。

### 模块 3：LM Head 与 logits

- 使用 `nn.Linear(D,V,bias=False)` 把每个隐藏向量投影到词表大小。
- 推导 `[B,S,D] @ [D,V] -> [B,S,V]`，同时说明实际 `nn.Linear.weight` 保存为 `[V,D]`。
- 通过固定权重让部分 logits 可手算，并与显式 `hidden @ lm_head.weight.T` 对照。
- 解释每个 `logits[b,s,v]` 是位置 `(b,s)` 对词表条目 `v` 的未归一化分数。
- 强调 logits 可以为负、不要求落在 `[0,1]`，也不要求沿词表维求和为 1。

### 模块 4：稳定 softmax 与概率

- 从定义 `exp(logit) / sum(exp(logit))` 出发，但实现时先沿目标维减去最大值。
- 提供教学函数 `stable_softmax(logits, dim=-1)`，使用 `max(..., keepdim=True)` 保持广播语义清晰。
- 解释减去同一个最大值不改变概率比例，却能避免不必要的指数溢出。
- 用极大 logits 比较直接 `exp` 路径与稳定路径，观察非有限值或无效概率。
- 验证输出 shape 不变、值非负、沿词表维求和在明确容差内接近 1，并与 `torch.softmax` 对照。
- 说明 softmax 必须沿词表轴 `V` 执行；沿 batch 或序列轴即使代码可运行，语义也错误。

### 模块 5：temperature、argmax 与 greedy next-token

- 定义 temperature 调整为 `softmax(logits / temperature, dim=-1)`。
- 比较小于 1、等于 1 和大于 1 的正 temperature，观察最大概率和分布熵的变化。
- 实现只接受 `temperature > 0` 的教学函数，非正值抛出包含实际值的 `ValueError`。
- 使用 `last_logits = logits[:, -1, :]` 得到 `[B,V]`，再用 `argmax(dim=-1, keepdim=True)` 得到 `[B,1]`。
- 解释 argmax 选择不需要先算 softmax，因为正 temperature 下 softmax 保持排序；概率仍用于解释分布和后续采样。
- 本周只实现 greedy 选择，不实现随机采样、top-k、top-p、EOS 或完整生成循环。

### 模块 6：因果错位与 teacher forcing

- 使用固定 token 序列构造 `inputs = sequence[:, :-1]` 和 `targets = sequence[:, 1:]`。
- 对每个位置解释输入前缀与要预测的下一个 token，建立自回归因果关系。
- 说明 teacher forcing 在一次训练式示例中提供整段已知输入，但位置 `s` 的目标仍是原序列位置 `s+1`；具备上下文混合能力的因果模型还必须阻止当前位置读取未来信息。
- 本周不实现 attention 或 causal mask，只建立数据错位契约。Embedding 与 LM Head 示例本身不混合不同位置，因此不能冒充完整的上下文语言模型；第四周再由 causal attention 引入受约束的上下文聚合。
- 使用 `logits [B,S,V]` 与 `targets [B,S]` 说明词表预测轴与目标 ID 的对应关系，不展开训练循环、优化器或反向传播。

## 综合任务

综合任务构建一个确定性的微型 token-to-logits 与 next-token 选择数据流。教学模块由 `nn.Embedding(V,D)` 和无 bias 的 `nn.Linear(D,V)` 组成，权重使用固定小值赋予，不依赖随机初始化结果。它用于验证接口、形状、概率和 token 选择，不宣称已经具备跨位置上下文建模能力。

学习者必须先书面完成：

- 手工文本映射得到的 token 字符串和 token ID。
- `token_ids`、Embedding 权重、`hidden`、LM Head 权重、`logits`、`last_logits` 和 `next_token_ids` 的 shape、dtype、轴语义与元素数。
- 指定位置的 Embedding 查表 values。
- LM Head 的收缩维、权重转置方向和指定 logits values。
- 稳定 softmax 的最大值平移、概率和与 greedy token。
- 因果输入和目标的逐位置错位关系。

验证代码应：

- 在 CPU 上独立运行并使用固定输入、固定权重和明确断言。
- 对 Embedding 模块输出与直接权重索引做严格对照。
- 对 LM Head 输出与显式矩阵乘法做容差对照。
- 对教学 softmax 与 `torch.softmax` 做明确 `atol`/`rtol` 对照。
- 断言概率沿 `V` 维求和约为 1，且输出均为有限值。
- 断言 greedy 输出 shape 为 `[B,1]`，并能通过 `id_to_token` 映射回教学 token。
- 断言因果 inputs 与 targets 分别等于原序列去掉最后和第一个位置后的结果。

## 错误与调试教学

教程包含以下可控失败或语义错误观察：

- 手工词表遇到未知 token。
- Embedding 输入 token ID 越界。
- Embedding 输入使用浮点 dtype。
- LM Head 输入最后一维不等于 `D`。
- 直接对极大 logits 做 `exp` 导致非有限中间结果。
- softmax 沿错误轴计算，shape 合法但概率语义错误。
- temperature 为 0 或负数。
- 使用 `logits[:, -1]` 后忘记 `keepdim=True`，导致 next-token 输出为 `[B]` 而不是统一约定的 `[B,1]`。

每个失败先打印相关 token、shape、dtype、数值范围或维度，再阅读异常或断言结果。教程不通过静默替换未知 token、clamp 越界 ID、任意 reshape、随意转 dtype 或忽略非有限值来掩盖根因。

## 验收标准

最终验收沿用前两周形式：

- 10 道概念题，至少答对 8/10。
- 5 道形状、概率与因果错位推导题，至少答对 4/5。
- 1 个综合任务，所有断言通过，并能脱离代码口述完整数据流：

```text
文本边界 -> token_ids [B,S] -> hidden [B,S,D]
         -> logits [B,S,V] -> last_logits [B,V]
         -> next_token_ids [B,1]
```

- 能解释 softmax 为什么沿 `V` 维、为什么先减最大值、temperature 如何改变分布，以及 teacher forcing 的输入/目标为何错位。
- 可选扩展不计入通过条件。

## 可选扩展

- 实现最小 top-k sampling，仅作为完成主线后的练习，不放入综合任务或通过标准。
- 比较不同 temperature 下分布熵，但熵计算必须使用稳定且避免 `0 * log(0)` 的实现。
- 观察 Embedding 与 LM Head 权重绑定的概念，但不在主线实现；正式模型组合阶段再处理配置与权重映射。

## 写作与技术约束

- 面向具备 Python 基础并完成前两周教程的学习者。
- 使用 `B/S/V/D`，并与路线图统一形状记号保持一致。
- 每个 PyTorch API 第一次出现时解释用途、关键参数和返回结果。
- 必修示例使用确定性小张量；权重使用固定值或固定随机种子后再显式覆盖。
- 每段代码说明预期输出或明确观察项。
- 浮点对照使用明确的 `atol` 和 `rtol`；概率和检查也使用明确容差。
- 不把教学空格切分器描述成真实 Tokenizer，不引入 chat template 或特殊 token 细节。
- 不提前教授 Attention、causal mask 实现、训练循环、交叉熵细节、完整生成循环、随机采样或真实模型权重。
- 不声称 temperature 为 0 是普通 softmax 极限的可直接数值实现。
- 不声称先 softmax 再 argmax 是 greedy 选择所必需的。

## 文档验证

提交前检查：

- Markdown 围栏完整，目录和显式锚点可用。
- README、路线图与教程的相对链接存在。
- 六个模块、综合任务、最终验收、提示、答案、术语表和第四周预告齐全。
- 每个模块有两个练习，所有模块练习及最终验收题均有稳定编号、提示和答案。
- 不存在 `TODO`、`TBD`、未填写表格或说明性输出占位符。
- 必修代码可在锁定的 CPU PyTorch 环境运行，无下载或仓库生成物。
- 受控错误均被捕获或以安全断言观察，教程验证脚本仍以退出码 0 完成。
- 综合代码的 shape、dtype、数值、有限性、概率和、greedy 结果与因果错位断言全部通过。
- `uv run pytest` 继续通过现有仓库测试。
- 概念题、推导题和综合任务覆盖路线图中的第三周验收标准。

## 第四周衔接

下一周预告只说明：第四周将解释模型如何通过 causal self-attention 在每个位置聚合允许看到的上下文，并扩展到多头注意力与 GQA。第三周应明确留下接口：Embedding 产生 `hidden [B,S,D]`，后续 Decoder 保持该形状，最终 LM Head 再产生 `logits [B,S,V]`；不在本周提前实现注意力。
