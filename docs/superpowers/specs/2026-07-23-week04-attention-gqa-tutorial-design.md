# 第四周 Self-Attention 与 GQA 教程设计规格

## 当前状态与本次范围

当前草稿已经完成学习目标、学习方式、环境边界以及六个核心教学模块。本次续写保留这些内容和既有教学顺序，不重写已完成模块；工作范围仅为补齐综合任务、最终验收、提示、参考答案、术语速查表和下一周预告，并补充 README 与路线图中的第四周入口，最后执行本规格定义的完整验证。

## 目标

为完成前三周教程的初学者编写一份第四周自包含教程。教程继续采用 Predict-Run-Explain 学习闭环，从可手算的单头 scaled dot-product attention 出发，逐步连接 causal mask、多头拆分与合并、完整 Q/K/V 投影、Multi-Head Attention（MHA）、物理复制 GQA 和逻辑分组 GQA。

教程完成后，学习者应能：

- 解释 Q、K、V 的职责，并从矩阵乘法推导 attention scores 的最后两维 `[S,T]`。
- 正确使用 `sqrt(Dh)` 缩放点积，而不是误用模型宽度或序列长度。
- 在 softmax 前应用 causal mask，并验证未来位置的 attention 概率严格为 0。
- 在 `[B,S,D]` 与 `[B,H,S,Dh]` 之间正确拆头、转置和合头。
- 从 hidden 经 Q/K/V 投影、attention 和 output projection 得到 `[B,S,D]`。
- 解释 MHA 与 GQA 的 head 数关系，计算每个 KV head 服务的 query head 数。
- 实现并比较物理复制 K/V 与逻辑分组两种 GQA 计算。
- 拒绝 `D % Hq != 0`、`Hq % Hkv != 0`、head dimension 不一致和不可广播 mask 等无效配置。
- 说明本周没有 KV Cache，因此 `T=S`，同时理解保留 `S/T` 两个符号的意义。

## 教程形式

沿用前三周的单篇自包含长教程形式：

- 教程文件为 `docs/tutorials/week04-self-attention-gqa.md`；当前续写在既有草稿中补齐缺失章节。
- 更新 `README.md` 和 `docs/roadmap.md`，提供稳定入口并标记第四周教程可用。
- 示例、练习和验证代码全部放在 Markdown 内联代码块中。
- 不新增正式 `src/` 模块、独立练习脚本、Notebook 或测试文件。
- CPU 是完整必修路径；所有必修实验使用确定性小张量。
- 不下载模型、Tokenizer、checkpoint 或数据集，不依赖 Transformers 等高层库。
- 可选扩展可以与 `torch.nn.functional.scaled_dot_product_attention` 做数值对照，但不作为必修路径。

教程维持统一结构：学习目标、学习方式、环境检查、六个教学模块、综合任务、最终验收、提示、参考答案、术语与速查表、下一周预告。每个模块包含用途说明、运行前预测、确定性实验、输出解释、常见误区、两个稳定编号练习和模块验收。

## 范围边界

本周只覆盖 causal self-attention。明确不加入：

- padding mask；
- attention dropout；
- RoPE 或其他位置编码；
- KV Cache；
- cross-attention；
- 完整 Decoder Block；
- 训练循环或优化器；
- 真实 Qwen3 权重加载。

这些边界防止学习者同时处理过多概念。Week 5 再引入 RoPE，后续周次再引入 KV Cache、完整 Decoder 和真实模型接口。

## 统一符号与形状

主线统一使用：

- `B`：batch size。
- `S`：query 序列长度。
- `T`：key/value 序列长度。
- `D`：模型 hidden width。
- `Hq`：query head 数。
- `Hkv`：key/value head 数。
- `Dh`：每个 head 的宽度。
- `G=Hq/Hkv`：每个 KV head 服务的 query head 数。

关键数据流为：

```text
hidden          [B,S,D]
q projected     [B,S,Hq*Dh]
k/v projected   [B,S,Hkv*Dh]
q               [B,Hq,S,Dh]
k/v             [B,Hkv,T,Dh]
scores          [B,Hq,S,T]
probabilities   [B,Hq,S,T]
context         [B,Hq,S,Dh]
output          [B,S,D]
```

本周没有 KV Cache，因此所有 self-attention 必修示例中 `T=S`。教程仍保留不同字母，因为 scores 的倒数第二维枚举 query 位置，最后一维枚举可被查询的 key 位置；这个区分是后续 KV Cache 的前置认知。

完整投影主线选择 `D=Hq*Dh`。GQA 中 K/V 投影宽度为 `Hkv*Dh`，不要求等于 `D`。output projection 接收合并后的 query-head context `[B,S,Hq*Dh]` 并输出 `[B,S,D]`。

## 六模块教学结构

### 模块 1：单头 scaled dot-product attention

使用手工给定的小 Q/K/V，不引入线性投影。学习者先预测：

- `Q [S,Dh] @ K.T [Dh,T] -> scores [S,T]`；
- 每个 score 表示一个 query 位置与一个 key 位置的相似度；
- 除以 `sqrt(Dh)` 后沿 `T` 维 softmax；
- `probabilities [S,T] @ V [T,Dh] -> context [S,Dh]`。

实验必须包含可手算 values，并显式比较未缩放与已缩放 scores。受控错误展示 Q/K 的 `Dh` 不一致时矩阵乘法失败。

### 模块 2：causal mask

从一个短序列构造上三角布尔 mask。必须展示：

- mask shape `[S,T]` 如何广播到 batch/head 维；
- mask 在 softmax 前用 `masked_fill(mask, -inf)` 应用；
- 每个 query 位置只允许关注当前位置及过去位置；
- softmax 后上三角未来位置概率严格为 0；
- 每行允许区域的概率和约为 1。

教程必须解释为什么不能在 softmax 后简单把未来概率改为 0 而不重新归一化。受控错误展示不可广播 mask shape。

### 模块 3：多头拆分与合并

从 `hidden [B,S,D]` 出发，要求 `D % H == 0` 并令 `Dh=D/H`。教学顺序为：

```text
[B,S,D]
-> view [B,S,H,Dh]
-> transpose [B,H,S,Dh]
-> transpose [B,S,H,Dh]
-> contiguous/reshape [B,S,D]
```

必须用递增整数 Tensor 验证拆头再合头精确恢复原 values，并解释 transpose 后 non-contiguous 与安全 reshape 的关系。受控错误拒绝不能均匀拆头的配置。

### 模块 4：完整 MHA 数据流

引入 bias-free Q/K/V 和 output projections，从 `hidden [B,S,D]` 走通完整 causal MHA：

1. 三个线性投影；
2. 拆成 `Hq=Hkv=H` 个 heads；
3. 计算 scaled scores；
4. 应用 causal mask；
5. softmax 和加权求和；
6. 合头；
7. output projection。

固定所有教学权重，验证最终 output shape 为 `[B,S,D]`、未来概率为 0、每行概率和约为 1。教程明确 MHA 的所有 query heads 与 KV heads 一一对应。

### 模块 5：GQA 物理复制 K/V

引入 `Hq >= Hkv` 和 `G=Hq/Hkv`。先验证 `Hq % Hkv == 0`，再使用 `repeat_interleave(G, dim=1)` 将：

```text
k/v [B,Hkv,T,Dh] -> repeated k/v [B,Hq,T,Dh]
```

之后复用 MHA 计算路径。教程必须通过明确的 head 编号展示连续的 `G` 个 query heads 共享同一个 KV head，并用 `numel` 对比复制前后的逻辑元素数。受控错误拒绝不能整除的 `Hq/Hkv`。

### 模块 6：GQA 逻辑分组

不创建 `[B,Hq,T,Dh]` 的 K/V 复制张量。将 query reshape 为：

```text
q [B,Hq,S,Dh] -> grouped q [B,Hkv,G,S,Dh]
```

K/V 保持 `[B,Hkv,T,Dh]`，通过分组广播或 `einsum` 计算：

```text
grouped scores  [B,Hkv,G,S,T]
grouped context [B,Hkv,G,S,Dh]
```

最后恢复为 `[B,Hq,S,Dh]`。必须在相同输入与权重下验证逻辑分组版本和物理复制版本的 scores、probabilities、context 与最终 output 数值一致，并解释逻辑分组为何避免 K/V 的物理重复。

## 综合任务

综合任务使用一个确定性的微型 causal GQA forward，建议固定：

- `B=1`；
- `S=T=3`；
- `D=4`；
- `Hq=2`；
- `Hkv=1`；
- `Dh=2`；
- `G=2`。

综合任务必须从 `hidden [B,S,D]` 开始，包含固定 Q/K/V/output projection 权重，并同时运行物理复制与逻辑分组路径。学习者在运行前预测关键 shapes、mask、至少一项 score、未来概率、输出 shape 和 K/V 复制倍率。

代码必须断言：

- 所有中间 shape 符合统一数据流；
- causal mask 后所有未来概率为 0；
- 每个 query head、每个位置的概率和约为 1；
- 所有 probabilities 和 outputs 为有限值；
- 物理复制和逻辑分组结果在合理 float32 容差内一致；
- 最终 output 为 `[B,S,D]`；
- 物理复制 K/V 的逻辑 `numel` 是原始 K/V 的 `G` 倍。

综合任务必须再次声明：它实现的是一个最小 attention 子层数据流，不包含 RoPE、残差、归一化、MLP、MoE 或完整 Decoder Block。

## 错误处理

所有受控错误必须在执行危险运算前给出直接、可定位的信息：

- `D % Hq != 0`：说明模型宽度无法均匀拆给 query heads。
- `Hq % Hkv != 0`：说明 query heads 无法均匀分组给 KV heads。
- Q/K 的 `Dh` 不一致：同时报告两个最后维度。
- K/V 的 `T` 不一致：说明 attention 权重无法与 V 对齐。
- mask 不可广播到 scores：同时报告 mask 与 scores shape。
- 全部位置被 mask：禁止对全 `-inf` 行直接 softmax，避免产生 NaN。

教程不应把底层 PyTorch 异常作为唯一教学反馈；主线 helper 或综合任务应先验证关键不变量。单独的受控错误实验可以展示底层异常，并解释它对应的 shape 规则。

## 练习与最终验收

每个模块提供两个稳定编号练习 `M1-E1` 至 `M6-E2`，题目、提示和答案分别使用显式 Markdown 锚点。练习覆盖：

- score 与 context shape 推导；
- `sqrt(Dh)` 缩放；
- causal mask values 和概率；
- 拆头、转置、合头；
- MHA 完整数据流；
- GQA 组大小和 head 映射；
- 物理复制与逻辑分组的 shape、数值和存储差异。

最终验收包含：

- 10 道概念题 `C1-C10`；
- 5 道 shape 或数值推导题 `S1-S5`；
- 每题对应的提示与有理由的参考答案；
- 建议评分规则与通过标准；
- 一张覆盖 Q/K/V、scores、mask、softmax、context、拆头/合头、MHA 和 GQA 的术语速查表。

## 验证策略

教程完成后必须执行：

1. 使用锁定的 Python 3.11 与 PyTorch 2.7.1 CPU 环境逐块执行所有 Python 代码围栏。
2. 检查所有代码块无未捕获异常，所有断言通过。
3. 检查 Markdown 围栏成对闭合。
4. 检查练习、概念题、推导题的 question/hint/answer 锚点完整且唯一。
5. 检查所有内部链接和 README/roadmap 入口有效。
6. 扫描 `TODO`、`TBD`、占位符和未完成文本。
7. 运行 `uv run pytest`，确保现有环境测试无回归。
8. 运行 `git diff --check` 并审查最终差异。

数值比较必须使用适合 float32 的显式容差。未来概率应直接验证为 0；概率和、两种 GQA 路径以及参考实现对照使用约 `1e-6` 级别容差，具体值按运算路径确定并在代码中写明。

## 非目标

本周不追求生产级 attention kernel、性能优化、FlashAttention、混合精度、GPU benchmark、动态 batch padding、KV Cache 或真实 Qwen3 参数兼容。物理复制与逻辑分组的比较用于建立 GQA 语义和内存直觉，不宣称 Python 教学实现代表高性能内核。

## Week 5 衔接

下一周引入 RoPE。Week 4 结束时，学习者必须能解释：

- attention 在没有显式位置旋转时如何只依据 Q/K 内容和 causal mask 工作；
- RoPE 将作用于 Q/K 而不是 V 的哪一部分接口；
- 为什么加入 RoPE 不改变外部 `[B,H,S,Dh]` shape；
- 为什么本周保留的 `S/T` 区分对后续 KV Cache 仍然有用。
