# 第七周完整 Dense Decoder 教程设计规格

## 目标

为完成前六周教程的初学者编写一份第七周自包含教程。教程继续采用 Predict-Run-Explain 学习闭环，把第五周的 causal GQA 路径与第六周的 SwiGLU MLP 按 pre-norm 和 residual connection 的正确顺序组合成 Dense Decoder Layer，再堆叠为微型 Dense Causal LM，完成从 token IDs `[B,S]` 到 logits `[B,S,V]` 的首次端到端 forward。

教程完成后，学习者应能：

- 区分 pre-norm、子层更新和 identity residual 三条语义不同的数据路径。
- 写出 attention block 与 MLP block 的正确操作顺序，并解释两次 residual 使用的基准张量不同。
- 将 RMSNorm、QK Norm、RoPE、causal GQA 和 SwiGLU 组合为一个 Dense Decoder Layer。
- 从 `[B,S,D]` 推导单层及多层 Decoder stack 的 shape，并在每个边界检查 dtype 和 device。
- 解释 causal mask 为什么在每一层 attention 中都必须生效，而不能只在模型入口检查一次。
- 组合 token embedding、Decoder stack、final RMSNorm 和 LM Head，得到 `[B,S,V]` logits。
- 比较 1 层与多层模型的逐层 hidden states，而不把 hidden norm 变化误说成必然单调增大。
- 关闭或开启 attention/MLP 更新项，观察各 residual 边界的 hidden-state 范数。
- 记录有稳定名称的中间张量，并从最终 logits 偏差回溯到首个不一致边界。
- 为第八周把 Dense MLP 替换为稀疏 MoE 子层做准备。

## 教程形式

- 新增 `docs/tutorials/week07-dense-decoder.md`。
- 更新 `README.md` 和 `docs/roadmap.md`，提供第七周稳定入口。
- 示例、练习、综合任务和验证代码全部放在 Markdown 内联代码块中。
- CPU 是完整必修路径；使用固定随机种子和确定性 FP32 小张量，不下载模型、Tokenizer、checkpoint 或数据集。
- 不新增正式 `src/` 模块、Notebook、独立练习脚本或测试文件。
- 保持六个模块、综合任务、`C1-C10`、`S1-S5`、提示、答案、速查表和下一周预告的统一结构。
- 每个 Python 围栏必须自行 import、定义所需函数并构造数据，可以独立执行，不能隐式依赖前一个围栏的进程状态。
- 环境命令沿用仓库既有写法，但本次文档工作不调整 uv、Python、PyTorch 或锁文件。

## 范围边界

本周只覆盖无训练、无缓存的 Dense Decoder forward。明确不加入：

- MoE router、Top-K、专家分发、容量限制、负载均衡或辅助损失；
- KV Cache、增量解码、采样、生成循环或 beam search；
- padding mask、变长 batch、packed sequence 或 sliding-window attention；
- dropout、训练、反向传播、损失函数、优化器或梯度检查点；
- FlashAttention、SDPA 快速路径、tensor parallel、fused kernel 或性能 benchmark；
- Tokenizer、真实 checkpoint、权重转换或 Qwen3-30B-A3B 的未核实具体配置数字。

Embedding 与 LM Head 权重绑定只放在可选扩展中，主线使用两组独立参数，以便先隔离 Decoder 组合和逐层调试。接入真实模型时，层数、hidden size、head 数、intermediate size、RoPE 配置、norm epsilon、bias 配置和权重命名必须来自目标 `config.json` 与参考实现，不能把教学值写死为官方配置。

## 正确数据流

单层采用 Qwen3-style pre-norm 顺序：

```text
layer input x                 [B,S,D]
attention norm               [B,S,D]
causal GQA update             [B,S,D]
after attention residual      h = x + attention_update
MLP norm                      [B,S,D]
SwiGLU update                 [B,S,D]
layer output                  y = h + mlp_update
```

必须保留两个不同 residual 基准：attention 更新加回原始 `x`；MLP 更新加回 attention residual 后的 `h`。不能写成把两个更新都直接加到原始 `x`，也不能把 norm 放在 residual 相加之后冒充 pre-norm。

完整模型数据流：

```text
token_ids [B,S]
-> embedding [B,S,D]
-> decoder layer 0 [B,S,D]
-> ...
-> decoder layer L-1 [B,S,D]
-> final RMSNorm [B,S,D]
-> LM Head [B,S,V]
```

Attention 内部继续使用第五周确定的顺序：input RMSNorm 已由 block 外部完成，随后 Q/K/V projection、QK Norm、RoPE、causal GQA、head merge 和 output projection。MLP 内部继续使用第六周确定的 `SiLU(gate) * up -> down`。本周不重新推导这些组件的全部数学细节，而是聚焦接口契约、组合顺序和偏差定位。

## 六模块结构

### 模块 1：Pre-norm 与 residual contract

- 用最小可手算子层说明 `x + F(norm(x))`，并对比 post-norm `norm(x + F(x))` 的操作顺序。
- 建立 residual 相加契约：两侧 shape、dtype 和 device 必须完全一致，不依赖广播或隐式类型提升。
- 解释 identity skip 即使子层更新关闭仍然存在；实验开关只控制更新项 `alpha * F(...)`。
- 验证 `alpha=0` 时 block 输出严格等于 residual 基准，`alpha=1` 时执行完整更新。
- 观察更新前后 hidden norm，但明确 residual 相加既可能增大也可能减小范数。

### 模块 2：Attention residual block

- 复用第五周 causal GQA 语义，封装 `input RMSNorm -> attention -> residual add`。
- 明确 Q/K 使用 QK Norm 与 RoPE，V 不使用；mask 在 score softmax 前应用。
- 从 `[B,S,D]` 追踪 Q `[B,Hq,S,Dh]`、K/V `[B,Hkv,S,Dh]`、scores `[B,Hq,S,S]` 和更新 `[B,S,D]`。
- 在 norm 输出、Q/K/V、scores、probabilities、attention update 和 residual 输出处检查 shape、dtype、device 与有限性。
- 验证未来位置概率为 0，关闭 attention 更新时 block 输出等于输入。

### 模块 3：SwiGLU residual block

- 复用第六周无 bias SwiGLU，封装 `post-attention RMSNorm -> SwiGLU -> residual add`。
- 追踪 gate/up/product `[B,S,I]`、down update `[B,S,D]` 和 residual 输出 `[B,S,D]`。
- 明确 MLP residual 基准是 attention block 输出，不是原始 layer input。
- 要求 gate/up shape 完全相等，down update 与 residual shape、dtype、device 完全一致。
- 验证关闭 MLP 更新时输出等于 attention block 输出，并比较开启前后的 hidden norm。

### 模块 4：完整 Dense Decoder Layer

- 将两个 block 按 attention 在前、MLP 在后的顺序组合为单层模块。
- 使用独立的 input RMSNorm 和 post-attention RMSNorm 参数，不能共享或复用同一个模块实例。
- 提供 attention/MLP 更新开关，只用于受控实验，不改变主线架构定义。
- 比较四种组合：全部开启、仅 attention、仅 MLP、全部关闭；全部关闭时层输出严格等于层输入。
- 构造显式参考路径与模块路径，对齐 `attention_norm`、`attention_update`、`after_attention`、`mlp_norm`、`mlp_update` 和 `layer_output`。

### 模块 5：Decoder stack 与逐层 trace

- 使用 `nn.ModuleList` 堆叠多个参数互不共享的 Decoder Layer。
- 解释第 `l+1` 层输入就是第 `l` 层输出，所有层保持 `[B,S,D]`，但 values 通常不同。
- 比较相同 embedding 输入经过 1 层与多层后的输出，并记录每层输入、两次 residual 后状态和层输出。
- 定义稳定的层级 trace 名称，例如 `layers.1.mlp_update`，按执行顺序比较两次 forward。
- 实现“首个偏差”函数：先检查 key 顺序和 shape/dtype/device，再用明确 `atol/rtol` 找到第一个数值不一致张量。
- 故意扰动第 2 层 down projection；第 2 层之前的 trace 必须一致，首次偏差应在 `layers.1.mlp_update`，后续差异只是传播结果。

### 模块 6：微型 Dense Causal LM

- 验证 token IDs 为 `[B,S]` 整数 Tensor，值域位于 `[0,V)`，再执行 embedding lookup。
- 组合 embedding、Decoder stack、final RMSNorm 和无 bias LM Head。
- 建立端到端 shape ledger：`[B,S] -> [B,S,D] -> L x [B,S,D] -> [B,S,D] -> [B,S,V]`。
- 检查所有模块参数和关键中间张量位于同一 device，并解释浮点 hidden dtype 与整数 token dtype 的区别。
- 对同一前缀构造两条只在未来 token 不同的序列，验证 causal 模型在前缀位置的 logits 不变。
- 可选扩展说明 embedding/LM Head 权重绑定的参数关系与配置方式，但不纳入主线实现和必修断言。

## 综合任务

综合任务固定使用：

```text
B=2, S=4, V=11
D=8, I=12
Hq=2, Hkv=1, Dh=4
L=2
dtype=float32, device=cpu
bias=False
```

使用固定 token IDs、固定参数初始化和 shared causal mask，完整执行：

```text
token IDs -> embedding
-> 2 x (input RMSNorm -> QKV -> QK Norm -> RoPE -> causal GQA
        -> output projection -> residual
        -> post-attention RMSNorm -> SwiGLU -> residual)
-> final RMSNorm -> LM Head -> logits
```

必须断言：

- token IDs 为 `[2,4]`，embedding 和每层输出为 `[2,4,8]`，logits 为 `[2,4,11]`；
- 每个关键边界的 shape、dtype 和 device 符合契约，所有浮点中间张量有限；
- 每层 attention probabilities 为 `[2,2,4,4]`，未来位置概率严格为 0，每行概率和约为 1；
- GQA head 映射、QK Norm 轴、RoPE 作用对象和 SwiGLU gate/up/down 布局延续前两周定义；
- 单层显式参考路径与 `nn.Module` 路径的六个关键边界数值对齐；
- 关闭单层两个更新项时输出严格等于输入；分别开启 attention 或 MLP 时只引入对应更新；
- 记录四种开关组合的 hidden norm，但不要求任何固定单调关系；
- 相同 embedding 输入经过 1 层与 2 层后 shape 相同，第二层 trace 明确显示 values 被继续更新；
- 两条仅在最后一个 token 不同的序列，其前 3 个位置 logits 在明确容差内一致；
- 正常模型与复制模型的完整 trace 和 logits 一致；扰动复制模型第 2 层 down projection 后，首个差异定位到 `layers.1.mlp_update`；
- 非整数 token IDs、越界 token ID、错误 hidden width、非法 head 配置、错误 mask shape、residual 两侧不匹配和非有限中间值被明确拒绝。

综合任务不执行采样或 loss 计算；logits 是本周端到端 forward 的终点。

## Trace 与首次偏差定位

Trace 只用于教学调试，不作为高性能推理接口。记录顺序至少包含：

```text
embedding
layers.0.attention_norm
layers.0.attention_update
layers.0.after_attention
layers.0.mlp_norm
layers.0.mlp_update
layers.0.output
...
final_norm
logits
```

需要深入 attention 时，可在当前层补充 Q/K/V、scores 和 probabilities；默认层级 trace 先定位到子层边界，再进入该子层细查。比较器先拒绝 key 集合或顺序不同，再逐项检查 shape、dtype、device 和数值。报告必须包含首个不同 key、最大绝对差和两侧 shape，避免只输出最终 logits 的总误差。

## 错误处理

教程在危险运算前明确拒绝：

- `B/S/V/D/I/L` 非正，或 `D != Hq*Dh`、`Hq % Hkv != 0`、`Dh` 为奇数；
- token IDs 不是 rank-2 整数 Tensor，或存在负数和 `>= V` 的值；
- hidden、norm weight、投影权重、mask 或 residual 两侧 shape 不符合预期；
- residual 两侧 dtype/device 不完全一致，或浮点模块错误接收整数 hidden states；
- causal mask 不能广播到 scores，或产生整行完全屏蔽；
- gate/up shape 不完全相等，down projection 不能映回 `D`；
- Decoder stack 实际层数与配置 `L` 不一致，或意外共享层实例/参数；
- trace key、顺序或元数据不一致；
- 任一关键浮点中间张量出现 NaN 或 Inf。

错误信息应包含当前层号、边界名称、实际值与期望值，使学习者能先定位层，再定位子层和具体运算。

## 练习与验收

每个模块提供两个稳定编号练习 `M1-E1` 至 `M6-E2`，并给出独立提示与有理由的参考答案。最终验收包含：

- 10 道概念题，至少答对 8/10；
- 5 道公式、shape 或调试推导题，至少答对 4/5；
- 综合任务全部断言通过；
- 能脱离代码画出单层两次 pre-norm、两次子层更新和两次 residual 的完整数据流；
- 能从 `[B,S]` 写到 `[B,S,V]` 的端到端 shape ledger；
- 能解释为什么关闭两个更新项后单层是 identity，以及为什么开启更新后范数不保证增大；
- 能根据 trace 判断最终 logits 偏差来自 embedding、某层 attention、某层 MLP、final norm 还是 LM Head；
- 能说明下一周将 Dense Decoder 中哪个边界替换为 MoE，以及 attention 路径为什么保持不变。

## 验证策略

完成后执行：

1. 独立执行教程中的所有 Python 代码围栏。
2. 检查 Markdown 围栏闭合、目录链接和显式锚点。
3. 检查 12 道模块练习、10 道概念题和 5 道推导题均有唯一 question/hint/answer 锚点。
4. 检查 README、路线图和教程的相对链接存在。
5. 扫描 `TODO`、`TBD`、待补充文本和未完成占位符。
6. 检查每个 residual、层边界、final norm 和 logits 都有 shape/dtype/device 契约。
7. 检查 trace 扰动实验确实把首次偏差定位到预期 key，而不是仅比较最终 logits。
8. 在可用环境中运行仓库测试；环境不可用时单独记录，不修改环境配置。
9. 运行 `git diff --check` 并审查最终差异。

## Week 8 衔接

下一周进入 MoE Router 与 Top-K。第七周结束时，学习者已经拥有可端到端运行、可逐层追踪的 Dense Causal LM。第八周将保持 attention、residual、Decoder stack、final norm 和 LM Head 接口不变，先把 token hidden states 展平为 `[N,D]`，学习 router logits、Top-K expert indices/weights 与归一化，再为后续用 sparse MoE 替换当前 Dense SwiGLU 子层做准备。
