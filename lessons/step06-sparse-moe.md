# Step 06：让每个 token 只调用最合适的专家

<!-- checkpoint: step06-ready -->

上一章让 prompt 中的 token 通过 causal Attention 读取了已经出现的上下文。现在继续沿 decoder layer 向前，把每个位置更新后的表示送进 Qwen3 MoE 的稀疏前馈分支：

```text
post-attention normalized hidden_states [B,S,D]
  -> Router logits [B*S,E]
  -> softmax + top-k
  -> selected expert IDs [B*S,K]
  -> routing weights [B*S,K]
  -> selected SwiGLU experts
  -> weighted sum
  -> MoE output [B,S,D]
```

这一章实现 `moe.py` 中的 `Qwen3MoeRouter`、`Qwen3MoeExperts` 和 `Qwen3SparseMoeBlock`。它们共同回答三个问题：一个 token 应该交给哪些专家、每个专家怎样变换 token、多个专家的结果怎样合回一个向量。

先划清本章边界。输入已经经过 decoder layer 外部的 `post_attention_layernorm`，所以 MoE 内部不重复做 RMSNorm；输出也暂时不与 residual 相加，那条连接会在下一章组装完整 decoder layer 时出现。

本章新增三个维度：

```text
T = B * S                    展平后的 token 数
E = num_experts              专家总数
K = num_experts_per_tok      每个 token 选择的专家数
M = moe_intermediate_size    每个专家的中间宽度
```

对于 `Qwen3-30B-A3B`：

```text
D = 2048
E = 128
K = 8
M = 768
```

模型每层拥有 128 个可选专家，但每个 token 只运行其中 8 个。这就是“稀疏”的含义：参数池很大，单个 token 实际激活的计算路径却只占一小部分。

## 一层 Sparse MoE 需要哪些真实权重

以第 0 层为例，本章读取三块 tensor：

```text
model.layers.0.mlp.gate.weight
model.layers.0.mlp.experts.gate_up_proj
model.layers.0.mlp.experts.down_proj
```

它们的 shape 是：

```text
Router             [E,D]   = [128,2048]
Expert gate + up   [E,2M,D] = [128,1536,2048]
Expert down        [E,D,M]  = [128,2048,768]
```

第一维都是 expert ID。`gate.weight[37]` 是 Router 给专家 37 打分时使用的向量，`experts.gate_up_proj[37]` 与 `experts.down_proj[37]` 则组成专家 37 自己的前馈网络。

这里的 `gate` 容易产生歧义：

```text
mlp.gate.weight              -> Router，决定去哪些专家
experts.gate_up_proj         -> 专家内部的 SwiGLU gate/up 投影
```

二者都叫 gate，但职责完全不同。前者在专家之间做选择，后者在某一个专家内部控制中间特征。

## Router 先接住真实权重与配置

<!-- checkpoint: step06-router-weights -->

左侧 `Qwen3MoeRouter.__init__()` 检查 Router 权重必须是 `[E,D]`，并保存三个决定路由行为的 config 字段：

```text
num_experts         -> 一共有多少个候选专家
num_experts_per_tok -> 每个 token 保留几个专家
norm_topk_prob      -> 是否重新归一化入选专家的权重
```

Router 本质上是一个没有 bias 的 Linear：

```text
token [D] @ router_weight.T [D,E] -> logits [E]
```

但我们没有直接把它包装成 `Linear`，因为 [[Router]] 的 forward 不会停在投影结果；它紧接着还要执行 [[softmax]]、[[top-k]] 和可选的重新归一化。把这些动作放在一个小类里，更容易看清“分数怎样变成执行路径”。

与 Attention 一样，`moe.py` 不负责从 Safetensors 分片里查参数。仍然由外部加载 tensor，再交给对应模块：

```text
checkpoint.py 负责：参数名 -> NumPy tensor
moe.py        负责：Router 与 Experts 怎样使用这些 tensor
```

## 为什么进入 Router 前要把 batch 与 sequence 展平

MoE 对每个 token 独立做路由。一个位置选择了专家 12，不会直接限制相邻位置选择什么，因此 Router 不需要保留 batch 和 sequence 两个独立维度。

```text
hidden_states [B,S,D]
  -> reshape
tokens [T,D], T = B*S
```

例如 `B=2, S=3` 时，Router 看到 6 个 token rows：

```text
[batch 0, position 0]
[batch 0, position 1]
[batch 0, position 2]
[batch 1, position 0]
[batch 1, position 1]
[batch 1, position 2]
```

展平只改变索引方式，不混合 token 的 hidden dimensions。处理完成后按原来的 `B` 和 `S` reshape 回去即可。

这也指出 Attention 与 MoE 的核心差别：Attention 沿 sequence 维让 token 读取其他位置；MoE 沿 hidden 维独立变换每个 token，只是不同 token 可以选择不同参数路径。

## 从 128 个分数中选出 top-8

<!-- checkpoint: step06-routing -->

左侧 Router forward 接收 `[T,D]`，先得到：

```text
logits [T,E]
```

第 `t` 行包含 token `t` 对全部专家的原始偏好。接着沿最后一维做稳定 softmax：

```text
probabilities[t,e] = softmax(logits[t,:])[e]
```

代码先减去每行最大值，再调用 `exp()`，与 Step 05 的 Attention softmax 使用同样的数值稳定原则。每一行 128 个概率之和为 1，但此时还没有稀疏化，所有专家都有一个概率。

随后按概率从高到低排序，只保留前 `K=8` 个：

```text
selected_experts [T,K]
routing_weights  [T,K]
```

假设一个小模型有 5 个专家，某个 token 的概率是：

```text
expert ID       0     1     2     3     4
probability   0.05  0.40  0.10  0.30  0.15
```

当 `K=2` 时，结果是：

```text
selected_experts = [1, 3]
routing_weights  = [0.40, 0.30]
```

Qwen3-30B-A3B 的 `norm_topk_prob` 为 true，所以还会在入选的两个概率之间重新归一化：

```text
[0.40, 0.30] / 0.70 -> [0.5714, 0.4286]
```

重新归一化后，每个 token 的 `K` 个 routing weights 之和重新变成 1。没有入选的专家不会参与本次前向，也不会出现在返回数组中。

需要区分两件事：softmax 决定相对概率，top-k 决定真正执行哪些分支。只计算 logits 和概率还不算稀疏；跳过未入选专家，才真正减少了单个 token 的专家计算量。

## 专家权重为什么把 128 个 MLP 堆在一起

<!-- checkpoint: step06-expert-weights -->

每个 Qwen3 MoE expert 都是一个 SwiGLU 前馈网络：

```text
x [D]
  -> gate projection [M]
  -> up projection   [M]
  -> SiLU(gate) * up [M]
  -> down projection [D]
```

如果分别保存 128 个 Python 对象，会产生大量重复结构。真实 checkpoint 把同类权重沿 expert 维堆成两个三维 tensor：

```text
gate_up_proj [E,2M,D]
down_proj    [E,D,M]
```

`gate_up_proj` 又把同一专家的 gate 和 up 两次投影融合在一起。一次矩阵乘法先得到 `[2M]`，再从中间切开：

```text
x [N,D] @ gate_up_proj[e].T [D,2M]
  -> gate_up [N,2M]
  -> split
gate [N,M], up [N,M]
```

这里 `N` 表示当前被分配给专家 `e` 的 token 数，不是整批的 `T`。不同专家收到的 `N` 可以不同，也可能为 0。

左侧 `Qwen3MoeExperts.__init__()` 用 config 检查两个三维 shape。尤其要注意 `down_proj` 的最后两维是 `[D,M]`，因为 Linear 权重仍然遵循 `[output,input]`；执行时转置成 `[M,D]`，把中间宽度投影回 hidden size。

## SwiGLU 怎样完成一个专家内部的变换

专家拿到自己负责的 token 后，核心公式是：

```text
gate = x @ W_gate.T
up   = x @ W_up.T

hidden = SiLU(gate) * up
output = hidden @ W_down.T
```

其中：

```text
SiLU(z) = z * sigmoid(z)
```

这不是普通的“两层 Linear 中间加激活”。`up` 分支提供候选特征，经过 SiLU 的 `gate` 分支逐元素调节这些特征，再由 down projection 混合回 `D` 维。

课程实现使用稳定的 sigmoid 写法：先计算 `exp(-abs(gate))`，再根据正负区间组合结果。这样在 gate 绝对值很大时，不会直接计算容易溢出的 `exp(-gate)`。

虽然权重是融合保存的，数学上仍然可以把前 `M` 行理解成 `W_gate`，后 `M` 行理解成 `W_up`。融合只减少一次独立投影调用，没有改变 SwiGLU 公式。

## 为什么按 expert 聚合 token，而不是按 token 循环

<!-- checkpoint: step06-experts -->

最直观但低效的写法是：遍历每个 token，再逐个运行它选中的 8 个专家。这样会产生大量很小的矩阵乘法。

左侧实现反过来遍历 expert ID：

```text
for expert e:
  找出所有选择了 e 的 token
  把这些 token 收集成 [N,D]
  一次运行 expert e
  乘各自 routing weight
  累加回对应 token
```

`np.where(selected_experts == expert_index)` 同时返回：

```text
token_indices    -> 哪些 token 选择了这个专家
top_k_positions  -> 该专家位于各 token 的第几个入选槽位
```

第二个索引用于找到正确的 routing weight：

```text
routing_weights[token_indices, top_k_positions]
```

一个 token 会出现在多个专家批次中，因为它选择了 `K` 个专家。每个专家输出先乘自己的权重，再累加到同一个 output row：

```text
moe_output[t] =
    weight[t,0] * expert[selected[t,0]](x[t])
  + weight[t,1] * expert[selected[t,1]](x[t])
  + ...
  + weight[t,K-1] * expert[selected[t,K-1]](x[t])
```

没有任何 token 被拼接成更宽的最终向量。无论选择多少专家，加权合并后的结果始终是 `[D]`，因此整个输出保持 `[T,D]`。

:::principle top-k 路由最终计算的混合公式

设 Router logits 为 `z_e`，对全部专家做 softmax 后得到 `p_e`。`TopK(p)` 选出的专家集合记为 `I`。当配置要求重新归一化时，入选专家的最终权重是：

```text
alpha_e = p_e / sum(j in I, p_j)
        = exp(z_e) / sum(j in I, exp(z_j))    e in I
```

第二个等号说明：重新归一化后，未入选专家已经从分母中消失。当前 token 的 MoE 输出就是：

```text
y(x) = sum(e in I, alpha_e * expert_e(x))
```

每个 `expert_e(x)` 都是 `[D]`，加权求和后仍是 `[D]`，所以 top-k 表示选择并混合多条计算路径，而不是把多个专家输出拼接起来。

若关闭重新归一化，入选权重之和可能小于 1。此时丢弃的专家概率质量也会影响输出整体幅度，而不仅仅是改变专家之间的相对比例。

:::endprinciple

代码还会直接跳过本批次没有收到 token 的专家。对于真实的 128 个专家，这条分支正是稀疏执行的关键：未命中的专家权重不会参加矩阵乘法。

本章没有实现训练时的负载均衡损失，也没有人为限制每个专家最多接收多少 token。推理只需要复现已经训练好的 Router 决策；训练如何避免少数专家过载，不属于当前 forward 主线。

## 组装 Router、Experts 与 shape 恢复

<!-- checkpoint: step06-block -->

左侧 `Qwen3SparseMoeBlock` 把本章数据流串起来。构造函数接收三块真实权重，分别建立 Router 与 Experts；forward 则只做四步：

```text
1. 检查 hidden_states 是 [B,S,D]
2. reshape 为 tokens [T,D]
3. Router 选择专家，Experts 计算并合并
4. reshape 回 [B,S,D]
```

现在可以把 Step 03 的 RMSNorm 与本章 MoE 接起来：

```python
from qwen3_moe import Qwen3SparseMoeBlock, RMSNorm

prefix = "model.layers.0"
post_attention_norm = RMSNorm(
    checkpoint.load_tensor(f"{prefix}.post_attention_layernorm.weight"),
    config.rms_norm_eps,
)
moe = Qwen3SparseMoeBlock(
    config,
    checkpoint.load_tensor(f"{prefix}.mlp.gate.weight"),
    checkpoint.load_tensor(f"{prefix}.mlp.experts.gate_up_proj"),
    checkpoint.load_tensor(f"{prefix}.mlp.experts.down_proj"),
)

normalized = post_attention_norm(hidden_states)
moe_output = moe(normalized)

print(normalized.shape)  # (B, S, 2048)
print(moe_output.shape)  # (B, S, 2048)
```

输入输出 shape 相同，内部计算路径却因 token 而异。某个表示数学推理的 token 可能偏向一组专家，另一个表示自然语言连接词的 token 可能选择另一组；Router 的学习参数决定这种分工，而不是代码预先给专家写死类别。

需要注意，课程代码强调数据流清晰，不是生产级高性能 MoE kernel。真实推理框架会进一步按设备、专家和 token 批量调度，避免扫描所有专家并减少数据搬运；这些优化不会改变本章的 Router、SwiGLU 与加权合并公式。

## MoE 在完整 decoder layer 中处于哪里

<!-- checkpoint: step06-package -->

最后从 `__init__.py` 导出三个类：

```python
from qwen3_moe import Qwen3MoeExperts, Qwen3MoeRouter, Qwen3SparseMoeBlock
```

把本章放回 Step 00 的 decoder layer：

```text
x [B,S,D]
  -> input RMSNorm
  -> causal GQA Attention             Step 05
  -> residual add                     下一章
  -> post-attention RMSNorm            Step 03
  -> Router [T,E]                      Step 06
  -> softmax + top-k [T,K]             Step 06
  -> selected SwiGLU experts           Step 06
  -> routing-weighted sum [B,S,D]      Step 06
  -> residual add                     下一章
```

先记住本章的六条规则：

1. Router 对每个 token 独立产生 `E` 个分数，Attention 才负责 token 之间的信息交换。
2. softmax 在全部 `E` 个专家上计算，top-k 再把执行路径稀疏到 `K` 个专家。
3. `norm_topk_prob=true` 时，入选专家的 routing weights 会重新归一化到和为 1。
4. 每个专家是 `SiLU(gate) * up` 后接 down projection 的 SwiGLU MLP。
5. 真实权重把全部专家堆成 `[E,2M,D]` 与 `[E,D,M]`，但只运行本批次实际命中的 expert slices。
6. 多个专家输出按 routing weights 相加，不是拼接，所以 MoE 前后都保持 `[B,S,D]`。

回到完整推理地图，现在一个 decoder layer 的两种核心变换都已经具备：

```text
prompt hidden states
  -> Attention：从已经出现的 token 读取上下文       Step 05
  -> MoE：为每个 token 选择参数路径并独立变换       Step 06
  -> residual connections + decoder layer          下一章
  -> repeat all layers
  -> final norm / logits / next token
```

下一章会把 RMSNorm、Attention、MoE 和两条 residual connection 组装成完整 `Qwen3DecoderLayer`。需要重新确认这条局部数据流在全模型中的位置时，可以回到 [Step 00 的完整推理地图](../step00/)。
