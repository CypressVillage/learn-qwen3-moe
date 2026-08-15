# Step 07：把 Attention、MoE 与两条 residual 组装成一层

<!-- checkpoint: step07-ready -->

前两章分别完成了 decoder layer 中的两种核心变换：Attention 让每个 token 读取已经出现的上下文，Sparse MoE 再为每个 token 选择合适的参数路径。现在把它们接回同一条 hidden states 主干：

```text
x [B,S,D]
  -> input RMSNorm
  -> causal GQA Attention
  -> + x                         第一条 residual
  -> post-attention RMSNorm
  -> Sparse MoE
  -> + attention result          第二条 residual
  -> layer output [B,S,D]
```

这一章实现 `model.py` 中的 `Qwen3DecoderLayer`。它不引入新的张量运算，而是回答一个同样关键的问题：已经实现的基础层、Attention 和 MoE，究竟以什么顺序组成真实 Qwen3 MoE 的一层。

先划清本章边界。我们只组装一个 decoder layer，不创建 Embedding，不循环 48 层，也不接 final RMSNorm 与 LM Head。输入是上一层传来的 hidden states 和本次 prompt 的 position IDs，输出仍然是 `[B,S,D]`，可以继续交给下一层。

把一层前向写成公式会更清楚：

```text
x0 = input hidden states
a  = Attention(RMSNorm(x0), position_ids)
x1 = x0 + a

m  = MoE(RMSNorm(x1))
x2 = x1 + m
```

最终返回 `x2`。注意第二个 RMSNorm 接收的是 `x1`，第二条 residual 保存的也必须是 `x1`，而不是最初的 `x0`。

## 一层 Decoder Layer 拥有哪些真实权重

以第 0 层为例，`Qwen3DecoderLayer` 会接住 11 块 tensor：

```text
model.layers.0.input_layernorm.weight

model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_norm.weight
model.layers.0.self_attn.k_norm.weight

model.layers.0.post_attention_layernorm.weight

model.layers.0.mlp.gate.weight
model.layers.0.mlp.experts.gate_up_proj
model.layers.0.mlp.experts.down_proj
```

它们不是 11 个彼此平行的操作，而是分属于四个子模块：

```text
input_layernorm             1 块
self_attention              6 块
post_attention_layernorm    1 块
sparse_moe                  3 块
```

对于 `Qwen3-30B-A3B`，关键 shape 是：

```text
input_layernorm             [2048]
q_proj                      [4096,2048]
k_proj / v_proj             [512,2048]
o_proj                      [2048,4096]
q_norm / k_norm             [128]
post_attention_layernorm    [2048]
router                      [128,2048]
expert gate_up              [128,1536,2048]
expert down                 [128,2048,768]
```

这一层本身不需要理解每块权重内部怎样计算。shape 检查和具体数学已经由 Step 03、Step 05、Step 06 的模块负责；Decoder Layer 的职责是把权重交给正确的模块，并维护正确的数据流。

## 组装四个已经实现的子模块

<!-- checkpoint: step07-components -->

左侧 `Qwen3DecoderLayer.__init__()` 先保存 `hidden_size`，然后创建：

```text
self.input_layernorm
self.self_attention
self.post_attention_layernorm
self.mlp
```

这里的 `mlp` 实际是 `Qwen3SparseMoeBlock`。Qwen3 配置和权重命名沿用了 Transformer 中常见的 `mlp` 字段，但在 MoE 层里，它不是一个所有 token 共享的普通 MLP，而是 Router 加多个 SwiGLU Experts。

构造函数显式接收全部权重，是为了让参数归属保持可见：

```text
q_proj_weight              -> Qwen3Attention
q_norm_weight              -> Qwen3Attention 内部的 QK Norm
router_weight              -> Qwen3SparseMoeBlock 的 Router
gate_up_proj_weight        -> Qwen3SparseMoeBlock 的 Experts
```

`Qwen3DecoderLayer` 不重复检查这些复杂 shape。创建 `RMSNorm`、`Qwen3Attention` 和 `Qwen3SparseMoeBlock` 时，各模块会继续执行自己已有的检查。这样责任边界保持一致：

```text
Decoder Layer   负责模块顺序与 residual
Attention       负责 Q/K/V、RoPE、GQA 与 causal mask
Sparse MoE      负责 Router、top-k、Experts 与加权合并
RMSNorm         负责最后一维的尺度归一化
```

本章也没有直接从 `SafetensorsCheckpoint` 读取权重。和前几章一样，模型组装代码接收已经加载好的 NumPy tensor；“参数名对应哪个 tensor”与“tensor 怎样参与计算”仍然分开。

## 第一条主干：pre-norm Attention 再加 residual

<!-- checkpoint: step07-attention-residual -->

左侧 `_attention_block()` 完成三步：

```text
residual = hidden_states
normalized = input_layernorm(hidden_states)
attention_output = self_attention(normalized, position_ids)
output = residual + attention_output
```

这叫 pre-norm 结构，因为 RMSNorm 位于 Attention 之前：

```text
x -> Norm -> Attention -> Add
|                         ^
+------ residual ---------+
```

不要把它误写成：

```text
Norm(x + Attention(x))
```

后者是不同的 post-norm 数据流，会改变送入 Attention 的数值，也会改变 residual 主干的传播方式。读取真实模型代码时，`input_layernorm` 这个名字也在提醒我们：它归一化的是 Attention 的输入。

为什么要先保存 `residual`？因为 `hidden_states` 变量接下来会依次被归一化和 Attention 输出覆盖，而相加时需要的是未经这两个操作的原始 `x0`：

```text
x1 = x0 + Attention(RMSNorm(x0))
```

两条分支的 shape 都是 `[B,S,D]`。Attention 内部虽然经历了 heads 拆分、causal scores 和 Value 聚合，但输出投影已经把结果恢复到 hidden size，因此可以逐元素相加。

residual add 不是拼接，也不会把 hidden size 从 2048 变成 4096。它把 Attention 产生的“上下文更新量”加回原表示，让主干既保留进入本层的信息，也吸收当前层读取到的新上下文。

`position_ids` 只传给 Attention，因为 RoPE 需要知道每个 token 的位置。MoE 对每个 token 独立变换 hidden dimensions，不做跨位置读取，因此不需要 position IDs。

## 第二条主干：pre-norm MoE 再加 residual

<!-- checkpoint: step07-moe-residual -->

第一条 residual 得到 `x1` 后，`_moe_block()` 使用完全对应的结构：

```text
residual = x1
normalized = post_attention_layernorm(x1)
moe_output = mlp(normalized)
x2 = residual + moe_output
```

公式是：

```text
x2 = x1 + MoE(RMSNorm(x1))
```

这里最容易出现的错误，是仍然把最初的 `x0` 加回来：

```text
错误：x2 = x0 + MoE(RMSNorm(x1))
正确：x2 = x1 + MoE(RMSNorm(x1))
```

Attention 的结果已经进入当前层状态，MoE 必须在这个更新后的状态上继续工作。第二条 residual 也围绕同一个 `x1` 展开，才能保持层内数据流连续。

为什么 Attention 和 MoE 前各有一个 RMSNorm，而不是整层只做一次？因为两次变换接收的输入不同：

```text
input_layernorm(x0)             -> 稳定 Attention 输入
post_attention_layernorm(x1)    -> 稳定 MoE 输入
```

`post_attention_layernorm` 的意思不是“对整个 Attention residual 结构做最终输出归一化”，而是“在 Attention 之后、MoE 之前进行归一化”。它仍然是第二个子块的 pre-norm。

MoE 输出与输入同为 `[B,S,D]`。内部的 Router 会把前两维展平、选出 top-k experts、运行 SwiGLU 并按 routing weights 合并，最后已经 reshape 回原 shape，所以 Decoder Layer 只需要执行逐元素 residual add。

推理阶段没有 dropout，因此两个子块之间不需要额外的随机操作。课程代码直接展示决定输出的主路径。

## 串起完整 Decoder Layer forward

<!-- checkpoint: step07-forward -->

左侧 `__call__()` 先检查输入必须是 `[B,S,D]`，再按固定顺序调用两个子块：

```python
hidden_states = self._attention_block(hidden_states, position_ids)
hidden_states = self._moe_block(hidden_states)
```

最终代码把第二行直接作为返回值。完整 shape 流是：

```text
input                         [B,S,D]
input RMSNorm                 [B,S,D]
Attention                     [B,S,D]
first residual add            [B,S,D]
post-attention RMSNorm        [B,S,D]
Sparse MoE                    [B,S,D]
second residual add           [B,S,D]
output                        [B,S,D]
```

Decoder Layer 只显式检查 hidden states。`position_ids` 的 batch 与 sequence shape 会继续由 `Qwen3Attention` 检查，因此这里不复制同一条验证逻辑。

现在可以从真实 checkpoint 组装第 0 层：

```python
from qwen3_moe import Qwen3DecoderLayer

prefix = "model.layers.0"
layer = Qwen3DecoderLayer(
    config,
    checkpoint.load_tensor(f"{prefix}.input_layernorm.weight"),
    checkpoint.load_tensor(f"{prefix}.self_attn.q_proj.weight"),
    checkpoint.load_tensor(f"{prefix}.self_attn.k_proj.weight"),
    checkpoint.load_tensor(f"{prefix}.self_attn.v_proj.weight"),
    checkpoint.load_tensor(f"{prefix}.self_attn.o_proj.weight"),
    checkpoint.load_tensor(f"{prefix}.self_attn.q_norm.weight"),
    checkpoint.load_tensor(f"{prefix}.self_attn.k_norm.weight"),
    checkpoint.load_tensor(f"{prefix}.post_attention_layernorm.weight"),
    checkpoint.load_tensor(f"{prefix}.mlp.gate.weight"),
    checkpoint.load_tensor(f"{prefix}.mlp.experts.gate_up_proj"),
    checkpoint.load_tensor(f"{prefix}.mlp.experts.down_proj"),
)

layer_output = layer(hidden_states, position_ids)
print(layer_output.shape)  # (B, S, 2048)
```

这一调用会执行完整的一层 prefill：所有 prompt token 先通过 causal Attention 读取各自可见的上下文，再分别经过 Router 和选中的 Experts。它还不是完整模型，因为目前只创建了第 0 层，也没有把最终 hidden states 映射成 vocabulary logits。

## 一层完成后，为什么还要重复 48 次

单个 decoder layer 不会一次完成全部推理。`Qwen3-30B-A3B` 的 `num_hidden_layers` 是 48，hidden states 会依次穿过不同权重的 48 层：

```text
embedding output
  -> layer 0
  -> layer 1
  -> layer 2
  -> ...
  -> layer 47
  -> final RMSNorm
  -> LM Head
  -> logits
```

每层结构相同，但参数不同。第 0 层的 `q_proj.weight` 与第 1 层的 `q_proj.weight` 是两块独立 tensor，各层 Router 和 Experts 也各自学习不同的变换。

两条 residual connection 让 hidden states 始终保留一条逐层延续的主干。可以把每层理解为在当前状态上添加两个更新：

```text
当前状态
  + 一次上下文读取更新
  + 一次稀疏前馈更新
  = 下一层状态
```

这只是帮助理解的视角，不意味着不同层的更新可以随意交换。第 `n+1` 层读取的是第 `n` 层完整输出，因此层顺序和每层内部的 Attention/MoE 顺序都必须保持。

本章暂时不实现层循环，是为了让“一个 layer 怎样工作”和“模型怎样管理很多 layers”保持两个清晰阶段。下一章会在完整 Causal LM 中创建 Embedding、全部 decoder layers、final norm 与 LM Head。

## Decoder Layer 回到完整推理地图

<!-- checkpoint: step07-package -->

最后从 `__init__.py` 导出 `Qwen3DecoderLayer`：

```python
from qwen3_moe import Qwen3DecoderLayer
```

把这一章放回 Step 00 的完整链路：

```text
text
  -> Tokenizer                              Step 02
  -> token IDs [B,S]
  -> Embedding [B,S,D]                      Step 03
  -> Decoder Layer 0
       -> RMSNorm + Attention + residual    Step 05 + Step 07
       -> RMSNorm + Sparse MoE + residual   Step 06 + Step 07
  -> Decoder Layer 1
  -> ...
  -> Decoder Layer 47
  -> final norm / logits                    下一章
  -> next token
```

先记住本章的六条规则：

1. Qwen3 Decoder Layer 是 pre-norm 结构，两个 RMSNorm 都位于各自变换之前。
2. 第一条 residual 是 `x0 + Attention(RMSNorm(x0))`。
3. 第二条 residual 必须围绕更新后的 `x1`，即 `x1 + MoE(RMSNorm(x1))`。
4. Attention 接收 position IDs，MoE 不接收，因为只有 Attention 使用 RoPE 做跨位置读取。
5. Attention 和 MoE 都恢复到 `[B,S,D]` 后再逐元素相加，residual 不改变 tensor shape。
6. 单层只完成一次上下文更新和一次稀疏前馈更新，完整模型还要按顺序运行全部 48 层。

回到推理主线，现在 prompt hidden states 已经能够通过一整个 Qwen3 MoE decoder layer：

```text
prompt hidden states
  -> causal context reading
  -> first residual
  -> top-k expert transformation
  -> second residual
  -> one decoder layer output              Step 07 完成
  -> repeat all decoder layers             下一章
  -> final norm / logits / next token
```

下一章会实现无 KV Cache 的完整 `Qwen3MoeForCausalLM` prefill，把 token IDs 经过 Embedding 和全部 Decoder Layers，最终投影成 vocabulary logits。需要重新确认层循环在全模型中的位置时，可以回到 [Step 00 的完整推理地图](../step00/)。
