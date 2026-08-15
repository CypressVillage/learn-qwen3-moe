# Step 05：让每个 token 只读取已经出现的上下文

<!-- checkpoint: step05-ready -->

上一章把 position IDs 写进了 Query 和 Key。现在把前面准备的基础层与 RoPE 组装起来，第一次完成一整段 Attention 数据流：

```text
normalized hidden_states [B,S,D]
  -> q_proj / k_proj / v_proj
  -> split heads + QK Norm
  -> RoPE(Query, Key)
  -> GQA scaled dot-product attention
  -> causal mask + softmax
  -> weighted Value
  -> merge heads + o_proj
  -> attention output [B,S,D]
```

这一章实现 `attention.py` 中的 `Qwen3Attention`。它处理 prompt 的 prefill：序列中所有 token 一起进入 Attention，每个位置可以读取自己和已经出现的前文，但不能偷看未来。

先划清本章边界。输入已经经过 decoder layer 外部的 `input_layernorm`，所以 `Qwen3Attention` 不重复做那次 RMSNorm；本章内部仍然会实现 Qwen3 特有的 QK Norm。我们也暂时不保存 KV Cache，增量 decode 会在后面的 cache 章节接入。

本章继续使用四个维度：

```text
B   = batch size
S   = sequence length
D   = hidden_size
Hd  = head_dim
Hq  = Query head count
Hkv = Key/Value head count
```

对于 `Qwen3-30B-A3B`：

```text
D   = 2048
Hd  = 128
Hq  = 32
Hkv = 4
```

最容易忽略的事实是，`D` 不等于 `Hq * Hd`。这里 `32 * 128 = 4096`，所以 Query 投影会先把每个 2048 维 hidden vector 扩成 4096 个数，拆成 heads；最后再由输出投影压回 2048 维。

## 一层 Attention 需要哪些真实权重

以第 0 层为例，`model.safetensors.index.json` 会把下面六块权重映射到具体分片：

```text
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_norm.weight
model.layers.0.self_attn.k_norm.weight
```

它们的 shape 并不相同：

```text
q_proj [Hq * Hd, D]   = [4096, 2048]
k_proj [Hkv * Hd, D]  = [ 512, 2048]
v_proj [Hkv * Hd, D]  = [ 512, 2048]
o_proj [D, Hq * Hd]   = [2048, 4096]
q_norm [Hd]           = [128]
k_norm [Hd]           = [128]
```

这组 shape 已经把 GQA 写进权重本身：模型真的只保存 4 个 Key heads 和 4 个 Value heads，并不是先保存 32 个再在运行时删掉。

<!-- checkpoint: step05-weights -->

左侧 `Qwen3Attention.__init__()` 先根据 config 算出 Query 总宽度与 KV 总宽度，再逐块检查真实权重。这样如果拿错层、拿错模型或交换了 Q/K 权重，错误会在 Attention 创建时直接暴露，而不是等到中间某次 reshape 才变成难懂的尺寸异常。

检查通过后，构造函数只组装已经实现过的积木：

```text
四个投影     -> Linear
Q/K 归一化   -> RMSNorm
位置旋转     -> RotaryEmbedding
```

`q_proj`、`k_proj`、`v_proj` 和 `o_proj` 都没有 bias，因此直接把 checkpoint tensor 交给 `Linear`。QK Norm 则使用 config 中同一个 `rms_norm_eps`，但 Q 和 K 各自拥有学习到的 weight。

这里没有让 `attention.py` 自己读取 Safetensors。模块边界仍然与 Step 03 一致：

```text
checkpoint.py 负责：参数名 -> NumPy tensor
attention.py  负责：这些 tensor 怎样完成 Attention
```

## Q、K、V 投影后为什么要立刻拆 heads

输入是已经归一化的：

```text
hidden_states [B,S,2048]
```

三个投影分别得到：

```text
Query values [B,S,4096]
Key values   [B,S, 512]
Value values [B,S, 512]
```

这些最后一维仍然把所有 heads 挤在一起。下一步按 config 拆开：

```text
Query [B,S,32,128]
Key   [B,S, 4,128]
Value [B,S, 4,128]
```

Qwen3 的 QK Norm 不是对 4096 或 512 个数整体做一次 RMSNorm，而是独立归一化每个 head 的 `Hd = 128` 个数。也就是说 RMSNorm 必须在拆 heads 之后执行：

```text
[B,S,H,Hd]
        ^
        `-- 只沿每个 head 的最后 128 维计算
```

Value 不做这一步。QK Norm 稳定的是后面用于点积的 Query 和 Key；Value 保存将被读取和汇总的内容。

<!-- checkpoint: step05-projections -->

左侧 `_project_query_key_value()` 先复用三个 `Linear`，再 reshape。`RMSNorm` 原本就只沿最后一维工作，所以它可以直接接收 `[B,S,H,Hd]`。

归一化完成后，代码把 head 维移到 sequence 前面：

```text
Query [B,S,32,128] -> [B,32,S,128]
Key   [B,S, 4,128] -> [B, 4,S,128]
Value [B,S, 4,128] -> [B, 4,S,128]
```

这样最后两个维度就是后面矩阵乘法需要的 `[S,Hd]`。batch 和 head 会作为批量维度保留下来。

要特别区分两种 Norm：decoder layer 的 `input_layernorm` 作用于 `[B,S,D]`，在进入 Attention 之前稳定 hidden states；这里的 `q_norm` 与 `k_norm` 作用于拆开的 `[B,S,H,Hd]`，专门稳定 Q/K 点积。它们位置不同，权重也不同。

## 点积前把 RoPE 接回数据流

到这里，Query 和 Key 已经有正确的 head shape，但还需要带上各自 token 的位置。Step 04 已经完成全部旋转细节，本章不重新实现公式，只负责在正确时刻调用它：

```text
projected Q/K
  -> split heads
  -> QK Norm
  -> RoPE
  -> Q @ K.T
```

<!-- checkpoint: step05-positions -->

左侧 `_apply_positions()` 从 `position_ids [B,S]` 生成 cosine/sine，再把同一套位置规则应用到 Query 和 Key。RoPE 允许 Q/K head 数分别是 32 和 4，因此这一段不需要先扩展 KV heads。

顺序不能随意交换。QK Norm 先调整每个 head 的尺度，RoPE 再旋转坐标；旋转后的 Q/K 才进入点积。Value 仍然既不做 QK Norm，也不做 RoPE。

这一步体现了累计实现的意义：上一章单独验证的
`apply_rotary_position_embedding()`，现在已经不再是孤立公式，而是 Attention forward 中连接投影与点积的一环。

## GQA 怎样让 32 个 Query heads 共享 4 组 K/V

普通 Multi-Head Attention 通常让 Q、K、V 拥有相同 head 数。Qwen3 使用 Grouped-Query Attention：

```text
32 Query heads / 4 KV heads = 每组 8 个 Query heads
```

共享关系可以想成：

```text
Query heads  0..7   -> Key/Value head 0
Query heads  8..15  -> Key/Value head 1
Query heads 16..23  -> Key/Value head 2
Query heads 24..31  -> Key/Value head 3
```

每个 Query head 仍然独立产生自己的 Query，因此能提出不同的“检索问题”；同组的 8 个 Query heads 共用同一份 Key 和 Value，减少 KV 权重与以后 KV Cache 的体积。

课程实现使用 `np.repeat(..., axis=1)`，在点积前把 K/V 的 head 维从 4 展开到 32：

```text
Key   [B, 4,S,Hd] -> [B,32,S,Hd]
Value [B, 4,S,Hd] -> [B,32,S,Hd]
```

这一步只是为了让 NumPy 的批量矩阵乘法直接对齐。它不会创造新的学习参数；同组的 8 份内容完全相同。后续实现 KV Cache 时，cache 仍然只需要保存原始 4 个 KV heads。

## Query 与所有 Key 的点积变成读取分数

扩展 head 数后，每个 Query head 都可以和对应的整段 Key 做矩阵乘法：

```text
Query [B,Hq,S,Hd]
Key.T [B,Hq,Hd,S]
  -> scores [B,Hq,S,S]
```

scores 的最后两个维度分别表示：

```text
倒数第 2 维：哪个 Query 位置正在读取
最后一维：  它正在给哪个 Key 位置打分
```

原始点积还要除以 `sqrt(Hd)`：

```text
scores = (Q @ K.T) / sqrt(Hd)
```

如果 `Hd` 很大，未缩放点积的绝对值也容易变大，softmax 会过早变得极端。乘上 `Hd ** -0.5` 可以把不同 head dimension 下的分数尺度拉回更稳定的范围。

## causal mask 为什么是 Attention 的必要部分

prefill 时整段 prompt 同时参与矩阵乘法，所以位置 0 的 Query 在数值上也能和位置 3 的 Key 做点积。但自回归语言模型不能让前面的 token 读取未来，否则训练和生成时就会看到不该知道的答案。

长度为 4 时，允许读取的位置形成一个下三角矩阵：

```text
             Key position
             0  1  2  3
Query 0      ✓  ×  ×  ×
Query 1      ✓  ✓  ×  ×
Query 2      ✓  ✓  ✓  ×
Query 3      ✓  ✓  ✓  ✓
```

代码用 `np.triu(..., k=1)` 找出主对角线上方的未来位置，把对应 score 替换成负无穷：

```text
future score = -inf
softmax(-inf) = 0
```

当前位置可以读取自己，所以主对角线不会被遮住。causal mask 约束的是信息方向，不是在删除 token。

## softmax 把分数变成读取比例

mask 之后，对每个 Query 位置最后一维的所有 Key scores 做 softmax：

```text
probabilities[b,h,q,:] = softmax(scores[b,h,q,:])
```

每一行 probabilities 都满足：

```text
未来位置的概率 = 0
允许位置的概率 >= 0
整行概率之和 = 1
```

左侧实现先减去每一行最大值，再计算指数。这不会改变 softmax 结果，却能避免较大的正 score 在 `exp()` 时溢出。点积、softmax 和 Value 加权都使用 `float32`，优先保证 CPU 教学实现的数值稳定。

最后让 probabilities 对 Value 做加权求和：

```text
probabilities [B,Hq,S,S]
Value         [B,Hq,S,Hd]
  -> attended [B,Hq,S,Hd]
```

对某个 Query 位置来说，输出不再只是它自己的向量，而是按照注意力概率汇总出的上下文。到这里，token 才真正完成了“读取前文”。

<!-- checkpoint: step05-causal-attention -->

左侧 `_scaled_dot_product_attention()` 把 GQA 共享、缩放点积、causal mask、稳定 softmax 与 Value 加权放在同一段连续的数据流里。这里没有额外接受手写 attention mask，因为本章只处理没有 padding 的 prompt；首先把自回归因果关系看清，批量 padding 可以在完整模型整合时再扩展。

## 合并 heads 并回到 hidden size

Attention kernel 返回：

```text
attended [B,32,S,128]
```

decoder layer 的 residual branch 需要 `[B,S,D]`，所以先把 sequence 维移回来，再把所有 Query heads 合并：

```text
[B,32,S,128]
  -> transpose [B,S,32,128]
  -> reshape   [B,S,4096]
```

注意，合并后是 4096，不是 2048。最后的 `o_proj [2048,4096]` 才把所有 heads 的上下文混合并投影回模型 hidden size：

```text
[B,S,4096] -> o_proj -> [B,S,2048]
```

<!-- checkpoint: step05-forward -->

左侧 `Qwen3Attention.__call__()` 先检查输入必须是 `[B,S,D]`，并要求 `position_ids` 与前两个维度完全一致。随后依次调用本章刚实现的三个阶段，合并 heads，再执行输出投影。

现在可以用一层真实权重把 Step 03 到 Step 05 接起来：

```python
import numpy as np

from qwen3_moe import Qwen3Attention, RMSNorm

prefix = "model.layers.0"
input_norm = RMSNorm(
    checkpoint.load_tensor(f"{prefix}.input_layernorm.weight"),
    config.rms_norm_eps,
)
attention = Qwen3Attention(
    config,
    checkpoint.load_tensor(f"{prefix}.self_attn.q_proj.weight"),
    checkpoint.load_tensor(f"{prefix}.self_attn.k_proj.weight"),
    checkpoint.load_tensor(f"{prefix}.self_attn.v_proj.weight"),
    checkpoint.load_tensor(f"{prefix}.self_attn.o_proj.weight"),
    checkpoint.load_tensor(f"{prefix}.self_attn.q_norm.weight"),
    checkpoint.load_tensor(f"{prefix}.self_attn.k_norm.weight"),
)

normalized = input_norm(hidden_states)
position_ids = np.arange(normalized.shape[1], dtype=np.int64)[None, :]
attention_output = attention(normalized, position_ids)

print(normalized.shape)        # (1, S, 2048)
print(attention_output.shape)  # (1, S, 2048)
```

输入输出 shape 相同不代表数据没有变化。输入中每个位置主要携带自己的表示；输出中每个位置已经按 causal attention 汇总了可见上下文。保持 `[B,S,D]` 是为了下一步能把结果加回 residual，并继续送入 MoE。

当前实现是 prefill Attention。它会为长度 `S` 的 prompt 建立完整 `[S,S]` score 矩阵，但不会把 K/V 返回或保存。之后接入 KV Cache 时，核心点积规则不会改变，只是 decode 阶段的 Query 长度会变成 1，并把新 K/V 追加到历史 cache。

## Attention 在完整 decoder layer 中处于哪里

<!-- checkpoint: step05-package -->

最后从 `__init__.py` 导出 `Qwen3Attention`：

```python
from qwen3_moe import Qwen3Attention
```

把本章放回 Step 00 的 decoder layer：

```text
x [B,S,D]
  -> input RMSNorm                 Step 03
  -> Q/K/V projections             Step 05
  -> QK Norm                       Step 05
  -> RoPE                          Step 04，Step 05 接入
  -> GQA + causal softmax          Step 05
  -> weighted Value + o_proj       Step 05
  -> attention_output [B,S,D]
  -> residual add                  后续 model 章节
  -> post-attention RMSNorm
  -> MoE                           下一章
```

先记住本章的五条 shape 规则：

1. Q 投影宽度是 `Hq * Hd`，K/V 投影宽度是 `Hkv * Hd`，它们不必等于 hidden size。
2. QK Norm 只沿拆 head 后的 `Hd` 工作，Value 不做 QK Norm，也不做 RoPE。
3. GQA 让每 `Hq / Hkv` 个 Query heads 共用一个 KV head；真实 KV 权重与 cache 都只保留 `Hkv` 份。
4. causal scores 的 shape 是 `[B,Hq,S,S]`，每个 Query 位置只能在自己和前文上分配概率。
5. heads 合并后先得到 `Hq * Hd`，再由 `o_proj` 回到 `[B,S,D]`。

回到完整推理地图，现在 prompt 中的 token 已经第一次互相交换信息：

```text
prompt
  -> token IDs [B,S]                    Step 02
  -> hidden states [B,S,D]              Step 03
  -> 带位置的 Query / Key                Step 04
  -> causal GQA context [B,S,D]         Step 05
  -> residual + MoE                     下一章
  -> decoder layers / logits / next token
```

下一章会进入 Qwen3 MoE 的稀疏前馈分支：读取 Router 与 Expert 权重，为每个 token 选择少量专家，再把专家输出按 routing probability 合并。需要重新确认 Attention 为什么处在 MoE 之前时，可以回到 [Step 00 的完整推理地图](../step00/)。
