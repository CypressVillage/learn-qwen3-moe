# Step 08：把整段 prompt 推到 vocabulary logits

<!-- checkpoint: step08-ready -->

上一章已经能让 hidden states 穿过一个完整 Decoder Layer。现在沿着 Step 00 的推理地图继续向外组装，把模型入口、48 层主干和输出端全部接起来：

```text
token IDs [B,S]
  -> Embedding [B,S,D]
  -> Decoder Layer 0
  -> Decoder Layer 1
  -> ...
  -> Decoder Layer 47
  -> final RMSNorm [B,S,D]
  -> LM Head [B,S,V]
  -> logits
```

这一章实现 `Qwen3MoeForCausalLM` 的无 KV Cache [[prefill]]。它接收整段 prompt 的 token IDs，一次计算 prompt 中所有位置，最终为每个位置返回一组覆盖完整 vocabulary 的 [[logits]]。

先划清本章边界。我们会得到 `[B,S,V]`，但暂时不从中选 token。`argmax`、temperature、top-k 与随机采样都属于下一章。当前目标只有一个：把 token IDs 完整推过 Qwen3 MoE Transformer，得到模型对“每个位置之后应该出现什么”的原始分数。

完整数据流可以写成：

```text
input_ids = prompt token IDs
position_ids = 0, 1, 2, ..., S-1

h0 = Embedding(input_ids)
h1 = DecoderLayer0(h0, position_ids)
h2 = DecoderLayer1(h1, position_ids)
...
h48 = DecoderLayer47(h47, position_ids)

normalized = FinalRMSNorm(h48)
logits = LMHead(normalized)
```

这里没有新的 Attention 或 MoE 数学。Step 08 的重点是模型级组装：哪些权重位于所有层之外，怎样按正确编号创建全部层，以及 `[B,S,D]` 怎样最终变成 `[B,S,V]`。

## 完整 Causal LM 多了哪些模型级权重

<!-- checkpoint: step08-components -->

左侧新增的 `Qwen3MoeForCausalLM.__init__()` 接收五部分内容：

```text
config
model.embed_tokens.weight
48 个 Qwen3DecoderLayer
model.norm.weight
lm_head.weight
```

对于 `Qwen3-30B-A3B`，模型级 tensor 的关键 shape 是：

```text
model.embed_tokens.weight    [151936,2048]
model.norm.weight            [2048]
lm_head.weight               [151936,2048]
```

中间的每个 Decoder Layer 仍然持有上一章介绍的 11 块 tensor。完整模型不是把 48 层权重拼成一块新矩阵，而是保存一个按执行顺序排列的 layer 列表：

```text
self.layers[0]   -> model.layers.0.*
self.layers[1]   -> model.layers.1.*
...
self.layers[47]  -> model.layers.47.*
```

构造函数先检查四条模型级约束：

```text
Embedding shape      == [vocab_size, hidden_size]
Decoder Layer 数量   == num_hidden_layers
final norm shape     == [hidden_size]
LM Head shape        == [vocab_size, hidden_size]
```

这些检查守住完整模型的边界。各层内部更细的 Q/K/V、Router 和 Experts shape，仍然由 `Qwen3DecoderLayer` 及其子模块负责。

之后四个字段分别使用已经实现过的积木：

```text
self.embed_tokens = Embedding(...)
self.layers = layers
self.norm = RMSNorm(...)
self.lm_head = Linear(...)
```

Embedding 与 LM Head 的 weight shape 看起来相同，但计算方向不同：

```text
Embedding:
token ID -> 从 [V,D] 中查一行 -> hidden vector [D]

LM Head:
hidden vector [D] -> 与 [V,D] 的每一行做投影 -> logits [V]
```

因此模型入口把离散 ID 变成 hidden states，模型出口再把 hidden states 变成 vocabulary 上的分数。它们处在同一个词表空间的两端。

`tie_word_embeddings` 为 true 的模型可以让这两端共享同一块 weight；`Qwen3-30B-A3B` 的配置为 false，因此真实 checkpoint 中有独立的 `lm_head.weight`。

## 从真实 checkpoint 装载模型外壳

<!-- checkpoint: step08-checkpoint-loading -->

到目前为止，各模块构造函数都接收已经加载好的 NumPy tensor。完整模型需要读取的参数很多，如果继续在调用处逐个写出，会让“权重名字怎样组织”淹没主线。

所以 `from_checkpoint()` 把 Step 01 的 `SafetensorsCheckpoint` 与模型结构接起来：

```text
读取 model.embed_tokens.weight
  -> 按 layer_index 创建全部 Decoder Layers
  -> 读取 model.norm.weight
  -> 读取或共享 lm_head.weight
  -> 创建 Qwen3MoeForCausalLM
```

模型级参数名直接对应真实仓库：

```python
embed_tokens_weight = checkpoint.load_tensor("model.embed_tokens.weight")
norm_weight = checkpoint.load_tensor("model.norm.weight")
lm_head_weight = checkpoint.load_tensor("lm_head.weight")
```

`SafetensorsCheckpoint` 会根据 index 找到 tensor 所在分片，只读取对应 byte range。`Qwen3MoeForCausalLM` 不需要知道它来自第几个 `.safetensors` 文件。

LM Head 有一个分支：

```python
lm_head_weight = (
    embed_tokens_weight
    if config.tie_word_embeddings
    else checkpoint.load_tensor("lm_head.weight")
)
```

如果配置要求共享词嵌入，就直接把 Embedding weight 交给 `Linear`，不再寻找独立 LM Head。否则必须读取 `lm_head.weight`。判断依据是配置，而不是猜 checkpoint 中有没有这个名字。

这个类方法会把完整模型权重装入内存。真实 `Qwen3-30B-A3B` 需要与模型规模相匹配的内存资源；课程实现优先展示清晰的数据流，理解和验证时仍可使用相同 shape 关系的小配置与小 tensor。

## 按 layer index 找齐 48 层权重

<!-- checkpoint: step08-decoder-loading -->

`from_checkpoint()` 中最重要的一行是层列表推导式：

```python
layers = [
    cls._load_decoder_layer(config, checkpoint, layer_index)
    for layer_index in range(config.num_hidden_layers)
]
```

它把配置中的 `num_hidden_layers` 变成真实执行顺序。对于 48 层模型，`range(48)` 产生 `0` 到 `47`，不会出现 `model.layers.48`。

`_load_decoder_layer()` 先建立当前层的参数名前缀：

```python
prefix = f"model.layers.{layer_index}"
```

当 `layer_index == 12` 时，后续名字自然变成：

```text
model.layers.12.input_layernorm.weight
model.layers.12.self_attn.q_proj.weight
model.layers.12.self_attn.k_proj.weight
...
model.layers.12.mlp.experts.down_proj
```

这些 tensor 按上一章构造函数规定的顺序交给 `Qwen3DecoderLayer`。这里要同时保持两种顺序正确：

```text
层与层之间：0 -> 1 -> 2 -> ... -> 47
一层参数之间：norm -> Attention weights -> norm -> MoE weights
```

参数名不会决定执行顺序。Safetensors index 只是一个 name-to-location 映射；真正让第 7 层在第 8 层之前运行的，是 Python 列表中的位置和稍后的 `for layer in self.layers`。

`load = checkpoint.load_tensor` 只是把重复的方法访问缩短：

```python
load(f"{prefix}.self_attn.q_proj.weight")
```

它没有缓存或批量读取语义，每次调用仍然按参数名读取一块 tensor。

现在 Step 01 的权重目录终于接到了完整模型结构：

```text
Safetensors 参数名
  -> model-level weights
  -> layer 0 的 11 块权重
  -> layer 1 的 11 块权重
  -> ...
  -> layer 47 的 11 块权重
  -> 可执行的模块树
```

## 为整段 prompt 建立 position IDs

<!-- checkpoint: step08-positions -->

调用者只需要提供 token IDs，不必再手工创建 position IDs。`_position_ids()` 根据 `[B,S]` 输入生成：

```text
token_ids:
[[  9707,  18830,   9999],
 [  198,    198,   151645]]

position_ids:
[[0, 1, 2],
 [0, 1, 2]]
```

代码先建立一行位置：

```python
positions = np.arange(sequence_length, dtype=np.int64)
```

再广播到 batch：

```python
np.broadcast_to(positions, (batch_size, sequence_length))
```

同一个 batch 中每条独立 prompt 都从位置 0 开始。这里没有处理 padding mask，因此课程当前假设输入 batch 中的序列已经具有可直接 prefill 的相同长度；padding 与更复杂的 batch 管理不属于当前主线。

position IDs 会原样传给所有 Decoder Layers。每层都有自己的 RoPE 模块，但同一个 token 在不同层中的绝对位置不变：prompt 第 5 个 token 在第 0 层和第 47 层都使用 position ID 4。

为什么不让每一层自己生成位置？因为位置属于这次模型调用的输入语义，不属于某一层的私有参数。模型入口生成一次，再沿层循环传递，数据来源更清楚，也为后续 decode 阶段使用非零起始位置留下边界。

当前 prefill 总从 0 开始。等引入 KV Cache 后，新 token 的 position ID 必须接在缓存长度之后，那时模型不能再简单地只看本次输入长度。

## 串起完整 prefill forward

<!-- checkpoint: step08-forward -->

左侧 `__call__()` 先检查输入必须是二维 `[B,S]`，而且 prompt 至少有一个 token。之后完整前向只有五步：

```python
position_ids = self._position_ids(token_ids)
hidden_states = self.embed_tokens(token_ids)
for layer in self.layers:
    hidden_states = layer(hidden_states, position_ids)
hidden_states = self.norm(hidden_states)
logits = self.lm_head(hidden_states)
```

对应 shape 流是：

```text
token_ids                      [B,S]
position_ids                   [B,S]
Embedding                      [B,S,D]
Decoder Layer 0                [B,S,D]
...
Decoder Layer 47               [B,S,D]
final RMSNorm                  [B,S,D]
LM Head                        [B,S,V]
```

层循环中每次都覆盖 `hidden_states`：

```python
for layer in self.layers:
    hidden_states = layer(hidden_states, position_ids)
```

这不是让所有层并行读取同一个 Embedding 输出。第 `n+1` 层接收的是第 `n` 层完整执行 Attention、MoE 和两条 residual 后的结果。

48 层之后还有一个 final RMSNorm。它不属于第 47 层的 `post_attention_layernorm`：

```text
layer 47 post_attention_layernorm
  -> 只归一化 layer 47 的 MoE 输入

model.norm
  -> 归一化全部 Decoder Layers 的最终输出
  -> 直接送入 LM Head
```

漏掉 `model.norm` 会让 LM Head 接收到与训练时不同尺度的 hidden states。它必须位于层循环之后、输出投影之前。

LM Head 是一个无 bias 的 `Linear`。对每个 `[D]` hidden vector，它都会产生 `[V]` logits：

```text
logits[b, s, v]
```

表示 batch 中第 `b` 条序列，在位置 `s` 之后把 vocabulary token `v` 作为下一个 token 的原始分数。logits 不是概率，不要求和为 1，也可以是负数。

由于 causal mask 保证位置 `s` 只能读取 `0..s`，所以一次 prefill 能同时得到每个前缀的预测：

```text
logits[:,0,:]     看过第 0 个 token 后的预测
logits[:,1,:]     看过第 0..1 个 token 后的预测
...
logits[:,-1,:]    看完整段 prompt 后的 next-token 预测
```

真正继续生成时，最关心的是最后一项 `logits[:, -1, :]`。但本章仍返回完整 `[B,S,V]`，因为这是 Causal LM prefill 的自然输出，也能清楚展示每个 prompt 位置都完成了预测。

:::principle 为什么位置 s 的 logits 预测的是 token s+1

输入 token 是 `[t0, t1, ..., t(S-1)]`。由于 causal mask，位置 `s` 的 hidden state `h_s` 最多只能包含前缀 `[t0, ..., ts]` 的信息，因此 LM Head 在这个位置表示的条件分布是：

```text
logits[:, s, :] -> P(t(s+1) | t0, ..., ts)
```

位置和目标由此向右错开一格：

```text
位置 0 hidden state   -> 预测 t1
位置 1 hidden state   -> 预测 t2
...
位置 S-1 hidden state -> 预测 prompt 后的第一个新 token
```

所以 `logits[:, -1, :]` 不是在重建 prompt 的最后一个 token。它已经看过完整 prompt，给出的是继续生成所需的下一 token 分布。Step 09 对 sequence 维取最后一项，正是利用这个对齐关系。

:::endprinciple

## 从模型目录跑到 logits

现在可以把前几章的入口连在一起：

```python
from pathlib import Path

import numpy as np

from qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3Tokenizer,
    SafetensorsCheckpoint,
)

checkpoint_dir = Path("Qwen3-30B-A3B")
config = Qwen3MoeConfig.from_json(checkpoint_dir / "config.json")
checkpoint = SafetensorsCheckpoint.from_directory(checkpoint_dir)
tokenizer = Qwen3Tokenizer.from_directory(checkpoint_dir)
model = Qwen3MoeForCausalLM.from_checkpoint(config, checkpoint)

token_ids = np.asarray([tokenizer.encode("MoE 是怎样工作的？")])
logits = model(token_ids)

print(token_ids.shape)  # (1, S)
print(logits.shape)     # (1, S, 151936)
```

这段调用覆盖了当前已经实现的整条主干：

```text
文本
  -> Qwen3 byte-level BPE
  -> token IDs
  -> Embedding
  -> 48 x (GQA Attention + Sparse MoE + residual)
  -> final RMSNorm
  -> LM Head
  -> vocabulary logits
```

课程代码使用 NumPy 展示真实计算关系，没有量化、设备切分、算子融合或高性能 kernel。直接装载并运行 30B 级模型需要充足内存和很长计算时间；学习各模块时应优先用小配置验证 shape 与数值流。

## logits 回到完整推理地图

<!-- checkpoint: step08-package -->

最后从 `__init__.py` 导出 `Qwen3MoeForCausalLM`：

```python
from qwen3_moe import Qwen3MoeForCausalLM
```

把这一章放回 Step 00 的完整链路：

```text
text
  -> Tokenizer                                  Step 02
  -> token IDs [B,S]
  -> Embedding [B,S,D]                          Step 03 + Step 08
  -> 48 Decoder Layers                          Step 05~08
       -> causal GQA Attention
       -> Sparse MoE
       -> two residual connections
  -> final RMSNorm                              Step 08
  -> LM Head
  -> logits [B,S,V]                             Step 08 完成
  -> select from logits                         下一章
  -> next token
```

先记住本章的七条规则：

1. 完整 Causal LM 的入口是 `[B,S]` token IDs，出口是 `[B,S,V]` vocabulary logits。
2. Embedding、全部 Decoder Layers、final RMSNorm 与 LM Head 必须严格按数据流顺序连接。
3. Decoder Layers 按 `0..num_hidden_layers-1` 装载和执行，每层结构相同但权重独立。
4. prompt prefill 的 position IDs 是每条序列从 0 开始的 `0..S-1`，同一组位置传给所有层。
5. `model.norm` 位于完整层循环之后，不等于最后一层内部的 `post_attention_layernorm`。
6. LM Head 把最后一维从 hidden size 投影到 vocabulary size，logits 是原始分数而不是概率。
7. 继续生成只需要最后位置的 `logits[:, -1, :]`，但本章保留所有位置的输出。

回到推理主线，现在整段 prompt 已经第一次完整穿过 Qwen3 MoE：

```text
prompt token IDs
  -> all prompt positions prefill
  -> all decoder layers
  -> final hidden states
  -> vocabulary logits for every position     Step 08 完成
  -> choose one next-token ID                  下一章
  -> decode token text
```

下一章会实现从最后位置 logits 选择 next token，先从 greedy decoding 开始，再说明 temperature 与采样怎样改变选择过程。需要重新确认 logits 在全局链路中的位置时，可以回到 [Step 00 的完整推理地图](../step00/)；需要回看单层内部的数据流，可以返回 [Step 07 的 Decoder Layer](../step07/)。
