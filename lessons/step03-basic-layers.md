# Step 03：让 token IDs 第一次进入模型权重

<!-- checkpoint: step03-ready -->

上一章把文字变成了 Qwen3 认识的 token IDs。现在继续沿着 Step 00 的推理地图往前走，让这些整数第一次参加张量计算：

```text
prompt 文本
  -> Tokenizer
  -> input_ids [B,S]
  -> Embedding
  -> hidden_states [B,S,D]
```

这一章会实现 [[Embedding]]、[[RMSNorm]] 和 `Linear`。它们都放在 `layers.py`，因为后面的 Attention、MoE 和完整模型会反复使用这些小积木。

先划清边界：本章只准备基础计算，不提前实现 Attention，也不创建完整 decoder layer。我们先看清 token ID 怎样变成向量，以及一个 `[B,S,D]` tensor 怎样在不打乱前两维的情况下被归一化和投影。

## 先找到 Embedding 的真实权重

Step 01 已经能按名字读取 Safetensors tensor。现在要拿的第一块真正参与 forward 的权重是：

```text
model.embed_tokens.weight
```

在 `Qwen3-30B-A3B` 中，它可以看成一张很大的表：

```text
Embedding weight [V,D]

V = vocab_size  = 151936
D = hidden_size = 2048
```

每一行对应一个 token ID，每一行里有 `D` 个浮点数。Tokenizer 输出 ID 的意义，到这里才真正落到模型参数上：ID `42` 就选择第 `42` 行，而不是把数字 `42` 本身当成某种语义数值。

```python
embedding_weight = checkpoint.load_tensor("model.embed_tokens.weight")
```

从 Step 01 的读取器出来后，这块权重已经是 NumPy array。`Embedding` 不需要再次理解 Safetensors，只要接住它，并确认 shape 是二维的 `[V,D]`。

<!-- checkpoint: step03-embedding-weight -->

左侧的构造函数只保存 `weight`。这里没有复制整张表，因为真实 Embedding 已经很大；推理层只需要引用 checkpoint 读出的 array。

这也体现了模块边界：

```text
checkpoint.py 负责：参数名 -> NumPy tensor
layers.py     负责：NumPy tensor 怎样参与计算
```

如果把“去哪个分片找参数”的逻辑写进 `Embedding`，后面的每一种层都要重复认识模型目录，代码很快就会缠在一起。

## Embedding 不是矩阵乘法，而是查表

假设 batch 里有两条序列，每条有三个 token：

```text
input_ids [B,S] =
[
  [12, 58, 91],
  [12, 77,  3],
]
```

Embedding 对每个位置做同一件事：拿 token ID 去选择权重的一行。

```text
12 -> weight[12] -> 长度为 D 的向量
58 -> weight[58] -> 长度为 D 的向量
91 -> weight[91] -> 长度为 D 的向量
```

NumPy 的高级索引正好表达了整批查表：

```python
hidden_states = weight[input_ids]
```

`input_ids` 原有的 `[B,S]` 会保留下来，每个整数位置再多出一维 `D`，因此结果是 `[B,S,D]`。

<!-- checkpoint: step03-embedding-lookup -->

左侧 `Embedding.__call__()` 在查表前守住两件事。

第一，token IDs 必须是整数。浮点数 `12.0` 看起来能对应第 12 行，但把连续值偷偷当下标会掩盖上游错误，所以这里不自动转换。

第二，每个 ID 都必须满足：

```text
0 <= token_id < V
```

越界通常意味着 tokenizer 与 checkpoint 不配套，或者序列在别处被破坏。与其让 NumPy 给出一个难以连接到全局数据流的索引错误，不如直接说明它超出了 Embedding 词表。

现在可以把前三章第一次接起来：

```python
import numpy as np

from qwen3_moe import Embedding, Qwen3Tokenizer, SafetensorsCheckpoint

tokenizer = Qwen3Tokenizer.from_file(checkpoint_dir / "tokenizer.json")
checkpoint = SafetensorsCheckpoint.from_directory(checkpoint_dir)
embedding = Embedding(checkpoint.load_tensor("model.embed_tokens.weight"))

input_ids = np.array([tokenizer.encode("MoE 是什么？")], dtype=np.int64)
hidden_states = embedding(input_ids)

print(input_ids.shape)      # [B,S]
print(hidden_states.shape)  # [B,S,D]
```

到这里，文字已经变成模型内部会一路传递的 hidden states。后面的层不再看 token 字符串，也不会知道某个位置原来是汉字还是标点；它们只处理这些向量。

## 为什么每个 decoder layer 都需要 RMSNorm

hidden states 接下来会连续穿过很多层。每次 Attention、MoE 和 residual add 都会改变向量数值，如果尺度不断漂移，后续点积和投影就会越来越难稳定。

Qwen3 在 Attention 和 MoE 前使用 RMSNorm。对一个位置的 hidden vector `x [D]`，它计算：

```text
rms(x) = sqrt(mean(x^2) + epsilon)

RMSNorm(x) = x / rms(x) * weight
```

这里与 LayerNorm 有一个重要区别：RMSNorm 不减均值，只根据平方均值调整整体尺度，再乘上一组学习到的 `weight [D]`。

虽然输入通常是 `[B,S,D]`，公式只沿最后一维 `D` 工作：

```text
[B,S,D]
     ^
     `-- 每个 token 的 D 个数独立计算 mean square
```

batch 之间不会混在一起，不同 token 位置之间也不会互相求平均。输出 shape 仍然是 `[B,S,D]`，这样才能继续送进 Attention 或 MoE。

<!-- checkpoint: step03-rmsnorm -->

左侧实现里，`axis=-1` 表示始终沿最后一维计算，`keepdims=True` 则把结果保留成 `[B,S,1]`。这样归一化系数可以通过广播乘回每个 hidden dimension。

中间计算先转成 `float32`。真实 checkpoint 常用 BF16 保存权重和激活，但平方、求均值和开平方对精度更敏感。课程里的 CPU 实现用 `float32` 完成这段计算，先保证数值过程清楚稳定。

`epsilon` 来自 `config.rms_norm_eps`。它不是可有可无的装饰：当一个向量接近全零时，`epsilon` 防止除数变成零。

真实模型里会多次看到一维 Norm 权重，例如：

```text
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
model.norm.weight
```

它们使用同一个 `RMSNorm` 计算，只是各自加载不同参数。

## Linear 改变最后一维的表示空间

Embedding 负责查表，RMSNorm 负责调整尺度，而后续绝大多数学习到的变换都由 Linear 完成。

例如 Attention 要从 hidden states 生成 Query：

```text
hidden_states [B,S,D]
  -> q_proj Linear
  -> query [B,S,O]
```

一块 Linear 权重按 Safetensors 中的布局保存为 `[O,I]`：

```text
I = input features
O = output features
```

输入最后一维必须是 `I`。计算时使用权重转置：

```text
output = input @ weight.T

[B,S,I] @ [I,O] -> [B,S,O]
```

与 RMSNorm 一样，`B` 和 `S` 不参与权重维度匹配。Linear 只把每个位置最后面的向量从 `I` 维投影到 `O` 维。

<!-- checkpoint: step03-linear -->

左侧 `Linear` 同时接受可选 bias。Qwen3 的很多投影没有 bias，但把这条基础计算写完整并不会改变无 bias 路径：当 `bias is None` 时，直接返回矩阵乘法结果。

现在可以用 Step 01 的真实权重做一次局部投影：

```python
from qwen3_moe import Linear, RMSNorm

norm = RMSNorm(
    checkpoint.load_tensor("model.layers.0.input_layernorm.weight"),
    config.rms_norm_eps,
)
q_proj = Linear(
    checkpoint.load_tensor("model.layers.0.self_attn.q_proj.weight")
)

normalized = norm(hidden_states)
query_values = q_proj(normalized)
```

这还不是完整 Attention。Query 之后还要拆 heads，Key/Value 也要各自投影，并加入 QK Norm、RoPE、causal mask 和注意力加权。本章只证明最底层的数据动作已经成立：一个 `[B,S,D]` tensor 可以沿最后一维被归一化，再被真实权重投影到新的表示空间。

## 三个基础层怎样被后续模块复用

<!-- checkpoint: step03-package -->

最后从 `__init__.py` 导出三个类。包入口现在可以提供：

```python
from qwen3_moe import Embedding, Linear, RMSNorm
```

它们在完整推理地图中的位置不是并排出现一次，而是会被反复复用：

```text
input_ids [B,S]
  -> Embedding
  -> hidden_states [B,S,D]
  -> RMSNorm
  -> Attention 内部多个 Linear
  -> residual add
  -> RMSNorm
  -> Router / Expert 内部多个 Linear
  -> residual add
  -> ...
  -> Final RMSNorm
  -> LM Head Linear
  -> logits [B,S,V]
```

先抓住三个 shape 规则：

1. `Embedding` 把 `[B,S]` 变成 `[B,S,D]`，新增的是 hidden dimension。
2. `RMSNorm` 只沿最后一维计算，输入输出 shape 不变。
3. `Linear` 只替换最后一维，`[B,S,I]` 变成 `[B,S,O]`。

这三条规则会贯穿后面所有章节。看到复杂模块时，先把它拆回“查表、归一化、最后一维投影”，很多 shape 就不再神秘。

回到 Step 00，本章刚刚完成了从整数世界进入向量世界的一步：

```text
prompt
  -> token IDs [B,S]          Step 02
  -> hidden states [B,S,D]    Step 03 Embedding
  -> normalized / projected   Step 03 RMSNorm + Linear
  -> 带位置信息的 Q/K          下一章
  -> Attention / MoE / ...
  -> logits
```

下一章会实现 RoPE。Attention 在比较 Query 和 Key 之前，必须知道每个 token 位于序列的什么位置。我们会从 position IDs 出发，构造旋转频率，并把位置信息写进 Q 和 K。忘记基础层为什么存在时，就回到 [Step 00 的完整推理地图](../step00/) 看它们怎样支撑整条 forward 链路。
