# Step 04：把 token 的位置写进 Query 和 Key

<!-- checkpoint: step04-ready -->

上一章让 token IDs 进入 Embedding，并准备了 RMSNorm 与 Linear。现在继续沿着 Attention 内部的数据流往前走：

```text
hidden_states [B,S,D]
  -> Q / K Linear projection
  -> reshape into heads [B,H,S,Hd]
  -> RoPE
  -> 带位置信息的 Query / Key
```

这一章实现 `rope.py`。它不读取新权重，也不计算完整 Attention，只负责两件事：从 `position_ids [B,S]` 构造每个位置的旋转角度，再用这些角度同时旋转 Query 和 Key。

先记住本章的四个维度：

```text
B  = batch size
S  = sequence length
H  = attention head count
Hd = head_dim
```

在 `Qwen3-30B-A3B` 中，`Hd = 128`。Query 有 32 个 heads，Key 只有 4 个 heads，但每个 head 的最后一维都是 128。RoPE 只旋转这个 `Hd`，不会把不同 batch、token 或 head 混在一起。

## 为什么只有 token 向量还不够

Tokenizer 和 Embedding 告诉模型“这是什么 token”，却没有告诉它“这个 token 在哪里”。同一个 token 出现在序列开头和结尾时，Embedding 查到的是同一行权重。

如果 Attention 直接计算 Query 与 Key 的点积，它只能比较内容，无法仅凭这一步区分顺序：

```text
"猫 追 老鼠"
"老鼠 追 猫"
```

两句话包含相近的 token，但位置关系完全不同。Attention 需要一种方式，让位置进入 Q/K 点积，同时又不为每个可能的位置学习一张巨大参数表。

[[RoPE]]，也就是 Rotary Position Embedding，选择旋转每个 attention head。位置不同，旋转角度不同；Query 和 Key 使用同一套旋转规则，因此它们的点积会自然带上相对位置差。

```text
position 2 的 Query 与 position 5 的 Key
  -> 各自按位置旋转
  -> 点积中保留 5 - 2 的相对距离信息
```

本章不推导完整三角公式，只抓住实现所需的事实：每两个坐标组成一个二维平面，位置决定这个平面旋转多少角度。

## 先为不同维度准备不同频率

Qwen3 的 `config.json` 已经给出 RoPE 的两个关键参数：

```json
{
  "head_dim": 128,
  "rope_theta": 1000000.0
}
```

`head_dim` 决定要旋转多少个坐标。因为坐标要两两组成平面，所以 Step 01 已经要求它必须是偶数。

`rope_theta` 决定频率跨度。不是所有旋转平面都用同一个速度：前面的维度随位置变化得快，后面的维度变化得慢，让一个 head 能同时表达短距离和长距离关系。

对偶数维索引 `i = 0, 2, 4, ...`，先计算：

```text
inverse_frequency[i] = 1 / theta^(i / head_dim)
```

`head_dim = 128` 时，只需要 64 个逆频率，因为每个频率会服务一个二维旋转平面。

<!-- checkpoint: step04-inverse-frequencies -->

左侧 `RotaryEmbedding.__init__()` 用 `np.arange(0, head_dim, 2)` 建立这些偶数维索引，再一次性算出 `inverse_frequencies [Hd/2]`。

这组数不是模型训练出的权重。只要 `head_dim` 和 `rope_theta` 确定，它就可以由公式重新构造，所以 `rope.py` 不需要向 checkpoint 请求任何 tensor。

这里把逆频率保存在对象里，是因为每一层、每次生成都会使用同样的频率规则。位置会变化，频率表本身不会变化。

## position IDs 怎样变成旋转角度

处理一段长度为 4 的 prompt 时，最简单的 position IDs 是：

```text
position_ids [B,S] = [[0, 1, 2, 3]]
```

每个位置乘上所有逆频率：

```text
angles[b, s, i] = position_ids[b, s] * inverse_frequencies[i]

[B,S,1] * [Hd/2] -> [B,S,Hd/2]
```

位置 0 乘任何频率都是 0，所以它的 cosine 全为 1、sine 全为 0，旋转后向量保持不变。后续位置则按各自的位置编号逐渐转动。

当前实现采用 Qwen3/Transformers 使用的 half-split 布局。64 个角度会复制一次，从 `[B,S,Hd/2]` 扩成 `[B,S,Hd]`：

```text
[a0, a1, ..., a63]
  -> [a0, a1, ..., a63, a0, a1, ..., a63]
```

接着分别取 cosine 和 sine，得到：

```text
cosine [B,S,Hd]
sine   [B,S,Hd]
```

<!-- checkpoint: step04-position-angles -->

左侧 `RotaryEmbedding.__call__()` 要求 position IDs 是二维整数 tensor，并拒绝负位置。这样 batch 中每条序列都能拥有自己的位置编号，同时让 shape 直接对齐后面 Query/Key 的 `B` 与 `S`。

中间计算使用 `float32`。position ID 可能在长上下文里变得很大，角度和三角函数比普通索引更需要稳定精度；最终结果会在乘法时广播到 Q/K。

可以先单独构造这一部分：

```python
import numpy as np

from qwen3_moe import Qwen3MoeConfig, RotaryEmbedding

config = Qwen3MoeConfig.from_json(checkpoint_dir / "config.json")
rope = RotaryEmbedding(config.head_dim, config.rope_theta)

position_ids = np.arange(4, dtype=np.int64)[None, :]
cosine, sine = rope(position_ids)

print(cosine.shape)  # (1, 4, 128)
print(sine.shape)    # (1, 4, 128)
```

注意，`max_position_embeddings` 不是用来创建一张固定长度的 cosine/sine 大表。这里根据本次输入的 position IDs 即时计算，因此 prefill 可以传 `[0, 1, ..., S-1]`，以后 decode 时也可以直接传下一个真实位置 `S`。

## 把一个 head 拆成旋转平面

有了 cosine 和 sine，还需要明确向量中的哪些坐标互相配对。

为便于观察，假设 `Hd = 4`：

```text
x = [x0, x1, x2, x3]
```

Qwen3 使用的布局把前后两半配对：`x0` 与 `x2` 是一个平面，`x1` 与 `x3` 是另一个平面。将每个平面旋转 90 度方向所需的辅助向量是：

```text
rotate_half(x) = [-x2, -x3, x0, x1]
```

也就是先把最后一维平均切成两半，再交换位置，并给原来的后半部分加负号。

<!-- checkpoint: step04-rotate-half -->

左侧 `_rotate_half()` 只操作最后一维。无论前面是 `[B,H,S]` 还是其他批量维度，它都不会改变这些维度。

现在二维旋转可以写成一个适合整批 tensor 的公式：

```text
rotated(x) = x * cosine + rotate_half(x) * sine
```

当 angle 为 0 时，`cosine = 1`、`sine = 0`，结果仍是 `x`。当位置变化时，同一个平面里的两个坐标会互相混合，但向量的 shape 不变。

:::principle RoPE 的点积为什么只留下相对位置

把一个二维坐标对在位置 `p` 的旋转写成矩阵 `R(p * omega)`。Query 位于 `p`，Key 位于 `r` 时，旋转后的点积是：

```text
(R(p * omega) q)^T (R(r * omega) k)
= q^T R(p * omega)^T R(r * omega) k
= q^T R((r - p) * omega) k
```

第二步使用了旋转矩阵的性质：转置等于反向旋转，而连续两次旋转的角度可以相加。于是 Query 的绝对角度 `p * omega` 与 Key 的绝对角度 `r * omega` 合并后，只剩位置差 `(r - p)`。

真实 head 不只包含一个二维平面，而是用多组 `omega` 同时旋转。高频坐标对位置变化更敏感，低频坐标覆盖更长距离；Attention 点积因此能从多个尺度感知相对位置。

`_rotate_half()` 使用前半与后半配对，而不是把相邻元素配对，但它实现的仍是同一组独立二维旋转。

:::endprinciple

## 同一套位置旋转应用到 Query 和 Key

Attention 拆 heads 后，Query 和 Key 的 shape 分别是：

```text
query [B,Hq,S,Hd]
key   [B,Hkv,S,Hd]
```

Qwen3 使用 GQA，所以 `Hq = 32`、`Hkv = 4`。RoPE 不能要求 Query 和 Key 的 head 数相同；它只要求两者的 batch、sequence length 和 `head_dim` 相同。

而刚才生成的 cosine/sine 是 `[B,S,Hd]`，没有 head 维。应用前插入一个长度为 1 的维度：

```text
[B,S,Hd] -> [B,1,S,Hd]
```

NumPy 广播会让同一个 token 位置的旋转参数作用到所有 Query heads 和所有 KV heads：

```text
query [B,32,S,Hd] * cosine [B,1,S,Hd]
key   [B, 4,S,Hd] * cosine [B,1,S,Hd]
```

<!-- checkpoint: step04-apply -->

左侧 `apply_rotary_position_embedding()` 先检查 Q/K 是否共享 `B`、`S` 和 `Hd`，但有意允许 head count 不同。然后分别应用同一个旋转公式，返回 shape 不变的 `rotated_query` 与 `rotated_key`。

用一个很小的 tensor 可以把整条链路接起来：

```python
rng = np.random.default_rng(0)
query = rng.standard_normal((1, 32, 4, config.head_dim), dtype=np.float32)
key = rng.standard_normal((1, 4, 4, config.head_dim), dtype=np.float32)

rotated_query, rotated_key = apply_rotary_position_embedding(
    query,
    key,
    cosine,
    sine,
)

print(rotated_query.shape)  # (1, 32, 4, 128)
print(rotated_key.shape)    # (1, 4, 4, 128)
```

旋转不会改变 head 数、序列长度或 head dimension。它改变的是每个位置上 Q/K 坐标的方向，让下一步点积能够感知相对位置。

还有一个容易混淆的边界：RoPE 只作用于 Query 和 Key，不旋转 Value。位置关系要进入的是 attention score `Q @ K.T`；Value 保存被读取的内容，后面由 attention probabilities 加权汇总。

## RoPE 在完整 Attention 中处于哪里

<!-- checkpoint: step04-package -->

最后从 `__init__.py` 导出 `RotaryEmbedding` 和 `apply_rotary_position_embedding`：

```python
from qwen3_moe import RotaryEmbedding, apply_rotary_position_embedding
```

把本章放回 Step 00 的 Attention 路线：

```text
hidden_states [B,S,D]
  -> RMSNorm                         Step 03
  -> q_proj / k_proj / v_proj        Step 03 的 Linear
  -> split heads [B,H,S,Hd]
  -> QK Norm                         后续 Attention 章节
  -> RoPE(Q, K, position_ids)         Step 04
  -> attention scores = Q @ K.T
  -> causal mask + softmax
  -> weighted Value
```

本章已经完成“把位置写入 Q/K”，但还没有真的让 token 读取前文。缺少的部分包括：加载 Q/K/V 投影和 QK Norm 权重、处理 GQA、构造 causal mask、计算 softmax，并把各个 heads 合回 hidden states。

先记住本章的三条 shape 规则：

1. `position_ids [B,S]` 生成 `cosine/sine [B,S,Hd]`。
2. cosine/sine 插入 head 维后，以 `[B,1,S,Hd]` 广播到任意数量的 Q heads 和 KV heads。
3. RoPE 只旋转最后一维，Query 和 Key 的 shape 都保持不变。

回到整条推理地图，现在数据已经走到：

```text
prompt
  -> token IDs [B,S]             Step 02
  -> hidden states [B,S,D]       Step 03
  -> Query / Key heads           Linear + reshape
  -> 带位置信息的 Query / Key     Step 04 RoPE
  -> 读取前文的 Attention         下一章
  -> MoE / logits / next token
```

下一章会把这些积木组装成完整的 GQA Attention：从真实 Q/K/V 权重开始，加入 QK Norm、RoPE 与 causal mask，第一次让序列中的 token 真正读取前文。需要重新确认 RoPE 在全局中的位置时，可以回到 [Step 00 的完整推理地图](../step00/)。
