# Step 10：KV Cache 数据结构

<!-- checkpoint: step10-ready -->

上一章已经能从完整 prompt 的最后位置 logits 选出一个 next-token ID。若要继续生成，最直接的方法是把新 token 追加到 prompt 后，再把整段序列重新送进模型：

```text
[t0,t1,t2] -> model -> t3
[t0,t1,t2,t3] -> model -> t4
[t0,t1,t2,t3,t4] -> model -> t5
```

这样虽然正确，却会反复计算旧 token 的 Attention Key 和 Value。Step 10 先建立 [[KV Cache]] 的数据边界，让每个 Decoder Layer 都能记住已经算过的状态：

```text
prompt prefill
  -> layer 0 Key/Value [B,Hkv,S,Dh]
  -> layer 1 Key/Value [B,Hkv,S,Dh]
  -> ...
  -> layer L-1 Key/Value [B,Hkv,S,Dh]

one new token
  -> new Key/Value [B,Hkv,1,Dh]
  -> append to every layer cache
```

本章只实现缓存容器，不修改 Attention，也不让模型真正使用缓存。Step 11 才会把它接进 GQA Attention，Step 12 再贯穿完整模型。

## 为什么缓存 Key 和 Value

Attention 对当前位置的 Query，需要读取所有历史位置的 Key 和 Value：

```text
Attention(Q_new, K_history, V_history)
```

生成下一个 token 时，旧 token 的隐藏状态没有变化，它们投影出的 Key 和 Value 也不会变化。因此可以保存历史结果，只为新 token 计算一次投影：

```text
不缓存：每轮重新计算 K0..Kt 与 V0..Vt
有缓存：复用 K0..K(t-1)、V0..V(t-1)，只计算 Kt、Vt
```

我们不缓存 Query。旧 Query 只用于生成旧位置输出，预测新 token 时只需要当前新位置的 Query。

:::principle KV Cache 节省了什么，又为什么没有让 Attention 变成 O(1)

没有缓存时，每生成一个 token，都要让不断增长的完整前缀重新经过 Q/K/V 投影。使用缓存后，旧位置的 Key 和 Value 不再投影，只为新 token 计算新的 Q/K/V。

但新 Query 仍然要与全部历史 Key 做点积，并用全部历史 Value 加权。因此单个 decode step 的 Attention 工作量仍随当前上下文长度 `S` 线性增长，而不是常数时间。

缓存本身大约保存下面数量的元素：

```text
2 * L * B * Hkv * S * Dh
```

`2` 来自 Key 和 Value，`L` 是层数。GQA 只缓存 `Hkv` 个头，而不是扩展后的 `Hq` 个头，所以占用会降为按 Query heads 缓存时的 `Hkv / Hq`。当前配置 `Hkv=4`、`Hq=32`，缓存量只有后者的八分之一；重复到 32 heads 只发生在当轮 Attention 计算时。

:::endprinciple

## 每层拥有独立槽位

<!-- checkpoint: step10-storage -->

Qwen3 的每个 Decoder Layer 都有自己的 Attention 投影权重，所以同一个 token 在不同层产生的 Key/Value 不同。`KVCache` 因而按 `num_hidden_layers` 建立两组槽位：

```python
self._keys = [None] * num_hidden_layers
self._values = [None] * num_hidden_layers
```

刚创建时，每一层都还没有执行 prefill，所以槽位是 `None`。经过一次完整 prompt 后，结构会变成：

```text
layer 0 -> key_0 [B,Hkv,S,Dh], value_0 [B,Hkv,S,Dh]
layer 1 -> key_1 [B,Hkv,S,Dh], value_1 [B,Hkv,S,Dh]
...
```

这里保存的是 GQA 尚未扩展的 KV heads，shape 为 `[B,Hkv,S,Dh]`，而不是 Query heads 的 `[B,Hq,S,Dh]`。因为多个 Query heads 会共享同一组 Key/Value，缓存扩展前的张量可以避免重复占用内存。

构造函数要求层数为正整数。层索引也集中经过 `_validate_layer_index()` 检查，避免负数被 Python 当作“从末尾读取”，或把状态写进不存在的层。

## 读取一层缓存

<!-- checkpoint: step10-lookup -->

`get(layer_index)` 返回两种结果：

```text
None                  这一层还没有缓存
(cached_key, value)   这一层已经完成过至少一次更新
```

Key 和 Value 必须成对出现。实现内部虽然使用两张列表，但公开接口不会让调用者只拿到其中一个。后续 Attention 可以用同一个分支区分首次 prefill 与后续 decode：

```python
cached = cache.get(layer_index)
if cached is None:
    # 第一次写入
else:
    # 追加到历史状态
```

## 沿 sequence 维追加

<!-- checkpoint: step10-append -->

新 Key/Value 进入缓存前必须满足：

```text
都是四维 [B,Hkv,Snew,Dh]
Key 与 Value shape 完全相同
Snew 至少为 1
```

首次更新直接保存张量。后续更新则沿 `axis=2` 拼接：

```python
key = np.concatenate((cached_key, key), axis=2)
value = np.concatenate((cached_value, value), axis=2)
```

为什么是 `axis=2`？各维含义是：

```text
axis 0  batch
axis 1  KV heads
axis 2  sequence
axis 3  head dimension
```

假设 prefill 保存 4 个 token，decode 再送入 1 个 token：

```text
[B,Hkv,4,Dh] + [B,Hkv,1,Dh] -> [B,Hkv,5,Dh]
```

追加前还会确认 batch、KV head 数和 head dimension 与历史缓存一致。只有 sequence length 可以增长。否则拼接出的状态就不再属于同一条模型执行链路。

`update()` 返回追加后的完整 Key/Value，而不只返回新片段。这样 Step 11 的 Attention 可以立即让当前 Query 读取全部上下文：

```text
new query [B,Hq,Snew,Dh]
  x complete cached key [B,Hkv,Stotal,Dh]
  -> attention scores [B,Hq,Snew,Stotal]
```

## 缓存长度是绝对位置起点

<!-- checkpoint: step10-length -->

`sequence_length` 收集已经写入层的 sequence length：

```python
lengths = {key.shape[2] for key in self._keys if key is not None}
```

空缓存返回 0。所有已写入层长度相同，就返回该长度；若长度不一致则报错。

这个值之后有两个用途：

```text
1. 新 token 的 position IDs 从 cached_length 开始
2. causal mask 知道左侧已有多少历史 Key
```

例如 prompt 长度为 4，下一次只输入一个 token，那么它的绝对位置不是 0，而是 4：

```text
cached length = 4
new position IDs = [4]
```

RoPE 必须使用这个绝对位置，否则每个 decode token 都会被错误地旋转成位置 0。

为什么检查所有已填充层长度一致？一次模型调用结束后，每层都应缓存同样多的 token。如果 layer 0 是 5、layer 1 是 4，说明执行在中途被打断或调用方式错误，继续生成会让层间上下文错位。

## 用小张量观察追加过程

```python
import numpy as np

from qwen3_moe import KVCache

cache = KVCache(num_hidden_layers=2)

prefill_key = np.zeros((1, 2, 3, 4), dtype=np.float32)
prefill_value = np.ones((1, 2, 3, 4), dtype=np.float32)
cache.update(0, prefill_key, prefill_value)
cache.update(1, prefill_key, prefill_value)

print(cache.sequence_length)  # 3

decode_key = np.full((1, 2, 1, 4), 2.0, dtype=np.float32)
decode_value = np.full((1, 2, 1, 4), 3.0, dtype=np.float32)
key, value = cache.update(0, decode_key, decode_value)

print(key.shape)    # (1, 2, 4, 4)
print(value.shape)  # (1, 2, 4, 4)
```

在只更新 layer 0、尚未更新 layer 1 的中间时刻，不应读取 `sequence_length`，因为两层暂时不一致。完整模型会先在调用开始前读取旧长度，再依次更新各层，最后恢复一致状态。

## KV Cache 回到完整推理地图

<!-- checkpoint: step10-package -->

从包入口导出：

```python
from qwen3_moe import KVCache
```

把它放回 Step 00 的地图：

```text
text -> token IDs -> full prompt prefill -> logits -> next token     Step 00~09
                         |
                         +-> save per-layer K/V [B,Hkv,S,Dh]         Step 10

next token -> project new K/V -> append cache -> attend to history  后续章节
```

先记住本章的六条规则：

1. KV Cache 按 Decoder Layer 分开保存，因为每层有独立的 Attention 投影。
2. 缓存 shape 是 `[B,Hkv,S,Dh]`，保留未扩展的 GQA KV heads。
3. Key 与 Value 必须成对、同 shape 保存。
4. 新状态只沿 sequence 维 `axis=2` 追加，其他维必须保持一致。
5. 空缓存长度为 0；完整模型调用前后，各层缓存长度必须一致。
6. cached length 将成为后续 position IDs 和非方形 causal mask 的起点。

现在我们只是有了“记忆容器”，Attention 仍然每次只看本次输入。下一章会修改 GQA Attention：把新 Key/Value 写入缓存，让新 Query 读取历史状态，并把原来的方形 causal mask 改成适配 prefill 与 decode 的矩形 mask。需要回顾为什么 next token 会进入下一轮，可以返回 [Step 09 的 token selection](../step09/)；需要重新查看全链路，可以回到 [Step 00 的推理地图](../step00/)。
