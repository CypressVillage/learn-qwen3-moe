# Step 11：支持 prefill 与 decode 的 Cached Attention

<!-- checkpoint: step11-ready -->

Step 10 已经准备好按层保存 `[B,Hkv,S,Dh]` Key/Value 的 `KVCache`，但现有 `Qwen3Attention` 还完全不知道它的存在。本章把两者接起来，让 Attention 同时支持两种执行方式：

```text
prefill：一次输入整段 prompt，建立初始缓存
decode：只输入新 token，让 Query 读取历史缓存
```

数据流将从原来的方形 Attention：

```text
Q [B,Hq,S,S] x K/V from same S tokens
```

变成允许 query length 与 key length 不同：

```text
Q_new [B,Hq,Snew,Dh]
K_all [B,Hkv,Scached+Snew,Dh]
V_all [B,Hkv,Scached+Snew,Dh]
scores [B,Hq,Snew,Scached+Snew]
```

## Attention 怎样知道自己属于哪一层

`KVCache` 按 Decoder Layer 保存状态，所以 Attention 除了接收 `cache`，还必须接收 `layer_index`：

```python
attention.cached(
    hidden_states,
    position_ids,
    cache=cache,
    layer_index=7,
)
```

`cached()` 把两个参数设为必填：

```text
attention(...)         普通无缓存 Attention
attention.cached(...)  读写指定层的 KV Cache
```

保留无缓存路径很重要。Step 05 的完整 prompt Attention 仍然是合法用法，小张量理解和一次性 prefill 不必强制创建缓存。Cached Attention 是在原接口上的增量能力，而不是另写一套数学实现。

## 为什么在 RoPE 之后写入缓存

<!-- checkpoint: step11-cache-update -->

当前 Attention 的顺序是：

```text
hidden states
  -> Q/K/V projection
  -> QK Norm
  -> RoPE(Q,K)
  -> cache.update(rotated K, V)
  -> scaled dot-product attention
```

缓存保存已经应用 RoPE 的 Key。原因是历史 token 的绝对位置固定，旋转后的 Key 在后续 decode 中不会变化。若缓存旋转前的 Key，每轮还要重新为全部历史位置应用 RoPE，失去缓存的一部分意义。

Value 不参与 RoPE，直接与旋转后的 Key 一起追加：

```python
query, key = self._apply_positions(query, key, position_ids)
key, value = cache.update(layer_index, key, value)
```

注意 Query 不进入缓存。当前 Query 在这一轮完成 Attention 后就不再需要；下一轮会为下一个 token 产生新的 Query。

prefill 时缓存原本为空：

```text
new K/V [B,Hkv,S,Dh] -> cache -> complete K/V [B,Hkv,S,Dh]
```

decode 时缓存已有 prompt：

```text
cached K/V [B,Hkv,Scached,Dh]
new K/V    [B,Hkv,Snew,Dh]
  -> append
complete   [B,Hkv,Scached+Snew,Dh]
```

GQA 的 `np.repeat` 仍然发生在缓存更新之后。这样缓存只保存 `Hkv` 个头，真正计算 Attention 时才临时扩展到 `Hq` 个头。

## 方形 causal mask 为什么不再够用

<!-- checkpoint: step11-causal-mask -->

无缓存 prefill 中，Query 和 Key 都来自同一段长度 `S`，mask 是方形 `[S,S]`：

```text
      K0 K1 K2 K3
Q0     ·  x  x  x
Q1     ·  ·  x  x
Q2     ·  ·  ·  x
Q3     ·  ·  ·  ·
```

decode 时假设缓存已有 4 个 token，本次输入 1 个新 token。Query length 是 1，Key length 是 5：

```text
      K0 K1 K2 K3 K4
Q4     ·  ·  ·  ·  ·
```

mask 必须是 `[1,5]`，而不是 `[1,1]`。当前 Query 可以读取所有历史 Key 和自己的 Key，没有未来位置需要遮挡。

如果一次 decode 输入两个新 token，缓存长度为 4：

```text
      K0 K1 K2 K3 K4 K5
Q4     ·  ·  ·  ·  ·  x
Q5     ·  ·  ·  ·  ·  ·
```

矩形上三角线的起点要向右平移 `cached_length`：

```python
cached_length = key_length - query_length
k = cached_length + 1
```

随后构造：

```python
np.triu(np.ones((query_length, key_length), dtype=bool), k=k)
```

当没有缓存时，`key_length == query_length`，继续使用 Step 05 的普通方形 mask。当有历史 Key 时，才切换到矩形 mask。两条路径最终都广播到 Attention scores 的 `[B,Hq,Squery,Skey]`。

## 用同一组权重比较两条路径

下面的关键数值事实是：一次处理完整序列，与先 prefill 前缀再 cached decode 后缀，最后位置输出应一致。

```python
import numpy as np

from qwen3_moe import KVCache, Qwen3Attention

# attention 使用同一份 config 与权重构造
full_positions = np.asarray([[0, 1, 2]])
full_output = attention(hidden_states, full_positions)

cache = KVCache(num_hidden_layers=1)
attention.cached(
    hidden_states[:, :2],
    np.asarray([[0, 1]]),
    cache=cache,
    layer_index=0,
)
decode_output = attention.cached(
    hidden_states[:, 2:],
    np.asarray([[2]]),
    cache=cache,
    layer_index=0,
)

np.testing.assert_allclose(
    decode_output[:, -1],
    full_output[:, -1],
    rtol=1e-5,
    atol=1e-5,
)
```

这个等价性依赖三件事同时正确：

1. 历史 Key 已经带有原来的 RoPE 绝对位置。
2. 新 token 使用接续的 position ID，而不是重新从 0 开始。
3. 矩形 causal mask 允许新 Query 读取全部历史位置。

本章由调用者显式提供 position IDs。完整模型自动从 `cache.sequence_length` 生成位置属于 Step 12。

## Cached Attention 回到完整推理地图

现在的数据流是：

```text
prompt hidden states
  -> project Q/K/V
  -> RoPE positions 0..S-1
  -> save K/V in layer cache
  -> square causal Attention

new hidden state
  -> project Q/K/V
  -> RoPE position S
  -> append K/V to layer cache
  -> rectangular Attention over positions 0..S
```

先记住本章的六条规则：

1. 普通 `__call__` 保留无缓存路径，`cached()` 要求传入 cache 与 layer index。
2. 缓存保存应用 RoPE 后的 Key，以及未旋转的 Value。
3. Query 不缓存，因为每轮只需要当前新位置的 Query。
4. GQA Key/Value 在未 repeat 的 `[B,Hkv,S,Dh]` 形态下缓存。
5. cached decode 的 Query 和 Key 长度可以不同，因此 causal mask 必须允许矩形 shape。
6. cached path 与 full-sequence path 在相同位置的输出应数值一致。

Attention 已经具备缓存能力，但 Decoder Layer 和完整模型还没有传入 `cache` 与 `layer_index`，也不会自动生成接续位置。下一章会把缓存贯穿全部层，实现完整模型级的 prefill 与 cached decode。需要回顾缓存容器，可以返回 [Step 10 的 KV Cache](../step10/)；需要查看最初的 GQA 计算，可以返回 [Step 05 的 Attention](../step05/)。
