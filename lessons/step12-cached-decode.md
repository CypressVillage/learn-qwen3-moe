# Step 12：完整模型的 Cached Decode

<!-- checkpoint: step12-ready -->

Step 11 已经让单个 `Qwen3Attention.cached()` 能把新 Key/Value 追加到指定层，并让新 Query 读取完整历史。真实生成需要把同一个动作穿过所有 Decoder Layers：

```text
new token IDs [B,Snew]
  -> Embedding
  -> layer 0 cached Attention -> update cache[0] -> MoE
  -> layer 1 cached Attention -> update cache[1] -> MoE
  -> ...
  -> final RMSNorm -> LM Head
  -> logits [B,Snew,V]
```

本章实现模型级 `cached()` 路径。它既能用空缓存完成 [[prefill]]，也能用已有缓存执行 [[decode]]：

```text
empty cache + prompt       = cached prefill
filled cache + new tokens  = cached decode
```

## Decoder Layer 传递缓存

<!-- checkpoint: step12-decoder-layer -->

普通 Decoder Layer 的 Attention block 是：

```text
residual
  + Attention(RMSNorm(hidden_states), position_ids)
```

缓存版本只改变 Attention 调用：

```text
residual
  + Attention.cached(
      RMSNorm(hidden_states),
      position_ids,
      cache,
      layer_index,
    )
```

之后的 Sparse MoE 与第二条 residual 完全不变。KV Cache 只优化 Attention 对历史上下文的读取，不会跳过当前 token 在任何层的 MoE 计算。

`layer_index` 由完整模型枚举层列表时提供：

```python
for layer_index, layer in enumerate(self.layers):
    hidden_states = layer.cached(
        hidden_states,
        position_ids,
        cache,
        layer_index,
    )
```

这样第 0 层只更新 `cache[0]`，第 1 层只更新 `cache[1]`。所有层共享一个 `KVCache` 对象，但各自拥有独立槽位。

我们保留原来的 `layer(hidden_states, position_ids)` 无缓存路径。课程中的完整 prefill 数学仍能直接运行，而生成循环会明确使用 `layer.cached()`。

## 绝对位置从缓存长度开始

<!-- checkpoint: step12-positions -->

无缓存模型每次都生成：

```text
position IDs = 0,1,...,S-1
```

这对重复整段 prompt 是正确的，却不适合只输入新 token。假设缓存已经保存 5 个位置，本次输入两个 token：

```text
cache.sequence_length = 5
new sequence length   = 2
position IDs          = 5,6
```

代码把旧缓存长度作为起点：

```python
start = cache.sequence_length
positions = np.arange(start, start + sequence_length)
```

然后广播到 batch：

```text
[5,6] -> [B,2]
```

为什么必须在进入第 0 层前读取长度？执行过程中，每经过一层，该层缓存就会先增长。如果在层循环内部重新读取全局长度，各层会看到不一致的中间状态。正确顺序是：

```text
1. 调用开始，所有层长度都为 Scached
2. 生成一次 position IDs
3. 逐层追加相同的 Snew
4. 调用结束，所有层长度都为 Scached + Snew
```

因此 `cache.sequence_length` 同时是层间一致性检查和绝对位置来源。

## 一个入口同时完成 prefill 与 decode

<!-- checkpoint: step12-model -->

模型级 `cached(token_ids, cache)` 保留普通 forward 的 shape 检查，然后执行：

```python
position_ids = self._cached_position_ids(token_ids, cache)
hidden_states = self.embed_tokens(token_ids)
for layer_index, layer in enumerate(self.layers):
    hidden_states = layer.cached(
        hidden_states, position_ids, cache, layer_index
    )
hidden_states = self.norm(hidden_states)
return self.lm_head(hidden_states)
```

### 第一次调用：prefill

```python
cache = KVCache(config.num_hidden_layers)
prompt_logits = model.cached(prompt_ids, cache)
```

输入和输出：

```text
prompt IDs [B,S]
positions  [B,S] = 0..S-1
logits     [B,S,V]
cache      每层 [B,Hkv,S,Dh]
```

这次仍然要计算整段 prompt，因为缓存一开始什么都没有。prefill 的价值不仅是产生第一个 next-token logits，也是在所有层建立历史 Key/Value。

### 后续调用：decode

```python
decode_logits = model.cached(next_token_ids[:, None], cache)
```

如果旧长度为 `S`：

```text
new IDs    [B,1]
position   [B,1] = S
logits     [B,1,V]
cache      每层从 S 增长到 S+1
```

完整模型仍会让这个新 token 穿过所有 Decoder Layers，但每层 Attention 不再重算旧 token 的 K/V。

## 与完整序列重算对齐

模型级正确性可以用相同原则检查：

```python
full_logits = model(np.asarray([[10, 20, 30]]))

cache = KVCache(config.num_hidden_layers)
model.cached(np.asarray([[10, 20]]), cache)
decode_logits = model.cached(np.asarray([[30]]), cache)

np.testing.assert_allclose(
    decode_logits[:, -1],
    full_logits[:, -1],
    rtol=1e-5,
    atol=1e-5,
)
```

full path 的最后位置与 cached path 的 decode 输出应一致。前者重新处理 `[10,20,30]`，后者复用 `[10,20]` 的每层 K/V，只处理 `[30]`。

由于 NumPy 浮点运算顺序可能略有差异，应使用容差比较，而不是要求每一位二进制完全相同。

## Cached Decode 回到完整推理地图

```text
prompt token IDs [B,S]
  -> model.cached(prompt, empty cache)             prefill
  -> logits [B,S,V] + per-layer K/V length S
  -> choose first next token

next token IDs [B,1]
  -> model.cached(new token, existing cache)       decode
  -> logits [B,1,V] + per-layer K/V length S+1
  -> choose another token
```

先记住本章的六条规则：

1. Decoder Layer 只把 cache 和 layer index 传进 Attention，MoE 与 residual 结构不变。
2. 所有层共享一个 KV Cache，但每层写入自己的槽位。
3. 新位置从调用开始时的 `cache.sequence_length` 接续。
4. 空 cache 加整段 prompt 是 cached prefill；已有 cache 加新 token 是 cached decode。
5. decode 输入 `[B,1]` 时，模型只返回新位置的 `[B,1,V]` logits。
6. cached decode 的对应位置 logits 应与完整序列重算结果数值一致。

模型现在已经能高效地前进一步，但调用者仍需手动完成“选 token、扩一维、再次调用、检查停止条件”。下一章会把这些动作组成自回归生成循环。需要回顾矩形 mask，可以返回 [Step 11 的 Cached Attention](../step11/)；需要回顾 token 选择策略，可以返回 [Step 09](../step09/)。
