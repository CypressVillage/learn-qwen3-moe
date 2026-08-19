# Step 13：自回归生成循环

<!-- checkpoint: step13-ready -->

到 Step 12 为止，我们已经拥有生成循环需要的全部局部动作：

```text
model.cached(prompt, empty cache) -> prefill logits
greedy / sampling                -> next-token ID
model.cached(new token, cache)   -> decode logits
```

本章把它们按正确顺序重复执行，直到遇到停止条件：

```text
prefill once
  -> choose token
  -> append token
  -> stop? yes -> return
  -> cached decode only that token
  -> choose token
  -> repeat
```

课程实现先聚焦单条 prompt，输入 shape 固定为 `[1,S]`。batch 生成还要处理不同序列在不同时间结束、padding 与 finished mask，会遮住最核心的自回归数据流。

## Prefill 只发生一次

<!-- checkpoint: step13-loop -->

`generate_token_ids()` 首先复制输入，避免修改调用者持有的数组：

```python
generated = prompt_token_ids.astype(np.int64, copy=True)
```

若 `max_new_tokens == 0`，立即返回 prompt，不创建缓存，也不调用模型。

真正生成时，根据模型层数建立 KV Cache：

```python
cache = KVCache(len(model.layers))
logits = model.cached(generated, cache)
```

这是唯一一次把整段 prompt 送入模型。调用结束后：

```text
logits shape          [1,S,V]
cache sequence length S
generated shape       [1,S]
```

之后每轮都只处理一个新 token，不再把 `generated` 整体送回模型。

## 从当前 logits 选择 token

循环支持两种策略：

```text
temperature is None   -> greedy_next_token(logits)
temperature is float  -> sample_next_token(logits, temperature, rng)
```

`None` 明确表示确定性 greedy，正浮点数表示 [[temperature]] sampling。它复用 Step 09 已经实现并验证过的函数，没有在循环内部复制 softmax 或采样逻辑。

两个选择函数都返回 `[1]`：

```text
logits [1,Scurrent,V]
  -> last position [1,V]
  -> next_token [1]
```

随后扩出 sequence 维并追加到结果：

```python
generated = np.concatenate(
    (generated, next_token[:, None]),
    axis=1,
)
```

shape 变化：

```text
generated [1,S+t] + next_token [1,1]
  -> generated [1,S+t+1]
```

## 为什么先检查 EOS，再做下一次 decode

追加 token 后立即检查：

```python
if eos_token_id is not None and next_token[0] == eos_token_id:
    break
```

EOS 已经是最终输出的一部分，但不需要再把它送进模型预测 EOS 后面的 token。先停止可以省掉一次没有用途的 decode。

若没有命中 EOS，才执行：

```python
logits = model.cached(next_token[:, None], cache)
```

如果当前 token 已经用完 `max_new_tokens` 配额，同样直接结束，不再为一个永远不会选择的后续 token 计算 logits。

注意传给模型的是 `[1,1]` 的 `next_token`，不是不断增长的 `generated`：

```text
错误：model.cached(generated, cache)       历史 token 会被重复追加
正确：model.cached(next_token[:,None], cache) 只追加新位置
```

每次 decode 后 cache 长度增加 1，下一轮 logits 也只包含新位置 `[1,1,V]`。`last_token_logits()` 对 prefill 的 `[1,S,V]` 和 decode 的 `[1,1,V]` 都适用。

:::principle generated、cache 与 logits 为什么总会短暂错开一个 token

prefill 完成后，prompt 的每个 token 都已经进入模型：

```text
generated length = S
cache length     = S
current logits   = 预测位置 S
```

从 logits 选出 `tS` 并追加到 `generated` 后，这个 token 已经属于输出，却还没有经过模型：

```text
generated length = S + 1
cache length     = S
```

只有执行 `model.cached(tS, cache)`，缓存才增长到 `S+1`，返回的 logits 则预测位置 `S+1`。因此循环始终在“选择一个 token”和“让这个 token 进入模型”之间交替。

这个状态不变量解释了两个常见错误：把完整 `generated` 再次传入会把历史重复追加；已经达到 EOS 或生成上限后继续 decode，则会为一个永远不会被选择的后续 token 做无用计算。

:::endprinciple

## 两个停止条件

循环最多运行 `max_new_tokens` 次。这个上限不是最终总长度，而是 prompt 之外允许新增的 token 数：

```text
prompt length = 20
max_new_tokens = 8
最终长度最多 = 28
```

提前停止条件是 EOS：

```text
达到 eos_token_id      提前结束
未达到 EOS             最多生成 max_new_tokens
eos_token_id is None   只依赖长度上限
```

`max_new_tokens` 必须是非负整数。负数没有清晰语义；布尔值虽然在 Python 中属于整数子类，也被显式拒绝，避免 `True` 被意外解释为生成一个 token。

## Greedy 与可复现采样

确定性生成：

```python
output_ids = generate_token_ids(
    model,
    prompt_ids,
    max_new_tokens=32,
    eos_token_id=eos_id,
)
```

temperature sampling：

```python
output_ids = generate_token_ids(
    model,
    prompt_ids,
    max_new_tokens=32,
    eos_token_id=eos_id,
    temperature=0.8,
    rng=np.random.default_rng(7),
)
```

同一个 RNG 会在循环中持续前进，从而产生一串随机选择。不能每轮重新 `default_rng(7)`，否则每一步都会从同一个随机状态起点采样。

## 用一个可控模型观察循环

生成函数只依赖模型的两个公开特征：

```text
model.layers             用来确定缓存层数
model.cached(ids, cache) 返回 logits
```

因此可以用小型可控对象验证调用顺序：

```python
class TinyModel:
    layers = [object()]

    def __init__(self):
        self.calls = []

    def cached(self, token_ids, cache):
        self.calls.append(token_ids.copy())
        logits = np.zeros((1, token_ids.shape[1], 5))
        logits[:, -1, len(self.calls)] = 10.0
        return logits

model = TinyModel()
result = generate_token_ids(
    model,
    np.asarray([[0, 1]]),
    max_new_tokens=3,
)

print([call.shape for call in model.calls])
# [(1, 2), (1, 1), (1, 1)]
```

第一次调用是 prompt prefill；之后每次都只有一个 token。生成 3 个 token 只需要 1 次 prefill 和 2 次后续 decode，因为最后一个 token 选出后不再需要预测下一个位置。

## 生成循环回到完整推理地图

<!-- checkpoint: step13-package -->

从包入口导出：

```python
from qwen3_moe import generate_token_ids
```

完整循环现在已经闭合：

```text
prompt IDs
  -> cached prefill
  -> logits
  -> choose next token
  -> append to output
  -> EOS? stop
  -> cached decode new token only
  -> repeat
```

先记住本章的七条规则：

1. prefill 只执行一次，用整段 prompt 建立每层 KV Cache。
2. 每轮都从当前 logits 的最后位置选择一个 token。
3. 结果数组追加 `[1,1]`，模型下一轮也只接收这个新 token。
4. 命中 EOS 后先停止，不再做无用的 decode。
5. `max_new_tokens` 限制新增长度，不包含 prompt 长度。
6. `temperature=None` 表示 greedy；正 temperature 表示随机采样。
7. 同一个 RNG 应贯穿整个循环，才能得到可复现的随机序列。

到这里，模型内部的生成链路已经完整。最后一章会把模型目录、config、Safetensors、Tokenizer、prompt 文本、生成循环与 decode 串成一个端到端入口，让输入和输出都回到普通字符串。需要回顾模型级缓存，可以返回 [Step 12](../step12/)；需要回顾选择策略，可以返回 [Step 09](../step09/)。
