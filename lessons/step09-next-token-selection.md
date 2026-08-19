# Step 09：从 logits 选择 next token

<!-- checkpoint: step09-ready -->

上一章已经让整段 prompt 穿过完整 Qwen3 MoE，并得到 `[B,S,V]` vocabulary logits。现在沿着 Step 00 的推理地图向前走一步，把最后位置的分数变成一个真正的 next-token ID：

```text
prompt token IDs [B,S]
  -> Qwen3MoeForCausalLM
  -> logits [B,S,V]
  -> final-position logits [B,V]
  -> greedy argmax 或 temperature sampling
  -> next-token IDs [B]
```

这一章只解决“一次模型调用之后怎样选一个 token”。我们会实现 greedy decoding、temperature softmax 和随机采样，但还不会把新 token 接回模型形成循环，也不会引入 KV Cache。自回归循环属于 Step 13，缓存相关的数据结构与 Attention 改造会从 Step 10 开始。

先区分三个经常混在一起的对象：

```text
logits         模型输出的任意实数分数，不要求和为 1
probabilities  对 logits 做 softmax 后的概率，沿 vocabulary 求和为 1
token ID       greedy 或 sampling 最终选出的一个整数
```

Step 08 的出口仍然是模型空间里的分数；Step 09 才第一次做生成策略决策。同一份模型 logits 可以交给不同选择策略，因此“模型算出了什么”和“我们怎样从中选 token”应保持为两个模块。

## 为什么只读取最后位置

<!-- checkpoint: step09-final-logits -->

prefill 返回 `[B,S,V]`，因为 causal LM 会同时为 prompt 中每个位置产生预测：

```text
logits[:,0,:]    读完 token 0 后，对下一个 token 的预测
logits[:,1,:]    读完 token 0..1 后，对下一个 token 的预测
...
logits[:,-1,:]   读完整段 prompt 后，对 next token 的预测
```

现在要继续完整 prompt，所以真正需要的是最后一项：

```python
return logits[:, -1, :]
```

shape 因而从：

```text
[B,S,V] -> [B,V]
```

这里保留 batch 维和 vocabulary 维，只移除 sequence 维。对于 batch 中的每条 prompt，都会得到一行独立的 vocabulary 分数。

`last_token_logits()` 在切片前检查三个边界：

```text
输入必须是三维 [B,S,V]
S 必须至少为 1
V 必须至少为 1
```

如果 sequence length 为 0，`-1` 没有可指向的位置；如果 vocabulary size 为 0，后续既不能 argmax，也不能形成概率分布。尽早在 logits 边界报错，会比让 NumPy 在更深处产生难懂错误更清楚。

要注意，最后位置不是“词表中最后一个 token”：

```text
logits[:, -1, :]
           ^  sequence 的最后位置

logits[:, :, -1]
           ^  vocabulary 的最后一个 token，含义完全不同
```

Step 09 的所有选择函数都通过 `last_token_logits()` 使用同一条 shape 规则，避免 greedy 和 sampling 各自切出不同位置。

## Greedy decoding：永远选最高分

<!-- checkpoint: step09-greedy -->

最直接的选择方法是 greedy decoding：对每一行 `[V]` logits 找最大值所在的 token ID。

```python
return np.argmax(last_token_logits(logits), axis=-1)
```

假设最后位置的 logits 是：

```text
token ID       0      1      2      3
logit        -0.4    2.1    0.8    1.7
```

最大 logit 是 `2.1`，位于 token ID `1`，所以 greedy 结果为：

```text
next_token_id = 1
```

对 batch 输入：

```text
final logits [B,V]
  -> argmax(axis=-1)
  -> token IDs [B]
```

`axis=-1` 很重要。它表示每条序列都沿自己的 vocabulary 维选择，而不是在整个 batch 的所有数字中只找一个全局最大值。

greedy 有三个直接特点：

1. 相同 logits 总会得到相同 token ID。
2. 不需要先计算 softmax，因为 softmax 保持分数大小顺序。
3. 它只关心第一名，不关心第一名领先第二名多少。

例如下面两组 logits 都会选择 token 0：

```text
[10.0, 0.0, -1.0] -> token 0，优势非常明显
[ 1.0, 0.99, 0.98] -> token 0，三个候选几乎相同
```

第二种情况下模型其实很不确定，但 greedy 仍然固定选择第一名。这让输出稳定、易于调试，却也可能让文本缺少变化，或者在生成循环中反复进入同一种局部选择。

## Temperature 怎样改变概率分布

<!-- checkpoint: step09-temperature -->

如果希望模型在多个合理候选之间做随机选择，需要先把 logits 变成概率。带 [[temperature]] 的 softmax 是：

```text
p_i = exp(logit_i / T) / sum_j exp(logit_j / T)
```

其中 `T` 必须是有限正数：

```text
0 < T < 1    分布更尖锐，更偏向最高分 token
T = 1        标准 softmax
T > 1        分布更平坦，低分 token 获得更多机会
```

代码先缩放最后位置的 logits：

```python
scaled_logits = last_token_logits(logits) / temperature
```

较小的 temperature 会放大 logits 之间的差距，较大的 temperature 会缩小差距。对于任意正 temperature，除法不会改变 logits 的排序，因此如果最后仍然做 argmax，结果与未缩放时相同。temperature 的实际意义在于改变后续 sampling 使用的概率。

### 为什么 softmax 前要减最大值

直接计算 `exp(logit)` 可能溢出。例如 `exp(1000)` 已经远超常见浮点数范围。但 softmax 对所有 logits 同时减去相同常数不敏感：

```text
softmax(x) = softmax(x - c)
```

所以代码选择每行最大值作为 `c`：

```python
shifted_logits = scaled_logits - np.max(
    scaled_logits,
    axis=-1,
    keepdims=True,
)
```

减完以后，每行最大值恰好为 0，其余值都小于等于 0：

```text
最大的 exp(0) = 1
其余 exp(负数) 位于 0 和 1 之间
```

这样就避免了正方向指数溢出。之后再归一化：

```python
unnormalized = np.exp(shifted_logits)
probabilities = unnormalized / np.sum(
    unnormalized,
    axis=-1,
    keepdims=True,
)
```

`keepdims=True` 让每行总和保持 `[B,1]`，从而能广播回 `[B,V]`。最终每条 batch 的 vocabulary 概率之和都是 1。

举一个三 token 的直观例子。假设 logits 为：

```text
[2.0, 1.0, 0.0]
```

temperature 变小时，token 0 的概率会更接近 1；temperature 变大时，三个概率会逐渐靠近。temperature 不会把不可能 token 自动过滤掉，它只调节相对分布的尖锐程度。

本章没有实现 top-k 或 top-p。它们会先截断候选集合，再重新归一化；temperature 则保留完整 vocabulary，只改变各 token 的相对概率。为了让一章只引入一组核心概念，这里先把最基础的 categorical sampling 链路写完整。

为什么 `temperature=0` 不被接受？数学上除以 0 没有定义。工程接口常把 temperature 0 特判成 greedy，但本课程已经提供明确的 `greedy_next_token()`，因此两个行为保持独立：

```text
greedy_next_token(logits)              确定性 argmax
sample_next_token(logits, temperature) 正 temperature 随机采样
```

## 从 categorical distribution 采样

<!-- checkpoint: step09-sampling -->

`next_token_probabilities()` 已经为 batch 中每条 prompt 产生一行 `[V]` 概率。`sample_next_token()` 接下来要从每一行 categorical distribution 中抽取一个 token ID。

NumPy 的 `Generator.choice()` 接收候选数量和对应概率：

```python
rng.choice(vocabulary_size, p=batch_probabilities)
```

当 `vocabulary_size == 4` 时，候选就是整数 `0,1,2,3`。返回值天然就是 vocabulary token ID，不需要再从 token 字符串反查。

代码逐 batch 采样：

```python
[
    rng.choice(vocabulary_size, p=batch_probabilities)
    for batch_probabilities in probabilities
]
```

输入和输出 shape 是：

```text
probabilities [B,V]
  -> 每一行独立抽一次
  -> next-token IDs [B]
```

这里的 Python 循环是有意保留的。NumPy `choice` 的单次 `p=` 参数表示一条概率分布，而 batch 中每一行通常不同。课程实现优先让“每条 prompt 按自己的分布抽一次”清晰可见，不提前引入复杂的向量化采样技巧。

### 为什么 RNG 是参数

函数允许调用者传入 `np.random.Generator`：

```python
rng = np.random.default_rng(7)
token_ids = sample_next_token(logits, temperature=0.8, rng=rng)
```

固定 seed 能让教学示例、调试和数值检查复现同一串随机选择。如果不传入 RNG，函数会创建默认 generator：

```python
rng = np.random.default_rng()
```

此时多次运行可能得到不同 token，这正是 sampling 的预期行为。

“随机”不等于“均匀”。某个 token 概率为 0.7，就应当比概率为 0.1 的 token 更常被抽到；但单次采样仍可能选中后者。概率描述的是大量重复试验中的频率，不是对某一次结果的保证。

## 从模型 logits 得到一个 token ID

现在可以把 Step 08 和 Step 09 接起来：

```python
from pathlib import Path

import numpy as np

from qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3Tokenizer,
    SafetensorsCheckpoint,
    greedy_next_token,
    sample_next_token,
)

checkpoint_dir = Path("Qwen3-30B-A3B")
config = Qwen3MoeConfig.from_json(checkpoint_dir / "config.json")
checkpoint = SafetensorsCheckpoint.from_directory(checkpoint_dir)
tokenizer = Qwen3Tokenizer.from_directory(checkpoint_dir)
model = Qwen3MoeForCausalLM.from_checkpoint(config, checkpoint)

prompt_ids = np.asarray([tokenizer.encode("MoE 是怎样工作的？")])
logits = model(prompt_ids)

greedy_id = greedy_next_token(logits)
sampled_id = sample_next_token(
    logits,
    temperature=0.8,
    rng=np.random.default_rng(7),
)

print(greedy_id.shape)   # (1,)
print(sampled_id.shape)  # (1,)
```

此时 `greedy_id[0]` 或 `sampled_id[0]` 就是一个可交给 tokenizer decode 的 vocabulary ID。但要注意：这段代码只产生一个 token，还没有把它追加回 `prompt_ids` 再运行模型。

当前数据流停在这里：

```text
text
  -> tokenizer.encode
  -> prompt token IDs [B,S]
  -> full-model prefill
  -> logits [B,S,V]
  -> final logits [B,V]
  -> greedy / temperature sampling
  -> one next-token ID per prompt [B]
```

如果立刻重复调用 `greedy_next_token(logits)`，只会从同一份 logits 得到同一个结果。要预测再下一个 token，必须把刚选出的 ID 纳入模型上下文并计算新的 logits。那正是后续 cached decode 与生成循环要解决的问题。

## Token selection 回到完整推理地图

<!-- checkpoint: step09-package -->

最后从 `__init__.py` 导出四个入口：

```python
from qwen3_moe import (
    greedy_next_token,
    last_token_logits,
    next_token_probabilities,
    sample_next_token,
)
```

它们对应一条由底向上的选择链路：

```text
last_token_logits
  -> 从 [B,S,V] 取出 [B,V]

greedy_next_token
  -> 对 [B,V] 做 argmax

next_token_probabilities
  -> temperature scaling + stable softmax

sample_next_token
  -> 从每行 categorical distribution 抽取 token ID
```

把 Step 09 放回 Step 00 的完整地图：

```text
text
  -> Tokenizer                                  Step 02
  -> token IDs [B,S]
  -> Embedding + 48 Decoder Layers              Step 03~08
  -> final RMSNorm + LM Head
  -> logits [B,S,V]                             Step 08
  -> final-position logits [B,V]                Step 09
  -> greedy argmax / temperature sampling
  -> next-token IDs [B]                         Step 09 完成
  -> append token and continue                  后续章节
```

先记住本章的七条规则：

1. 继续完整 prompt 时只使用 `logits[:, -1, :]`，因为它代表读完最后一个输入 token 后的预测。
2. greedy decoding 直接沿 vocabulary 维做 argmax，不需要先计算 softmax。
3. temperature 必须是有限正数；小于 1 会让分布更尖锐，大于 1 会让分布更平坦。
4. 正 temperature 不改变 logits 排序，因此它主要影响 sampling，而不影响 argmax 结果。
5. stable softmax 先减去每行最大值，避免指数溢出，再沿 vocabulary 维归一化。
6. sampling 为 batch 中每条 prompt 使用各自的 categorical distribution，并返回 `[B]` token IDs。
7. 注入带 seed 的 NumPy RNG 可以复现采样，但单次随机结果不等于概率排名。

回到推理主线，我们现在已经能从 prompt 得到第一个新 token：

```text
prompt
  -> prefill logits
  -> choose next-token ID                       Step 09 完成
  -> remember previous Key / Value              下一章
  -> process only the newly selected token
  -> repeat until stopping condition
```

下一章会实现 KV Cache 数据结构，先保存每一层已经计算过的 Key 和 Value，为后续只处理新 token 的 cached Attention 建立边界。需要重新确认 logits 从哪里来时，可以返回 [Step 08 的 Causal LM Prefill](../step08/)；需要查看完整生成链路，可以回到 [Step 00 的推理地图](../step00/)。
