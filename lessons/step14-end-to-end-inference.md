# Step 14：端到端 Qwen3 MoE 推理

<!-- checkpoint: step14-ready -->

前十四个 Step 已经依次实现了完整推理链路中的所有局部模块：

```text
config + Safetensors                         Step 01
text <-> token IDs                           Step 02
Embedding / RMSNorm / Linear                 Step 03
RoPE                                         Step 04
GQA Attention                                Step 05
Sparse MoE                                   Step 06
Decoder Layer                                Step 07
Causal LM prefill -> logits                  Step 08
next-token selection                         Step 09
KV Cache                                     Step 10
Cached Attention                             Step 11
full-model cached decode                     Step 12
autoregressive generation loop               Step 13
```

最后一章不再增加新的矩阵运算，而是把入口和出口都接回普通文本：

```text
model directory + prompt string
  -> load config / weights / tokenizer / model
  -> encode prompt
  -> generate token IDs
  -> decode token IDs
  -> output string
```

这一步看似只是“粘合代码”，却决定了用户能否从一个真实 Qwen3 MoE 模型目录出发，完整走过我们逐章建立的数据流。

## 一个模型目录包含什么

<!-- checkpoint: step14-loading -->

`_load_model_directory()` 接收本地模型目录，并读取三类资产：

```text
config.json                       架构尺寸与推理配置
model.safetensors.index.json      权重名到 shard 的映射
*.safetensors                     实际张量数据
tokenizer.json                    vocabulary、BPE merges、special tokens
```

加载顺序是：

```python
config = Qwen3MoeConfig.from_json(directory / "config.json")
checkpoint = SafetensorsCheckpoint.from_directory(directory)
tokenizer = Qwen3Tokenizer.from_file(directory / "tokenizer.json")
model = Qwen3MoeForCausalLM.from_checkpoint(config, checkpoint)
```

为什么先读取 config？模型只有知道层数、hidden size、head 数、expert 数和各投影 shape，才能解释 Safetensors 中的权重。

为什么 Tokenizer 必须来自同一个目录？Embedding 与 LM Head 都按 vocabulary token ID 排列。若换成另一套 vocabulary，即使字符串看起来相似，同一个整数也可能指向完全不同的 token。

辅助函数把 tokenizer 与模型作为一对返回：

```text
tokenizer 负责 string <-> token IDs
model     负责 token IDs -> logits
```

导入放在函数内部，是为了避免包初始化时形成 `generation -> model -> ... -> generation` 的循环依赖。只有真正调用端到端入口时，才装载这些较高层模块。

## 从 prompt 字符串开始

<!-- checkpoint: step14-text -->

公开入口是：

```python
generate_text(
    model_directory,
    prompt,
    max_new_tokens,
    eos_token_id=None,
    temperature=None,
    seed=None,
    skip_special_tokens=True,
)
```

第一步编码 prompt：

```python
prompt_ids = tokenizer.encode(prompt)
```

Tokenizer 返回 Python `list[int]`，模型需要带 batch 维的 NumPy 数组：

```python
np.asarray([prompt_ids], dtype=np.int64)
```

shape 从：

```text
list length S -> [1,S]
```

空字符串不一定总会编码出 token。若结果为空，模型无法执行 prefill，因此入口会在创建缓存前给出明确错误。

## 生成策略怎样穿过端到端入口

greedy 模式不需要 RNG：

```python
temperature=None
```

sampling 模式在入口处根据 seed 创建一次 generator：

```python
rng = np.random.default_rng(seed)
```

然后把它交给 Step 13 的循环，在所有生成步之间持续复用。相同模型、prompt、temperature 与 seed 会重放相同的采样序列。

所有生成参数原样传给 `generate_token_ids()`：

```python
generated_ids = generate_token_ids(
    model,
    prompt_array,
    max_new_tokens,
    eos_token_id=eos_token_id,
    temperature=temperature,
    rng=rng,
)
```

`eos_token_id` 由调用者显式提供，因为不同使用方式可能把 `<|endoftext|>`、`<|im_end|>` 或其他 special token 作为停止标记。可以从同目录 tokenizer 查询：

```python
tokenizer.token_to_id("<|im_end|>")
```

端到端入口不擅自猜测停止 token，避免把普通文本中的合法 special token 语义混为一谈。

## 为什么返回完整文本

生成循环返回的是 prompt 加 continuation 的完整 token IDs：

```text
[prompt IDs | generated IDs]
```

最后统一 decode：

```python
tokenizer.decode(
    generated_ids[0],
    skip_special_tokens=skip_special_tokens,
)
```

因此 `generate_text()` 返回完整文本，而不是只返回新增后缀。这与 token ID 层的返回值保持一致，也让调用者可以直接打印最终上下文。

默认 `skip_special_tokens=True`，EOS 等控制 token 不会显示在文本里。若正在调试 token 边界，可以设为 `False` 查看它们。

## 运行真实模型目录

<!-- checkpoint: step14-package -->

从包入口导入：

```python
from qwen3_moe import generate_text
```

greedy 示例：

```python
text = generate_text(
    "Qwen3-30B-A3B",
    "MoE 是怎样工作的？",
    max_new_tokens=64,
    eos_token_id=151645,
)
print(text)
```

temperature sampling 示例：

```python
text = generate_text(
    "Qwen3-30B-A3B",
    "用一个比喻解释 KV Cache：",
    max_new_tokens=96,
    eos_token_id=151645,
    temperature=0.8,
    seed=7,
)
print(text)
```

真实 `Qwen3-30B-A3B` 权重很大。本课程实现使用 NumPy 和 CPU，目标是透明展示推理过程，而不是提供生产级吞吐。运行完整模型需要足够内存，速度也会明显慢于成熟推理框架。

## 一次调用内部发生了什么

把 `generate_text()` 展开，可以得到最终完整地图：

```text
prompt string
  -> byte-level BPE Tokenizer
  -> prompt IDs [1,S]
  -> Embedding [1,S,D]
  -> 逐层：RMSNorm
            -> Q/K/V projection + QK Norm + RoPE
            -> GQA causal Attention
            -> residual
            -> RMSNorm
            -> Sparse MoE Router + top-k Experts
            -> residual
  -> final RMSNorm + LM Head
  -> prefill logits [1,S,V]
  -> choose first next token
  -> append token ID
  -> cached decode [1,1] through all layers
  -> choose / append / repeat
  -> EOS or max_new_tokens
  -> Tokenizer.decode
  -> output string
```

其中 prefill 与 decode 共享同一套模型权重，区别只在输入长度、position IDs、KV Cache 和 causal mask shape。

## 从 Step 00 回看完整课程

Step 00 先画地图、创建空文件骨架，没有提前实现任何后续模块。之后每一章都沿真实数据流填入一段：

```text
输入资产 -> 文本编码 -> 张量基础 -> 位置 -> 上下文读取
         -> 专家计算 -> 层组装 -> 完整 logits -> token 决策
         -> 缓存结构 -> 缓存 Attention -> 模型 decode
         -> 生成循环 -> 文本输出
```

现在左侧最终 checkpoint 与 `src/qwen3_moe/` 当前源码完全一致；更早的 checkpoint 都是它的逐行子序列。这意味着课程不是展示十五份互不相干的示例，而是从空骨架累计长成同一个完整推理框架。

最后记住端到端推理的八条规则：

1. config、Safetensors 与 Tokenizer 必须来自同一个模型目录。
2. Tokenizer 的 token IDs 必须与 Embedding 和 LM Head 的 vocabulary 排列一致。
3. prompt 至少要编码出一个 token，才能执行 causal prefill。
4. prefill 建立全部层的 KV Cache，decode 只输入新 token。
5. greedy 不使用 RNG；sampling 用同一个带 seed RNG 贯穿循环。
6. EOS token 由调用者按模型使用方式显式指定。
7. 最终 decode 默认跳过 special tokens，并返回 prompt 加 continuation 的完整文本。
8. 本课程优先保证 NumPy 推理过程清晰正确，不以生产级性能为目标。

主线课程到这里完成。需要重新建立整体视角，可以回到 [Step 00 的完整推理地图](../step00/)；需要追踪生成循环，可以返回 [Step 13](../step13/)；需要定位模型内部任意一个局部模块，可以通过顶部课程导航回到对应章节。
