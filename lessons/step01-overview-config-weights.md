# Step 01：先读说明书，再找到第一块权重

<!-- checkpoint: step01-empty-workspace -->

上一章看完了整张推理地图，左侧也留下了一排空文件。

这一章先不急着算 Attention。模型要跑起来，至少得先回答两个更朴素的问题：这台 Qwen3 MoE 到底长什么样？几百亿参数又分别放在哪个文件里？

答案就在下载好的模型目录里：`config.json` 是说明书，`model.safetensors.index.json` 和 16 个 [[Safetensors]] 分片是零件箱。我们先把这两样读懂。

## 模型得先知道自己长什么样

打开 `config.json`，里面有很多字段。架构类名、Transformers 版本、默认 dtype 都能在这里出现，但真正决定模型结构的只有一部分：

```json
{
  "model_type": "qwen3_moe",
  "vocab_size": 151936,
  "hidden_size": 2048,
  "num_hidden_layers": 48,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "num_experts": 128,
  "num_experts_per_tok": 8
}
```

先看这些数字在推理地图上的位置：

```text
token ID 的范围                 <- vocab_size
hidden_states [B,S,D]           <- hidden_size
重复多少次 Decoder Layer        <- num_hidden_layers
Q/K/V 怎样拆成 heads            <- attention heads / KV heads / head_dim
Router 给多少个专家打分         <- num_experts
每个 token 真正跑几个专家       <- num_experts_per_tok
```

这些值不能散落在各个文件里。否则 `attention.py` 以为有 32 个 Query heads，`model.py` 却按另一组数字创建权重，shape 迟早会对不上。

所以先在 `config.py` 里放一个 `Qwen3MoeConfig`。它只是一个数据类，不做矩阵计算，职责是把后面所有模块要共用的结构参数放在同一个地方。

<!-- checkpoint: step01-config-contract -->

左侧现在出现了配置字段。暂时不用逐个背，先抓住三组：模型有多宽、有多深；Attention 怎样分头；MoE 有多少专家、每个 token 选几个。

`extra_fields` 也值得看一眼。真实 `config.json` 往往比这一课需要的字段更多。我们不因为暂时没用到某个字段就拒绝整份配置，而是把它留下来，等后面的章节需要时再接进正式结构。

<!-- checkpoint: step01-config-validation -->

## 先守住配置的结构关系

字段摆好了，但一组数字能塞进数据类，不代表它们真的能组成一台 Qwen3 MoE。每次创建配置实例后，`__post_init__()` 都会立刻检查后续计算依赖的结构关系。

左侧的 `__post_init__()` 只守住几条后续计算真正依赖的关系：所有尺寸都必须是正整数；Query heads 必须能按组共享 KV heads；RoPE 要把 `head_dim` 分成两半，所以它必须是偶数；每个 token 选择的专家数也不能超过专家总数。

这里没有把配置写成一份几十条规则的考试卷。校验最重要的任务，是让错误的结构尽早停下来，同时别让大量边界检查盖住主线。

## 把 JSON 变成代码里的配置

结构关系守住以后，还得真的把文件读进来。`from_json()` 做的事情很直接：

```text
读取 config.json
  -> 解析成 Python dict
  -> 取出 Qwen3MoeConfig 认识的字段
  -> mlp_only_layers 转成 tuple
  -> 其余内容放进 extra_fields
  -> 创建 Qwen3MoeConfig
```

现在可以这样读配置：

```python
config = Qwen3MoeConfig.from_json(checkpoint_dir / "config.json")

print(config.num_hidden_layers)   # decoder 要循环多少次
print(config.num_experts)         # Router 要给多少个专家打分
print(config.num_experts_per_tok) # 每个 token 选几个专家
```

到这里，`config.json` 已经处理完成。我们知道该创建多少层 Decoder、每层有多少个 Attention heads 和专家，以及各种 hidden states 应该是什么 shape。

但配置只描述模型的结构，不包含模型在训练中学到的参数值。Embedding 表、Attention 投影矩阵和专家权重这些真正参与计算的数字，仍然保存在模型目录的 Safetensors 文件里。只有把它们读出来并放到对应模块中，这台只有结构的“空模型”才能开始推理。

因此，Step 01 的后半段要从 `config.py` 转到 `checkpoint.py`，解决第二个模型资产问题：给定一个参数名，怎样从一个或多个 Safetensors 文件中找到并读出对应的 NumPy tensor。后面的 Embedding、Attention 和 MoE 都会依赖这个入口加载自己的权重。

```text
config.json
  -> 告诉代码模型由哪些模块组成、每个 tensor 应该是什么 shape

Safetensors 权重
  -> 提供这些 tensor 在训练后得到的具体数值
```

## 为什么不让模型代码直接读取 Safetensors

直接使用文件时，后面的模型代码会同时遇到几类细节：权重可能是单文件，也可能被拆成多个分片；参数名要先经过 index 才能找到分片；找到分片后还要解析 header、计算 byte offset、处理 dtype，最后才能得到 NumPy tensor。如果 Embedding、Attention 和 MoE 都自己处理这些步骤，文件格式细节就会散落到整个模型实现里。

所以这里把权重读取问题收拢成一个很小的接口：我们定义一个 `SafetensorsCheckpoint` 类。它不是 Safetensors 格式自带的类，也不是从 Transformers 里复制来的，而是这个课程为了完整推理链路自己定义的一层权重访问接口。

```text
SafetensorsCheckpoint
  输入：模型目录
  保存：参数名 -> TensorInfo 的索引
  提供：tensor_info(name)、load_tensor(name)

后续模型模块
  只提交参数名 -> 得到 NumPy tensor
  不关心单文件、分片、header 和 byte offset
```

它要解决的核心问题不是“怎样计算模型”，而是“怎样给模型提供一个稳定的按名字取权重入口”。这个边界也让权重目录的组织方式和后面的矩阵计算彼此独立。

<!-- checkpoint: step01-checkpoint-abstraction -->

左侧先建立这层抽象的基本状态。`TensorInfo` 描述一个 tensor 在磁盘上的位置和格式；`SafetensorsCheckpoint` 保存模型目录与整张参数索引。此时还没有读取任何权重 payload，`_DTYPE_WIDTHS` 和 `_NUMPY_DTYPES` 只是为后面的长度校验与 NumPy 解码准备格式信息。

## 一个参数到底在哪个分片

Qwen3-30B-A3B 的权重没有塞进一个巨大的文件，而是拆成了 16 个分片。模型初始化时如果想拿：

```text
model.layers.0.self_attn.q_proj.weight
```

总不能把 16 个文件挨个翻一遍。`model.safetensors.index.json` 就是目录：

```json
{
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00016.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00016.safetensors",
    "model.layers.47.mlp.gate.weight": "model-00016-of-00016.safetensors"
  }
}
```

左边是参数名，右边是它所在的分片。

<!-- checkpoint: step01-index-discovery -->

`SafetensorsCheckpoint.from_directory()` 先看目录里有没有 index。有，就按 `weight_map` 组织分片；没有，就接受一个单文件 checkpoint。这样后面的模型代码只按参数名取权重，不需要知道它来自一个文件还是 16 个文件。`keys()` 用来查看已有参数名，`tensor_info(name)` 则把未知名称统一变成清晰的错误。

这里要分清两层信息：

```text
index：这个参数在哪个分片？
header：它在分片里的哪一段 bytes？dtype 和 shape 是什么？
```

index 只解决了第一层。下一步要打开分片的 header。

## Safetensors 其实很朴素

一个 Safetensors 文件可以先看成三段：

```text
前 8 bytes：header 有多长
接下来：    JSON header
最后：      紧挨着存放的 tensor bytes
```

header 里每个 tensor 大概长这样：

```json
{
  "model.embed_tokens.weight": {
    "dtype": "BF16",
    "shape": [151936, 2048],
    "data_offsets": [0, 622329856]
  }
}
```

`data_offsets` 不是文件开头的绝对位置，而是相对于 tensor 数据区的偏移。真正读取时，要先跳过 8 bytes 和整段 header，再加上 tensor 自己的起始 offset。

<!-- checkpoint: step01-header-validation -->

左侧先从 `_read_index()` 开始：它收集 `weight_map` 涉及的所有分片，分别读取 header，再确认 index 指向的参数确实存在。随后 `_read_shard()` 读前 8 bytes 得到 header 长度，并把 JSON header 变成一组 `TensorInfo`。每个 `TensorInfo` 记住六件事：名字、dtype、shape、分片名、起止 offset，以及数据区从哪里开始。

这一步还没有加载权重本体。即使模型有几百亿参数，读取目录时也只需要读 index 和各分片前面很小的 header。我们现在拿到的是一张地图，不是把整座仓库搬进内存。

代码还顺手检查了一件最关键的事：`shape` 和 `dtype` 推出来的 byte 数，必须刚好等于 offsets 覆盖的长度。比如 `[4,4]` 的 `F32` tensor 应该占：

```text
4 * 4 * 4 bytes = 64 bytes
```

如果 header 只给了 60 bytes，继续 reshape 只会得到一块残缺权重，所以当场报错更好。

## 只拿现在需要的 tensor

目录和 header 都读完后，`load_tensor(name)` 就很自然了：

```text
参数名
  -> TensorInfo
  -> 打开对应分片
  -> seek 到那段 bytes
  -> 按 dtype 解释
  -> reshape 成记录的 shape
```

<!-- checkpoint: step01-load-tensor -->

关键点是“只读这一段”。要拿 Embedding，就只读取 Embedding 对应的 byte range，不会先把 16 个分片全部装进内存。

Qwen3 权重常用 BF16。NumPy 没有在这里直接保留 BF16 array，所以代码先把每个 16-bit word 放到一个 32-bit word 的高 16 位，再把它看成 `float32`。这样后面的 CPU 小张量实现可以直接计算。

<!-- checkpoint: step01-package -->

最后把这一章完成的类型写进 `__init__.py`。`Qwen3MoeConfig` 是模型结构入口，`SafetensorsCheckpoint` 是按名字读取权重的入口，`TensorInfo` 则描述一个权重在分片中的元数据。这样使用者不需要知道它们分别定义在哪个模块，可以直接从 `qwen3_moe` 包导入。

完整使用方式只有几行：

```python
from qwen3_moe import Qwen3MoeConfig, SafetensorsCheckpoint

config = Qwen3MoeConfig.from_json(checkpoint_dir / "config.json")
checkpoint = SafetensorsCheckpoint.from_directory(checkpoint_dir)

info = checkpoint.tensor_info("model.embed_tokens.weight")
embedding_weight = checkpoint.load_tensor(info.name)

print(info.dtype, info.shape, info.shard)
```

现在回到 Step 00 的地图：

```text
模型目录
  |-- config.json ----------------> Qwen3MoeConfig
  `-- index + Safetensors --------> 按名字读取 NumPy tensor

文本 -> [Tokenizer] -> token IDs -> [Embedding / Attention / MoE] -> logits
          下一章                    后续章节使用本章读出的配置和权重
```

这一章还没有让模型生成任何东西，但地基已经有了。后面创建 Embedding、Attention 和 Expert 时，模块结构来自 `config`，参数数值来自 `checkpoint`。它们不再需要关心 JSON 怎样解析，也不需要关心某个权重藏在第几个分片。

下一章继续顺着真实数据流往前走，开始实现 Tokenizer。模型目录已经能读，接下来该把用户输入的文字变成它真正接收的 token IDs 了。忘记这一块在全局哪里时，就回到 [Step 00 的完整推理地图](../step00/)。
