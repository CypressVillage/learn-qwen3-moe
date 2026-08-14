# Step 00：Qwen3 MoE 概览

<!-- checkpoint: step00-empty-repository -->

这一章不实现模型。我们先像查看一张施工图那样，从 Hugging Face 上的目标模型仓库出发，认识我们拿到了哪些模型资产、需要编写哪些模块解释这些资产，以及它们怎样协作，把 prompt 变成下一个 token。

左侧现在是我们的空实现仓库，不是 Hugging Face 模型仓库。继续向下阅读时，Python 文件会按介绍顺序逐个出现；它们暂时都保持空白，只建立模块边界，不提前放入后续章节才会讲解的实现。

## 1. 我们从 Hugging Face 仓库拿到什么

本课程以 [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B) 这一类 Qwen3 MoE checkpoint 为目标。下载后的目录不是一份可以直接阅读执行的 NumPy 程序，而是一组描述模型、保存参数和定义文本编码方式的资产。

省略具体分片数量后，可以把仓库看成：

```text
Qwen3-30B-A3B/
|-- config.json
|-- generation_config.json
|-- tokenizer.json
|-- tokenizer_config.json
|-- merges.txt
|-- vocab.json
|-- model.safetensors.index.json
|-- model-00001-of-NNNNN.safetensors
|-- model-00002-of-NNNNN.safetensors
|-- ...
|-- README.md
`-- LICENSE
```

仓库版本可能增减辅助文件，权重分片数量也可能变化，但推理最关心的边界保持一致：

| 模型仓库资产 | 它告诉我们什么 |
| --- | --- |
| `config.json` | 模型层数、hidden size、Attention heads、KV heads、专家数量、RoPE 参数等结构信息 |
| `model.safetensors.index.json` | 每个参数名位于哪个 Safetensors 分片 |
| `model-*.safetensors` | Embedding、Attention、Norm、Router 和 Experts 的真实 tensor 数值 |
| `tokenizer.json`、`vocab.json`、`merges.txt` | 文本如何切分并映射为与权重配套的 token IDs |
| `tokenizer_config.json` | special tokens、chat template 和 tokenizer 行为配置 |
| `generation_config.json` | EOS 以及 temperature、top-k、top-p 等默认生成设置 |
| `README.md`、`LICENSE` | 模型用法、限制和授权条件，不直接参与 forward 计算 |

这里有一个重要区别：`config.json` 是数据，不是模型代码；Safetensors 保存权重，也不描述完整计算过程。下载 checkpoint 只意味着原材料已经到齐，并不意味着我们已经拥有可执行的推理引擎。

```text
Hugging Face 模型仓库 = 配置 + tokenizer 资产 + 权重 + 元数据
本课程实现仓库         = 读取这些资产并执行 Qwen3 MoE 推理的代码
```

## 2. 为什么需要自己的推理模块

要让上面的静态资产真正运行，代码必须依次回答：配置怎样变成层和 shape，参数怎样从分片中取出，文本怎样变成 token IDs，每个 decoder layer 怎样计算，以及如何不断选择 next token。

模型资产与本课程模块的对应关系如下：

| Hugging Face 资产或运行时状态 | 本课程模块 | 执行职责 |
| --- | --- | --- |
| `config.json` | `config.py` | 读取、解析并校验模型结构 |
| Safetensors index 和 shards | `checkpoint.py` | 定位并读取权重 tensor |
| Tokenizer 相关文件 | `tokenizer.py` | 文本与 token IDs 互转 |
| Embedding、Norm、Linear、MLP 权重 | `layers.py` | 执行可复用的基础张量计算 |
| Attention 和 RoPE 权重与配置 | `attention.py`、`rope.py` | 执行上下文混合与位置旋转 |
| Router 和 Expert 权重 | `moe.py` | 选择专家、dispatch、计算并合并 |
| 全部结构与参数 | `model.py` | 组装 decoder layers 和完整模型 |
| `generation_config.json` 和用户参数 | `generation.py` | 驱动生成循环、采样和停止 |
| 推理时产生的历史 Key/Value | `cache.py` | 保存并复用已经计算过的上下文 |

KV Cache 是唯一不来自模型仓库的这一类数据：它是在 forward 过程中动态产生的运行时状态。接下来我们逐个建立这些代码边界。

## 3. 逐个认识工程模块

推理并不是一个巨大的 `model.py`。文本处理、权重读取、Attention、MoE、缓存和生成循环有不同的输入输出，也会在不同阶段独立验证。把它们拆开，后续才能沿真实数据流逐章实现，而不会一次面对整台机器。

<!-- checkpoint: step00-config -->

### `config.py`：模型的结构说明书

配置决定模型有多宽、多深，以及每层怎样计算。这里最终会读取并校验 hidden size、层数、词表大小、Attention heads、KV heads、专家数量和每个 token 选择的专家数等参数。

其他模块不能把这些数字写死。它们都应从同一份配置建立张量 shape 和层结构，因此 `config.py` 是整个工程最先确定的边界。

<!-- checkpoint: step00-checkpoint -->

### `checkpoint.py`：权重仓库的入口

真实 Qwen3 MoE 权重通常分布在多个 Safetensors shard 中。这个模块最终负责读取索引、检查文件、按参数名定位 shard，并把磁盘上的数组交给模型装配代码。

它只回答“某个权重在哪里、怎样安全读取”，不负责解释这个权重参与哪种数学计算。

<!-- checkpoint: step00-tokenizer -->

### `tokenizer.py`：文本与 token IDs 的边界

Transformer 不直接接收字符串。Tokenizer 按词表、预分词规则、BPE merge 和 special token 定义，把 Unicode 文本编码成整数 ID，并在生成结束后把 IDs 解码回文本。

```text
"MoE" -> encode -> [44, 78, 36] -> input_ids [B, S]
```

具体 ID 必须与 checkpoint 配套；用字符编号或任意 bytes 代替，即使模型数学正确，也无法得到真实模型的输出。

<!-- checkpoint: step00-layers -->

### `layers.py`：可复用的基础积木

Embedding、Linear、RMSNorm 和 Dense MLP 会被多个高层模块反复使用。这个文件最终提供这些小而稳定的 NumPy 原语，让 Attention、MoE 和整模型只组合模块，不重复实现底层计算。

例如 Embedding 把 `input_ids [B,S]` 查表为 `hidden_states [B,S,D]`；RMSNorm 稳定数值范围，但不改变 shape。

<!-- checkpoint: step00-rope -->

### `rope.py`：把位置写进 Q 和 K

RoPE 根据 position IDs 生成旋转频率，并旋转 Attention 的 Query 和 Key 成对通道。它不向 hidden states 直接添加位置向量，而是让旋转后的点积能够表达 token 之间的相对位置。

这个模块只处理位置和旋转，不负责完整 Attention。

<!-- checkpoint: step00-attention -->

### `attention.py`：让 token 读取上下文

这里最终实现 QK Norm、Grouped Query Attention、RoPE 接入、causal mask 和输出投影。输入与输出都保持 `[B,S,D]`，但中间会拆成 Query heads 与较少的 Key/Value heads：

```text
hidden_states [B,S,D]
  -> Q [B,Hq,S,Dh]
  -> K [B,Hkv,S,Dh]
  -> V [B,Hkv,S,Dh]
  -> context [B,S,D]
```

Attention 负责 token 之间交换信息。causal mask 保证当前位置不能读取未来 token。

<!-- checkpoint: step00-moe -->

### `moe.py`：让每个 token 只经过少数专家

Sparse MoE 位于 decoder layer 的前馈分支。Router 为每个 token 计算专家分数，只选择 top-k 专家；随后 dispatch token、执行各专家的 SwiGLU MLP，再按 routing weight 合并回原顺序。

```text
hidden_states [B,S,D] -> tokens [T,D]
router_logits [T,E]   -> top-k indices / weights [T,K]
expert outputs        -> combine -> [B,S,D]
```

Attention 仍然是稠密的。MoE 的稀疏性只发生在前馈计算中。

<!-- checkpoint: step00-cache -->

### `cache.py`：保存不会变化的历史 K/V

第一轮 prefill 会计算 prompt 全部 token 的 Key/Value。之后每生成一个 token，历史 K/V 都不会变化，因此 cache 只追加新位置，避免重复计算整个前缀。

```text
prefill: input S tokens -> cache length S
decode:  input 1 token  -> cache length S+1
decode:  input 1 token  -> cache length S+2
```

Cache 改变计算方式和 Attention 的 key length，但不能改变同一位置的数学结果。

<!-- checkpoint: step00-model -->

### `model.py`：组装 decoder layer 和完整模型

这个文件把基础层、Attention、MoE、残差连接、最终 RMSNorm 和 LM Head 接成完整 Qwen3 MoE。它还根据配置判断某层使用 Dense MLP 还是 Sparse MoE，并从 checkpoint 装配相应权重。

单个 decoder layer 的 shape 主干始终是：

```text
x [B,S,D]
  -> RMSNorm -> Attention -> residual add
  -> RMSNorm -> Dense MLP / Sparse MoE -> residual add
  -> layer_output [B,S,D]
```

<!-- checkpoint: step00-generation -->

### `generation.py`：把一次 forward 变成连续文本

模型一次 forward 只输出 logits，不会自行不断生成。这个模块最终负责取最后位置的 logits，应用 temperature、top-k、top-p 或 greedy decoding，追加 next token，并检查 EOS 与长度上限。

它也是 prefill、逐 token decode 和 KV Cache 生命周期真正汇合的地方。

<!-- checkpoint: step00-package -->

### `__init__.py`：公开已经完成的能力

包入口只导出课程已经实现并验证过的公共接口。尚未讲到的模块不会返回伪结果，也不会提前暴露只有名字、没有行为的 stub。

到这里，右侧已经形成完整的空文件树。接下来再把这些模块放进同一条推理链路，理解数据如何穿过它们。

<!-- checkpoint: step00-module-flow -->

## 4. 模块怎样连成一次完整推理

从依赖关系看，可以把工程分成四层：

| 层次 | 文件 | 作用 |
| --- | --- | --- |
| 模型资产 | `config.py`、`checkpoint.py`、`tokenizer.py` | 解释模型结构、读取权重、定义文本与 IDs 的对应关系 |
| 数学组件 | `layers.py`、`rope.py`、`attention.py`、`moe.py` | 完成实际张量计算 |
| 模型组装 | `cache.py`、`model.py` | 保存跨轮状态并组合 decoder layers |
| 生成控制 | `generation.py`、`__init__.py` | 驱动自回归循环并提供公共入口 |

它们的主要调用关系是：

```text
generation.py
  |-- tokenizer.py       读取 tokenizer 资产，文本 <-> token IDs
  |-- model.py           token IDs -> logits
       |-- config.py     读取 config.json，决定结构与 shape
       |-- checkpoint.py 读取 Safetensors 权重
       |-- layers.py     Embedding / Norm / Linear / MLP
       |-- attention.py  上下文混合
       |    |-- rope.py  位置旋转
       |    `-- cache.py 保存运行时历史 K/V
       `-- moe.py        router / experts / combine
```

这个分层也解释了课程顺序：先确认配置和权重，再让文本进入模型；随后实现基础层、位置、Attention 和 MoE；最后组装完整模型与生成循环，并接入 KV Cache 优化。

## 5. 从 prompt 到输出 token 的完整地图

```text
prompt 文本
  |
  v
Tokenizer.encode
  |
  v
input_ids [B, S]
  |
  v
Token Embedding -> hidden_states [B, S, D]
  |
  v
+-------------------------------------------------------------+
| 重复 L 次 Qwen3 MoE Decoder Layer                           |
|                                                             |
|  RMSNorm -> GQA Attention + RoPE + causal mask + KV Cache   |
|          -> residual add                                    |
|          -> RMSNorm                                         |
|          -> Sparse MoE 或 Dense MLP                         |
|          -> residual add                                    |
+-------------------------------------------------------------+
  |
  v
Final RMSNorm -> LM Head -> logits [B, S, V]
  |
  v
取最后位置 [B, V] -> temperature / top-k / top-p
  |
  v
next_token [B, 1]
  |
  +----> 追加到序列，更新 KV Cache，继续下一轮
  |
  v
遇到 EOS 或长度上限 -> Tokenizer.decode -> 输出文本
```

先记住五条边界：

1. 模型输入不是字符串，而是整数 `input_ids`。
2. 一次 `forward()` 输出不是文本，而是每个位置对整个词表的 `logits`。
3. MoE 不替代 Attention；它替代的是部分 decoder layer 的 Dense MLP 分支。
4. 自回归生成是模型外部的循环，模型只完成一轮 token IDs 到 logits 的变换。
5. KV Cache 连接相邻 forward，让后续轮次只计算新 token。

## 6. 全程使用的 shape 符号

| 符号 | 含义 | 典型张量 |
| --- | --- | --- |
| `B` | batch size | 同时处理的序列数 |
| `S` | 当前输入序列长度 | `input_ids [B,S]` |
| `T` | 展平后的 token 数，`T=B*S` | MoE dispatch 使用 |
| `D` | residual hidden size | `hidden_states [B,S,D]` |
| `V` | vocabulary size | `logits [B,S,V]` |
| `L` | decoder layer 数量 | Transformer 主循环 |
| `Hq` | Query head 数量 | Q projection |
| `Hkv` | Key/Value head 数量 | GQA 的 K/V projection |
| `Dh` | 每个 attention head 的宽度 | Q/K/V 最后一维 |
| `E` | routed expert 总数 | router logits `[T,E]` |
| `K` | 每个 token 激活的专家数 | top-k indices `[T,K]` |

不要把 `D = Hq * Dh` 当成所有 Qwen3 checkpoint 都必须满足的规则。实现应读取真实配置，并按权重 shape 建立投影边界。

## 7. 后续每章怎样使用这张地图

后续章节遵循同一套阅读方法：

1. 先指出当前文件在完整数据流中的位置。
2. 明确它接收什么、返回什么，以及必须维持的 shape。
3. 用小张量和 NumPy 原语实现并验证局部计算。
4. 把新模块接回累计推理框架。
5. 未实现的下游能力继续保持不可用。

下一章从工程最底层的模型资产开始：[Step 01：配置和权重目录](/step01/)。在 token 真正流动之前，我们先确认 checkpoint 描述了怎样的模型，以及每个参数存放在哪里。
