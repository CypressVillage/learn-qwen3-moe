# Step 00：先看清 Qwen3 MoE 怎样跑起来

<!-- checkpoint: step00-empty-repository -->

这一章先不写实现。

我们从 Hugging Face 上真实的 [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B) 仓库出发，看看下载一个模型时究竟拿到了什么。然后顺着一条 prompt 往前走：它怎样变成 token，怎样穿过 Attention 和 MoE，最后怎样变成下一个 token。

右侧现在还是一个空仓库。读到一个模块，它就出现一个空文件。这里只把整台机器的零件摆到桌上，后面的章节再沿着数据流，需要什么，就把什么填进去。

## 1. 下载下来的模型并不会自己运行

打开 `Qwen3-30B-A3B` 的文件列表，最显眼的是这些内容：

```text
Qwen3-30B-A3B/
|-- config.json
|-- generation_config.json
|-- tokenizer.json
|-- tokenizer_config.json
|-- merges.txt
|-- vocab.json
|-- model.safetensors.index.json
|-- model-00001-of-00016.safetensors
|-- model-00002-of-00016.safetensors
|-- ...
|-- model-00016-of-00016.safetensors
|-- README.md
`-- LICENSE
```

先把它们分成三堆。

`config.json` 是模型的说明书。真实配置里能看到 `Qwen3MoeForCausalLM`、48 层 decoder、128 个 routed experts、每个 token 选择 8 个专家，以及 Attention heads、KV heads、RoPE 等结构参数。

`tokenizer.json`、`vocab.json`、`merges.txt` 和 `tokenizer_config.json` 负责文本。它们决定一段 Unicode 字符串怎样切成与这份模型配套的 token IDs，也保存 special tokens 和 chat template。

16 个 `model-*.safetensors` 才是参数本体。Embedding、Attention 投影、RMSNorm、Router 和每个 Expert 的数值都在里面。`model.safetensors.index.json` 像目录一样，告诉我们一个参数名应该去哪个分片找。

剩下的 `generation_config.json` 保存默认生成参数，`README.md` 和 `LICENSE` 说明模型怎样使用、有什么限制。它们很重要，但不会参加一次 forward 的张量计算。

所以，下载 checkpoint 只是把原料搬回来了：

```text
Hugging Face 仓库：配置 + tokenizer 资产 + 权重 + 元数据
我们要写的代码：读懂这些资产，并把计算真正跑起来
```

`config.json` 不是 Python 模型，Safetensors 也不会告诉 CPU 先做 Attention 还是先做 MoE。中间缺的这部分，就是这门课要亲手实现的推理框架。

## 2. 这些文件逼着我们写出哪些模块

先从仓库里的原料往里走。

<!-- checkpoint: step00-config -->

### `config.py`：先读懂模型说明书

看到 `config.json`，第一件事不是算矩阵，而是确认自己面对的是一台怎样的模型：hidden size 多大，有多少层，Q head 和 KV head 各有多少，一共有多少专家，每个 token 激活几个专家。

这些数字会决定后面几乎所有 tensor 的 shape，所以放进 `config.py`，只读一次，其他模块都从这里拿，不能各写一份常量。

<!-- checkpoint: step00-checkpoint -->

### `checkpoint.py`：按名字找到真实权重

接着是 16 个 Safetensors 分片。模型组装时会问：“`model.layers.0.self_attn.q_proj.weight` 在哪里？”

`checkpoint.py` 读取 index，找到对应分片，再从文件里取出 tensor。它只管权重在哪里、dtype 和 shape 是什么，不管这个权重接下来怎样参与 Attention。

<!-- checkpoint: step00-tokenizer -->

### `tokenizer.py`：让文字进入模型

用户给我们的是字符串，Embedding 接受的却是整数。中间需要 Tokenizer 按 Qwen3 自己的词表和 merge 规则做转换：

```text
"MoE 是什么？" -> encode -> [token_id, token_id, ...]
```

这里不能拿字符编号或随便找一套 BPE 顶替。token ID 会直接用来查 Embedding 表，编号只要错一个，后面的数学全对也没用。

<!-- checkpoint: step00-layers -->

### `layers.py`：先准备反复使用的小积木

token IDs 进入模型后，第一步是查 Embedding，得到 `hidden_states [B,S,D]`。之后每一层还会反复使用 RMSNorm、Linear 和 MLP。

这些计算本身不负责“理解上下文”或“选择专家”，但 Attention 和 MoE 都离不开它们，所以统一放在 `layers.py`。

<!-- checkpoint: step00-rope -->

### `rope.py`：告诉 Attention token 在哪里

Attention 只看 Q 和 K 的点积，本身不知道谁在前、谁在后。RoPE 根据 position IDs 旋转 Q 和 K，让点积带上相对位置信息。

`rope.py` 只准备并应用这次旋转。它是 Attention 里面的一步，不是另一套独立的模型层。

<!-- checkpoint: step00-attention -->

### `attention.py`：让当前 token 读取前文

Qwen3 的 Attention 会把 hidden states 投影成 Q、K、V，给 Q/K 做 Norm 和 RoPE，再用 causal mask 保证当前位置看不到未来。

这份模型使用 GQA：Query heads 多，Key/Value heads 少。多个 Query head 共用一组 K/V，最后再合并回原来的 hidden size。

```text
hidden_states [B,S,D]
  -> Q / K / V
  -> QK Norm + RoPE
  -> causal attention
  -> context [B,S,D]
```

Attention 做的是 token 之间的信息交换。它回答“当前 token 应该从前文读什么”。

<!-- checkpoint: step00-moe -->

### `moe.py`：给不同 token 派不同专家

读完上下文以后，每个 token 还要经过前馈网络。Qwen3 MoE 不让所有 token 都跑同一份大 MLP，而是先用 Router 打分，再从 128 个 routed experts 中选出 8 个。

```text
hidden_states
  -> Router 打分
  -> 每个 token 选择 top-8 experts
  -> 只执行被选中的 Expert MLP
  -> 按 routing weight 合并
```

这就是“稀疏”的位置。Attention 仍然会处理所有 token；MoE 替换的是 decoder layer 里的前馈分支，不是 Attention。

<!-- checkpoint: step00-cache -->

### `cache.py`：别把已经算过的前文再算一遍

第一次处理 prompt 时，Attention 会为所有位置算出 K 和 V。生成第二个 token 时，旧位置的 K/V 不会变化，没必要重新计算。

KV Cache 把它们留下来：

```text
prefill: 整段 prompt -> 建立长度 S 的 cache
decode:  只输入新 token -> cache 追加到 S+1
decode:  再输入新 token -> cache 追加到 S+2
```

Cache 不改变模型应该得到的结果，只是避免重复劳动。

<!-- checkpoint: step00-model -->

### `model.py`：把一层层 decoder 接起来

现在零件已经差不多齐了。`model.py` 负责创建 Embedding，重复 48 次 decoder layer，最后接 RMSNorm 和 LM Head。

一层 decoder 的主干并不复杂：

```text
x
  -> RMSNorm -> Attention -> 加回 x
  -> RMSNorm -> MoE       -> 再加回去
```

shape 从 `[B,S,D]` 进去，仍然以 `[B,S,D]` 出来。这样同一种 layer 才能一层接一层堆下去。

<!-- checkpoint: step00-generation -->

### `generation.py`：模型只算一次，生成循环让它继续说

完整模型的一次 forward 不会返回一句话，只会返回 `logits [B,S,V]`。logits 表示每个位置对整个词表中每个 token 的打分。

`generation.py` 取最后一个位置的 logits，选出 next token，把它追加到序列，再带着 KV Cache 调一次模型。如此循环，直到遇到 EOS 或长度上限。

<!-- checkpoint: step00-package -->

### `__init__.py`：把已经做好的能力交给使用者

最后留一个很薄的包入口。课程做到哪，它就导出哪些真正可用的类和函数。还没实现的东西继续保持不可用，不先摆一个会返回假结果的壳。

右侧现在已经有了完整的空文件树。它还不能运行，这很正常。这一章只需要先记住每个问题放在哪个文件里，后面再一块一块填上。

<!-- checkpoint: step00-module-flow -->

## 3. 把零件放回同一台机器

从调用关系看，`generation.py` 在最外面控制循环，`model.py` 在里面完成一次 forward：

```text
generation.py
  |-- tokenizer.py       文本 <-> token IDs
  `-- model.py           token IDs -> logits
       |-- config.py     决定模型结构和 shape
       |-- checkpoint.py 提供真实权重
       |-- layers.py     Embedding / Norm / Linear
       |-- attention.py  读取上下文
       |    |-- rope.py  写入位置信息
       |    `-- cache.py 复用历史 K/V
       `-- moe.py        Router / Experts / 合并
```

这里最容易混淆的是两层循环。

`model.py` 里面是层循环：同一批 hidden states 依次穿过 48 个 decoder layers。

`generation.py` 外面是 token 循环：一次 forward 选出一个 next token，再把这个 token 送回模型，继续生成下一个。

一内一外，合起来才是完整推理。

## 4. 从 prompt 到输出 token

假设用户输入一句 prompt。它走过的完整路径是：

```text
prompt 文本
  |
  v
Tokenizer.encode
  |
  v
input_ids [B,S]
  |
  v
Embedding -> hidden_states [B,S,D]
  |
  v
+----------------------------------------------------------+
| 重复 L 次 Decoder Layer                                  |
|                                                          |
| RMSNorm -> GQA Attention + RoPE + causal mask + KV Cache |
|         -> residual add                                  |
| RMSNorm -> Sparse MoE                                    |
|         -> residual add                                  |
+----------------------------------------------------------+
  |
  v
Final RMSNorm -> LM Head -> logits [B,S,V]
  |
  v
取最后位置 logits [B,V]
  |
  v
greedy / temperature / top-k / top-p
  |
  v
next_token [B,1]
  |
  +----> 追加到序列，更新 KV Cache，再跑一轮
  |
  v
遇到 EOS 或长度上限
  |
  v
Tokenizer.decode -> 输出文本
```

先抓住五件事：

1. 模型吃的是 token IDs，不是字符串。
2. 一次 forward 给出的是 logits，不是最终文本。
3. Attention 让 token 读取前文，MoE 让 token 选择前馈专家，两者不是一回事。
4. 自回归生成发生在模型外部，一轮只选一个 next token。
5. KV Cache 把相邻两轮 forward 接起来，让模型不必重算整段前文。

以后每实现一个模块，都回来看看它在这张图的哪个位置。只要这条主线没有丢，局部的矩阵 shape 就不会变成一堆孤立公式。

## 5. 先认识几个 shape 字母

后面的代码会一直使用这些缩写：

| 符号 | 含义 | 最常见的位置 |
| --- | --- | --- |
| `B` | batch size | `input_ids [B,S]` |
| `S` | 当前序列长度 | prompt 或本轮输入的 token 数 |
| `D` | hidden size | `hidden_states [B,S,D]` |
| `V` | vocabulary size | `logits [B,S,V]` |
| `L` | decoder layer 数量 | 模型内部的层循环 |
| `E` | routed expert 数量 | Router 对专家的打分 |
| `K` | 每个 token 选择的专家数 | top-k routing |

暂时不用背更多。等写到 Attention 时再引入 heads 和 head dimension，写到 MoE 时再展开 token dispatch。现在知道 `input_ids -> hidden_states -> logits -> next_token` 这条 shape 主线就够了。

下一章从最靠近磁盘的地方开始：[Step 01：配置和权重目录](../step01/)。我们先让代码读懂 `config.json`，再从 Safetensors 分片中准确找到一个真实 tensor。
