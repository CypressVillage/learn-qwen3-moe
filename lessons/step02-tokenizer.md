# Step 02：把一句话变成模型认识的 token IDs

<!-- checkpoint: step02-ready -->

上一章已经能读懂 `config.json`，也能按名字找到 Safetensors 里的真实权重。现在沿着 Step 00 的推理地图，从用户真正会交给我们的东西开始：一段文字。

模型的第一层是 Embedding。它不会接收 Python 字符串，只会接收整数 ID：

```text
"MoE 是什么？"
  -> Tokenizer
  -> [token_id, token_id, ...]
  -> Embedding 查表
  -> hidden_states [B,S,D]
```

这一章要填上中间的 Tokenizer。目标不是发明一种新的分词方法，而是准确执行 `Qwen3-30B-A3B` 已经保存好的 byte-level BPE 规则。只有这样得到的 ID，才会指向训练时对应的 Embedding 行。

## 先打开 tokenizer.json

模型目录里同时有 `tokenizer.json`、`vocab.json`、`merges.txt` 和 `tokenizer_config.json`。它们不是四套 tokenizer，而是同一套规则的不同保存形式。

这一章直接读取 `tokenizer.json`，因为它已经把运行 encode 所需的内容放在了一起：

```text
tokenizer.json
  |-- model.vocab          token 字符串 -> token ID
  |-- model.merges         BPE 合并顺序
  |-- pre_tokenizer        一段文字先怎样粗切分
  `-- added_tokens         <|im_start|> 等 special tokens
```

`tokenizer_config.json` 里的 chat template 很重要，但它解决的是“怎样把多轮对话排成一段 prompt”，不是“这段 prompt 怎样变成 token IDs”。为了沿真实数据流一次只解决一个问题，本章先不实现聊天模板。

<!-- checkpoint: step02-assets -->

右侧的 `from_file()` 先取出 BPE 模型的 `vocab` 和 `merges`，再找到正则预切分规则与 special tokens。

词表的方向是：

```text
token 字符串 -> token ID
```

encode 正好需要这个方向。decode 则需要反过来查，所以构造函数同时建立：

```text
vocabulary: token -> ID
tokens:      ID -> token
```

merge 列表也不会当成普通集合。它在文件中的先后顺序就是优先级，越靠前 rank 越小，BPE 遇到多个可合并的相邻 pair 时，要先执行 rank 最小的那一个。

到这里我们只是把规则读进内存，还没有处理任何文字。接下来先解释 byte-level 里的“byte”。

## Unicode 字符先变成 UTF-8 bytes

如果直接给所有 Unicode 字符编号，词表会非常庞大，而且总会遇到训练词表里没见过的新字符。Qwen3 的办法是先把文字编码成 UTF-8 bytes。

例如一个汉字通常会变成三个 byte。byte 的取值只有 256 种，所以无论输入中文、英文还是 emoji，底层都能表示：

```text
文本字符
  -> UTF-8
  -> 0 到 255 的 byte 序列
```

但 BPE 的词表 key 是字符串，不能直接把任意 byte 当成可打印文本保存。于是 byte-level BPE 再建立一张固定映射，把 256 个 byte 各自映射到一个可见 Unicode 字符。

<!-- checkpoint: step02-byte-alphabet -->

右侧 `_byte_alphabet()` 建立的就是这张双向表：

```text
byte_encoder:  0..255 -> 可见字符
byte_decoder:  可见字符 -> 0..255
```

ASCII 中适合直接显示的字符尽量保留，其余 byte 映射到 256 之后的 code point。空格因此常会在词表里显示成 `Ġ`，换行也会显示成另一个特殊外观的字符。

这里要特别注意：`Ġ` 不是模型凭空插入的语义标记。它只是某个原始 byte 的可见替身。decode 时把映射倒过来，就能拿回原始 UTF-8 bytes。

## BPE 之前为什么还要预切分

我们当然可以把整篇文章一次性交给 BPE，但那会允许 merge 跨过所有边界，搜索范围也会不断增大。真实 Qwen3 tokenizer 会先用 `tokenizer.json` 里的正则，把文字切成较小片段。

这些片段大致会把连续字母、数字、标点、空白和换行分开处理。这里说“大致”，是因为真正的规则就在模型资产里，不应该靠我们重新写一份相似规则来猜。

<!-- checkpoint: step02-pretokenizer -->

所以 `_find_split_pattern()` 会进入 `pre_tokenizer` 配置，找到 Qwen3 保存的 `Split` 正则，再交给 `regex` 执行。

这里使用第三方 `regex` 而不是 Python 标准库的 `re`，因为 tokenizer 规则会使用 Unicode 属性，例如按所有语言中的字母和数字分类。标准库 `re` 不能完整解释这套语法。

预切分只决定 BPE 每次处理哪一小段，并不会直接决定最终 token。每个片段仍要先转成 UTF-8 bytes，再经过真正的 merge。

## BPE 做的事就是反复合并

假设一个预切分片段经过 byte 映射后，暂时是：

```text
h | e | l | l | o
```

一开始每个字符都是一个 piece。然后查看所有相邻 pair：

```text
(h,e) (e,l) (l,l) (l,o)
```

如果 merge 表里同时允许其中几对，就选择 rank 最小的一对，把它合起来，再重新查看新的相邻 pair。这个过程一直持续到没有 pair 可以继续合并。

<!-- checkpoint: step02-bpe -->

右侧 `_merge_token()` 正是在做这件事：

```text
字符序列
  -> 找当前 rank 最小的相邻 pair
  -> 合并这对 piece
  -> 重新寻找
  -> 没有可用 merge 时停止
```

BPE 不是“查到一个最长单词就结束”。最终边界由整张 merge rank 表共同决定。两个看起来相同的字符片段，只要前面的空格、大小写或 byte 形式不同，都可能得到不同 token。

代码还会缓存一个预切分片段的 BPE 结果。生成过程中常会重复遇到空格、标点和常见词，没有必要每次从单字节重新合并。这个 cache 只复用 tokenizer 的确定性结果，不会改变 token IDs。

## 把整条 encode 链路接起来

现在可以把前面的步骤串成真正的 `encode()`：

```text
输入字符串
  -> 分离 special tokens
  -> 正则预切分普通文本
  -> 每个片段编码成 UTF-8 bytes
  -> bytes 映射成可见字符
  -> 按 merge rank 执行 BPE
  -> 每个最终 piece 查 vocabulary
  -> token IDs
```

<!-- checkpoint: step02-encode -->

`_encode_text()` 负责普通文本。它对每个正则片段执行 byte 映射和 BPE，最后在 `vocabulary` 中查 ID。

外层 `encode()` 还要先识别 special tokens。例如聊天 prompt 里可能出现：

```text
<|im_start|>user
MoE 是什么？<|im_end|>
```

`<|im_start|>` 不能被当成普通的 `<`、字母和 `|` 再走 BPE。它在 `added_tokens` 里已经有一个完整 ID，所以代码按完整字符串匹配，直接追加对应 ID；两边的普通文字再各自进入 `_encode_text()`。

这份实现不会自动添加 BOS 或 EOS。原因很简单：Qwen3 的 tokenizer 资产决定 token 怎样编码，但“这次调用是否要在两端额外加 token”属于更外层的 prompt 或生成策略。悄悄添加反而会让调用者看不清真正送进模型的序列。

现在可以这样使用：

```python
from qwen3_moe import Qwen3Tokenizer

tokenizer = Qwen3Tokenizer.from_file(checkpoint_dir / "tokenizer.json")
input_ids = tokenizer.encode("MoE 是什么？")

print(input_ids)
print(len(input_ids))  # 这就是后面的序列长度 S
```

得到的每个 ID 都必须落在模型词表范围内。下一章创建 Embedding 后，`input_ids` 中的数字会直接选择 `model.embed_tokens.weight` 的对应行。

## decode 沿原路返回

生成循环最后拿到的仍然是 token IDs。要把它们展示给用户，Tokenizer 需要沿 encode 的方向倒着走：

```text
token IDs
  -> 查回 token 字符串
  -> 可见字符映射回 bytes
  -> 拼接完整 byte 序列
  -> UTF-8 解码
  -> 文本
```

<!-- checkpoint: step02-decode -->

为什么要先拼 bytes，最后统一做 UTF-8 decode？因为一个汉字的三个 UTF-8 bytes 可能落在不同 token 里。逐 token 解码会把一个完整字符拆坏；先把 byte 序列接起来，字符边界才会恢复。

special token 不属于 byte alphabet。遇到它时，代码先输出已经积累的普通 bytes，再决定保留这个标记，还是在 `skip_special_tokens=True` 时跳过它。

encode 和 decode 现在形成了一个很有用的检查：

```python
text = "MoE 是什么？"
assert tokenizer.decode(tokenizer.encode(text)) == text
```

这个等式不能证明 tokenizer 与官方实现的每一个边界都相同，但至少能确认 byte 映射和 UTF-8 往返没有丢失输入。真正的 token 边界仍由同一份 `tokenizer.json` 的正则、vocab 和 merges 决定。

<!-- checkpoint: step02-package -->

最后从 `__init__.py` 导出 `Qwen3Tokenizer`。包入口现在提供了三块已经实现的能力：

```python
from qwen3_moe import Qwen3MoeConfig, Qwen3Tokenizer, SafetensorsCheckpoint
```

回到 Step 00 的完整地图，本章刚刚填上了最左边的一段：

```text
prompt 文本
  -> Qwen3Tokenizer.encode
  -> input_ids [B,S]
  -> Embedding
  -> hidden_states [B,S,D]
  -> Attention / MoE / ...
  -> logits
  -> next token IDs
  -> Qwen3Tokenizer.decode
  -> 输出文本
```

现在文字已经变成了模型认识的整数，但这些整数还没有参加任何张量计算。下一章会实现 Embedding、RMSNorm 和 Linear 这些基础层，让 `input_ids` 第一次进入权重，变成贯穿 decoder 的 `hidden_states [B,S,D]`。如果 token、ID 和 hidden states 的关系开始混在一起，就回到 [Step 00 的完整推理地图](/step00/) 再看一次主线。
