# Step 01：<span class="title-unit">配置</span>和权重目录

<!-- checkpoint: step01-empty-workspace -->

Step 00 已经画出从文本到 next token 的完整推理地图。这一课先不计算 Attention，也不假装模型已经能够生成文本，只完成最靠近磁盘的两个真实边界：读取架构配置，以及检查 Safetensors 权重目录。

## 1. 当前走到哪里

回到 Step 00 的完整控制流：

```text
文本 prompt
  -> Tokenizer.encode
  -> input_ids [B,S]
  -> Qwen3MoeForCausalLM.forward
  -> logits [B,S,V]
  -> 取最后位置 logits [B,V]
  -> greedy 或 sampling 选择 next_token [B,1]
  -> 追加 token，更新 KV Cache，重复 forward
  -> Tokenizer.decode
  -> 生成文本
```

一次 `forward()` 只负责从 token IDs 得到 logits。Tokenizer、next-token 选择、停止条件和生成循环都位于模型调用边界之外；KV Cache 则连接相邻两次模型调用，避免重复计算已经处理过的历史 token。

本课只点亮 token 进入模型之前的“装机检查”：

```text
[config.json 已可读] + [权重目录已可检查]
        |
        v
文本 -> [Tokenizer 未实现] -> token IDs -> [完整 MoE forward 未实现]
     -> [logits 不可用] -> [生成循环不可用]
```

## 2. 地图上还有什么能力未完成

下载或拿到一个 checkpoint 目录，并不等于模型已经装配完成。完整推理至少还缺少：

| 组件 | 职责 | 完成步骤 |
| --- | --- | --- |
| Tokenizer | 文本与兼容 token IDs 互转 | Step 02 |
| Embedding、Linear、RMSNorm | 建立 hidden states 与基础层 | Step 03 |
| RoPE、Attention、GQA、QK Norm | 让 token 读取合法上下文 | Step 04-06 |
| Expert、Router、Top-K、dispatch | 执行稀疏 MoE 计算 | Step 07-09 |
| 完整模型与权重映射 | 把外部参数严格装入内部模块 | Step 10 |
| 生成与 KV Cache | 选择、追加、停止和复用历史 | Step 11-14 |

Step 01 不会返回“shape 看起来正确”的伪 logits。未实现能力必须保持不可用，直到对应 contract 被真实代码和测试满足。

## 3. 最小输入、输出和 shape ledger

<!-- checkpoint: step01-config-contract -->

本课使用仓库内可再分发的微型 checkpoint。它只用于检查格式，不具备语言能力。

| 边界 | 输入 | 输出 | shape/数量 |
| --- | --- | --- | --- |
| `Qwen3MoeConfig.from_json` | `config.json` | 结构化配置 | 标量字段与 layer 列表 |
| `SafetensorsCheckpoint.from_directory` | checkpoint 目录 | 参数目录 | `N=2` 个 metadata entry |
| `tensor_info(name)` | 参数名 | dtype、shape、分片、offset | 一个 `TensorInfo` |
| `load_tensor(name)` | 参数名 | NumPy CPU array | Embedding 为 `[4,4]` |

微型配置采用 `V=16, D=4, L=2, Hq=2, Hkv=1, Dh=2, E=4, K=2`。这些数字只属于本课示例，不是 Qwen3 MoE 模型族的固定常量。

<!-- checkpoint: step01-config-validation -->

配置校验覆盖正整数、GQA 头整除、偶数 `head_dim`、专家 Top-K 上界、Dense-only layer 范围、布尔字段和正数 epsilon/theta。特别注意：Qwen3 配置不能套用 Dense 示例的 `D = Hq * Dh` 约束；目标 checkpoint 可以使用宽于 residual hidden size 的 Query 投影。

## 4. 用小数据看懂 Safetensors

<!-- checkpoint: step01-index-discovery -->

Safetensors 单个分片的物理布局可以简化为：

```text
8-byte little-endian header length
JSON header
raw tensor bytes
```

header 中每个参数记录：

```json
{
  "model.embed_tokens.weight": {
    "dtype": "F32",
    "shape": [4, 4],
    "data_offsets": [0, 64]
  }
}
```

<!-- checkpoint: step01-header-validation -->

`[4,4]` 一共有 `4 * 4 = 16` 个元素，`F32` 每个元素 4 bytes，因此必须占 `16 * 4 = 64` bytes。若 offset 只覆盖 60 bytes，读取器会在构造目录时失败，而不是读出一个残缺数组。

分片 checkpoint 另有一个 index：

```json
{
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.layers.1.mlp.gate.weight": "model-00002-of-00002.safetensors"
  }
}
```

index 只回答“参数在哪个文件”；shape、dtype 和 byte offsets 仍来自对应分片 header。读取目录时只解析小型 JSON header，不会把所有 tensor payload 装入内存。

## 5. 最终代码边界

后续步骤会逐渐补全相同边界，调用方向保持稳定：

```python
def generate(model, tokenizer, prompt, sampling, max_new_tokens):
    input_ids = tokenizer.encode(prompt)
    cache = None
    for _ in range(max_new_tokens):
        logits, cache = model(input_ids, cache=cache)
        next_token = sample(logits[:, -1, :], sampling)
        input_ids = append_token(input_ids, next_token)
        if should_stop(next_token, tokenizer):
            break
    return tokenizer.decode(input_ids)


class Qwen3MoeForCausalLM:
    def forward(self, input_ids, cache=None):
        hidden_states = self.embed_tokens(input_ids)
        position_ids = build_position_ids(input_ids, cache)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        attention_mask = build_causal_mask(input_ids, cache)
        for layer in self.layers:
            hidden_states, cache = layer(
                hidden_states, position_embeddings, attention_mask, cache
            )
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states), cache
```

权重加载也分成两个阶段：

1. Step 01 验证配置、索引、header、dtype、shape、offset，并按名称读取一个 tensor。
2. Step 10 定义外部参数名到内部模块名的显式映射，报告 missing、unexpected 和 mismatched 参数。

本课不会根据字符串猜测权重该装到哪个模块，也不会用静默 reshape/transpose 掩盖结构不一致。

### 累计 checkpoint

本课不是在最后一次性展示完整源码。右侧仓库从两个空文件开始，随着正文依次加入配置 contract、结构校验、权重目录发现、header 校验和单 tensor 读取。每个 checkpoint 都是下一版的行级子序列，因此已经出现的代码不会被暗中改写或删除。

<!-- checkpoint: step01-load-tensor -->

核心使用方式：

```python
from qwen3_moe import Qwen3MoeConfig, SafetensorsCheckpoint

config = Qwen3MoeConfig.from_json(checkpoint_dir / "config.json")
checkpoint = SafetensorsCheckpoint.from_directory(checkpoint_dir)
for name in checkpoint.keys():
    print(checkpoint.tensor_info(name))
embedding = checkpoint.load_tensor("model.embed_tokens.weight")
```

`load_tensor(name)` 只打开参数所在分片并读取对应 byte range。BF16 payload 会明确转换为 NumPy `float32`，因为当前锁定的 NumPy 版本没有原生 `bfloat16` dtype；不支持的 dtype 会明确失败。

## 6. 与 NumPy 原语对照

微型 Embedding 权重来自：

```python
expected = np.arange(16, dtype=np.float32).reshape(4, 4) / 10.0
```

读取结果通过 `np.testing.assert_array_equal(actual, expected)` 做逐元素对照，而不是只比较 shape。16 个值从 `0.0` 到 `1.5`，等差数列求和为：

```text
(0.0 + 1.5) * 16 / 2 = 12.0
```

这里的逐元素对照验证的是：读取器对 payload bytes 的 dtype 解释、元素数量和 reshape 都正确。它不验证模型数学，也不代表权重已经能装入完整 Transformer。

下一步自然进入 Tokenizer：权重目录已经可检查，但模型真正接收的是整数 token IDs。Step 02 将从文本、UTF-8 bytes 和 BPE 资源出发，建立可与目标 tokenizer 对齐的 encode/decode 边界。需要重新确认全局位置时，回到 [Step 00 完整推理地图](/step00/)。
