# 07：最小自回归 greedy 生成

上一章完成的是一次模型 forward。本章只增加最小控制循环：取最后位置 logits，用 `argmax` 选择一个 token，追加到输入，再次运行模型。

## 为什么需要生成循环

`TinyDenseCausalLM.forward(input_ids)` 返回 `[B,S,V]` logits，不会自动延长序列。语言模型一次只给出“每个已有位置之后更倾向哪个词表项”的分数。要得到多个新 token，调用方必须重复：

```text
input_ids
-> model
-> logits[:, -1, :]
-> argmax
-> next_token [B,1]
-> append
-> repeat
```

这也划清三个边界：模型负责算 logits，选择策略负责从 logits 选 token，生成循环负责维护不断增长的 `input_ids`。

## 它在整机中的位置

```text
手工整数 token IDs
  -> TinyDenseCausalLM.forward
  -> logits
  -> 取最后位置
  -> greedy argmax
  -> 追加 token ID
  -> 再次 forward
```

本章位于模型外部，不修改 `TinyDenseCausalLM` 的公共 API。独立示例是 [`examples/run_tiny_greedy.py`](../../examples/run_tiny_greedy.py)；它与单次 forward 示例 [`examples/run_tiny_dense.py`](../../examples/run_tiny_dense.py) 分开，避免混淆两个概念。

## Shape ledger

设当前长度为 `S_t`，词表大小为 `V`：

| 步骤 | 表达式 | shape |
| --- | --- | --- |
| 当前序列 | `input_ids` | `[B,S_t]` |
| 一次 forward | `logits = model(input_ids)` | `[B,S_t,V]` |
| 最后位置分数 | `logits[:, -1, :]` | `[B,V]` |
| greedy 选择并保留序列维 | `argmax(dim=-1, keepdim=True)` | `[B,1]` |
| 追加 | `cat((input_ids, next_token), dim=1)` | `[B,S_t+1]` |

`keepdim=True` 很关键。`torch.cat(..., dim=1)` 要求两边都是 rank-2，且 batch 维相同。每轮只增加一个 token，所以长度从 `S_t` 变成 `S_t + 1`。

## 最小 greedy 控制流

下面完整嵌入示例实际调用的 `greedy_generate()`：

<!-- source-sync: examples/run_tiny_greedy.py::greedy_generate -->
```python
def greedy_generate(
    model: nn.Module, input_ids: Tensor, max_new_tokens: int
) -> tuple[Tensor, list[Tensor]]:
    """Append one greedy token per step and expose each step for teaching."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    generated = input_ids
    new_tokens: list[Tensor] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            new_tokens.append(next_token)
            generated = torch.cat((generated, next_token), dim=1)
    return generated, new_tokens
```

### 1. 校验 max_new_tokens

负数步数没有合理语义，因此函数在进入推理前直接抛出 `ValueError`。`0` 合法：循环不执行，最终序列就是输入，轨迹为空。

### 2. 初始化结果与逐步轨迹

`generated = input_ids` 保存当前完整序列；`new_tokens` 单独记录每一步选择出的 `[B,1]` token。前者用于下一轮模型输入，后者用于教学、打印和测试每一步。

### 3. 进入 inference_mode

整个循环位于 `torch.inference_mode()` 内，不建立 autograd 图，也不保存训练反向传播所需状态。这里是纯推理控制流。

### 4. 每轮运行模型并只取 last logits

`model(generated)` 返回当前长度上的 `[B,S_t,V]` logits。切片 `logits[:, -1, :]` 只保留最后位置 `[B,V]`，因为该位置已经因果地读取全部现有前缀，对应本轮“下一个 token”的分数。

### 5. 用 argmax 选择 token，并以 keepdim 保持 rank

`.argmax(dim=-1, keepdim=True)` 沿词表维选择最高分 ID，同时直接保留长度维，结果为 `[B,1]`。这样无需额外 `unsqueeze`，即可与 rank-2 的 `generated` 拼接。greedy 不引入随机性，适合观察和测试控制流，但不是唯一生成策略。

### 6. 记录 token，再拼接到当前序列

`new_tokens.append(next_token)` 先保存本轮轨迹；`torch.cat((generated, next_token), dim=1)` 再把序列长度从 `S_t` 增加到 `S_t+1`。下一轮会重新计算增长后的完整序列，因为这里没有 KV Cache。

### 7. 返回最终序列和完整生成轨迹

循环结束后，`generated` 包含原始输入与全部新增 token；`new_tokens` 按执行顺序保留恰好 `max_new_tokens` 个 `[B,1]` 张量。

查看完整文件：[`examples/run_tiny_greedy.py`](../../examples/run_tiny_greedy.py)。循环调用的模型实现见 [`src/qwen3_moe/model.py`](../../src/qwen3_moe/model.py)，单次 forward 对照见 [`examples/run_tiny_dense.py`](../../examples/run_tiny_dense.py)。

## 本章明确没有什么

这个最小循环有意不包含以下能力：

- **没有 Tokenizer**：输入和输出都是手工整数 token IDs，不在文本与 token 之间转换。
- **没有 EOS**：循环不会因为生成了某个“结束”ID而提前停止，只运行固定 `max_new_tokens` 步。
- **没有 sampling**：没有 temperature、top-k、top-p 或随机抽样，只有确定性的 `argmax`。
- **没有 KV Cache**：每一轮把增长后的完整序列重新送进模型，重复计算全部前缀。
- **没有语言语义保证**：模型随机初始化，没有真实 checkpoint；生成的 ID 只证明控制流可运行。

这些不是遗漏，而是本章的范围边界。先把一次 forward 和循环的关系看清，再在后续课程引入生产能力。

## 最小实验

运行仓库示例：

```bash
uv run python examples/run_tiny_greedy.py
```

也可以现在直接运行下面的独立实验。它调用真实模型，但不创建任何新源码文件：

```bash
uv run python - <<'PY'
import torch
from qwen3_moe import DenseConfig, TinyDenseCausalLM

torch.manual_seed(7)
config = DenseConfig(
    vocab_size=11,
    hidden_size=8,
    intermediate_size=12,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
    rope_theta=100.0,
)
model = TinyDenseCausalLM(config).eval()
generated = torch.tensor([[1, 2, 3], [4, 5, 6]])

with torch.inference_mode():
    for step in range(3):
        logits = model(generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        print(f"step={step} next_shape={tuple(next_token.shape)} next={next_token.squeeze(-1).tolist()}")
        generated = torch.cat((generated, next_token), dim=1)

print("final shape:", tuple(generated.shape))
print("final IDs:", generated.tolist())
PY
```

预期结构性结果：batch 始终为 2；每步 `next_token` 都是 `[2,1]`；三步后长度从 3 增加到 6；所有新 ID 都在 `[0, vocab_size)`。具体 ID 由随机初始化权重决定，不应解释成文本。

## 对应测试

本章复用真实模型测试来保证循环所依赖的边界：

- [`tests/test_model.py`](../../tests/test_model.py) 的 `test_tiny_dense_model_runs_end_to_end_on_cpu()`：保证输入 `[B,S]` 能得到 `[B,S,V]` logits。
- 同文件的 `test_causal_prefix_logits_do_not_depend_on_a_future_token()`：保证最后位置预测建立在合法因果前缀上。
- 同文件的 `test_model_rejects_invalid_token_ids()`：保证追加后再次输入模型的 token 必须在词表范围内且为整数。

[`tests/test_greedy_example.py`](../../tests/test_greedy_example.py) 验证每步长度 `+1`、`next_token` 为 `[B,1]`、ID 在词表范围、batch 不变，以及固定种子和相同输入得到确定结果。随完整测试一起运行：

```bash
uv run pytest
uv run python examples/run_tiny_greedy.py
```

## 常见误解

1. **“一次 forward 会生成整段序列。”** 一次 forward 只返回当前输入各位置的 logits；循环才让序列增长。
2. **“应该对所有 `[B,S,V]` logits 做 argmax 再全部追加。”** 每轮只使用最后位置 `[B,V]`，每个 batch 追加一个 token。
3. **“`argmax` 默认输出就是 `[B,1]`。”** 默认 `keepdim=False` 时沿词表维 argmax 后是 `[B]`；真实函数显式传入 `keepdim=True` 保留长度维。
4. **“greedy 输出可复现就说明模型有语言能力。”** 确定性只来自固定参数、固定输入和非随机选择，不代表语义质量。
5. **“每轮完整 forward 就是高效生成。”** 没有 KV Cache 时会重复计算前缀；本章只展示正确控制流，不讨论效率优化。
6. **“整数 ID 可以直接当文本读。”** 没有 Tokenizer 就没有可靠的文本编码和解码关系。

## 深入教程

- [Week 3：Token、logits 与 Causal LM](../tutorials/week03-token-logits-causal-lm.md)：深入理解词表、最后位置 logits 和 next-token 边界。
- [Week 7：Dense Decoder](../tutorials/week07-dense-decoder.md)：回顾每次循环内部执行的完整 Dense forward。
- [00：LLM 推理整机导览](00-inference-overview.md)：回看 Tokenizer、模型、选择策略和循环各自的职责边界。

## 回到整体

Dense 主线现在形成了最小闭环：

```text
当前 IDs [B,S]
-> Dense forward [B,S,V]
-> 最后位置 [B,V]
-> greedy token [B,1]
-> 新 IDs [B,S+1]
-> 重复
```

你应留下四项证据：生成循环位置图、每轮 shape ledger、一次真实 `TinyDenseCausalLM` 循环调用，以及长度/范围/确定性中的至少一项检查。

下一章不再扩展生成能力，而是说明 [Dense MLP 到 MoE 的稳定替换边界](08-dense-to-moe.md)。
