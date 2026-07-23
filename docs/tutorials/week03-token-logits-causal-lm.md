# 第三周教程：Token、Logits 与因果语言建模

**目录**

- [这周要学会什么](#goals)
- [学习方式](#method)
- [开始前的环境检查](#environment-check)
- [模块 1：手工微型词表与 Tokenizer 边界](#module-1)
- [模块 2：Embedding 查表](#module-2)
- [模块 3：LM Head 与 logits](#module-3)
- [模块 4：稳定 softmax 与概率](#module-4)
- [模块 5：temperature 与 greedy next-token](#module-5)
- [模块 6：因果错位与 teacher forcing](#module-6)
- [综合任务：走通微型 token-to-logits 数据流](#capstone)
- [最终验收](#acceptance)
- [提示](#hints)
- [参考答案](#answers)
- [术语与速查表](#glossary)
- [下一周预告](#next-week)

<a id="goals"></a>
## 这周要学会什么

前两周建立了 Tensor、矩阵乘法和 `nn.Module` 基础。本周第一次走通语言模型输出端的数据流：

```text
文本边界 -> token_ids [B,S] -> hidden [B,S,D]
         -> logits [B,S,V] -> last_logits [B,V]
         -> next_token_ids [B,1]
```

完成后，你应该能够：

1. 区分文本处理、整数 token ID 和神经网络 `forward` 的边界。
2. 解释 `nn.Embedding(V,D)` 怎样把 `[B,S]` 变成 `[B,S,D]`。
3. 解释 LM Head 怎样把 `[B,S,D]` 变成 logits `[B,S,V]`。
4. 区分 logits、概率和最终选择的 token ID。
5. 实现减最大值的稳定 softmax，并验证概率沿 `V` 维求和约为 1。
6. 解释正 temperature 如何改变概率分布，但不改变 greedy argmax 的排序。
7. 从最后一个位置取得 `[B,V]`，并保留二维输出得到 `[B,1]`。
8. 构造 next-token 任务的输入与目标错位，并正确解释 teacher forcing。

本周继续使用微型词表和固定小张量。CPU 是完整主路径；不下载真实 Tokenizer、模型、checkpoint 或数据集，也不实现 Attention 和完整生成循环。

<a id="method"></a>
## 学习方式

建议投入 **6-10 小时**。每段核心代码继续遵循 **Predict -> Run -> Explain**：

1. **Predict：** 写出 values、shape、dtype、目标轴和预期 token。
2. **Run：** 保留完整输出或受控错误。
3. **Explain：** 用轴语义解释结果，不只说“代码能跑”。

统一记号：`B` 是 batch，`S` 是序列长度，`V` 是词表大小，`D` 是隐藏宽度。凡是 softmax、argmax 或索引，都要先说清操作的是哪一维。

<a id="environment-check"></a>
## 开始前的环境检查

从仓库根目录运行锁定环境：

```bash
uv --version
uv sync --locked --python 3.11.15
uv run python scripts/check_environment.py
uv run pytest
```

环境检查总体应为 `[PASS]`；CPU-only 机器上的 CUDA `[SKIPPED]` 不阻塞本周学习。保存教程代码后用 `uv run python <文件路径>` 运行，不调用系统 Python。

<a id="module-1"></a>
## 模块 1：手工微型词表与 Tokenizer 边界

真实 Tokenizer 会处理子词、特殊 token 和规范化。本周只用空格切分的手工词表观察边界，**不能把它当作 Qwen Tokenizer 的实现**。

```python
import torch

token_to_id = {"<pad>": 0, "我": 1, "喜欢": 2, "学习": 3, "模型": 4, "。": 5}
id_to_token = {token_id: token for token, token_id in token_to_id.items()}

def encode(text):
    tokens = text.split()
    unknown = [token for token in tokens if token not in token_to_id]
    if unknown:
        raise ValueError(f"unknown token: {unknown[0]!r}")
    return torch.tensor([token_to_id[token] for token in tokens], dtype=torch.int64)

def decode(token_ids):
    return " ".join(id_to_token[int(token_id)] for token_id in token_ids)

token_ids = encode("我 喜欢 学习 。")
print("token ids:", token_ids)
print("shape/dtype:", tuple(token_ids.shape), token_ids.dtype)
print("decoded:", decode(token_ids))

try:
    encode("我 喜欢 未知词")
except ValueError as error:
    print("encode error:", error)
```

预期 `token_ids` 为 `[1,2,3,5]`，shape 是 `(4,)`，dtype 是 `torch.int64`。编码/解码发生在模型边界；神经网络 `forward` 接收数值 Tensor，不负责切分原始字符串。

### 常见误区

- token 是字符串片段，token ID 是词表中的整数索引，两者不是同一种对象。
- 手工词表遇到未知 token 时应明确失败；静默替换会隐藏输入问题。
- `decode` 的职责是把 ID 映射回片段，不代表一定能逐字恢复原始空格和格式。

### 练习

<a id="m1-e1-question"></a>
**M1-E1**（[提示](#m1-e1-hint) · [答案](#m1-e1-answer)）：写出 `"我 学习 模型 。"` 的 token 列表、ID、shape 和 dtype，再解码验证。

<a id="m1-e2-question"></a>
**M1-E2**（[提示](#m1-e2-hint) · [答案](#m1-e2-answer)）：说明为什么模型 `forward` 应接收 token ID 而不是原始字符串，以及教学空格切分器不能代表真实 Qwen Tokenizer 的两个原因。

### 模块 1 验收

1. 能否区分 token、token ID 和 Tensor？
2. 能否说明 Tokenizer 位于模型数值 `forward` 之外？
3. 能否让未知 token 错误包含具体输入？

<a id="module-2"></a>
## 模块 2：Embedding 查表

`nn.Embedding(V,D)` 保存权重 `[V,D]`。每个整数 ID 选择同编号的一行，因此输入 `[B,S]` 保留两个轴，并新增宽度 `D`：

```text
token_ids [B,S] -> hidden [B,S,D]
```

```python
import torch
from torch import nn

embedding = nn.Embedding(6, 3)
with torch.no_grad():
    embedding.weight.copy_(torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ]))

token_ids = torch.tensor([[1, 2, 3], [4, 5, 2]], dtype=torch.int64)
hidden = embedding(token_ids)
direct = embedding.weight[token_ids]

print("weight shape:", tuple(embedding.weight.shape))
print("token/hidden shape:", tuple(token_ids.shape), tuple(hidden.shape))
print(hidden)
torch.testing.assert_close(hidden, direct, atol=0, rtol=0)

for bad_ids in (torch.tensor([[1.0, 2.0]]), torch.tensor([[1, 6]])):
    try:
        embedding(bad_ids)
    except (RuntimeError, IndexError) as error:
        print("embedding error:", type(error).__name__, str(error).splitlines()[-1])
```

这里 `V=6`，合法 ID 是 `0-5`。Embedding 输入必须是支持的整数索引 dtype；浮点 ID 没有“第 1.5 行”的查表语义。越界 ID 也不能靠 `clamp` 静默修复，应回到编码或词表契约排查。

### 常见误区

- Embedding 查表不是 `token_ids @ weight`；ID 是索引，不是连续特征。
- `[B,S] -> [B,S,D]` 不会把 `V` 轴复制到输出，`V` 只决定表有多少行。
- `padding_idx` 等真实配置以后按模型需求引入，本周不猜测官方设置。

### 练习

<a id="m2-e1-question"></a>
**M2-E1**（[提示](#m2-e1-hint) · [答案](#m2-e1-answer)）：`nn.Embedding(8,4)` 接收 `token_ids [2,5]`，写出权重与输出 shape，并解释每个轴。

<a id="m2-e2-question"></a>
**M2-E2**（[提示](#m2-e2-hint) · [答案](#m2-e2-answer)）：给定三行固定 Embedding 权重和 ID `[2,0,2]`，手算输出 values；解释重复 ID 为什么得到重复行。

### 模块 2 验收

1. 能否推导 `[B,S] -> [B,S,D]`？
2. 能否用 `weight[token_ids]` 对照模块输出？
3. 能否分别诊断错误 dtype 与越界 ID？

<a id="module-3"></a>
## 模块 3：LM Head 与 logits

LM Head 把每个长度为 `D` 的隐藏向量投影成 `V` 个词表分数：

```text
hidden [B,S,D] -> logits [B,S,V]
```

PyTorch `nn.Linear(D,V,bias=False)` 的权重实际保存为 `[V,D]`，因此等价显式计算是 `hidden @ weight.T`。

```python
import torch
from torch import nn

lm_head = nn.Linear(3, 4, bias=False)
with torch.no_grad():
    lm_head.weight.copy_(torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]))

hidden = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]]])
logits = lm_head(hidden)
explicit = hidden @ lm_head.weight.T

print("weight shape:", tuple(lm_head.weight.shape))
print("logits:", logits)
print("logits shape:", tuple(logits.shape))
torch.testing.assert_close(logits, explicit, atol=0, rtol=0)

try:
    lm_head(torch.ones(1, 2, 5))
except RuntimeError as error:
    print("LM Head error:", str(error).splitlines()[-1])
```

第一个隐藏向量 `[1,2,3]` 产生 logits `[1,2,3,6]`。这些数只是未归一化分数：可以为负、可以大于 1，也不要求总和为 1。

### 常见误区

- logits 的最后一维 `V` 表示候选词表条目，不是隐藏特征 `D`。
- `nn.Linear.weight` 是 `[V,D]`，显式 matmul 时必须转置。
- “最大 logit”表示当前规则下最偏好的候选，不表示它已经是概率。

### 练习

<a id="m3-e1-question"></a>
**M3-E1**（[提示](#m3-e1-hint) · [答案](#m3-e1-answer)）：`hidden [2,4,3]` 进入 `nn.Linear(3,7,bias=False)`，写出权重、输出 shape 和收缩维。

<a id="m3-e2-question"></a>
**M3-E2**（[提示](#m3-e2-hint) · [答案](#m3-e2-answer)）：对隐藏向量 `[a,b]` 和权重行 `[[1,0],[0,1],[1,1]]` 手算三个 logits，并说明为什么其和不必为 1。

### 模块 3 验收

1. 能否解释 `[B,S,D] -> [B,S,V]`？
2. 能否写出 `hidden @ lm_head.weight.T`？
3. 能否区分 logits 与概率？

<a id="module-4"></a>
## 模块 4：稳定 softmax 与概率

softmax 把同一位置的 `V` 个 logits 转成非负且和为 1 的概率。直接计算 `exp(logits)` 可能溢出；先减去该行最大值不会改变概率比例：

```python
import torch

def stable_softmax(logits, dim=-1):
    maximum = logits.max(dim=dim, keepdim=True).values
    shifted = logits - maximum
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)

logits = torch.tensor([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
probabilities = stable_softmax(logits, dim=-1)
reference = torch.softmax(logits, dim=-1)

print(probabilities)
print("row sums:", probabilities.sum(dim=-1))
print("finite:", torch.isfinite(probabilities).all().item())
torch.testing.assert_close(probabilities, reference, atol=1e-7, rtol=1e-6)

naive_exp = torch.exp(logits)
naive = naive_exp / naive_exp.sum(dim=-1, keepdim=True)
print("naive finite rows:", torch.isfinite(naive).all(dim=-1))

wrong_axis = stable_softmax(logits, dim=0)
print("wrong-axis column sums:", wrong_axis.sum(dim=0))
```

第二行减去最大值后变成 `[-2,-1,0]`，不再计算 `exp(1002)`。`dim=-1` 指最后的词表轴；沿 batch 轴做 softmax 虽然 shape 不变，却回答了错误问题。

### 常见误区

- softmax 不改变 shape，只改变最后一维数值的解释。
- 减最大值应使用 `keepdim=True`，以便最大值沿原轴广播回去。
- “代码没报错”不能证明 softmax 轴正确，必须检查哪一维的和为 1。

### 练习

<a id="m4-e1-question"></a>
**M4-E1**（[提示](#m4-e1-hint) · [答案](#m4-e1-answer)）：手算 logits `[0,0,0]` 的 softmax，并说明减最大值后的 values。

<a id="m4-e2-question"></a>
**M4-E2**（[提示](#m4-e2-hint) · [答案](#m4-e2-answer)）：对 logits `[B,S,V]=[2,3,5]`，softmax 应使用哪个 dim？输出 shape 和哪一维的和应约为 1？

### 模块 4 验收

1. 能否写出稳定 softmax 的四步计算？
2. 能否解释减最大值为何不改变结果？
3. 能否用有限性、概率和及参考实现三重验证？

<a id="module-5"></a>
## 模块 5：temperature 与 greedy next-token

正 temperature `T` 使用 `softmax(logits / T)`。`T<1` 通常让分布更尖，`T>1` 让分布更平；`T<=0` 不是本教程接受的数值输入。

```python
import torch

def probabilities_with_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return torch.softmax(logits / temperature, dim=-1)

def entropy(probabilities):
    safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    return -(probabilities * safe.log()).sum(dim=-1)

logits = torch.tensor([[[0.0, 1.0, 3.0], [1.0, 4.0, 2.0]]])
for temperature in (0.5, 1.0, 2.0):
    probabilities = probabilities_with_temperature(logits[:, -1, :], temperature)
    print(temperature, probabilities, "max=", probabilities.max().item(),
          "entropy=", entropy(probabilities).item())

last_logits = logits[:, -1, :]
next_token_ids = last_logits.argmax(dim=-1, keepdim=True)
print("last/next shapes:", tuple(last_logits.shape), tuple(next_token_ids.shape))
print("next token ids:", next_token_ids)

for bad_temperature in (0.0, -1.0):
    try:
        probabilities_with_temperature(last_logits, bad_temperature)
    except ValueError as error:
        print("temperature error:", error)
```

这里最后一个位置的最大 logit 位于 ID 1，因此输出 `[[1]]`。正 temperature 下，除法和 softmax 都保持 logits 的大小顺序，所以 greedy argmax 不必先计算概率。

保留 `keepdim=True` 让输出为 `[B,1]`，便于以后沿序列轴追加；若省略则得到 `[B]`。

### 常见误区

- temperature 改变概率差距，不会为 greedy 选择创造新的最大项。
- temperature 为 0 不能直接代入除法。
- 本周只选择一个 next token，不实现 EOS、循环追加或随机采样。

### 练习

<a id="m5-e1-question"></a>
**M5-E1**（[提示](#m5-e1-hint) · [答案](#m5-e1-answer)）：比较 `T=0.5` 与 `T=2` 时 logits `[0,2]` 除温度后的差距，判断哪个分布更尖。

<a id="m5-e2-question"></a>
**M5-E2**（[提示](#m5-e2-hint) · [答案](#m5-e2-answer)）：`logits [3,4,10]` 取最后位置并 argmax，写出 `last_logits` 和保留/不保留维度时的输出 shape。

### 模块 5 验收

1. 能否解释 temperature 对最大概率和熵的方向性影响？
2. 能否拒绝非正 temperature？
3. 能否得到统一的 `next_token_ids [B,1]`？

<a id="module-6"></a>
## 模块 6：因果错位与 teacher forcing

给定完整序列，next-token 任务把输入去掉最后一项，把目标去掉第一项：

```text
完整序列: [我, 喜欢, 学习, 。]
inputs:   [我, 喜欢, 学习]
targets:  [喜欢, 学习, 。]
```

```python
import torch

sequence = torch.tensor([
    [1, 2, 3, 5],
    [1, 2, 4, 5],
], dtype=torch.int64)
inputs = sequence[:, :-1]
targets = sequence[:, 1:]

print("sequence shape:", tuple(sequence.shape))
print("inputs:", inputs, tuple(inputs.shape))
print("targets:", targets, tuple(targets.shape))
assert torch.equal(inputs, sequence[:, :-1])
assert torch.equal(targets, sequence[:, 1:])

fake_logits = torch.zeros(2, 3, 6)
assert fake_logits.shape[:2] == targets.shape
print("prediction positions align with targets")
```

teacher forcing 表示示例中已知的正确 token 被放入输入序列；每个位置的目标仍是下一个 token。它不等于允许模型读取未来信息。真正具有上下文混合能力的模型还需要 causal mask；第四周学习 causal self-attention。

本周的 Embedding 加 LM Head 不混合不同序列位置，因此它只是验证 token-to-logits 接口，不能冒充完整的上下文语言模型。

### 常见误区

- inputs 与 targets shape 相同，但 values 错开一个位置。
- 目标是整数 ID `[B,S]`，logits 为每个目标提供 `V` 个候选 `[B,S,V]`。
- teacher forcing 是数据供给方式，不是取消因果约束。

### 练习

<a id="m6-e1-question"></a>
**M6-E1**（[提示](#m6-e1-hint) · [答案](#m6-e1-answer)）：对序列 `[2,4,1,3,5]` 写出 inputs、targets 和各自长度。

<a id="m6-e2-question"></a>
**M6-E2**（[提示](#m6-e2-hint) · [答案](#m6-e2-answer)）：解释为什么 logits `[B,S,V]` 能与 targets `[B,S]` 逐位置对应，以及 teacher forcing 为什么仍需要因果结构。

### 模块 6 验收

1. 能否无代码写出左移/右移后的 values？
2. 能否解释 logits 与 target 在前两个轴上的对应？
3. 能否区分 teacher forcing 与未来信息泄漏？

<a id="capstone"></a>
## 综合任务：走通微型 token-to-logits 数据流

先预测所有 shape、指定 values、概率和 greedy token，再运行。这个模块没有跨位置上下文混合，只验证接口与数值规则。

```python
import torch
from torch import nn

token_to_id = {"<pad>": 0, "我": 1, "喜欢": 2, "学习": 3, "模型": 4, "。": 5}
id_to_token = {token_id: token for token, token_id in token_to_id.items()}

class TinyTokenToLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 3)
        self.lm_head = nn.Linear(3, 6, bias=False)

    def forward(self, token_ids):
        if token_ids.dtype != torch.int64:
            raise TypeError(f"token_ids must be int64, got {token_ids.dtype}")
        hidden = self.embedding(token_ids)
        return hidden, self.lm_head(hidden)

model = TinyTokenToLogits().eval()
with torch.no_grad():
    model.embedding.weight.copy_(torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
    ]))
    model.lm_head.weight.copy_(torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
    ]))

full_sequences = torch.tensor([[1, 2, 3, 5], [1, 2, 4, 5]], dtype=torch.int64)
token_ids = full_sequences[:, :-1]
targets = full_sequences[:, 1:]

with torch.inference_mode():
    hidden, logits = model(token_ids)
    probabilities = torch.softmax(logits - logits.max(dim=-1, keepdim=True).values, dim=-1)
    last_logits = logits[:, -1, :]
    next_token_ids = last_logits.argmax(dim=-1, keepdim=True)
    assert tuple(token_ids.shape) == (2, 3)
    assert tuple(hidden.shape) == (2, 3, 3)
    assert tuple(logits.shape) == (2, 3, 6)
    assert tuple(last_logits.shape) == (2, 6)
    assert tuple(next_token_ids.shape) == (2, 1)
    assert token_ids.dtype == targets.dtype == torch.int64
    torch.testing.assert_close(hidden, model.embedding.weight[token_ids], atol=0, rtol=0)
    torch.testing.assert_close(logits, hidden @ model.lm_head.weight.T, atol=0, rtol=0)
    torch.testing.assert_close(probabilities, torch.softmax(logits, dim=-1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2, 3), atol=1e-6, rtol=0)
    assert torch.isfinite(probabilities).all()
    assert torch.equal(token_ids, full_sequences[:, :-1])
    assert torch.equal(targets, full_sequences[:, 1:])

decoded_next = [id_to_token[int(token_id)] for token_id in next_token_ids[:, 0]]
print("shape flow:", tuple(token_ids.shape), tuple(hidden.shape), tuple(logits.shape),
      tuple(last_logits.shape), tuple(next_token_ids.shape))
print("next ids/tokens:", next_token_ids[:, 0].tolist(), decoded_next)
print("probability sums:", probabilities.sum(dim=-1))
print("all capstone checks passed")
```

脱离代码口述：整数 ID 查表得到 hidden；LM Head 对最后一维投影得到每个位置的词表 logits；softmax 只沿 `V`；greedy 只看当前输入最后位置；因果 targets 比 inputs 向后错一项。

<a id="acceptance"></a>
## 最终验收

先独立作答，需要时查看同编号提示，最后再核对答案。

### A. 概念题

1. <a id="c1-question"></a>**C1**（[提示](#c1-hint) · [答案](#c1-answer)）：Tokenizer 边界与神经网络 `forward` 各自处理什么？
2. <a id="c2-question"></a>**C2**（[提示](#c2-hint) · [答案](#c2-answer)）：为什么 Embedding 是查表而不是 token ID 与权重做矩阵乘法？
3. <a id="c3-question"></a>**C3**（[提示](#c3-hint) · [答案](#c3-answer)）：解释 `[B,S] -> [B,S,D]` 中保留和新增的轴。
4. <a id="c4-question"></a>**C4**（[提示](#c4-hint) · [答案](#c4-answer)）：LM Head 的 logits `[B,S,V]` 中每个轴表示什么？
5. <a id="c5-question"></a>**C5**（[提示](#c5-hint) · [答案](#c5-answer)）：logits 为什么不是概率？
6. <a id="c6-question"></a>**C6**（[提示](#c6-hint) · [答案](#c6-answer)）：稳定 softmax 为什么减最大值，为什么必须沿 `V` 维？
7. <a id="c7-question"></a>**C7**（[提示](#c7-hint) · [答案](#c7-answer)）：temperature 小于或大于 1 时通常怎样影响分布？
8. <a id="c8-question"></a>**C8**（[提示](#c8-hint) · [答案](#c8-answer)）：greedy argmax 为什么不必先计算 softmax？
9. <a id="c9-question"></a>**C9**（[提示](#c9-hint) · [答案](#c9-answer)）：为什么推荐 `argmax(..., keepdim=True)`？
10. <a id="c10-question"></a>**C10**（[提示](#c10-hint) · [答案](#c10-answer)）：teacher forcing 与允许读取未来 token 有何区别？

### B. 形状、概率与错位推导题

1. <a id="s1-question"></a>**S1**（[提示](#s1-hint) · [答案](#s1-answer)）：`Embedding(10,4)` 接收 `[3,5]`，写出权重、hidden shape 和合法 ID 范围。
2. <a id="s2-question"></a>**S2**（[提示](#s2-hint) · [答案](#s2-answer)）：`hidden [2,6,4]` 进入 `Linear(4,9,bias=False)`，写出权重、logits shape 和显式 matmul。
3. <a id="s3-question"></a>**S3**（[提示](#s3-hint) · [答案](#s3-answer)）：对 `[1000,1001,1002]` 写出稳定 softmax 平移后的 logits，并指出概率最大项。
4. <a id="s4-question"></a>**S4**（[提示](#s4-hint) · [答案](#s4-answer)）：`logits [2,7,11]` 经最后位置选择与保维 argmax 后，两个 shape 分别是什么？
5. <a id="s5-question"></a>**S5**（[提示](#s5-hint) · [答案](#s5-answer)）：对序列 `[1,4,2,6]` 写出 causal inputs、targets，并说明对应 logits 的前两维。

### 通过标准

- 概念题至少 **8/10**。
- 推导题至少 **4/5**。
- 综合任务全部断言通过。
- 能不看答案口述 `[B,S] -> [B,S,D] -> [B,S,V] -> [B,V] -> [B,1]`，并解释概率轴与因果错位。

<a id="hints"></a>
## 提示

### 模块练习提示

<a id="m1-e1-hint"></a>**M1-E1：** 按字典逐词替换，四个 token 产生长度 4 的一维整数 Tensor。

<a id="m1-e2-hint"></a>**M1-E2：** 分开考虑字符串规则与数值算子；真实 Tokenizer 还涉及子词和特殊 token。

<a id="m2-e1-hint"></a>**M2-E1：** 表有 `V` 行、每行 `D` 个数；输入轴原样保留。

<a id="m2-e2-hint"></a>**M2-E2：** 每个 ID 独立选择同编号行，ID 相同不会自动去重。

<a id="m3-e1-hint"></a>**M3-E1：** `nn.Linear` 权重先写 `[out,in]`，输出只替换最后一维。

<a id="m3-e2-hint"></a>**M3-E2：** 每一行权重与 `[a,b]` 点积；归一化发生在 softmax，不在 LM Head。

<a id="m4-e1-hint"></a>**M4-E1：** 三项完全相等，指数也相等。

<a id="m4-e2-hint"></a>**M4-E2：** `V` 是最后一维，所以使用 `dim=-1` 或 `dim=2`。

<a id="m5-e1-hint"></a>**M5-E1：** 分别得到 `[0,4]` 与 `[0,1]`，比较两项差距。

<a id="m5-e2-hint"></a>**M5-E2：** `[:, -1, :]` 删除 `S` 轴；argmax 默认再删除 `V` 轴。

<a id="m6-e1-hint"></a>**M6-E1：** inputs 去末项，targets 去首项。

<a id="m6-e2-hint"></a>**M6-E2：** 前两轴标记同一批预测位置；第三轴提供每个位置的候选分数。

### 最终验收提示

<a id="c1-hint"></a>**C1：** 一个处理字符串与 ID，另一个处理 Tensor 数值。

<a id="c2-hint"></a>**C2：** ID 表示行号，不表示长度为 `V` 的连续向量。

<a id="c3-hint"></a>**C3：** batch 和位置都仍存在，每个位置多出一个表示向量。

<a id="c4-hint"></a>**C4：** 最后轴枚举词表候选。

<a id="c5-hint"></a>**C5：** 检查范围、符号和求和约束。

<a id="c6-hint"></a>**C6：** 平移不改变指数比例；归一化候选必须发生在词表轴。

<a id="c7-hint"></a>**C7：** 除以小数放大差距，除以较大数缩小差距。

<a id="c8-hint"></a>**C8：** 正数缩放和 softmax 都保持排序。

<a id="c9-hint"></a>**C9：** 想一想以后怎样把新 ID 沿序列轴追加。

<a id="c10-hint"></a>**C10：** 已知完整训练样本不代表每个位置可访问完整样本。

<a id="s1-hint"></a>**S1：** 合法索引从 0 开始，到行数减 1。

<a id="s2-hint"></a>**S2：** `Linear(in,out)` 保存 `[out,in]`。

<a id="s3-hint"></a>**S3：** 减去 1002。

<a id="s4-hint"></a>**S4：** 先删除 `S` 轴，再把 `V` 轴压成长度 1。

<a id="s5-hint"></a>**S5：** 三个输入位置对应三个 next-token 目标。

<a id="answers"></a>
## 参考答案

### 模块练习答案

<a id="m1-e1-answer"></a>**M1-E1：** token 为 `我/学习/模型/。`，ID 为 `[1,3,4,5]`，shape `[4]`，dtype `int64`；解码得到用空格连接的同一组教学 token。

<a id="m1-e2-answer"></a>**M1-E2：** Tokenizer 负责字符串切分和 ID 映射，模型 `forward` 负责 Tensor 计算。真实 Qwen Tokenizer 使用子词规则并处理特殊 token 等配置，不是简单空格切分。

<a id="m2-e1-answer"></a>**M2-E1：** 权重 `[8,4]`，输出 `[2,5,4]`；2 是 batch，5 是位置，4 是每个位置的表示宽度。

<a id="m2-e2-answer"></a>**M2-E2：** 输出依次是权重第 2、0、2 行；重复 ID 两次执行相同查表，因此第 1 与第 3 个输出向量相同。

<a id="m3-e1-answer"></a>**M3-E1：** 权重 `[7,3]`，输出 `[2,4,7]`；显式计算用 `[2,4,3] @ [3,7]`，收缩 3。

<a id="m3-e2-answer"></a>**M3-E2：** logits 为 `[a,b,a+b]`。它们是未归一化分数，没有非负或和为 1 的约束。

<a id="m4-e1-answer"></a>**M4-E1：** 减最大值后仍为 `[0,0,0]`，三个指数均为 1，所以概率均为 `1/3`。

<a id="m4-e2-answer"></a>**M4-E2：** 使用 `dim=-1`（即 2），输出仍为 `[2,3,5]`，每个 `[V]` 切片的和约为 1。

<a id="m5-e1-answer"></a>**M5-E1：** `T=0.5` 得 `[0,4]`，`T=2` 得 `[0,1]`；前者差距更大，softmax 更尖。

<a id="m5-e2-answer"></a>**M5-E2：** `last_logits` 为 `[3,10]`；不保维 argmax 为 `[3]`，保维后为 `[3,1]`。

<a id="m6-e1-answer"></a>**M6-E1：** inputs `[2,4,1,3]`，targets `[4,1,3,5]`，长度都为 4。

<a id="m6-e2-answer"></a>**M6-E2：** logits 与 targets 的 `[B,S]` 标记相同预测位置，logits 的额外 `V` 轴提供候选。teacher forcing 提供正确历史输入，但因果结构仍必须阻止位置访问未来位置。

### 最终验收答案

<a id="c1-answer"></a>**C1：** Tokenizer 边界处理文本、token 和整数 ID；模型 `forward` 接收 ID Tensor 并执行 Embedding、Decoder、LM Head 等数值计算。

<a id="c2-answer"></a>**C2：** token ID 是离散行索引，Embedding 直接选择权重的一行。普通矩阵乘法要求输入本身是可相乘的数值特征，语义不同。

<a id="c3-answer"></a>**C3：** `B/S` 保留，因为每条序列的每个位置仍存在；每个 ID 被替换为长度 `D` 的向量，因此新增末轴。

<a id="c4-answer"></a>**C4：** `B` 是 batch，`S` 是预测位置，`V` 枚举该位置的全部词表候选分数。

<a id="c5-answer"></a>**C5：** logits 可以为任意实数，不限于 `[0,1]`，沿 `V` 维也不要求和为 1；softmax 后才成为本教程中的概率。

<a id="c6-answer"></a>**C6：** 减最大值避免指数溢出，且共同平移不改变归一化后的比例。沿 `V` 维归一化才表示同一位置各 token 候选的概率。

<a id="c7-answer"></a>**C7：** 正且小于 1 的 temperature 放大 logits 差距，分布通常更尖；大于 1 缩小差距，分布通常更平。

<a id="c8-answer"></a>**C8：** 除以正 temperature 和 softmax 都是严格保持大小顺序的变换，因此最大 logit 的索引不变。

<a id="c9-answer"></a>**C9：** 保维得到 `[B,1]`，可直接作为一个新序列位置沿 `S` 轴拼接；`[B]` 会丢失序列轴。

<a id="c10-answer"></a>**C10：** teacher forcing 把已知正确历史 token 作为训练式输入；因果结构仍限制每个位置只能使用允许的过去信息，不能读取未来位置。

<a id="s1-answer"></a>**S1：** 权重 `[10,4]`，hidden `[3,5,4]`，合法 ID 为 `0-9`。

<a id="s2-answer"></a>**S2：** 权重 `[9,4]`，logits `[2,6,9]`，显式式为 `hidden @ linear.weight.T`，即 `[2,6,4] @ [4,9]`。

<a id="s3-answer"></a>**S3：** 平移后为 `[-2,-1,0]`，第三项指数最大，因此第三个候选概率最大。

<a id="s4-answer"></a>**S4：** `last_logits` 为 `[2,11]`，保维 argmax 为 `[2,1]`。

<a id="s5-answer"></a>**S5：** inputs `[1,4,2]`，targets `[4,2,6]`；对应 logits 的前两维为 `[1,3]`（单 batch、三个预测位置）。

<a id="glossary"></a>
## 术语与速查表

| 边界/运算 | 输入 | 输出 | 关键检查 |
| --- | --- | --- | --- |
| 教学编码 | 文本 token | `token_ids [S]` | token 必须在词表中 |
| Embedding | `[B,S]` | `[B,S,D]` | 整数 dtype，ID 在 `[0,V)` |
| LM Head | `[B,S,D]` | `[B,S,V]` | `Linear(D,V)` 权重 `[V,D]` |
| softmax | logits `[... ,V]` | probabilities 同 shape | 沿 `V`，先减最大值 |
| temperature | logits / `T` | 调整后的概率 | `T>0` |
| greedy | last logits `[B,V]` | IDs `[B,1]` | `argmax(-1, keepdim=True)` |
| 因果错位 | sequence `[B,L]` | inputs/targets `[B,L-1]` | 去末项 / 去首项 |

调试顺序：先打印 token 与 ID，再查 ID 范围和 dtype；接着逐边界核对 `[B,S] -> [B,S,D] -> [B,S,V]`；概率异常先查 softmax 轴与有限性；next token shape 异常再查最后位置索引和 `keepdim`。

<a id="next-week"></a>
## 下一周预告

第四周将学习 causal self-attention、Multi-Head Attention 与 GQA。届时 Decoder 会在不改变外部 `hidden [B,S,D]` 接口的前提下，让每个位置聚合允许看到的上下文，再由 LM Head 产生 `[B,S,V]`。

开始前请确保你能解释：本周 Embedding 与 LM Head 示例为什么没有跨位置上下文能力，以及 causal mask 未来要解决什么问题。
