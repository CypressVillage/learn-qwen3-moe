# 01：模型外壳

本章沿 `TinyDenseCausalLM` 的最外层理解 token IDs、Embedding、hidden states、Final Norm、LM Head 和 logits。Decoder Layers 暂时仍是保持 `[B,S,D]` 接口的黑盒。

## 为什么需要模型外壳

神经网络不能直接对字符串做矩阵计算，也不能把内部宽度 `D` 的向量直接当作词表 ID。模型外壳负责两次空间转换：入口把离散 ID 查表成连续 hidden states；出口把 hidden states 投影成对整个词表的 logits。

如果删除 Embedding，整数 ID 会被误当成连续数值特征。如果删除 LM Head，模型虽有内部表示，却无法回答“词表中哪个 token 更合适”。

## 它位于整机的什么位置

```text
Tokenizer 输出
-> input_ids [B,S]
-> Embedding
-> hidden_states [B,S,D]
-> L x Decoder Layer（本章当黑盒）
-> final_norm [B,S,D]
-> lm_head
-> logits [B,S,V]
-> next-token 选择（模型 forward 之外）
```

`TinyDenseCausalLM.__init__()` 定义“有哪些部件”，`TinyDenseCausalLM.forward()` 定义“一次调用按什么顺序流过这些部件”。调用时应写 `model(input_ids)`，而不是把 `model.forward(input_ids)` 当作常规入口；前者经过 `nn.Module` 的标准调用机制。

## Shape ledger

设 `B=2, S=4, V=11, D=8, L=2`：

| 操作 | 参数 shape | 输入 shape | 输出 shape |
| --- | --- | --- | --- |
| `nn.Embedding(V,D)` | `embedding.weight [V,D]=[11,8]` | `input_ids [2,4]` | `[2,4,8]` |
| 每个 `DenseDecoderLayer` | 多组参数 | `[2,4,8]` | `[2,4,8]` |
| `RMSNorm(D)` | `weight [D]=[8]` | `[2,4,8]` | `[2,4,8]` |
| `nn.Linear(D,V,bias=False)` | `lm_head.weight [V,D]=[11,8]` | `[2,4,8]` | `[2,4,11]` |

Embedding 是查表：每个 ID 选择 `embedding.weight` 的一行，所以保留 `B/S`，新增 `D`。LM Head 是线性投影：对每个位置独立执行等价计算 `normalized @ lm_head.weight.T`，把最后一维 `D` 换成 `V`。

## 先读真实代码：外壳注册了哪些部件

先带着一个任务读构造函数：按执行顺序确认配置如何保存，Embedding、Decoder stack、Final Norm 和 LM Head 如何注册，以及可选的权重绑定究竟绑定了什么。

<!-- source-sync: src/qwen3_moe/model.py::TinyDenseCausalLM.__init__ -->
```python
def __init__(self, config: DenseConfig) -> None:
    super().__init__()
    self.config = config
    self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
    self.layers = nn.ModuleList(
        [DenseDecoderLayer(config) for _ in range(config.num_hidden_layers)]
    )
    self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    if config.tie_word_embeddings:
        self.lm_head.weight = self.embedding.weight
```

`super().__init__()` 必须最先建立 `nn.Module` 的注册机制，随后 `self.config` 保存已经验证过的结构参数。这个构造函数不检查运行时 `input_ids`；rank、dtype、范围和 device 检查属于 `forward()` 的入口职责。

模块按数据流顺序注册。`nn.Embedding(V,D)` 准备 `[V,D]` 查找表；列表推导式每次新建一个 `DenseDecoderLayer`，再由 `nn.ModuleList` 正确登记全部层参数；`RMSNorm(D)` 保持 hidden shape；无 bias 的 `nn.Linear(D,V)` 准备输出投影。注册顺序本身不执行数据，真正的调用顺序由 `forward()` 决定。

这些定义直接确定外壳 shape：ID `[B,S]` 经 Embedding 变成 `[B,S,D]`，每个 Decoder Layer 依靠内部 residual 保持 `[B,S,D]`，Final Norm 仍不改 shape，LM Head 最后得到 `[B,S,V]`。真实 `forward()` 还会创建 `positions [B,S]` 和 `attention_mask [1,1,S,S]` 供 Decoder 使用，它们不参与入口查表或出口投影。

如果 `tie_word_embeddings=True`，最后一行赋值让 `lm_head.weight` 与 `embedding.weight` 指向同一个 Parameter，而不是复制 values。构造阶段也没有 debug 分支；debug 是 `forward(return_debug=True)` 在真实模块调用过程中收集中间激活，不会改变这里注册的组件树。

完整文件：[`src/qwen3_moe/model.py`](../../src/qwen3_moe/model.py)、[`src/qwen3_moe/norms.py`](../../src/qwen3_moe/norms.py)、[`src/qwen3_moe/config.py`](../../src/qwen3_moe/config.py)。

## 最小实验：核对查表与输出投影

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
)
model = TinyDenseCausalLM(config).eval()
input_ids = torch.tensor([[1, 2, 1], [3, 4, 5]], dtype=torch.long)

with torch.inference_mode():
    logits, debug = model(input_ids, return_debug=True)
    explicit_logits = debug["final_norm"] @ model.lm_head.weight.T

embedding = debug["embedding"]

print("input/embedding/logits:", input_ids.shape, embedding.shape, logits.shape)
print("weights:", model.embedding.weight.shape, model.lm_head.weight.shape)
print("same ID, same lookup:", torch.equal(embedding[0, 0], embedding[0, 2]))
torch.testing.assert_close(embedding, model.embedding.weight[input_ids], atol=0, rtol=0)
torch.testing.assert_close(logits, explicit_logits, atol=1e-6, rtol=1e-6)
print("lookup and LM Head checks passed")
PY
```

这个实验调用了完整真实模型，但只验证外壳的两个局部事实。中间 Decoder 已经运行，不应据此说 `embedding` 直接进入 LM Head；显式投影使用的是所有 Decoder 之后的 `final_norm`。

## 对应测试

- [`test_tiny_dense_model_runs_end_to_end_on_cpu`](../../tests/test_model.py)：验证 `[2,4] -> [2,4,8] -> [2,4,11]`，以及每层保持 hidden contract。
- [`test_tied_embeddings_share_the_same_parameter`](../../tests/test_model.py)：验证开启 `tie_word_embeddings=True` 后，`embedding.weight is lm_head.weight`。
- [`test_model_rejects_invalid_token_ids`](../../tests/test_model.py)：验证 rank、整数 dtype、正 batch/sequence 和词表范围。
- [`test_rms_norm_matches_the_reference_formula`](../../tests/test_norms.py)：验证 Final Norm 使用的真实 `RMSNorm` 数值公式且保持 shape。

```bash
uv run pytest tests/test_model.py tests/test_norms.py
```

## 常见误解与受控错误

- **“ID 8 比 ID 4 大一倍，所以语义也更强。”** ID 只是行号，没有连续数值含义。
- **“Embedding 做的是普通 `input_ids @ weight`。”** 输入是索引表 `[B,S]`，不是 one-hot 特征；真实操作等价于 `weight[input_ids]`。
- **“LM Head 输出的是概率。”** 它输出任意实数 logits；softmax 或 token 选择在模型外另做。
- **“Embedding 和 LM Head 都是 `[V,D]`，所以一定共享权重。”** shape 和 values 相同都不等于同一个 Parameter；只有显式绑定后 `is` 才为真。
- **“Final Norm 改变 hidden 宽度。”** `RMSNorm(D)` 保持 `[B,S,D]`，LM Head 才把 `D` 换成 `V`。

受控观察越界 ID：

```bash
uv run python - <<'PY'
import torch
from qwen3_moe import DenseConfig, TinyDenseCausalLM

model = TinyDenseCausalLM(DenseConfig(11, 8, 12, 1, 2, 1, 4))
try:
    model(torch.tensor([[0, 11]], dtype=torch.long))
except ValueError as error:
    print(error)
PY
```

词表大小为 11 时合法范围是 `[0,11)`，即 0 到 10。不要用 `clamp` 把 11 静默改成 10；应回到 Tokenizer 或输入构造处修复契约。

## 深入教程

- [第三周：Token、Logits 与因果语言建模](../tutorials/week03-token-logits-causal-lm.md)：查表、LM Head、权重绑定、softmax 和 next-token shape。
- [第二周：矩阵乘法与 `nn.Module`](../tutorials/week02-matmul-nn-module.md)：`nn.Module`、Parameter、ModuleList、`nn.Linear` 权重方向和 inference mode。
- [第五周：RMSNorm、RoPE 与 QK Norm](../tutorials/week05-rmsnorm-rope-qk-norm.md)：深入 `RMSNorm` 的公式与数值稳定性。

## 回到完整推理流程

你现在能解释模型外壳：

```text
input_ids [B,S]
-> [Embedding 已理解]
-> [Decoder Layers 仍是黑盒]
-> [Final Norm 已定位]
-> [LM Head 已理解]
-> logits [B,S,V]
```

请留下本章证据：外壳位置图、参数与激活的 shape ledger、一次 `model(..., return_debug=True)` 调用，以及模型/Norm 测试结果。下一章进入 [02：Decoder Layer](02-decoder-layer.md)，打开中间黑盒，但仍不展开 Attention 内部公式。
