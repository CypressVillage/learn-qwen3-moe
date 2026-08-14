# 04：Attention，让当前位置读取左侧上下文

上一章已经说明了张量怎样变形。本章把这些操作串成真正的因果 Attention，并始终对照仓库中的 `GroupedQueryAttention`。

## 为什么需要 Attention

Embedding 只把每个 token ID 独立地变成向量。MLP 也只在同一个 token 位置内部变换特征。要让“当前位置”利用前文，例如让第 4 个 token 的表示受到第 1 至 3 个 token 影响，就需要一个跨 token 混合模块。

因果语言模型还多一条限制：位置 `i` 可以读取 `0..i`，不能偷看 `i+1..S-1`。Attention 同时解决“读哪些位置”和“按多大权重读取”两个问题。

## 它在整机中的位置

```text
input_ids [B,S]
  -> Embedding [B,S,D]
  -> Decoder Layer
       -> input_norm
       -> GroupedQueryAttention   <- 本章
       -> 第一次 residual add
       -> post_attention_norm
       -> SwiGLU
       -> 第二次 residual add
  -> Final Norm
  -> LM Head
  -> logits [B,S,V]
```

Attention 接收并返回相同 contract 的浮点张量 `[B,S,D]`，因此它的更新量可以和 residual 相加。它不接收 token ID，也不直接输出词表 logits。

## Shape ledger

记号：

- `B`：batch size
- `S`：sequence length
- `D`：hidden size
- `Hq`：query head 数，即 `num_attention_heads`
- `Hkv`：key/value head 数，即 `num_key_value_heads`
- `Dh`：每个 head 的宽度，即 `head_dim`
- `G = Hq / Hkv`：每个 KV head 服务的 query head 数

配置要求 `D = Hq * Dh`，并且 `Hq` 能被 `Hkv` 整除。

| 阶段 | 真实符号 | shape |
| --- | --- | --- |
| 输入 | `hidden_states` | `[B,S,D]` |
| Q 投影并拆 head | `query_projected` | `[B,S,Hq,Dh]` |
| K 投影并拆 head | `key_projected` | `[B,S,Hkv,Dh]` |
| V 投影并拆 head | `value`（transpose 前） | `[B,S,Hkv,Dh]` |
| QK Norm 后换轴 | `query_pre_rope` | `[B,Hq,S,Dh]` |
| QK Norm 后换轴 | `key_pre_rope` | `[B,Hkv,S,Dh]` |
| 位置旋转参数 | `cos`, `sin` | `[B,1,S,Dh]` |
| RoPE 后 | `query`, `key` | `[B,Hq,S,Dh]`, `[B,Hkv,S,Dh]` |
| 展开 KV heads | `repeated_key`, `repeated_value` | `[B,Hq,S,Dh]` |
| 两两位置分数 | `scores` | `[B,Hq,S,S]` |
| 因果 mask | `attention_mask` | 通常为 `[1,1,S,S]`，广播到 scores |
| softmax 权重 | `probabilities` | `[B,Hq,S,S]` |
| 加权读取 | `context` | `[B,Hq,S,Dh]` |
| 合并 heads | `merged` | `[B,S,D]` |
| 输出投影 | `update` | `[B,S,D]` |

## 真实 `forward()` 与执行顺序

下面完整代码就是本章 shape ledger 的执行主体。先通读返回值和 debug 名称，再按代码后的八个阶段逐段核对。

<!-- source-sync: src/qwen3_moe/attention.py::GroupedQueryAttention.forward -->
```python
def forward(
    self,
    hidden_states: Tensor,
    positions: Tensor,
    attention_mask: Tensor | None = None,
    *,
    return_debug: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    require_floating("attention input", hidden_states)
    if hidden_states.ndim != 3:
        raise ValueError("attention input must be rank-3 [B,S,D]")
    batch_size, sequence_length, hidden_size = hidden_states.shape
    if hidden_size != self.hidden_size:
        raise ValueError(
            f"attention expected hidden size {self.hidden_size}, got {hidden_size}"
        )
    if positions.shape != (batch_size, sequence_length):
        raise ValueError(
            f"positions must have shape {(batch_size, sequence_length)}"
        )
    if positions.device != hidden_states.device:
        raise ValueError("positions and hidden states must share a device")

    query_projected = self.q_proj(hidden_states).view(
        batch_size,
        sequence_length,
        self.num_attention_heads,
        self.head_dim,
    )
    key_projected = self.k_proj(hidden_states).view(
        batch_size,
        sequence_length,
        self.num_key_value_heads,
        self.head_dim,
    )
    value = self.v_proj(hidden_states).view(
        batch_size,
        sequence_length,
        self.num_key_value_heads,
        self.head_dim,
    )

    query_pre_rope = self.q_norm(query_projected).transpose(1, 2)
    key_pre_rope = self.k_norm(key_projected).transpose(1, 2)
    value = value.transpose(1, 2)
    cos, sin = self.rotary(positions, dtype=hidden_states.dtype)
    query, key = apply_rotary_pos_emb(
        query_pre_rope, key_pre_rope, cos, sin
    )

    repeated_key = key.repeat_interleave(self.group_size, dim=1)
    repeated_value = value.repeat_interleave(self.group_size, dim=1)
    scores = torch.matmul(query, repeated_key.transpose(-1, -2))
    scores = scores / math.sqrt(self.head_dim)

    if attention_mask is None:
        attention_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            diagonal=1,
        ).view(1, 1, sequence_length, sequence_length)
    if attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must have bool dtype")
    if attention_mask.device != hidden_states.device:
        raise ValueError("attention_mask and hidden states must share a device")
    try:
        expanded_mask = attention_mask.expand_as(scores)
    except RuntimeError as exc:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} cannot broadcast "
            f"to scores {tuple(scores.shape)}"
        ) from exc
    if expanded_mask.all(dim=-1).any():
        raise ValueError("attention_mask contains a fully blocked query row")

    masked_scores = scores.masked_fill(expanded_mask, float("-inf"))
    probabilities = torch.softmax(masked_scores.float(), dim=-1).to(scores.dtype)
    context = torch.matmul(probabilities, repeated_value)
    merged = context.transpose(1, 2).contiguous().view(
        batch_size, sequence_length, self.hidden_size
    )
    update = self.out_proj(merged)

    require_finite("attention scores", scores)
    require_finite("attention probabilities", probabilities)
    require_finite("attention update", update)
    if not return_debug:
        return update
    return update, {
        "query_projected": query_projected,
        "key_projected": key_projected,
        "query_pre_rope": query_pre_rope,
        "key_pre_rope": key_pre_rope,
        "query": query,
        "key": key,
        "value": value,
        "repeated_key": repeated_key,
        "repeated_value": repeated_value,
        "scores": scores,
        "probabilities": probabilities,
    }
```

### 1. QKV 投影

入口先验证 `hidden_states [B,S,D]` 是浮点 rank-3 张量、最后一维等于配置的 `hidden_size`，并确认 `positions [B,S]` 位于同一 device。随后同一个输入分别经过 `q_proj`、`k_proj`、`v_proj`：Q 表示当前位置想匹配什么，K 表示每个位置提供什么匹配特征，V 表示匹配后真正读取的内容。

### 2. Head 拆分

三次投影最初都把 head 宽度合并在最后一维。紧跟投影的 `.view(...)` 将 Q 拆为 `[B,S,Hq,Dh]`，将 K/V 拆为 `[B,S,Hkv,Dh]`。这里已经体现 GQA 的参数差异，但还没有把较少的 KV heads 展开。

### 3. QK Norm

`q_norm` 和 `k_norm` 在每个 head 的 `Dh` 维上分别规范化 Q/K，V 不做 QK Norm。之后 `.transpose(1, 2)` 把布局改为 Q `[B,Hq,S,Dh]`、K/V `[B,Hkv,S,Dh]`，让每个 head 可以独立执行后面的矩阵乘法。

### 4. RoPE

`self.rotary(positions, dtype=hidden_states.dtype)` 生成 `cos/sin [B,1,S,Dh]`，再由 `apply_rotary_pos_emb()` 旋转 Q/K。长度为 1 的 head 轴会分别广播到 `Hq` 和 `Hkv`；Q/K shape 不变，V 不旋转。位置 0 是恒等旋转，后续位置以不同角度进入匹配关系。

### 5. GQA

`repeat_interleave(self.group_size, dim=1)` 将 K/V 从 `Hkv` 个 heads 显式展开到 `Hq` 个 heads，使一组连续 query heads 共享同一个原始 KV head。随后 Q 与转置后的 K 在 `Dh` 上收缩，得到 `scores [B,Hq,S,S]`，再除以 `sqrt(Dh)` 控制点积尺度。显式复制便于教学观察，不代表生产实现最节省显存的形式。

### 6. Mask

没有外部 mask 时，代码在 hidden states 的 device 上创建 `[1,1,S,S]` 上三角 bool mask；`True` 表示禁止读取未来位置。自定义 mask 也必须是 bool、同 device，并能广播到 scores。`expand_as` 只建立广播视图；整行全为 `True` 会被拒绝，因为该 query 没有任何合法 key。

```text
query\key  0  1  2  3
0          可  禁 禁 禁
1          可  可 禁 禁
2          可  可 可 禁
3          可  可 可 可
```

### 7. Softmax

`masked_fill` 先把禁止位置设为 `-inf`，然后临时转成 float32 做 softmax，再转回 scores dtype，得到 `probabilities [B,Hq,S,S]`。每行合法位置的概率和为 1，未来位置概率为 0；`probabilities @ repeated_value` 随后沿 key-position 轴加权汇总，产生 `context [B,Hq,S,Dh]`。

### 8. Merge

`context.transpose(1, 2)` 先回到 `[B,S,Hq,Dh]`，`.contiguous().view(B,S,D)` 再合并 heads。`out_proj` 生成最终 `update [B,S,D]`，从而可以加入 residual。有限性检查覆盖 scores、概率和更新量；`return_debug=True` 只额外暴露关键中间张量，不改变主计算路径。

## 完整文件链接

- [`attention.py`](../../src/qwen3_moe/attention.py)：完整 GQA 模块，包括投影层定义和上述 `forward()`。
- [`norms.py`](../../src/qwen3_moe/norms.py)：QK Norm 使用的 `RMSNorm`。
- [`rope.py`](../../src/qwen3_moe/rope.py)：`RotaryEmbedding`、`rotate_half()` 和 `apply_rotary_pos_emb()`。
- [`config.py`](../../src/qwen3_moe/config.py)：head 数、`head_dim` 与 `hidden_size` 约束。
- [`decoder.py`](../../src/qwen3_moe/decoder.py)：Attention 更新如何接入第一次 residual add。

## 最小实验

下面直接运行真实 Attention，并检查最重要的三个事实：输出 shape 不变、未来概率为 0、每行概率和为 1。

```bash
uv run python - <<'PY'
import torch
from qwen3_moe import DenseConfig, GroupedQueryAttention

torch.manual_seed(7)
config = DenseConfig(
    vocab_size=11,
    hidden_size=8,
    intermediate_size=12,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=2,
    rope_theta=100.0,
)
attention = GroupedQueryAttention(config).eval()
hidden = torch.randn(1, 4, 8)
positions = torch.arange(4).view(1, 4)

with torch.inference_mode():
    update, debug = attention(hidden, positions, return_debug=True)

probabilities = debug["probabilities"]
future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
print("update:", update.shape)
print("probabilities:", probabilities.shape)
print("future probabilities:", probabilities.masked_select(future.view(1, 1, 4, 4)))
print("row sums:", probabilities.sum(dim=-1))
print("K heads -> repeated heads:", debug["key"].shape, debug["repeated_key"].shape)
PY
```

受控错误：把 `positions` 改成 `torch.arange(3).view(1, 3)`，调用会因位置 shape 不匹配而失败。这说明位置信息必须和每个 batch、每个序列位置一一对应。

## 对应测试

- [`tests/test_attention.py`](../../tests/test_attention.py) 的 `test_causal_gqa_masks_future_tokens_and_maps_head_groups()`：验证输出 contract、未来概率为 0、概率归一化，以及 GQA 的 head 映射。
- 同文件的 `test_attention_rejects_non_broadcastable_or_fully_blocked_masks()`：验证非法广播和整行屏蔽会被明确拒绝。
- [`tests/test_rope.py`](../../tests/test_rope.py)：验证 RoPE 保持 shape 和范数，位置 0 不旋转，后续位置发生旋转。
- [`tests/test_norms.py`](../../tests/test_norms.py)：验证 QK Norm 所用 `RMSNorm` 的公式、shape 和输入检查。

运行：

```bash
uv run pytest tests/test_attention.py tests/test_rope.py tests/test_norms.py
```

## 常见误解

1. **“Attention 的输出是概率。”** 概率只是中间量；最终输出是对 Value 加权、合并 heads、再经过 `out_proj` 的 `[B,S,D]` 更新量。
2. **“因果 mask 会删除未来 token。”** token 仍在同一批张量中，只是对应分数在 softmax 前被设为负无穷。
3. **“GQA 是减少 query head 数。”** 本实现保留 `Hq` 个 query heads，减少的是独立 K/V heads。
4. **“RoPE 改变 hidden size。”** RoPE 只旋转 Q/K 最后一维中的成对分量，shape 不变。
5. **“能通过 shape 测试就说明因果性正确。”** 错误 mask 也可能输出正确 shape；必须检查未来概率或前缀不受未来 token 影响。

## 深入教程

- [Week 4：Self-Attention 与 GQA](../tutorials/week04-self-attention-gqa.md)：点积、softmax、mask、multi-head 和 GQA 的更多手算与实验。
- [Week 5：RMSNorm、RoPE 与 QK Norm](../tutorials/week05-rmsnorm-rope-qk-norm.md)：规范化公式、旋转推导、受控错误和数值检查。
- [Week 2：矩阵乘法与 `nn.Module`](../tutorials/week02-matmul-nn-module.md)：如果 `matmul` 和线性层仍不熟悉，先复习这里。

## 回到整体

现在 Decoder Layer 的第一半已经展开：

```text
h = x + Attention(RMSNorm(x))
```

你应留下四项证据：一张 Attention 位置图、一份从 `[B,S,D]` 到 `[B,Hq,S,S]` 再回来的 shape ledger、一次 `return_debug=True` 的真实调用，以及一次因果概率测试或非法 mask 受控错误。

下一章学习第二半的 [SwiGLU MLP](05-mlp.md)：Attention 已经在 token 之间搬运信息，MLP 将在每个 token 位置内部变换这些特征。
