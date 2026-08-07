# 可安装 Dense 推理底座整改设计

## 目标

把前七周教程中已经验证的核心实现提取为可安装、可导入、可测试的 Python 包，使每周学习产物能够继续组合，而不是停留在 Markdown 代码块中。本轮完成 CPU FP32 可运行的微型 Dense Causal LM，为后续 MoE Router、专家分发和 KV Cache 提供稳定底座。

同时在个人 Obsidian 知识库中新增两篇长期维护的文档：一篇说明 PyTorch 模型实现的设备、精度和状态管理注意事项；一篇说明深度学习项目在 CPU、不同 CUDA 环境和不同机器之间如何管理依赖。

## 范围

本轮包含：

- 将项目改为标准 `src` layout 的可安装 Python 包。
- 提取配置、RMSNorm、RoPE、causal GQA、SwiGLU、Dense Decoder Layer 和 Tiny Dense Causal LM。
- 将 Week 5-7 综合任务中的关键断言迁移为 pytest。
- 新增一个 CPU 可运行的端到端示例。
- 更新 README、路线图和环境文档，明确代码产物、CPU 基线和后续 GPU 使用方式。
- 新增两篇外部知识库文档并更新对应 Obsidian 索引。

本轮不包含：

- MoE Router、Top-K、专家分发、聚合或 MoE Decoder。
- KV Cache、生成循环、Tokenizer、真实 checkpoint 或权重映射。
- CUDA 专用 kernel、FlashAttention、Triton、量化或性能优化。
- 未经实际 GPU 环境验证的 CUDA 依赖锁定配置。

## 包结构

```text
src/qwen3_moe/
├── __init__.py
├── config.py
├── norms.py
├── rope.py
├── attention.py
├── mlp.py
├── decoder.py
├── model.py
└── debug.py

tests/
├── test_config.py
├── test_norms.py
├── test_rope.py
├── test_attention.py
├── test_mlp.py
├── test_decoder.py
└── test_model.py

examples/
└── run_tiny_dense.py
```

`pyproject.toml` 移除 `package = false`，增加构建后端，使 `uv sync` 安装当前项目。包的公共 API 从 `qwen3_moe.__init__` 导出配置和主要模型类，内部调试辅助函数不作为稳定公共 API。

## 配置

`DenseConfig` 使用 dataclass 保存微型 Dense 模型所需结构参数：

- `vocab_size`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `rms_norm_eps`
- `rope_theta`
- `tie_word_embeddings`

构造时验证所有维度为正、`hidden_size == num_attention_heads * head_dim`、query head 数可被 key/value head 数整除、RoPE head dimension 为偶数。教学默认值只用于示例和测试，不伪装成 Qwen3-30B-A3B 官方配置。

## 模块接口

### RMSNorm

`RMSNorm.forward(x)` 保持输入 shape，沿最后一维计算。为降低低精度误差，归一化统计以 FP32 计算，再转换回输入 dtype 后乘权重。输入、权重的 device 必须一致，不硬编码 CPU 或 CUDA。

### RoPE

`RotaryEmbedding` 将 `inv_freq` 注册为 non-persistent buffer。`forward(positions, dtype)` 在 positions 所在设备上以 FP32 计算 cos/sin，再转换为调用方需要的浮点 dtype。`apply_rotary_pos_emb(q, k, cos, sin)` 不改变 Q/K shape。

### Causal GQA

`GroupedQueryAttention.forward(hidden_states, positions, attention_mask=None, return_debug=False)` 执行：

```text
Q/K/V projection
-> reshape heads
-> QK Norm
-> RoPE
-> GQA head expansion
-> causal attention
-> head merge
-> output projection
```

未传 mask 时模块创建上三角 causal mask，临时 Tensor 使用输入的 device。返回值默认只有 attention update；`return_debug=True` 时返回 update 和必要的中间张量字典。

### SwiGLU

`SwiGLU.forward(x, return_debug=False)` 执行 `down(silu(gate(x)) * up(x))`。默认仅返回 update，调试模式额外返回 gate、up 和 product。

### Dense Decoder

`DenseDecoderLayer.forward(hidden_states, positions, attention_mask=None, return_debug=False)` 使用 pre-norm：

```text
x -> input norm -> attention -> x + attention update
  -> post-attention norm -> SwiGLU -> residual + MLP update
```

调试模式记录稳定边界名，但正常推理不克隆和保存所有中间 Tensor。

### Tiny Dense Causal LM

`TinyDenseCausalLM.forward(input_ids, return_debug=False)` 默认返回 logits `[B,S,V]`。调试模式返回 `(logits, debug)`。模型内部根据 `input_ids.device` 创建 positions 和 mask，不执行 `.cpu()`、`.cuda()` 或隐式设备迁移。

可选的 embedding/LM Head 权重绑定由配置控制，默认关闭以保持教学边界清晰。

## CPU 与 GPU 策略

CPU FP32 是本轮唯一强制运行环境和数值基线。代码必须满足：

- 不在模型内部硬编码 `.cuda()` 或设备字符串。
- 由输入和模块参数决定设备；新 Tensor 使用 `x.new_*` 或显式 `device=x.device`。
- 固定模型状态使用 parameter 或 registered buffer，不使用无法随 `.to(...)` 迁移的普通 Tensor 属性。
- 测试默认只要求 CPU FP32；CUDA 测试只能作为可选跳过项。
- 后续 GPU 使用通过新虚拟环境安装匹配的 PyTorch wheel，再调用 `model.to(device, dtype)`，不重写模型结构。

## 依赖策略

本轮保留一个可复现的 CPU 开发基线，`pyproject.toml` 是人工维护的依赖来源，`uv.lock` 是精确解析结果。项目固定 Python 3.11.15 和 PyTorch 2.7.1，不手工编辑生成的 `requirements.txt`。

CUDA 驱动属于系统环境，不进入 Python 依赖。未来获得 GPU 后，先记录 GPU、驱动和目标 CUDA wheel，再新增经过实际验证的互斥 CPU/CUDA profile；不在当前无 GPU 环境中声称 CUDA profile 已可复现。

## 测试策略

测试覆盖：

- 配置合法性和非法 head/width 组合。
- RMSNorm shape、有限性和参考公式对齐。
- RoPE shape、旋转范数、position 差异和 buffer 迁移。
- causal mask 未来概率为零、概率和为一、GQA head 映射正确。
- SwiGLU 与 PyTorch 原语参考路径一致。
- Dense Decoder 两次 residual 的基准正确。
- Tiny 模型输出 `[B,S,V]`、前缀因果不变性、层实例不共享。
- 模型可以通过 `.to("cpu")` 运行，forward 内没有设备硬编码。
- 可编辑安装后可从仓库外语义上导入 `qwen3_moe`，并运行端到端示例。

所有浮点比较明确设置 `atol` 和 `rtol`。测试不下载模型、Tokenizer、checkpoint 或数据集。

## 文档产物

外部知识库新增：

- `📓学习笔记/LLM/PyTorch 模型实现注意事项.md`
- `⚙️环境配置/深度学习项目的多环境依赖管理.md`

两篇文档使用知识库现有 YAML frontmatter、Obsidian 双链和中文标题。第一篇重点解释 parameter/buffer、device/dtype、临时 Tensor、数值稳定性、推理模式、调试和测试分层；第二篇区分 Python 包、PyTorch wheel、CUDA runtime、NVIDIA 驱动、模型权重和机器配置，并给出 uv 管理建议和 CPU 到 GPU 的迁移流程。

项目文档更新：

- README 仓库地图增加 `src/`、模块测试和示例。
- README 快速开始说明 CPU 是完整支持路径，CUDA 不可用不是失败。
- roadmap 前七周的实现工作改为指向累计模块，而不是声明不创建正式模块。
- environment 文档说明当前 CPU 基线以及未来 GPU profile 的添加原则。

## 验收标准

- `uv sync` 能将项目安装为 `qwen3_moe` 包。
- `uv run pytest` 在 CPU 环境全部通过，CUDA 不可用时不失败。
- `uv run python examples/run_tiny_dense.py` 输出预期 logits shape 且全部有限。
- 可从测试和示例中导入同一份模块实现，没有复制 Week 5-7 的类定义。
- 两篇外部笔记存在并已加入对应索引。
- 文档不宣称当前机器拥有可用 GPU，也不宣称未验证的 CUDA 环境可复现。
- 不实现本轮范围以外的 MoE、KV Cache、真实权重或性能优化。
