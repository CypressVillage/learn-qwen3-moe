# 可视化 Qwen3 MoE 推理课程重写设计

## 状态

- 日期：2026-08-14
- 分支：`rewrite/visual-qwen3-moe-course`
- 目标模型族：Qwen3 MoE decoder-only causal language model
- 主要参考 checkpoint：Qwen3-30B-A3B 系列兼容配置与权重
- 教学实现：Python、PyTorch、NumPy 和少量标准库
- 网站形态：静态可部署的交互式教程与离线 trace 阅读器

## 背景

当前仓库已经具备一套可安装、可测试的微型 Dense 推理底座，包括 RMSNorm、RoPE、QK Norm、GQA、SwiGLU、Dense Decoder、Causal LM 和最小 greedy 生成。`docs/course/` 提供了从整机导览到 Dense 生成闭环的主线，`docs/tutorials/` 提供了更深入的 PyTorch、公式和小张量实验。

下一阶段不在现有主线上继续追加零散章节，而是彻底重写为一套沿 LLM 实际推理数据流前进的可视化课程。课程借鉴 `pi-from-scratch` 的教学机制：先展示最终系统框架，随后让正文、累计代码和执行 trace 同步推进；读者完成最后一步时，也完成一条真实的 Qwen3 MoE 推理路径。

这次重写不是把 Llama 2 教程中的名词替换成 Qwen3。Qwen3 MoE 需要额外处理 QK Norm、GQA、稀疏 MoE Router、Top-K 专家、token-to-expert 分发、专家输出聚合，以及由配置决定的 Dense/MoE 层布局。Tokenizer、权重格式和生成路径也必须以目标 checkpoint 的真实配置为准。

## 设计结论

课程采用 14 个累计步骤，而不是原提议的 12 个步骤。MoE Router 与专家分发分别占用独立步骤，完整模型组装与首次 token 生成也保持分离。这样每一步只引入一个主要认知负担，并且每一步都能留下独立、可验证、可视化的产出。

网站不在浏览器中运行 PyTorch。Python 脚本使用固定微型配置生成静态 trace，网站读取生成后的 JSON 或 TypeScript 数据。真实大模型权重只用于本地或服务器验证，不进入网站构建产物。

NumPy 和 PyTorch 的“只改 import”兼容限定为仓库自有的教学 API 子集，不承诺 NumPy 兼容完整 `torch`、autograd、CUDA 或 `nn.Module` 生态。

## 目标

- 让读者沿 `文本 -> token IDs -> hidden states -> Qwen3 MoE Transformer -> logits -> next token -> 自回归生成` 完成一次端到端推理。
- 第一步先介绍完整推理所需组件，并给出最终代码框架；后续步骤在相同边界上逐渐补全实现。
- 教程正文、累计代码、测试和网站 trace 始终来自可校验的仓库内容，避免手工复制导致漂移。
- 同一上层模型代码可以选择 NumPy 教学后端或 PyTorch 推理后端。
- 每个数学模块都先使用可手算的小张量，再连接到 Qwen3 MoE 的真实 shape 和参数命名。
- 先完成微型 checkpoint 的严格对齐，再讨论 Qwen3-30B-A3B 的量化、offload 和硬件限制。
- 网站在桌面和移动端都能完整阅读；交互用于解释数据流，而不是装饰页面。

## 非目标

- 不实现训练、反向传播、辅助负载均衡损失或分布式训练。
- 不重写完整 Transformers、vLLM、CUDA、Triton 或生产级推理服务。
- 不承诺 NumPy 后端加载或高效运行 Qwen3-30B-A3B。
- 不在浏览器下载或执行真实模型权重。
- 不把“输出了文字”作为正确性证明；正确性以 shape、数值对齐和首次偏差定位为准。
- 不声称单卡 A10 24GB 可以加载 Qwen3-30B-A3B 全量 BF16 权重。
- 不在新课程通过验收前删除现有 `docs/course/` 和 `docs/tutorials/`。

## 目标学习者

读者应具备 Python 基础语法、函数和类的知识，但可以不熟悉 PyTorch、Attention 或 MoE。课程不要求读者先系统学习线性代数；需要的矩阵、广播、softmax 和概率知识在首次使用时就地解释。

完成课程后，读者应能够：

- 解释一次模型 forward 与完整自回归生成的区别。
- 写出主要张量的 shape，并解释 batch、sequence、head、expert 和 hidden 轴。
- 从配置推导 GQA 分组、RoPE head dimension、专家数和每 token 激活专家数。
- 用 NumPy 或 PyTorch 实现教学版 Qwen3 MoE forward。
- 读取 Qwen tokenizer 资源、配置、Safetensors index 和需要的权重 tensor。
- 对比 cached 与 uncached logits，诊断 position offset 或 KV head 错误。
- 解释 temperature、top-k 和 top-p 在 logits 到 next token 之间的位置。
- 使用 trace 找到自有实现与参考实现首次不一致的模块和张量。

## 教学原则

### 沿数据流前进

课程不先分别讲完所有数学背景，再尝试组装模型。每一步都从当前推理流程中的下一个未实现边界出发：需要把文本变成 ID 时实现 Tokenizer，需要让 token 读取上下文时实现 Attention，需要替换 Dense MLP 时实现 Router 和 Experts。

### 先看整机，再进入零件

Step 01 展示最终生成循环、模型外壳、Decoder Layer、Attention、MoE、Cache 和 Sampling 的接口。初始框架允许包含 `NotImplementedError` 或教学伪代码，但模块边界、参数方向和调用关系必须与最终实现一致。

### 代码累计而不是示例堆叠

每一步展示当前时刻的完整仓库快照。后续步骤补全已有边界，不另起一套互不相干的示例实现。读者在右侧编辑器中看到的是“到当前步骤为止，整个项目是什么样子”，而不是孤立代码片段。

### 先小张量，再真实权重

所有核心模块先使用固定种子和微型配置验证。真实 checkpoint 的职责是最终对齐，不承担首次解释公式的任务。

### 正确性优先于性能

Router 和专家分发先写清晰循环版，再讨论排序分组、向量化或高性能 kernel。KV Cache 先使用易验证的追加式实现，再讨论预分配。

### 每一步都留下证据

每个 Step 必须包含位置图、shape ledger、累计代码 checkpoint、一次 trace、至少一个测试和一个常见错误。没有运行证据的“理解”不算完成。

## 统一形状记号

| 记号 | 含义 |
| --- | --- |
| `B` | batch size |
| `S` | 本次 forward 的 query token 数 |
| `P` | 调用前已缓存的历史 token 数 |
| `T` | Attention 使用的 key/value 总长度，`T=P+S` |
| `V` | vocabulary size |
| `D` | hidden size |
| `L` | decoder layer 数 |
| `Hq` | query head 数 |
| `Hkv` | key/value head 数 |
| `Dh` | head dimension |
| `G` | 每个 KV head 服务的 query head 数，`G=Hq/Hkv` |
| `I` | Dense MLP intermediate size |
| `Imoe` | 单个 MoE expert intermediate size |
| `E` | routed expert 总数 |
| `K` | 每个 token 选择的 expert 数 |
| `N` | 展平后的 token 数，`N=B*S` |

教程中的具体数值只来自当前微型配置或正在读取的 `config.json`。不得把网上文章或某个 checkpoint 的数字硬编码成整个 Qwen3 MoE 模型族的固定事实。

## Qwen3 MoE 架构边界

新课程以目标 checkpoint 的配置和参考实现为权威来源，至少覆盖以下行为：

- Decoder-only causal language model。
- Token embedding、Decoder stack、final RMSNorm 和无 bias LM head。
- Attention 使用独立 Q/K/V/O projection。
- Query head 数可以大于 KV head 数，形成 GQA。
- Q 和 K 在每个 head 的 `Dh` 维度上执行 RMSNorm。
- RoPE 作用于归一化后的 Q/K，位置和 scaling 行为来自配置。
- Decoder Layer 使用两次 pre-norm residual。
- Dense MLP 和 Expert 都使用无 bias SwiGLU。
- Router 将 `[N,D]` 投影成 `[N,E]`，softmax 后选择 Top-K experts。
- 是否重新归一化选中专家权重由 `norm_topk_prob` 决定。
- 专家输出按 Router 权重加权，并聚合回原 token 顺序。
- 每层使用 Dense MLP 还是 Sparse MoE 由 `decoder_sparse_step`、`mlp_only_layers` 和专家配置决定。
- KV Cache 只缓存每层旋转后的 K 和对应 V；cache head 数保持 `Hkv`，不存储重复到 `Hq` 后的副本。
- 训练使用的 Router auxiliary loss 不进入纯推理主线。

## 课程步骤

### Step 01：推理全景、配置和权重目录

核心问题：完整 LLM 推理到底需要哪些组件，它们怎样协作？

内容：

- 展示文本到生成文本的完整控制流。
- 展示最终项目文件树和总体 Python 框架。
- 区分 Tokenizer、模型 forward、next-token 选择和生成循环。
- 读取并校验 `config.json`。
- 读取 Safetensors index，列出参数名、shape、dtype 和所在分片。
- 提供按名称读取单个 tensor 的 `load_tensor(name)` 边界。
- 解释为什么第一步只能检查权重目录，完整装配要等模型结构完成。

运行状态：配置、索引和微型 checkpoint 检查可运行；完整 Qwen3 MoE forward 尚不可运行。

网站重点：整机流程图、模型参数树、权重分片地图和初始代码框架。

### Step 02：实现 Qwen Tokenizer

核心问题：文本如何稳定地变成与 checkpoint 一致的 token IDs？

内容：

- UTF-8 bytes 与 byte-to-unicode 映射。
- Qwen byte-level BPE 的预分词、merge ranks、encode 和 decode。
- 特殊 token 与普通文本 token 的边界。
- EOS、BOS、PAD 等 ID 只从 tokenizer/config 资源读取。
- Chat template 独立于基础 encode/decode，作为本章后半部分或扩展。
- 与官方兼容 tokenizer 做固定样例对齐，但课程运行不依赖高层模型 API。

运行状态：文本可以 round-trip，并输出真实兼容 token IDs。

网站重点：字符、UTF-8 bytes、初始 symbols、BPE merge 和最终 token IDs 的动画时间线。

### Step 03：Linear、Embedding 和 RMSNorm

核心问题：整数 token ID 怎样进入连续向量空间，隐藏状态怎样控制数值尺度？

内容：

- 建立 NumPy/PyTorch 教学 backend。
- 实现 Parameter、Linear、Embedding 和 RMSNorm 所需的最小推理接口。
- 解释 Embedding 查表与矩阵乘法的区别。
- RMSNorm 使用 FP32 统计后转换回输入 dtype。
- 从 checkpoint 加载 embedding 和 norm 权重。

运行状态：真实 token IDs 可以得到 `[B,S,D]` hidden states，并通过 RMSNorm 对齐。

网站重点：Embedding 行选择、RMS 计算、缩放前后数值与 shape。

### Step 04：实现 RoPE

核心问题：Attention 如何知道 token 的相对位置？

内容：

- 从配置计算 inverse frequencies。
- 构造 position IDs、cos 和 sin。
- 实现 split-half rotation 和 `apply_rotary_pos_emb`。
- 解释 prefill 与 decode 的 position offset。
- 为可能的 RoPE scaling 保留配置驱动接口，不提前硬编码特定策略。

运行状态：给定 Q/K 和 position IDs，可以产生 shape 不变的旋转结果。

网站重点：二维旋转、不同 position 的角度、旋转前后向量范数。

### Step 05：实现单头 Causal Attention

核心问题：一个 token 如何读取允许看到的上下文？

内容：

- 单头 Q/K/V projection。
- scaled dot-product attention。
- causal mask、稳定 softmax 和 context 聚合。
- 使用可手算张量逐项核对 scores 和 probabilities。
- 与 PyTorch 原语组合出的参考结果对齐。

运行状态：单头 Attention 对 `[B,S,Dh]` 返回相同 shape。

网站重点：mask 前后 scores、Attention 热力图和当前 token 的加权读取过程。

### Step 06：实现 Multi-Head Attention、GQA 和 QK Norm

核心问题：多个 Query heads 如何共享较少的 KV heads？

内容：

- Q/K/V projection 的不同输出宽度。
- `[B,S,H,Dh]` 与 `[B,H,S,Dh]` 的转换。
- head-dim QK Norm 的准确位置。
- Q/K 应用 RoPE。
- `Hq/Hkv` 分组和 K/V 逻辑扩展。
- 输出 heads 合并和 O projection。

运行状态：完成无 cache 的 Qwen3 风格 causal GQA。

网站重点：Query heads 到 KV heads 的分组图、每个 head 的 shape 和概率矩阵。

### Step 07：实现 SwiGLU Expert

核心问题：Attention 之后为什么还需要逐 token 非线性变换？

内容：

- gate、up、down 三条 projection。
- `silu(gate(x)) * up(x)`。
- Dense intermediate size 与 MoE expert intermediate size 的区别。
- 同一个 SwiGLU 计算核心如何服务 Dense MLP 和多个 Experts。

运行状态：Dense SwiGLU 和单个 Expert 都能独立运行。

网站重点：三个 projection、SiLU gate 和逐元素乘法。

### Step 08：实现 MoE Router 和 Top-K

核心问题：每个 token 如何选择少量专家？

内容：

- 将 `[B,S,D]` 展平为 `[N,D]`。
- Router projection 和 logits `[N,E]`。
- FP32 softmax、Top-K indices 和 weights `[N,K]`。
- 按 `norm_topk_prob` 决定是否重新归一化。
- 非法 `K>E`、NaN 和确定性 tie 行为。
- 输出 expert counts 和 Router entropy 供观察。

运行状态：Router 可以对每个 token 给出正确专家与权重。

网站重点：token-to-expert 连线、Top-K 选择、概率条形图和专家负载。

### Step 09：实现专家分发、执行和聚合

核心问题：选中的 token 怎样送入专家，并恢复原顺序？

内容：

- 先实现清晰的逐专家循环版。
- token index、top-k position 和 expert index 的关系。
- 每个专家使用独立 SwiGLU 权重。
- Router weight 加权与 `index_add` 聚合。
- 空专家、所有 token 命中同一专家、一个 token 命中多个专家。
- 循环版与分组版结果对齐。

运行状态：完整 Sparse MoE block 接收并返回 `[B,S,D]`。

网站重点：dispatch 矩阵、专家局部 batch、聚合回 token 的路径。

### Step 10：组装完整 Qwen3 MoE Transformer

核心问题：所有零件怎样组成真实模型的一次 forward？

内容：

- 组装 Attention、MoE/Dense MLP、两次 RMSNorm 和 residual。
- 按配置选择每层 MLP 类型。
- 组装 Embedding、Decoder stack、final norm 和 LM head。
- 完成外部权重名到自有模块权重名的显式映射。
- 报告 loaded、missing、unexpected 和 shape mismatch keys。
- 使用微型 checkpoint 与参考实现逐层对齐。

运行状态：微型 Qwen3 MoE checkpoint 可以输出 logits `[B,S,V]`。

网站重点：跨层张量流水线、Dense/MoE 层分布和首次偏差定位器。

### Step 11：生成第一个 Token

核心问题：模型的一次 forward 怎样转化成一个新 token？

内容：

- 输入真实 tokenizer 产生的 prompt IDs。
- 取 `logits[:, -1, :]`。
- 解释 logits、概率和 token ID 的区别。
- 使用 greedy argmax 生成第一个 token。
- 将 token ID decode 回文本片段。

运行状态：从文本 prompt 生成第一个可解码 token。

网站重点：最后位置 logits、候选 token 排名和首 token 选择。

### Step 12：实现自回归生成

核心问题：一个 token 怎样变成连续输出？

内容：

- 追加 next token 并重复模型调用。
- EOS、最大新 token 数和 batch 停止状态。
- 区分 prompt、generated IDs 和 decoded text。
- 记录无 cache 时每轮重复计算的完整前缀。

运行状态：无 KV Cache 的 greedy 生成闭环可运行。

网站重点：token 时间线、每轮输入长度和重复计算区域。

### Step 13：加入 Prefill、Decode 和 KV Cache

核心问题：怎样避免每一步重复计算整个前缀？

内容：

- 每层 K/V cache 的 `[B,Hkv,P,Dh]` contract。
- Prefill 一次处理 prompt，decode 每次处理新 token。
- position offset、`S=1` mask 和 `T=P+S`。
- cache 中保存旋转后的 K 与 V，不保存重复后的 `Hq` 副本。
- cached 与 uncached logits 逐 token 对齐。
- KV Cache 理论内存计算。

运行状态：prefill + cached decode 完成生成，并与无 cache 基线一致。

网站重点：每层 cache 增长、prefill/decode 切换和重复计算消除。

### Step 14：加入 Temperature、Top-K 和 Top-P

核心问题：模型怎样从候选分布中进行可控随机选择？

内容：

- temperature 对 logits 的缩放。
- 稳定 softmax。
- top-k 过滤。
- top-p 按概率排序、累计和与最小候选保留。
- 随机种子和 multinomial sampling。
- greedy 作为独立策略，而不是 temperature 为零的特殊浮点路径。
- 采样后 EOS 和停止条件。

运行状态：完整教学版文本生成接口可选择 greedy、temperature、top-k 和 top-p。

网站重点：过滤前后概率分布、累计概率曲线和抽样落点。

## Step 01 总体代码框架

第一步必须展示接近以下边界的框架，具体名称可以在实现阶段小幅调整，但职责不可漂移：

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
                hidden_states,
                position_embeddings,
                attention_mask,
                cache,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, cache


class DecoderLayer:
    def forward(self, hidden_states, position_embeddings, attention_mask, cache):
        hidden_states = hidden_states + self.self_attn(
            self.input_norm(hidden_states),
            position_embeddings,
            attention_mask,
            cache,
        )
        hidden_states = hidden_states + self.mlp(
            self.post_attention_norm(hidden_states)
        )
        return hidden_states, cache
```

框架的目的是让读者先拥有稳定地图，不要求第一步伪造一个已经可用的完整模型。未实现函数必须明确失败，不能返回形状正确但语义错误的占位 tensor。

## NumPy/PyTorch 双后端

### 兼容范围

上层课程代码通过仓库自有 backend 使用统一接口：

```python
from mini_qwen.backends import numpy as xp

# 或
from mini_qwen.backends import torch as xp
```

首期统一接口只覆盖推理所需子集：

- tensor 创建、转换和基础属性。
- reshape、transpose、concatenate、stack 和 broadcast。
- matmul、linear、embedding lookup。
- exp、sqrt、rsqrt、sin、cos、softmax 和 silu。
- sort、topk、cumsum、multinomial 和 index-add 类聚合。
- Parameter、Module、ModuleList、Linear、Embedding 和 RMSNorm 的最小状态管理。
- `state_dict`、严格权重加载和 dtype 转换所需功能。

### 不兼容范围

- NumPy backend 不提供 autograd、optimizer、CUDA 或完整 hook 系统。
- 不允许使用 `import numpy as torch` 假装完整兼容。
- PyTorch 专属优化只能位于 backend 或优化实现中，不能污染教学参考路径。
- 大模型实际运行、BF16/CUDA 和量化适配以 PyTorch backend 为主。

### 对照测试

每个数学模块至少包含三层测试：

- 可手算输入与预期结果。
- NumPy backend 与 PyTorch backend 对齐。
- PyTorch 教学实现与对应 PyTorch 原语或参考模型对齐。

## Tokenizer 和依赖边界

课程目标是不依赖 `transformers.generate()` 或高层模型实现。Tokenizer 可以在测试中使用兼容实现作为 oracle，但课程必须有自己的 encode/decode 路径。

仅使用 Python 标准库 `re` 很难准确覆盖 byte-level BPE 预分词中的 Unicode character properties。允许加入小型 `regex` 运行依赖，以保证与目标 tokenizer 资源一致。若最终证明目标 tokenizer 的 `tokenizer.json` 可以在不增加运行依赖的情况下被严格解释，可以再移除该依赖；不得为了维持“只有 torch 和 numpy”而接受静默 tokenization 偏差。

Chat template 与基础 tokenizer 分层：基础 tokenizer 负责文本与 token IDs；chat template 负责把消息结构转换成模型约定文本和特殊 token。Step 02 首先保证基础 encode/decode 对齐，再加入 chat template。

## 配置与权重加载

权重加载采用两阶段设计：

### Step 01：元数据与按名读取

- 解析 `config.json`。
- 解析单文件或分片 Safetensors index。
- 不加载全部 tensor 即可列出 key、dtype、shape 和分片。
- 支持按参数名读取一个 tensor。
- 在读取前检查文件存在性、offset、dtype 和 shape。

### Step 10：模型映射与完整装配

- 外部 key 到内部 key 使用显式映射，不依赖静默字符串猜测。
- shape 不一致立即报告参数名、期望 shape 和实际 shape。
- 不通过隐式 reshape 或 transpose 掩盖结构错误。
- 专家权重必须明确处理 expert 轴、gate/up packing 和 checkpoint 版本差异。
- 加载报告包含 loaded、missing、unexpected 和 mismatched。
- 微型 checkpoint 必须支持保存、分片、重新加载和数值复现。

Safetensors 的格式解析可以使用标准库实现教学读取器；若后续选择 `safetensors` 作为可选工程依赖，课程仍需解释文件头、offset、dtype 和分片索引，而不能把格式完全当作黑盒。

## 累计代码 Checkpoint

### 内容源

以下内容分别承担唯一职责：

- `lessons/stepXX-*.md`：教程正文。
- `src/mini_qwen/`：最终累计实现的权威源码。
- `tutorial/checkpoints/`：由脚本生成或验证的每个阅读节点代码快照。
- `tutorial/traces/`：由 Python 运行生成的静态 trace。
- `web/`：只负责渲染生成内容，不维护第二份手写 Python 源码。

### 快照策略

课程允许用后续实现替换早期 `NotImplementedError`，因此不强制所有 checkpoint 只能插入新行。网站必须显示从上一 checkpoint 到当前 checkpoint 的结构化 diff，并保持文件路径和符号身份稳定。

每个 checkpoint 包含：

```text
id
step
label
active_file
focus_range
repository_snapshot
trace_case
```

生成或验证脚本必须拒绝：

- Markdown 引用了不存在的 checkpoint。
- checkpoint 引用了不存在的文件或行范围。
- 源码片段与对应 checkpoint 不一致。
- 同一个 Step 最终 checkpoint 不能通过该 Step 声明的测试。

## Trace 系统

### 原则

Trace 是真实 Python 教学实现的运行记录，不是网站根据公式伪造的数据。所有官方 trace 使用固定版本、固定随机种子、固定微型配置和 CPU 生成。网站不因阅读行为发起模型请求。

### 事件结构

```json
{
  "id": "step06-gqa-012",
  "step": "step06",
  "kind": "tensor",
  "label": "QK Norm 后的 query",
  "source": {
    "file": "src/mini_qwen/attention.py",
    "line": 73,
    "symbol": "GroupedQueryAttention.forward"
  },
  "module": "layers.0.self_attn.q_norm",
  "inputs": [
    {"name": "query", "shape": [1, 4, 6, 8], "dtype": "float32"}
  ],
  "outputs": [
    {"name": "query_norm", "shape": [1, 4, 6, 8], "dtype": "float32"}
  ],
  "summary": {
    "min": -1.4,
    "max": 1.7,
    "mean": 0.02,
    "finite": true
  },
  "visualization": {
    "type": "head-grid",
    "data": []
  }
}
```

### Trace 类型

- `boundary`：Tokenizer、model、layer、generation 等调用边界。
- `tensor`：shape、dtype、device 和数值摘要。
- `attention`：mask、scores 和 probabilities。
- `routing`：Router logits、Top-K 和 expert counts。
- `dispatch`：token/expert 映射和聚合。
- `cache`：每层 K/V cache 的增长。
- `sampling`：logits 过滤、概率和选中 token。
- `error`：受控错误与触发的输入 contract。

完整 tensor 只允许用于足够小的教学 shape。较大 tensor 仅保存切片、统计值或可视化所需聚合数据。Trace 文件不得包含真实模型权重、访问 token、绝对本地路径或不可再分发的输入。

## 网站设计

### 技术方案

网站放在独立 `web/` workspace，采用 React 和支持静态构建的框架。实现阶段可以选择 Next.js static export 或等价方案，但必须满足：

- 构建时从仓库 Markdown、checkpoint 和 trace 生成内容。
- 线上阅读不需要 Python server。
- 桌面和移动端均可使用。
- 代码、trace 和正文之间的链接可测试。
- 不把大体积依赖或模型文件打入客户端。

### 桌面布局

桌面默认使用正文与实验台双栏布局：

- 左侧：课程正文、章节导航和 checkpoint 锚点。
- 右侧：代码、Trace、Tensor 和可视化标签页。
- 正文滚动到 checkpoint 时，右侧切换到对应仓库快照和源码焦点。
- 右侧提供锁定按钮，允许读者固定当前代码或 trace。

### 移动端布局

移动端不强行保留窄双栏：

- 正文保持主阅读流。
- 代码和 trace 通过底部抽屉或全屏面板打开。
- checkpoint 提供明确的“查看代码”和“查看运行”入口。
- 所有图表支持触摸、缩放或表格降级。

### 核心视图

- Repository：当前 checkpoint 的文件树和完整代码。
- Diff：本 checkpoint 相对上一 checkpoint 的新增与替换。
- Trace：按执行顺序单步前进、后退和自动播放。
- Tensor：shape、轴语义、dtype、统计值和小张量内容。
- Attention：因果 mask 和 head heatmap。
- Router：token-to-expert 路由与负载。
- Cache：prefill/decode 时间线和逐层 K/V 长度。
- Sampling：temperature、top-k、top-p 的概率变化。

### 视觉方向

网站应呈现“实验室工作台”而不是通用文档站：正文强调阅读，右侧强调运行状态；颜色用于区分 token、head、expert、cache 和错误，不使用无语义渐变。图表优先清楚表达轴、数值和因果关系，避免装饰性动画干扰阅读。

## 建议仓库结构

```text
learn-qwen3-moe/
├── lessons/
│   ├── README.md
│   ├── step01-overview-config-weights.md
│   ├── step02-tokenizer.md
│   └── ...
├── src/mini_qwen/
│   ├── backends/
│   │   ├── numpy.py
│   │   └── torch.py
│   ├── config.py
│   ├── checkpoint.py
│   ├── tokenizer.py
│   ├── layers.py
│   ├── rope.py
│   ├── attention.py
│   ├── moe.py
│   ├── decoder.py
│   ├── model.py
│   ├── cache.py
│   ├── sampling.py
│   └── generation.py
├── examples/
│   ├── step01_inspect_checkpoint.py
│   └── ...
├── tutorial/
│   ├── checkpoints/
│   └── traces/
├── scripts/
│   ├── generate_course_content.py
│   ├── generate_checkpoints.py
│   └── generate_traces.py
├── tests/
│   ├── backends/
│   ├── tokenizer/
│   ├── model/
│   ├── course/
│   └── web/
├── web/
└── docs/
    ├── course/       # 迁移完成前保留
    └── tutorials/    # 迁移完成前保留
```

`mini_qwen` 是暂定包名。实现前应确认是否继续使用现有 `qwen3_moe` 包名；在没有外部消费者或持久化格式约束时，不增加长期兼容层。

## 单步章节契约

每个 Step 必须按以下顺序组织：

1. 当前推理流程走到了哪里。
2. 当前缺少什么能力。
3. 最小输入、输出和 shape ledger。
4. 用可手算数据建立直觉。
5. 补全累计代码。
6. 沿真实源码执行一次 trace。
7. 与 NumPy、PyTorch 原语或参考实现对照。
8. 运行本 Step 测试。
9. 触发一个受控错误并解释 contract。
10. 回到完整推理地图，标记已完成和仍缺失部分。

章节结尾必须列出：

- 当前可以运行的命令。
- 当前仍不能运行的功能。
- 本 Step 新增或修改的文件。
- 验收问题和预期证据。
- 下一步为什么自然出现。

## 测试策略

### Python 实现

- 配置字段类型、范围和跨字段关系。
- Tokenizer 固定字符串、Unicode、特殊 token 和 round-trip。
- 每个模块的 shape、dtype、device 和有限性。
- NumPy/PyTorch differential tests。
- causal prefix invariance。
- GQA 分组和非法 head 配置。
- QK Norm 与 RoPE 执行顺序。
- Router Top-K、归一化策略和 expert counts。
- 专家 dispatch、空专家和聚合顺序。
- 微型 checkpoint 严格加载和逐层对齐。
- greedy、EOS、batch 停止和 sampling 确定性。
- cached/uncached logits 对齐。

### 课程内容

- 所有 Markdown 链接有效。
- checkpoint 锚点顺序与生成数据一致。
- 源码引用指向存在的文件、符号和行范围。
- 每个 Step 至少有一个可运行命令、测试和 trace。
- 文档中声明的 shape 与 trace shape 一致。
- 课程不把训练行为、量化后端或参考库行为误写成自有实现。

### 网站

- 静态构建成功。
- 每个 Step 可直接访问。
- checkpoint 滚动同步和锁定行为正确。
- 键盘可导航，代码标签和按钮具备可访问名称。
- 移动端不产生不可操作的横向页面滚动。
- 生成后的 HTML 包含必要标题、代码和无脚本降级内容。

## 内容生成与 CI

建议 CI 分为四个独立检查：

1. `python-tests`：运行 Python 模块和示例测试。
2. `course-sync`：重新生成 checkpoint/content/trace 元数据并检查工作树无差异。
3. `web-tests`：lint、组件测试和静态构建。
4. `link-check`：检查 Markdown、源码和站内链接。

Trace 不应在普通 CI 中依赖真实大模型或远程下载。仓库提交固定微型 trace；真实 checkpoint 对齐放入手动或有缓存的集成测试 profile。

## 迁移策略

重写采用并行迁移，不直接覆盖现有课程：

### 阶段一：建立新骨架

- 新增 `lessons/`、`tutorial/` 和 `web/`。
- 现有 `docs/course/` 继续作为默认入口。
- 完成 Step 01 的端到端网站原型。

### 阶段二：迁移 Dense 基础

- 将现有 RMSNorm、RoPE、GQA、SwiGLU 和 Dense Decoder 的正确实现迁移或重构进新累计代码。
- 保留现有测试作为行为基线。
- 完成 Step 02-07。

### 阶段三：完成 MoE 和生成

- 完成 Step 08-14。
- 接入微型 checkpoint 和真实兼容 tokenizer 对照。
- 增加 KV Cache 和采样 trace。

### 阶段四：切换入口

- 新课程满足验收标准后，根 README 切换到可视化课程。
- 旧 `docs/course/` 和 `docs/tutorials/` 标为 legacy/reference，或根据内容覆盖情况归档。
- 删除旧内容前检查外部链接、独有实验和仍被引用的源码说明。

## 实施里程碑

| 里程碑 | 范围 | 完成标准 |
| --- | --- | --- |
| M1 课程基础设施 | 内容 schema、checkpoint、trace schema、网站壳 | Step 01 在桌面和移动端完整可读 |
| M2 双后端基础 | Step 02-04 | Tokenizer、Embedding、RMSNorm、RoPE 对照通过 |
| M3 Attention 与 Dense | Step 05-07 | 单头、GQA、QK Norm 和 SwiGLU trace 可用 |
| M4 MoE | Step 08-09 | Router、dispatch 和聚合与朴素参考一致 |
| M5 完整模型 | Step 10-11 | 微型 checkpoint 输出 logits 并生成首 token |
| M6 生成系统 | Step 12-14 | 自回归、KV Cache 和采样完整通过 |
| M7 真实对齐与发布 | 真实 checkpoint profile、网站部署、入口迁移 | 至少一个可承载 checkpoint 逐层对齐，静态网站发布 |

## 风险与控制

### Tokenizer 精确兼容比预期复杂

控制：先固定 tokenizer 资源版本和 oracle 样例；允许使用 `regex`；把 chat template 与基础 BPE 分开验收。

### NumPy/PyTorch API 抽象膨胀

控制：只实现课程实际使用的最小子集；任何新 API 必须有两个 backend 的调用方和 differential test。

### 课程快照与最终源码漂移

控制：checkpoint 由脚本生成或严格校验；CI 重新生成后要求工作树无差异；网站不手写 Python 副本。

### Trace 文件过大

控制：只使用微型 shape；默认保存统计和切片；Attention、Router 等只保存可视化需要的数据。

### 真实权重无法在现有硬件完整加载

控制：先用自制微型 checkpoint 和可承载的小模型验证；30B 阶段只采用明确记录的量化或 CPU offload profile。

### 参考实现随版本变化

控制：记录参考库版本和目标 checkpoint revision；映射层显式处理差异；课程核心公式不直接依赖参考库内部私有 API。

### 全面重写导致现有内容丢失

控制：并行迁移；旧教程在新课程验收前不删除；建立独有内容清单后再归档。

## 最终验收标准

- 根入口可以清楚说明课程目标、14 步顺序和硬件边界。
- Step 01 展示完整推理框架，并能实际读取配置和权重目录。
- 每个 Step 都有正文、累计代码 checkpoint、trace、测试和受控错误。
- NumPy 和 PyTorch backend 在声明支持的教学 API 上通过 differential tests。
- Tokenizer 对固定中英文、Unicode 和特殊 token 样例与目标资源一致。
- QK Norm、RoPE、GQA 和 causal mask 的顺序与目标 Qwen3 MoE 行为一致。
- Router、Top-K、`norm_topk_prob`、专家分发和聚合均有数值测试。
- 微型 Qwen3 MoE checkpoint 可以严格加载并完成 `[B,S] -> [B,S,V]`。
- 可以从文本 prompt 生成首 token，并完成无 cache 与 cached 自回归生成。
- cached 与 uncached logits 在明确容差内一致。
- greedy、temperature、top-k 和 top-p 有确定性测试与可视化 trace。
- 网站静态构建通过，桌面和移动端均可完成课程阅读。
- 网站产物不包含真实模型权重、凭据或机器绝对路径。
- 至少一个可承载的兼容 checkpoint 与参考实现完成逐层数值对齐。
- 文档如实说明 Qwen3-30B-A3B 的权重存储、量化和硬件限制。

## 后续决策点

以下问题在进入实现前确定，但不阻塞本设计：

- 新累计包继续使用 `qwen3_moe`，还是更名为 `mini_qwen`。
- 网站采用 Next.js static export，还是更轻量的 Vite/React 静态方案。
- Safetensors 教学读取器是默认实现，还是将官方库作为可选快速路径。
- 真实数值对齐首先选择哪个可承载的 Qwen3 MoE checkpoint。

这些决策应遵循最小正确实现原则，不改变 14 步课程结构、Qwen3 MoE 架构边界和离线 trace 方案。
