# 16 周学习路线

## 使用说明

每周投入 6-10 小时。主线始终坚持：先写形状，再写公式；先做微型实现，再接真实权重；先验证正确性，再优化性能。

统一形状记号：

- `B`：batch size
- `S`：当前这次 forward 的输入/query 序列长度
- `P`：本次 forward 前已经缓存的历史 token 长度；无 KV Cache 时为 0
- `T`：attention 使用的 key/value 总长度，`T=P+S`
- `D`：hidden size
- `Hq`：query head 数
- `Hkv`：key/value head 数
- `Dh`：head dimension
- `E`：routed expert 总数
- `K`：每个 token 选择的 expert 数
- `I`：MLP intermediate size
- `V`：词表大小

具体数值必须以所使用模型的 `config.json` 为准，不把教学配置或网上文章中的数字硬编码成 Qwen3-30B-A3B 官方配置。

## 第 1 周：张量、形状与显存

完整教程：[第一周教程：张量、形状与内存](tutorials/week01-tensors-shapes-memory.md)

**概念**

- PyTorch tensor、rank、shape、stride、dtype、device、`numel()`。
- 索引、切片、reshape/view、transpose/permute 与 contiguous。
- 显存估算：元素数乘每元素字节；INT4 按位打包与向上取整。

**实现工作**

- 按完整教程完成各模块的 Predict-Run-Explain 实验，并记录预测、实际输出与解释。
- 建立 `token_ids [B,S]`、`embedding_weight [V,D]`、`hidden [B,S,D]`、投影与输出的 shape ledger，写明 rank、每一维语义、`numel` 和 dtype。
- 手算 FP32、FP16、BF16、INT8 与 packed INT4 内存，再用 PyTorch 的 `numel()`、`element_size()` 和可选 CUDA 指标做对照。
- 观察并解释非法 reshape、非连续 `view`、广播或矩阵乘法不匹配等可控错误；奇数个 INT4 与非法 reshape/错误理解均作为教程练习，不作为正式测试。
- 本周不创建正式 `src/` 模块或自动化测试；这些内容推迟到后续开始编写项目代码的实现阶段。

**实验**

- 先手算再比较 FP32、FP16、BF16、INT8、INT4 张量体积。
- 观察 transpose 后的 stride 和 contiguous 状态。
- 可用 CUDA 时先记录当前 allocated 基线并重置峰值统计，再分配和测量，比较 allocated/peak 增量；具体步骤见完整教程。

**验收标准**

- 完成教程中的 C1-C10 概念题并至少答对 8/10。
- 完成 S1-S5 形状推导题并至少答对 4/5。
- 完成综合任务：不看代码、输出或参考答案，能口述 `[B,S] -> [B,S,D] -> [B,S,H]` 数据流，核对 shape ledger、查表、投影、元素数与内存结果，并解释理论内存与 observed/peak memory 的差异。

**可选扩展**

- 写一个表格比较不同 `B/S/D/dtype` 的理论内存。

## 第 2 周：广播、矩阵乘法与 `nn.Module`

**概念**

- 广播规则、逐元素运算、`matmul`/`einsum`、批量矩阵乘法。
- `nn.Module`、`Parameter`、buffer、`state_dict`、train/eval 与 `no_grad`/`inference_mode`。
- 随机种子、初始化、CPU/GPU 和 dtype 转换。

**实现工作**

- 不依赖 `nn.Linear` 手写教学版线性层：`[B,S,Din] @ [Din,Dout] -> [B,S,Dout]`。
- 实现 shape assertion，并保存/加载微型 `state_dict`。
- 为广播正确和错误案例编写测试。

**实验**

- 比较手写线性层与 `nn.Linear(bias=False)` 的输出。
- 比较 FP32、FP16、BF16 的误差；不支持某 dtype 时记录而非强行运行。

**验收标准**

- 能逐维解释矩阵乘法中哪些维度收缩、哪些维度保留。
- 固定随机种子后，手写实现与 PyTorch 参考在给定容差内一致。
- 能区分 parameter、buffer 和普通属性。

**可选扩展**

- 用 `torch.profiler` 观察一次小矩阵乘法，但不提前做优化。

## 第 3 周：Token、Logits 与因果语言建模

**概念**

- Tokenizer 的编码/解码边界，Embedding 查表，LM Head。
- `logits [B,S,V]`、softmax、概率、temperature 与 argmax。
- 自回归因果建模和 teacher forcing 的输入/目标错位。

**实现工作**

- 用微型词表实现 Embedding 到 LM Head 的数据流。
- 实现稳定 softmax 和 greedy next-token 选择。
- 暂时用整数 token，不下载真实模型。

**实验**

- 改变 temperature，观察概率分布熵和最大概率。
- 对极大 logits 比较直接 `exp` 与减最大值后的稳定实现。

**验收标准**

- 能写出 `[B,S] -> [B,S,D] -> [B,S,V] -> [B,1]`。
- 概率沿词表维求和约为 1，极端输入不产生不必要的 NaN。
- 能说明 Tokenizer 不属于神经网络 forward 的哪一部分。

**可选扩展**

- 实现 top-k sampling 的最小版本。

## 第 4 周：Self-Attention 与 GQA

**概念**

- Q/K/V、scaled dot-product attention、causal mask。
- Multi-Head Attention 与 Grouped Query Attention；`Hq` 可大于 `Hkv`。
- 关键形状：当前 query 为 `q [B,Hq,S,Dh]`，attention 实际使用的 key/value 为 `k/v [B,Hkv,T,Dh]`，scores 为 `[B,Hq,S,T]`。本周尚无 KV Cache，因此 `P=0`、`T=S`。

**实现工作**

- 实现单头 attention，再扩展为多头与 GQA。
- 显式实现 K/V 按组服务多个 query heads 的逻辑。
- 对小张量写手算和参考实现测试。

**实验**

- 打印一个短序列的因果 attention 矩阵。
- 比较物理复制 K/V 与逻辑扩展的结果和内存。

**验收标准**

- 能解释为什么 scores 最后两维是 `[S,T]`。
- 未来位置的 attention 概率为 0，GQA 输出为 `[B,S,D]`。
- 能由 `Hq/Hkv` 算出每组 query head 数，并拒绝不能整除的配置。

**可选扩展**

- 与 `torch.nn.functional.scaled_dot_product_attention` 做数值对照。

## 第 5 周：RMSNorm、RoPE 与 QK Norm

**概念**

- RMSNorm 与 LayerNorm 的区别。
- RoPE 对成对通道施加旋转，以及 position 对 Q/K 的影响。
- QK Norm 在 RoPE/attention 路径中的位置应以目标模型实现和配置为准。

**实现工作**

- 手写 RMSNorm、RoPE 频率与旋转函数、Q/K normalization。
- 对奇偶通道、position offset、dtype 做显式检查。
- 用微型张量和参考公式建立测试。

**实验**

- 比较不同 position 的旋转结果和向量范数。
- 比较 FP32 计算 RoPE 频率后转换 dtype 与全程低精度的误差。

**验收标准**

- 写出 RMSNorm 输入输出均为 `[B,S,D]`。
- RoPE 不改变 Q/K 形状，并在容差内保持每对旋转分量的二范数。
- 能按顺序画出 Q projection、reshape、QK Norm、RoPE、attention。

**可选扩展**

- 研究长上下文 RoPE scaling 配置，但不在主线中硬编码。

## 第 6 周：SwiGLU 与 Dense MLP

**概念**

- Gate、up、down projection 和 SiLU。
- SwiGLU 数据流：`silu(gate(x)) * up(x)` 后接 down projection。
- 参数量、激活量与临时张量内存。

**实现工作**

- 实现无 bias 的教学版 SwiGLU：`[B,S,D] -> [B,S,I] -> [B,S,D]`。
- 编写维度不匹配和数值对齐测试。
- 增加参数量与峰值中间激活估算。

**实验**

- 比较 ReLU MLP、GELU MLP 和 SwiGLU 的输出分布。
- 改变 `I`，观察参数与激活内存的线性变化。

**验收标准**

- 能写出 gate/up/down 权重形状及矩阵乘法方向。
- 手写实现与由 PyTorch 原语拼成的参考实现一致。
- 能区分模型参数内存和一次 forward 的中间激活内存。

**可选扩展**

- 实现分块 MLP，观察内存与时间取舍。

## 第 7 周：完整 Dense Decoder

**概念**

- Pre-norm、残差连接、attention block、MLP block。
- Decoder stack、final norm、LM Head。
- 逐层数值验证与首次偏差定位。

**实现工作**

- 组合 RMSNorm、GQA、RoPE 和 SwiGLU 为 Dense Decoder Layer。
- 组合微型 Dense Causal LM，并添加 causal mask。
- 每个边界断言 shape、dtype 和 device。

**实验**

- 关闭/开启各残差支路，观察 hidden-state 范数。
- 用 1 层和多层微型模型比较逐层输出。

**验收标准**

- 完成 `[B,S] -> logits [B,S,V]` 的微型 dense forward。
- 所有组件测试和整层测试通过，且无需 GPU。
- 能从 logits 偏差回溯到首个不一致的中间张量。

**可选扩展**

- 实现 embedding 与 LM Head 权重绑定的可配置选项。

## 第 8 周：MoE Router 与 Top-K

**概念**

- Dense MLP 与稀疏 MoE；总参数和每 token 激活参数的区别。
- Router logits `[N,E]`，其中 `N=B*S`；top-k indices/weights `[N,K]`。
- softmax、Top-K 后归一化，以及路由配置必须来自目标模型。

**实现工作**

- 展平 token 维，实现 router projection、Top-K 选择和权重归一化。
- 处理稳定排序、dtype 与非法 `K>E`。
- 返回可观察的 expert counts。

**实验**

- 构造可手算 router logits，验证选中专家与权重。
- 改变 logits 尺度，观察路由置信度和专家分布。

**验收标准**

- 能解释 `[B,S,D] -> [N,D] -> [N,E] -> indices/weights [N,K]`。
- 每个 token 的选中权重和在规定归一化策略下约为 1。
- 能说明“只激活 K 个专家”为什么不减少权重存储到 K/E。

**可选扩展**

- 计算路由熵并画 expert 使用直方图。

## 第 9 周：专家分发、计算与聚合

**概念**

- Token-to-expert dispatch、expert batch、scatter/index_add 聚合。
- 同一个 token 被多个专家处理，输出按 router weight 加权。
- 空专家、重复专家和 token 顺序恢复。

**实现工作**

- 先写清晰循环版专家执行，再写向量化/分组版。
- 每个专家使用独立 SwiGLU 参数。
- 实现 `[N,D] -> K 条专家路径 -> [N,D]` 的加权聚合。

**实验**

- 用恒等专家或常数专家手算聚合结果。
- 构造所有 token 路由到同一专家和存在空专家的极端情况。

**验收标准**

- 聚合结果与朴素逐 token 参考实现一致。
- 无 token 的专家不会报错，输出 token 顺序不变。
- 能列出 dispatch 中最可能产生大临时张量的位置。

**可选扩展**

- 比较循环、布尔掩码和排序分组三种实现的耗时。

## 第 10 周：MoE Decoder 与负载观察

**概念**

- 用 MoE block 替换 Dense MLP 后的 Decoder 数据流。
- 路由负载、容量、负载不均衡；推理与训练辅助损失的边界。
- 模型配置中的共享/路由专家等字段以实际 Qwen3-MoE 配置为准。

**实现工作**

- 组合 Attention、MoE、残差与 normalization 为 MoE Decoder Layer。
- 添加路由统计，但不把训练辅助损失误加到推理输出。
- 构建多层微型 MoE Causal LM。

**实验**

- 固定输入观察不同层的 expert counts 和路由熵。
- 人为偏置 router，观察负载失衡对耗时的影响。

**验收标准**

- 微型 MoE 模型输出 logits `[B,S,V]`，测试覆盖多层和空专家。
- 能区分 router logits、router weights、expert outputs 三类张量。
- 能说明 30B 总参数与 A3B 激活参数的含义及局限。

**可选扩展**

- 输出每层路由统计 JSON，供后续 benchmark 使用。

## 第 11 周：配置驱动与权重映射

**概念**

- `config.json`、Safetensors、分片索引、参数命名和 tensor shape。
- 架构参数与运行参数分离；严格加载与缺失/多余 key。
- 权重转置、专家维排列和 tied weights 风险。

**实现工作**

- 定义最小模型配置 dataclass，并从 JSON 加载。
- 写“外部 key -> 自有模块 key”的显式映射表和 shape 校验器。
- 先用自制微型 checkpoint 测试保存、分片清单和加载。

**实验**

- 故意交换两个维度，确认加载器给出包含 key 和期望 shape 的错误。
- 只读取真实模型的配置/权重索引元数据时，先核对磁盘预算，不加载 tensor。

**验收标准**

- 不修改模型代码即可由配置创建不同尺寸的微型模型。
- 映射报告列出已加载、缺失、多余和 shape 不符的 key。
- 不把未验证的官方配置数字写死在实现中。

**可选扩展**

- 实现按需读取单个 Safetensors 分片的检查工具。

## 第 12 周：Tokenizer、Prefill 与生成循环

**概念**

- Tokenizer 特殊 token、chat template 与模型 forward 的边界。
- Prefill 一次处理 prompt；decode 每步处理新 token。
- Greedy、temperature、top-k/top-p、随机种子和停止条件。

**实现工作**

- 接入兼容 tokenizer，仅把 token IDs 交给自有模型。
- 实现无 KV Cache 的基线生成循环。
- 加入 EOS、最大新 token 数和确定性采样测试。

**实验**

- 比较 greedy 与不同 temperature 的输出 token 序列。
- 记录无 cache 时序列变长导致的每步延迟。

**验收标准**

- 能画出文本到 token、logits、采样、追加 token、解码文本的循环。
- 在微型模型上可重复生成，并正确停止。
- 明确 tokenizer/chat template 可以复用，但核心模型 forward 不依赖高层 `generate()`。

**可选扩展**

- 实现 repetition penalty，并写出它修改 logits 的位置。

## 第 13 周：KV Cache 与逐层数值对齐

**概念**

- 每层 K/V cache：本次调用前为 `[B,Hkv,P,Dh]`，追加当前 `k/v [B,Hkv,S,Dh]` 后为 `[B,Hkv,T,Dh]`，其中 `T=P+S`。
- position offset、causal mask 在 `S=1` decode 时的变化。
- Cache 内存与层数、上下文、dtype 的关系。

**实现工作**

- 为每层 attention 添加 cache 输入/输出。
- 实现 prefill + decode 路径和无 cache 基线。
- 逐层比较 hidden states、K/V 和 logits。

**实验**

- 对同一 token 序列比较 cached 与 uncached logits。
- 改变上下文长度，测 KV Cache 理论值、实测值与每 token 延迟。

**验收标准**

- cached/uncached 在明确容差内一致，cache 的 `T` 每步正确增长。
- 能从模型配置推导每 token、每层和全模型 KV Cache 字节数。
- 能诊断 position offset 或 K/V head 维度错误导致的首个偏差。

**可选扩展**

- 设计预分配 cache，比较与逐步 `cat` 的性能。

## 第 14 周：真实权重的安全接入与对齐

**概念**

- 参考实现作为数值 oracle，而非主线依赖。
- 分层加载、CPU offload、meta device 和内存峰值。
- “权重格式可读取”与“显存能容纳模型”是两回事。

**实现工作**

- 先对兼容的微型/较小 Qwen3-MoE checkpoint 完成端到端映射。
- 添加 hook 或调试接口，比较 embedding、单层 hidden state、final logits。
- 为 Qwen3-30B-A3B 制定仅元数据检查、量化运行或 CPU offload 路径。

**实验**

- 找出自有实现与参考实现首次超出容差的层和张量。
- 在下载任何大权重前估算下载大小、缓存副本和加载峰值空间。

**验收标准**

- 至少一个可承载 checkpoint 完成权重映射和逐层对齐。
- 映射逻辑不通过静默 reshape 掩盖结构错误。
- 文档和日志不声称单张 A10 可加载 Qwen3-30B-A3B 全量 BF16。

**可选扩展**

- 在有足够系统内存时测试 CPU offload，并记录每 token 延迟。

## 第 15 周：量化、显存核算与兼容层

**概念**

- Weight-only INT8/INT4、group size、scale/zero point、反量化。
- 量化权重格式、量化 kernel 和纯 PyTorch 教学实现之间的区别。
- 精度误差、磁盘体积、峰值显存和速度不一定同向改善。

**实现工作**

- 对微型线性层实现教学用 per-group 量化/反量化参考路径。
- 为真实量化 checkpoint 设计适配接口，不假装 packed 权重是 BF16 tensor。
- 建立权重、KV Cache、激活、临时缓冲区的预算表。

**实验**

- 比较 BF16/FP16/INT8/INT4 的线性层误差、体积和延迟。
- 在 A10 上尝试受支持的 Qwen3-30B-A3B 4-bit 运行方案；失败也记录后端、版本和错误。

**验收标准**

- 能解释理论 4-bit 体积为何仍小于实际进程显存需求。
- 量化误差测试有明确基线、`atol/rtol` 或任务指标。
- 能清楚分开“自研模型逻辑正确性”和“第三方量化 kernel 执行”。

**可选扩展**

- 比较不同 group size 的误差与元数据开销。

## 第 16 周：性能分析、整体验收与独立复现

**概念**

- Prefill throughput、decode tokens/s、首 token 延迟、单 token 延迟。
- Warm-up、同步、峰值显存、可复现实验与性能瓶颈。
- Python dispatch、专家负载、内存带宽和 kernel launch 开销。

**实现工作**

- 建立统一 benchmark：记录硬件、软件版本、配置、dtype、量化、`B/S`。
- 从空目录按文档重建环境并运行微型端到端推理。
- 整理架构图、shape 表、权重映射表和已知限制。

**实验**

- 比较本地 4060 与 A10 上可运行配置的 prefill/decode 指标。
- 对一个已定位瓶颈做单项优化，优化前后使用同一输入和测量方法。

**验收标准**

- 独立讲解并演示文本到新 token 的完整实现，包括 MoE 与 KV Cache。
- 所有核心组件有 shape 测试和数值测试，端到端结果有接受证据。
- 报告 Qwen3-30B-A3B 在 A10 上采用的实际运行方式，不夸大 BF16 可行性。
- 能列出尚未覆盖的生产能力和下一步学习方向。

**可选扩展**

- 针对已证明正确的热点尝试 `torch.compile` 或 Triton；保留未优化参考实现用于对照。

## 最终交付清单

- [ ] 微型配置可在 CPU 完成端到端推理。
- [ ] CUDA 可用时，可在 RTX 4060 完成日常测试和小模型实验。
- [ ] Dense 和 MoE Decoder 均有独立数值验证。
- [ ] Prefill、decode、KV Cache、采样和停止条件组成完整生成循环。
- [ ] 至少一个可承载 checkpoint 完成配置与权重映射验证。
- [ ] Qwen3-30B-A3B 使用量化或 CPU offload 的限制与数据被如实记录。
- [ ] 每个关键张量都有形状、dtype、device 和维度语义说明。
