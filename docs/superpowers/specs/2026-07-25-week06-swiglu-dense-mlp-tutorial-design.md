# 第六周 SwiGLU 与 Dense MLP 教程设计规格

## 目标

为完成前五周教程的初学者编写一份第六周自包含教程。教程继续采用 Predict-Run-Explain 学习闭环，从普通两层 Dense MLP 出发，逐步引入 SiLU、gate/up 双投影、逐元素门控、down projection、模块状态、参数量与中间激活内存核算，最终完成可数值验证的微型 SwiGLU 前馈子层。

教程完成后，学习者应能：

- 解释 Decoder 为什么在 attention 之外还需要逐 token 的前馈变换。
- 写出普通两层 MLP 与 SwiGLU 的公式，并指出二者的数据流差异。
- 写出 gate、up、down 三个无 bias 投影的权重 shape 和矩阵乘法方向。
- 解释 `silu(gate(x)) * up(x)` 为什么要求两个分支 shape 完全一致。
- 从 `[B,S,D]` 推导 gate/up 激活 `[B,S,I]`、逐元素乘积 `[B,S,I]` 和输出 `[B,S,D]`。
- 用函数版和 `nn.Module` 版实现同一个 SwiGLU，并与由 PyTorch 原语拼成的参考路径做数值对齐。
- 区分参数、输入输出激活和 forward 中间张量，手算参数量与理论字节数。
- 说明 `I` 对参数量和主要中间激活的线性影响，并识别同时存活张量对峰值内存的影响。
- 比较 ReLU MLP、GELU MLP 与 SwiGLU 的结构和输出分布，但不把单次微型实验推广为模型质量结论。
- 为第七周把 attention 子层和 MLP 子层组合成完整 Dense Decoder Layer 做准备。

## 教程形式

- 新增 `docs/tutorials/week06-swiglu-dense-mlp.md`。
- 更新 `README.md` 和 `docs/roadmap.md`，提供第六周稳定入口。
- 示例、练习、综合任务和验证代码全部放在 Markdown 内联代码块中。
- CPU 是完整必修路径；使用固定随机种子和确定性 FP32 小张量，不下载模型、Tokenizer、checkpoint 或数据集。
- 不新增正式 `src/` 模块、Notebook、独立练习脚本或测试文件。
- 保持六个模块、综合任务、`C1-C10`、`S1-S5`、提示、答案、速查表和下一周预告的统一结构。
- 环境命令沿用仓库既有写法，但本次文档工作不调整 uv、Python、PyTorch 或锁文件。

## 范围边界

本周只覆盖独立 Dense MLP/SwiGLU 子层。明确不加入：

- attention、GQA、RoPE 或 QK Norm 的重复实现；
- residual connection、pre-norm 或完整 Decoder Layer；
- MoE router、Top-K、专家分发或聚合；
- dropout、训练、反向传播、优化器或损失函数；
- tensor parallel、fused SwiGLU、Triton/CUDA kernel 或性能优化结论；
- KV Cache、生成循环、Tokenizer 或真实权重加载；
- Qwen3-30B-A3B 的未核实具体配置数字。

教程使用明确标注的微型 `D` 和 `I`。接入真实模型时，hidden size、intermediate size、bias 配置、激活函数和权重命名必须来自目标 `config.json` 与参考实现，不能把教学值写死为官方配置。

## 正确数据流

主线采用 Qwen3-style 无 bias SwiGLU：

```text
x                              [B,S,D]
gate projection                [B,S,I]
up projection                  [B,S,I]
SiLU(gate)                     [B,S,I]
SiLU(gate) * up                [B,S,I]
down projection                [B,S,D]
```

以 PyTorch `nn.Linear(in_features, out_features, bias=False)` 的参数布局表示：

```text
gate weight                    [I,D]
up weight                      [I,D]
down weight                    [D,I]
```

若使用显式右乘教学公式，则等价写为：

```text
gate = x @ W_gate              W_gate [D,I]
up   = x @ W_up                W_up   [D,I]
out  = hidden @ W_down         W_down [I,D]
```

教程必须明确区分这两种权重布局，不能把 `nn.Linear.weight` 的存储 shape 与 `x @ W` 教学 shape 混写。所有必修实现选择一种布局后保持一致，并在对照段解释二者通过转置对应。

SwiGLU 外部接口保持 `[B,S,D] -> [B,S,D]`，但本周不把这种 shape 保持误说成残差连接；输出与输入可以相加属于第七周的 Decoder 组合工作。

## 六模块结构

### 模块 1：Dense MLP 基线

- 说明 attention 负责 token 间信息混合，而 MLP 对每个 token 独立执行相同的特征变换。
- 从单个向量 `[D]` 扩展到 `[B,S,D]`，展示 batch 与 sequence 维被保留。
- 实现无 bias 的两层 MLP：`[B,S,D] -> [B,S,I] -> [B,S,D]`。
- 使用循环逐 token 参考与批量矩阵乘法结果对齐，证明 MLP 不混合 token 轴。
- 拒绝第一层输入宽度或第二层收缩宽度不匹配。

### 模块 2：SiLU 与激活函数

- 写出 `SiLU(x)=x*sigmoid(x)`，用手算值解释负数被平滑抑制而非全部截断。
- 将手写 SiLU 与 `torch.nn.functional.silu` 对齐。
- 对同一组小输入比较 ReLU、GELU 与 SiLU 输出，不宣称某种激活在所有模型中必然更优。
- 验证激活函数不改变 shape、dtype 或 device，并检查输出有限。
- 解释激活函数本身没有可学习参数。

### 模块 3：SwiGLU 门控路径

- 从单分支 MLP 过渡到 gate/up 两条独立投影路径。
- 实现 `hidden = silu(gate(x)) * up(x)`，强调乘法是逐元素运算，不是矩阵乘法。
- 用可手算权重展示 gate 分支控制 up 分支中每个位置、每个中间通道的通过程度。
- 验证 gate、up、SiLU 输出和乘积均为 `[B,S,I]`。
- 展示 shape 可广播但语义错误的危险案例，并要求两分支 shape 完全相等而不是依赖隐式广播。

### 模块 4：函数版与 `nn.Module` 版 SwiGLU

- 先用显式 Tensor 权重实现函数版 SwiGLU，再实现包含三个 `nn.Linear(..., bias=False)` 的教学模块。
- 将相同权重复制到两条实现路径，比较 gate、up、乘积和最终输出。
- 检查 `state_dict` 中恰好包含 gate/up/down 三组权重，并解释 parameter 与普通属性的区别。
- 验证 `.eval()` 与 `torch.inference_mode()` 的职责；本周无 dropout，因此 train/eval 输出应一致。
- 受控拒绝输入最后一维不等于 `D`、非法非正维度和不匹配参考权重。

### 模块 5：参数量与理论内存

- 推导无 bias SwiGLU 参数量：`D*I + D*I + I*D = 3DI`。
- 对比无 bias 普通两层 MLP 参数量 `2DI`，解释 SwiGLU 多出的 gate 投影。
- 若展示带 bias 变体，只作为公式扩展并明确不是本周 Qwen3-style 主线。
- 按 dtype 字节数计算参数理论体积，区分 FP32、FP16/BF16 与仅用于体积练习的 INT8。
- 强调参数理论体积不等于完整进程显存，也不包含梯度、优化器、allocator、kernel workspace 或框架开销。

### 模块 6：中间激活、输出分布与 `I` 扫描

- 建立 forward shape ledger：输入、gate、up、activated gate、乘积和输出。
- 分别计算每个张量的元素数与理论字节数，并讨论哪些张量可能同时存活。
- 改变 `I`，验证参数量和主要 `[B,S,I]` 中间激活元素数线性变化。
- 在相同 `B/S/D/I`、固定权重尺度和随机种子下比较 ReLU MLP、GELU MLP 与 SwiGLU 的均值、标准差、最小值、最大值和有限性。
- 明确输出分布实验只帮助观察结构差异，不用于判断训练后模型质量，也不作为硬件性能 benchmark。

## 综合任务

综合任务固定使用：

```text
B=2, S=3, D=4, I=6
dtype=float32, device=cpu
bias=False
```

使用固定输入和固定 gate/up/down 权重，完整执行：

```text
x -> gate/up projections -> SiLU(gate) -> elementwise product
  -> down projection -> output
```

同时实现并比较：

1. 逐 token 循环参考；
2. 显式 Tensor 矩阵乘法函数版；
3. 三个 `nn.Linear` 组成的模块版。

必须断言：

- 输入 `[2,3,4]`，gate/up/activated gate/乘积均为 `[2,3,6]`，输出为 `[2,3,4]`；
- 三条实现路径的所有对应中间张量和最终输出在明确 `atol/rtol` 内一致；
- 手写 SiLU 与 PyTorch 原语一致；
- gate/up 逐元素乘法没有发生广播；
- 改变一个 token 不会改变其他 token 的输出；
- 所有中间值与输出均有限；
- 模块 `state_dict` key 和权重 shape 符合设计；
- 参数总数等于 `3DI=72`；
- FP32 参数理论体积等于 `72*4=288` bytes；
- 每个主要中间张量的元素数和理论字节数与 shape ledger 一致；
- 将 `I` 从 6 改为 12 时，参数量和单个 `[B,S,I]` 张量元素数均翻倍；
- 错误输入宽度、错误 down weight 方向和可广播但不相等的 gate/up shape 被明确拒绝。

综合任务是独立 MLP 子层实验，不包含 input RMSNorm、post-attention RMSNorm、residual connection 或 attention 输出。

## 错误处理

教程在危险运算前明确拒绝：

- `D <= 0` 或 `I <= 0`；
- 输入最后一维与 gate/up 预期输入宽度不匹配；
- gate 与 up 权重的输入或输出宽度不一致；
- down 权重不能把 `I` 映射回 `D`；
- gate 与 up 分支输出 shape 不完全相同，即使 PyTorch 广播规则允许相乘；
- Tensor 权重 dtype/device 与输入不一致且没有显式转换策略；
- 任一关键中间张量出现 NaN 或 Inf；
- 参数量或内存公式把 bias、梯度、优化器状态或临时张量混入错误类别。

错误信息应包含实际 shape 和期望 shape，帮助学习者定位是输入宽度、投影方向还是分支 shape 出错。

## 练习与验收

每个模块提供两个稳定编号练习 `M1-E1` 至 `M6-E2`，并给出独立提示与有理由的参考答案。最终验收包含：

- 10 道概念题，至少答对 8/10；
- 5 道公式、shape、参数量或内存推导题，至少答对 4/5；
- 综合任务全部断言通过；
- 能脱离代码画出 gate、up、SiLU、逐元素乘法和 down projection 数据流；
- 能分别使用显式矩阵布局和 `nn.Linear.weight` 布局写出三个权重 shape；
- 能区分参数内存、单个激活张量体积与实际 forward 峰值内存；
- 能解释普通 MLP、GLU/SwiGLU 和未来 MoE expert MLP 之间的关系与边界。

## 验证策略

完成后执行：

1. 独立执行教程中的所有 Python 代码围栏。
2. 检查 Markdown 围栏闭合、目录链接和显式锚点。
3. 检查 12 道模块练习、10 道概念题和 5 道推导题均有唯一 question/hint/answer 锚点。
4. 检查 README、路线图和教程的相对链接存在。
5. 扫描 `TODO`、`TBD`、待补充文本和未完成占位符。
6. 在可用环境中运行仓库测试；环境不可用时单独记录，不修改环境配置。
7. 运行 `git diff --check` 并审查最终差异。

## Week 7 衔接

下一周组合完整 Dense Decoder。第六周结束时，学习者已经拥有两个外部 shape 都保持 `[B,S,D]` 的独立子层：第五周完成的带 normalization 与 position 的 attention 路径，以及本周完成的 SwiGLU MLP。第七周将引入 pre-norm 和 residual connection，把它们按正确顺序组合成 Dense Decoder Layer，再堆叠为微型 Dense Causal LM。
