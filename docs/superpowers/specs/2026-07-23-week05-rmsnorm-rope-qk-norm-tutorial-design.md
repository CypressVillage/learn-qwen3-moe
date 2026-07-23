# 第五周 RMSNorm、RoPE 与 QK Norm 教程设计规格

## 目标

为完成前四周教程的初学者编写一份第五周自包含教程。教程继续采用 Predict-Run-Explain 学习闭环，从可手算的 RMSNorm 与二维旋转出发，逐步连接逐 head QK Norm、向量化 RoPE、position offset、dtype 误差和带位置处理的微型 causal GQA。

教程完成后，学习者应能：

- 写出 RMSNorm 公式，解释它与 LayerNorm 的差别，并沿最后一维正确实现。
- 区分 Decoder 子层入口的 hidden-state RMSNorm 与 attention 内部逐 head 的 QK Norm。
- 解释 Qwen3 attention 的顺序：Q/K projection、reshape、QK Norm、transpose、RoPE、attention。
- 从二维旋转推导 RoPE，说明它只改变 Q/K 数值而不改变 shape。
- 构造 inverse frequencies、angles、cos/sin，并解释每个 shape 与广播轴。
- 验证 position 0 恒等、成对通道二范数保持、同位置 Q/K 点积保持和共同 position offset 不改变相对 attention scores。
- 解释频率计算使用 FP32 基准的原因，并量化低精度误差。
- 把 RMSNorm、QK Norm 和 RoPE 接入第四周的 causal GQA，保持物理复制与逻辑分组结果一致。

## 教程形式

- 新增 `docs/tutorials/week05-rmsnorm-rope-qk-norm.md`。
- 更新 `README.md` 和 `docs/roadmap.md`，提供第五周稳定入口。
- 示例、练习、综合任务与验证代码全部放在 Markdown 内联代码块中。
- CPU 是完整必修路径；使用确定性 FP32 小张量，不下载模型、Tokenizer、checkpoint 或数据集。
- 不新增正式 `src/` 模块、Notebook、独立练习脚本或测试文件。
- 保持六个模块、综合任务、`C1-C10`、`S1-S5`、提示、答案、速查表和下一周预告的统一结构。

## 范围边界

本周只覆盖归一化与位置旋转如何进入 attention。明确不加入：

- 完整 Decoder 残差路径；
- MLP、SwiGLU 或 MoE；
- KV Cache；
- padding mask、dropout 或 cross-attention；
- 长上下文 RoPE scaling 的具体策略；
- 真实 Qwen3 配置数字和权重加载；
- 生产级 fused kernel 或性能结论。

RMSNorm 的 `eps`、RoPE 的 `rope_theta`、head dimension 和 scaling 策略必须来自未来使用的目标配置；教程只使用明确标注的微型教学值。

## 正确数据流

主线采用以下顺序：

```text
hidden                         [B,S,D]
input RMSNorm                  [B,S,D]
Q projection                   [B,S,Hq*Dh]
K/V projection                 [B,S,Hkv*Dh]
reshape q                      [B,S,Hq,Dh]
reshape k/v                    [B,S,Hkv,Dh]
Q RMSNorm / K RMSNorm          shape 不变，沿 Dh
transpose q                    [B,Hq,S,Dh]
transpose k/v                  [B,Hkv,T,Dh]
RoPE(q, positions)             [B,Hq,S,Dh]
RoPE(k, positions)             [B,Hkv,T,Dh]
causal GQA scores              [B,Hq,S,T]
probabilities                  [B,Hq,S,T]
context                        [B,Hq,S,Dh]
output                         [B,S,D]
```

本周没有 KV Cache，因此 `P=0`、`T=S`。V 不执行 QK Norm，也不执行 RoPE。

教程采用与 Qwen3 参考实现一致的 split-half 旋转约定：将最后一维分成前后两半，`rotate_half([x1,x2])=[-x2,x1]`。教程必须指出 adjacent-pair 是另一种合法布局，但两种布局不能混用。

## 六模块结构

### 模块 1：RMSNorm

- 手写 `x * rsqrt(mean(x^2)+eps) * weight`。
- 输入输出保持 `[B,S,D]`，统计量沿 `D` 计算。
- 对比 LayerNorm：RMSNorm 不减均值，也不按方差重新居中。
- 使用全 1 与非均匀 weight，说明缩放前归一化值 RMS 约为 1，但最终输出 RMS 不必为 1。
- 受控拒绝错误 weight width 和非正 `eps`。

### 模块 2：QK Norm

- 将 RMSNorm 应用于 `[B,S,H,Dh]` 的最后一维。
- 明确每个 token、每颗 head 独立统计 `Dh`。
- 对比 hidden-state RMSNorm `[B,S,D]` 与 QK Norm `[B,S,H,Dh]`。
- 验证 Q/K shape 不变，V 未归一化。
- 展示沿错误轴归一化虽可运行但语义错误。

### 模块 3：二维旋转与 RoPE

- 从 `[x1,x2]` 的二维旋转矩阵推导 `x*cos(theta)+rotate_half(x)*sin(theta)`。
- 验证 position 0 恒等、旋转后 shape 不变、每对通道二范数保持。
- 验证同一角度旋转 Q/K 时点积保持。
- 拒绝奇数 `Dh`，解释通道必须成对。

### 模块 4：向量化频率与 position offset

- 构造 `inv_freq [Dh/2]`、`angles [B,S,Dh/2]` 和扩展后的 `cos/sin [B,S,Dh]`。
- 广播到 q/k `[B,H,S,Dh]`，保持 batch、head 与序列轴语义。
- 验证不同 position 改变 Q/K 数值但不改变 shape。
- 验证所有 Q/K position 同加 offset 时完整 scores 保持，因为相对位置不变。
- 拒绝错误 position dtype、负 position、长度不匹配和奇数 `Dh`。

### 模块 5：dtype 与计算顺序

- 以 FP32 计算 inverse frequencies、angles、cos/sin 作为教学基准。
- 在 CPU 支持时比较 BF16 路径与 FP32 后转换路径的最大绝对误差。
- 验证输出有限，不强迫运行不支持的 dtype。
- 使用非均匀 QK Norm weight 展示 `QK Norm -> RoPE` 与 `RoPE -> QK Norm` 通常不等价。

### 模块 6：接入 causal GQA

- 在第四周微型 GQA 前加入 input RMSNorm、QK Norm 与 RoPE。
- 同时保留物理复制 K/V 和逻辑分组两条计算路径。
- 检查 causal future probabilities 为 0、概率沿 `T` 求和约为 1、所有结果有限。
- 比较两条路径的 scores、probabilities、context 和 output。

## 综合任务

综合任务固定使用：

```text
B=1, S=T=3, D=8
Hq=2, Hkv=1, Dh=4, G=2
positions=[0,1,2]
```

使用固定 hidden、固定无 bias 投影和非均匀 norm weight，完整执行：

```text
hidden -> input RMSNorm -> Q/K/V projection -> reshape
       -> QK Norm -> transpose -> RoPE -> causal GQA
       -> merge heads -> output projection
```

必须断言：

- 所有关键 shape 与 shape ledger 一致；
- input RMSNorm、QK Norm 和 RoPE 均保持各自输入 shape；
- position 0 的 RoPE 为恒等变换；
- 每个 split-half 通道对的二范数在容差内保持；
- 同位置旋转前后的 Q/K 点积保持；
- 所有 position 同加 offset 后 scores 保持；
- causal future probabilities 严格为 0，沿 `T` 的和约为 1；
- 物理和逻辑 GQA 的 scores、probabilities、context、output 一致；
- 最终 output 为 `[1,3,8]` 且所有值有限；
- 使用非均匀 QK Norm weight 时，错误的 `RoPE -> QK Norm` 顺序与正确结果存在可观察差异。

综合任务是 attention 子层的数据流实验，不是完整 Decoder Block。

## 错误处理

教程在危险运算前明确拒绝：

- RMSNorm weight 最后一维与输入不匹配；
- `eps <= 0`；
- `Dh` 为奇数；
- Q/K 最后一维不同；
- position 不是整数 Tensor、包含负数或序列长度不匹配；
- cos/sin 与 Q/K 不可广播；
- `Hq % Hkv != 0`；
- causal mask 产生全屏蔽行；
- 任一关键中间张量出现 NaN 或 Inf。

## 练习与验收

每个模块提供两个稳定编号练习 `M1-E1` 至 `M6-E2`，并给出独立提示与有理由的参考答案。最终验收包含：

- 10 道概念题，至少答对 8/10；
- 5 道公式、shape 或数值推导题，至少答对 4/5；
- 综合任务全部断言通过；
- 能脱离代码画出正确数据流并说明每个归一化和旋转轴；
- 能区分 shape 合法但语义错误的归一化轴、旋转布局、position 广播和操作顺序。

## 验证策略

完成后执行：

1. 独立执行教程中的所有 Python 代码围栏。
2. 检查 Markdown 围栏闭合、目录链接和显式锚点。
3. 检查 12 道模块练习、10 道概念题和 5 道推导题均有唯一 question/hint/answer 锚点。
4. 检查 README、路线图和教程的相对链接存在。
5. 扫描 `TODO`、`TBD` 和未完成占位符。
6. 运行 `uv run pytest`。
7. 运行 `git diff --check` 并审查最终差异。

## Week 6 衔接

下一周引入 SwiGLU 与 Dense MLP。第五周结束时，学习者已经能得到一个正确加入 normalization 和 position 的 attention 输出 `[B,S,D]`；第六周将独立实现另一个保持 `[B,S,D]` 外部接口的前馈子层，为第七周组合完整 Dense Decoder 做准备。
