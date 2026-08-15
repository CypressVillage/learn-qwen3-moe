# 后日谈：从能跑到跑得动

后日谈聚焦 Qwen3 MoE 推理系统与性能优化，不计入 Step 00 至 Step 14 的主线进度。

- E00：30B 模型的参数、激活、KV Cache 与内存成本。
- E01：Safetensors mmap、延迟加载与分层权重驻留。
- E02：BF16、FP16、INT8 与 INT4 量化。
- E03：KV Cache 预分配、原地追加与分页管理。
- E04：分块 Attention、online softmax 与 Flash Attention 原理。
- E05：MoE token dispatch、专家分组与权重调度。
- E06：从纯 NumPy 参考实现到工业推理引擎。
