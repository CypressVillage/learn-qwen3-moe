# LLM 推理教程主线重构实施计划

## 目标

在保留现有七篇深入教程的前提下，新增按认知阶段组织的主线课程，使初学者先理解完整推理与生成流程，再沿真实 `src` 实现逐层深入。

## 工作项

1. 新增 `docs/course/README.md` 和 `00` 至 `08` 九个主线章节。
2. 更新根 `README.md`，将主线课程设为首次学习入口。
3. 更新 `docs/roadmap.md`，区分概念主线与 16 周实践节奏。
4. 为七篇旧教程增加主线位置、源码、测试和返回入口导航。
5. 新增最小 greedy 生成示例，保持一次 forward 示例不变。
6. 新增生成行为测试，验证 shape、长度、范围和确定性。
7. 检查 Markdown 链接、运行示例和完整测试。
8. 将主线 00-08 的关键真实函数放入正文，按执行顺序讲解，并增加源码片段同步检查。

## 实施约束

- 不删除或重命名旧教程。
- 不实现真实 Tokenizer、EOS、sampling、KV Cache、MoE 或权重加载。
- 主线章节不复制旧教程的大量练习，只提供必要解释和精确链接。
- 文档始终区分一次模型 forward 与自回归生成。
- 随机初始化模型的输出只用于观察控制流，不宣称具有语言意义。

## 验证命令

```bash
uv run pytest
uv run python examples/run_tiny_dense.py
uv run python examples/run_tiny_greedy.py
```
