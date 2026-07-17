# uv 与 PyTorch 双机环境设计

## 目标

为 RTX 4060 Laptop 8GB 和单张 A10 24GB 建立一套统一、可复现的 Python/PyTorch 学习环境。使用 `uv` 管理 Python、依赖和锁文件，同时导出传统 `requirements.txt` 供兼容工具使用。

当前仓库中的 `.venv` 使用 Python 3.13，缺少 pip 和 PyTorch，与现有环境文档要求的 Python 3.11 不一致。本设计将 Python 3.11 设为项目标准版本。

## 依赖来源

`pyproject.toml` 是唯一手工维护的依赖来源：

```toml
dependencies = [
  "torch>=2.6,<3",
]

[dependency-groups]
dev = [
  "pytest>=8,<10",
]
```

`uv.lock` 由 `uv lock`/`uv sync` 生成，用于锁定本地 RTX 4060 和 A10 服务器上的实际版本。`requirements.txt` 由以下命令生成，不手工修改。`--no-header` 避免在导出文件中嵌入输出路径，使不同目标路径的逐字节比较保持稳定：

```bash
uv export --format requirements-txt --no-dev --no-header --output-file requirements.txt
```

项目使用 `.python-version` 指定 Python 3.11。标准安装流程为：

```bash
uv python install 3.11
uv sync --python 3.11
uv run python scripts/check_environment.py
```

## PyTorch 与 CUDA 策略

通用依赖不按 GPU 型号拆分，也不把某个 CUDA wheel index 写死在 `requirements.txt`。Linux PyTorch wheel 自带其需要的 CUDA runtime；系统 NVIDIA 驱动负责提供兼容的驱动接口。

若某台机器需要使用 PyTorch 官方 CUDA 专用 index，应以当时的 PyTorch 官方安装选择器为准，并在机器级安装说明中明确记录，不修改项目的通用依赖来源。不得根据 `nvidia-smi` 显示的 CUDA Version 直接推断应该安装哪个 wheel，也不自动安装 NVIDIA 驱动或完整 CUDA Toolkit。

## 文件范围

新增：

```text
pyproject.toml
uv.lock
requirements.txt
.python-version
scripts/check_environment.py
tests/test_check_environment.py
```

修改：

```text
README.md
docs/environment.md
.gitignore
```

不在本次任务中安装 Transformers、Accelerate、量化库或模型下载工具。第一周教程只需要 PyTorch；后续依赖在对应学习阶段加入。

## 环境检查器

`scripts/check_environment.py` 输出：

- Python 版本。
- PyTorch 版本。
- PyTorch 编译时 CUDA runtime。
- `torch.cuda.is_available()`。
- GPU 名称、总显存和 Compute Capability。
- BF16 支持情况。
- 一个确定性的 CPU Tensor 运算结果。
- CUDA 可用时的确定性 CUDA Tensor 运算结果。

检查逻辑拆成可测试函数，命令行入口负责格式化输出和退出码。脚本不得下载模型、安装依赖或修改驱动。

## 状态与错误处理

- Python 不是 3.11：返回失败，提示运行 `uv python install 3.11` 和 `uv sync --python 3.11`。
- PyTorch 无法导入：返回失败，提示运行 `uv sync`，并保留导入错误摘要。
- PyTorch 可导入但 CUDA 不可用：CPU 检查通过，CUDA 状态标记为跳过；这不代表项目依赖安装失败。
- CUDA 可用但设备查询或张量运算失败：返回失败，显示原始异常摘要，并链接到环境排查文档。
- 没有 NVIDIA GPU：允许完成 CPU 验收，不伪造 GPU 检查结果。

## 测试策略

测试不依赖真实 GPU。通过小型替代对象或可注入的 PyTorch 接口覆盖：

- Python 3.11 接受，其他版本拒绝。
- PyTorch 缺失时输出清晰错误。
- CPU Tensor 检查结果正确。
- CUDA 不可用时正确跳过且总体成功。
- CUDA 可用时报告名称、总显存、Compute Capability 和 BF16。
- CUDA 查询或运算异常时返回失败。
- CLI 成功和失败退出码正确。

测试应验证行为和输出字段，不模拟 PyTorch 内部实现细节。

## 双机工作流

RTX 4060 与 A10 使用同一组命令：

```bash
git pull
uv sync --python 3.11
uv run python scripts/check_environment.py
```

两台机器应具有一致的 Python、PyTorch 和项目依赖版本。检查输出中的 GPU 名称、显存、Compute Capability 和可能的 BF16 状态可以不同。

## 验收标准

在当前工作区执行：

```bash
uv sync --python 3.11
uv run pytest
uv run python scripts/check_environment.py
uv export --format requirements-txt --no-dev --no-header --output-file /tmp/requirements.txt
diff requirements.txt /tmp/requirements.txt
```

验收要求：

- `uv sync` 创建 Python 3.11 环境并成功安装依赖。
- 所有测试通过。
- 环境检查器在 CPU 环境中成功运行，CUDA 缺失只标记为跳过。
- 导出的 `/tmp/requirements.txt` 与仓库中的文件完全一致。
- README 和环境文档中的命令与实际文件保持一致。

RTX 4060 和 A10 的 GPU 验证需分别在对应机器执行相同检查命令；当前无 GPU 的工作区不阻塞通用环境验收。
