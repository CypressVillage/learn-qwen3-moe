# 环境配置与双机工作流

## 项目标准环境

RTX 4060 Laptop、A10 服务器和 CPU 学习环境使用同一套项目依赖：

| 组件 | 锁定版本 |
| --- | --- |
| uv | 0.11.28 |
| Python | 3.11.15 |
| PyTorch | 2.7.1；Linux wheel 运行时报告 `2.7.1+cu126` |
| NumPy | 2.2.6 |
| CUDA runtime | 12.6，由 PyTorch wheel 依赖提供 |

NumPy 是显式运行时依赖，用于 PyTorch/NumPy 数据互操作，并避免缺少 NumPy 时的运行期警告。它不表示项目开始依赖 Transformers；Transformers、Accelerate、量化库和模型下载工具仍不属于当前环境。

## 依赖文件职责

- `pyproject.toml`：唯一手工维护的项目依赖来源。增加、删除或修改直接依赖时只编辑此文件。
- `uv.lock`：`uv` 解析出的精确锁文件，供可复现安装使用，提交到 Git。
- `requirements.txt`：从锁文件生成的无开发依赖兼容导出，供仍要求 requirements 格式的工具使用。绝不手工编辑。

日常安装必须使用 `--locked`。如果 `pyproject.toml` 与 `uv.lock` 不一致，命令应失败而不是悄悄改锁文件。

## 安装 uv

推荐 WSL2 Ubuntu 或原生 Linux。WSL2 仓库宜放在 `/home/<user>/...`，不要放在 `/mnt/c/...`；GPU 使用 Windows NVIDIA 驱动透传，不要在 WSL 内重复安装 Windows 驱动。

Ubuntu/WSL2 可先安装基础工具，再按 uv 官方安装方式安装：

```bash
sudo apt update
sudo apt install -y ca-certificates curl git
curl -LsSf https://astral.sh/uv/0.11.28/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

确认 `uv --version` 输出以 `uv 0.11.28` 开头。执行网络脚本前可查看 `https://astral.sh/uv/0.11.28/install.sh`。受管理服务器应使用管理员批准的安装方式。`pyproject.toml` 中的 `required-version = "==0.11.28"` 会拒绝其他 uv 版本，避免不同解析或导出行为产生未审查的差异。不要替换系统 `/usr/bin/python3`，也不需要手工创建或激活 `.venv`；`uv sync` 会管理项目环境。

## clone 或 pull 后的标准命令

在 RTX 4060、A10 和 CPU 机器上都执行同一组命令：

```bash
uv python install 3.11.15
uv sync --locked --python 3.11.15
uv run python scripts/check_environment.py
uv run pytest
```

首次 clone 后直接运行；已有仓库先 `git pull`，再运行同一组命令。`uv sync --locked` 安装锁定的运行时和开发依赖，`uv run` 自动使用项目 `.venv`，无需 `source .venv/bin/activate`。

## 临时代理

先尝试直接连接。只有当前网络确实无法访问依赖源时，才把代理变量临时前缀到单条命令，例如：

```bash
HTTP_PROXY=http://cachyos:7897 HTTPS_PROXY=http://cachyos:7897 uv sync --locked --python 3.11.15
```

这种写法只影响该命令，不要把代理持久写入项目文件、shell 配置或 Git 配置。示例不含凭据；不得提交代理用户名、密码、token、私有 URL 或其他凭据。

## 环境检查器

```bash
uv run python scripts/check_environment.py
```

检查器不下载模型、不安装软件，也不修改驱动。它依次报告：

- Python 版本和 PyTorch 导入状态。
- 确定性的 CPU Tensor 运算结果。
- CUDA 可用时的 PyTorch/CUDA runtime、GPU 名称、总显存、Compute Capability、BF16 支持和 CUDA Tensor 运算结果。

每行状态和进程退出码含义如下：

- `[PASS]`：该项通过。
- `[SKIPPED] CUDA`：CUDA 当前不可用，但 Python、PyTorch 和 CPU 路径正常；CPU-only 环境整体仍为成功。
- `[FAIL]`：Python、导入、CPU 运算或 CUDA 查询/运算失败。
- 最后一行 `[PASS] 环境检查 / Environment check` 对应退出码 `0`；`[FAIL]` 对应退出码 `1`，适合脚本和 CI 判断。

CPU 机器的正常结果可以包含 `[SKIPPED] CUDA`，并以总体 `[PASS]` 和退出码 `0` 结束。GPU 机器只有实际 CUDA smoke test 通过才能算 GPU 验证成功；检查器结果是权威依据。

## PyTorch wheel、CUDA runtime 与驱动

项目锁定的 Linux PyTorch wheel 报告 `torch==2.7.1+cu126`，并通过 Python 依赖携带 CUDA 12.6 runtime 库。它们不是系统 NVIDIA 驱动，也不要求为了普通 PyTorch 使用而先安装完整 CUDA Toolkit。

`nvidia-smi` 显示的 `CUDA Version` 是系统驱动可支持的最高 CUDA 版本提示，不是当前 Python 环境的 runtime 版本。当前 runtime 应以检查器输出的 `torch.version.cuda` 为准。

CUDA 12.x 的 minor-version compatibility 需要 NVIDIA 525 系列或更新驱动，优先使用更新且仍受支持的驱动。若 `nvidia-smi` 失败或驱动过旧，应由宿主机或服务器管理员处理，而不是修改锁文件或随意重装 CUDA Toolkit。

RTX 4060 和 A10 都受该 PyTorch 构建支持。A10 属于受支持的 Ampere 架构，但架构兼容不等于目标服务器已经验证；实际 A10 验证仍待在该机器运行 `nvidia-smi` 和环境检查器并看到 CUDA `[PASS]`。

## 双机工作流

本地和服务器只同步源码、锁文件和小型结果，不同步 `.venv`、模型缓存或权重。常见循环如下：

1. 在 RTX 4060/CPU 上 pull、locked sync、运行检查器和测试。
2. 通过 Git 把同一提交同步到 A10。
3. 在 A10 上运行完全相同的四条标准命令。
4. 只有检查器 CUDA 状态为 `[PASS]` 后才运行较大 GPU 实验。
5. 在周记中记录提交 SHA、检查器输出摘要、设备、dtype 和实验结果。

长时间服务器实验使用管理员提供的 `tmux`。不要提交模型权重、缓存、虚拟环境、benchmark 大文件或任何凭据。

## 有意更新依赖

普通 clone/pull 后不要运行未锁定的解析命令。只有确实要更新依赖时：

1. 编辑 `pyproject.toml` 中的直接依赖。
2. 运行未锁定的 `uv lock`，审查 `uv.lock` 的解析变化。
3. 运行未锁定的 `uv sync --python 3.11.15`，验证新环境。
4. 运行测试和环境检查器。
5. 用锁定、无 header 的规范命令重新生成兼容导出：

```bash
uv export --locked --format requirements-txt --no-dev --no-header --output-file requirements.txt
```

提交前审查 `pyproject.toml`、`uv.lock` 和 `requirements.txt` 的差异。`--no-header` 避免输出目标路径进入文件，使导出可以稳定地逐字节比较。

## 硬件范围

| 工作 | RTX 4060 8GB | A10 24GB | CPU |
| --- | --- | --- | --- |
| 单元测试、小张量验证 | 推荐 | 可用 | 推荐 |
| 微型 Dense/MoE 模型 | 推荐 | 推荐 | 可用 |
| BF16 小模型实验 | 视检查结果 | 推荐，但尚待实机验证 | 慢，仅作参考 |
| Qwen3-30B-A3B 全量 BF16 | 不可行 | 不可行 | 权重约 60GB 且极慢 |
| 量化/显存 benchmark | 小配置 | 后期主要设备 | 参考/offload |

“每 token 激活约 3B 参数”不表示只需保存 3B 参数。全部专家权重仍要位于 GPU、CPU 或两者之间；30B 参数的 BF16 权重理论体积约 60GB，尚未计入 KV Cache、激活和框架开销。

## 常见问题排查

### `uv` 或 Python 3.11.15 不可用

- 重新加载 shell，检查 `uv --version`。
- 运行 `uv python install 3.11.15`，不要修改系统 Python 符号链接。
- 受管理服务器若禁止下载 Python，联系管理员提供精确的 Python 3.11.15 解释器。

### locked sync 报锁文件不一致

- 确认 pull 到了同一提交，且 `pyproject.toml`、`uv.lock` 都没有本地误改。
- 日常安装不要去掉 `--locked` 让命令静默更新依赖。
- 只有有意更新依赖时才按上面的维护流程运行 `uv lock` 和未锁定的 `uv sync`。

### 下载超时或连接失败

- 先确认直接访问是否只是暂时失败，然后重试。
- 网络确需代理时，只给失败的单条命令加临时 `HTTP_PROXY`/`HTTPS_PROXY` 前缀。
- 不把代理设置或凭据写入仓库；代理仍失败时保留完整 uv 错误信息供网络管理员排查。

### `torch.cuda.is_available()` 为 `False`

- 先运行 `nvidia-smi`。失败通常表示驱动、WSL 透传或设备权限问题。
- 运行检查器确认 PyTorch wheel 版本和 `torch.version.cuda`；不要根据 `nvidia-smi` 的 CUDA Version 猜 wheel。
- 确认使用 `uv run`，避免误用系统 Python 或其他虚拟环境。
- CPU 机器上 CUDA `[SKIPPED]` 是正常成功状态；预期有 GPU 的机器则继续检查驱动是否为 NVIDIA 525 系列或更新版本。

### CUDA 检查为 `[FAIL]`

- 保留检查器打印的原始异常摘要，并记录 `nvidia-smi` 输出、驱动版本和 GPU 型号。
- 不要先删除锁文件、改装另一个 torch 或安装完整 CUDA Toolkit，这会掩盖实际驱动/设备问题。
- A10 在 CUDA smoke test 成功前保持“待验证”，不能只凭型号宣称通过。

### `CUDA out of memory`

- 记录模型、batch、序列长度、dtype 和显存统计。
- 先减小 batch、序列长度、隐藏维度、层数或专家数。
- 30B BF16 在 8GB 或 24GB 单卡 OOM 是容量限制，不是安装错误。

### BF16 报错或结果异常

- 查看检查器报告的 BF16 支持状态。
- 正确性参考先使用 CPU/FP32 小张量，再按硬件支持切换 dtype。
- 不假设所有算子、量化后端或 GPU 的 BF16 行为完全相同。

### pytest 失败

- 使用规范命令 `uv run pytest`，不要调用环境外的 `pytest`。
- 先处理第一个失败并保留完整 traceback；CUDA 不可用本身不应导致环境检查测试失败。

## 模型与磁盘

初始环境配置不下载模型。第 11 周以后下载 checkpoint 前，先检查许可证、文件总大小、`df -h`、缓存目录和可用 RAM；CPU offload 还需要额外内存。不得提交 token、私有 URL、权重或缓存。
