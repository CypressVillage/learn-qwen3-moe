# 环境配置与双机工作流

## 原则

- 推荐本地使用 WSL2 Ubuntu 22.04/24.04 或原生 Linux；服务器使用常见的 Ubuntu LTS。
- 两台机器都推荐使用 Python 3.11 独立虚拟环境；本文用 `uv` 获取指定版本，避免依赖发行版默认仓库是否提供 `python3.11` 包。
- 先验证 Python、PyTorch 和小测试，不在初始配置阶段下载任何模型。
- PyTorch 安装命令应以 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 当前给出的命令为准，不盲目复制过期 CUDA wheel 地址。
- NVIDIA 驱动由宿主机/服务器管理员维护；通常不需要为了使用 PyTorch wheel 单独安装完整 CUDA Toolkit。

## 本地：WSL2 或 Linux

### WSL2 前提

在 Windows PowerShell 中确认 WSL 已安装并更新：

```powershell
wsl --status
wsl --update
wsl -l -v
```

建议把仓库放在 WSL Linux 文件系统（例如 `/home/<user>/learn-qwen3-moe`），而不是 `/mnt/c/...`，以减少大量小文件和测试时的文件系统开销。WSL2 使用 Windows NVIDIA 驱动透传 GPU，不要在 WSL 中重复安装 Windows 显卡驱动。

### Linux 基础工具与 Python 3.11

Ubuntu/WSL2 中执行：

```bash
sudo apt update
sudo apt install -y ca-certificates curl git
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
uv python install 3.11
```

`uv` 把项目所需的 Python 与系统 Python 分开，因此同一流程适用于默认 Python 不同的 Ubuntu 22.04 和 24.04。执行网络脚本前可先在浏览器查看 `https://astral.sh/uv/install.sh`；受管理设备应遵循管理员批准的安装方式。如果系统已经提供可用的 Python 3.11，也可跳过 `uv python install` 并使用该解释器。不要替换系统 `/usr/bin/python3`。

### 创建虚拟环境

在仓库根目录执行：

```bash
cd /home/zbc/learn-qwen3-moe
uv venv --seed --python 3.11 .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python --version
python -m pip --version
```

预期 `python --version` 显示 `Python 3.11.x`，且 pip 路径位于当前仓库的 `.venv` 中。

### 检查 NVIDIA 驱动

```bash
nvidia-smi
```

应能看到 RTX 4060 Laptop、驱动版本和显存信息。`nvidia-smi` 显示的 `CUDA Version` 是驱动可支持的最高 CUDA 版本提示，不等于虚拟环境中 PyTorch 实际使用的 CUDA runtime 版本。

### 安装 PyTorch

根据 PyTorch 官方选择器选择 Linux、Pip、Python，以及驱动支持的 CUDA 版本。官方命令形式通常类似：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cuXXX
```

这里的 `cuXXX` 必须替换为官方当前提供且与你的驱动兼容的版本，不能原样执行。若暂时只做 CPU 学习，也可以先安装官方 CPU wheel。

后续任务创建 `pyproject.toml` 后，从仓库根目录安装项目与开发依赖：

```bash
python -m pip install -e '.[dev]'
```

`-e` 表示 editable install：修改 `src/learn_qwen3_moe` 后不必重复安装。当前文档任务尚未创建 `pyproject.toml`，因此应在后续项目配置任务完成后执行。

### 验证 PyTorch 与 CUDA

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
    print("test tensor:", x)
    print("allocated bytes:", torch.cuda.memory_allocated())
PY
```

CPU 安装时 `CUDA available: False` 是预期结果；CUDA 安装且驱动正常时应为 `True`。

## A10 服务器

### 登录并检查资源

本地创建 SSH key（已有 key 可跳过）：

```bash
ssh-keygen -t ed25519 -C "learn-qwen3-moe"
ssh-copy-id <user>@<server-host>
ssh <user>@<server-host>
```

在服务器上先检查：

```bash
nvidia-smi
df -h
free -h
command -v uv >/dev/null && uv python find 3.11 || python3 --version
```

确认看到 NVIDIA A10 及约 24GB 显存。还要确认系统内存和数据盘空间；CPU offload 需要足够 RAM，模型下载和缓存可能同时占用多个副本。

### 创建服务器虚拟环境

假设仓库位于服务器的 `~/learn-qwen3-moe`。服务器没有 `uv` 时，先按上面的“Linux 基础工具与 Python 3.11”步骤安装，或使用管理员提供的 Python 3.11：

```bash
cd ~/learn-qwen3-moe
uv python install 3.11
uv venv --seed --python 3.11 .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

同样从 PyTorch 官方选择器取得服务器适用的安装命令，然后验证：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("runtime CUDA:", torch.version.cuda)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY
```

后续 `pyproject.toml` 创建后执行：

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

### 推荐 SSH 配置

在本地 `~/.ssh/config` 中可配置别名：

```sshconfig
Host qwen-a10
    HostName <server-host>
    User <user>
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

之后使用：

```bash
ssh qwen-a10
```

长时间 benchmark 建议使用服务器已有的 `tmux`：

```bash
tmux new -s qwen-moe
```

不要在未获得许可时安装系统包、修改驱动或占满共享数据盘。

## 双机代码同步

推荐用 Git 远端同步源码和文档，但不要提交模型权重、虚拟环境、缓存或 benchmark 大文件。当前仓库已经初始化 Git；配置远端后即可在本地与服务器之间同步后续学习成果。

一次常见工作循环：

1. 在本地 RTX 4060/CPU 写实现并运行小测试。
2. 同步源码到服务器，不同步 `.venv` 和模型缓存。
3. SSH 登录 A10，激活服务器自己的 `.venv`。
4. 先运行同一组 CPU/小 CUDA 测试，再运行较大实验。
5. 把命令、版本、配置和结果摘要写入 `docs/progress.md` 的周记副本。

如果没有 Git 远端，可临时使用 `rsync`，并明确排除环境和模型目录。第一次必须加 `--dry-run`（短选项 `-n`）预览：

```bash
rsync -avn --exclude '.venv/' --exclude '__pycache__/' --exclude '.pytest_cache/' /home/zbc/learn-qwen3-moe/ qwen-a10:~/learn-qwen3-moe/
```

逐项检查预览输出、源路径和目标路径。源路径末尾的 `/` 表示同步目录内容；目标端同名文件会被本地版本覆盖。先备份服务器上的独立实验结果，初学阶段不要添加 `--delete`。确认预览无误后才去掉 `-n`：

```bash
rsync -av --exclude '.venv/' --exclude '__pycache__/' --exclude '.pytest_cache/' /home/zbc/learn-qwen3-moe/ qwen-a10:~/learn-qwen3-moe/
```

## 硬件角色与 dtype

| 工作 | 本地 RTX 4060 8GB | A10 24GB | CPU |
| --- | --- | --- | --- |
| 单元测试、小张量手算 | 推荐 | 可用 | 推荐 |
| 微型 Dense/MoE 模型 | 推荐 | 推荐 | 可用 |
| BF16 小模型实验 | 视支持情况 | 推荐 | 慢，仅作参考 |
| Qwen3-30B-A3B 全量 BF16 | 不可行 | 不可行 | 需约 60GB 权重且极慢，还需额外 RAM |
| Qwen3-30B-A3B 4-bit | 通常仍紧张 | 可作为后期实验，取决于后端与额外开销 | 可配合 offload，速度慢 |
| 吞吐/显存 benchmark | 小配置 | 主要设备 | 参考 |

不要仅凭“30B-A3B 每 token 激活约 3B 参数”估算加载显存。所有专家权重仍需驻留在 GPU、CPU 或二者之间。30B 参数按 BF16 的权重理论体积约为 `30e9 * 2 bytes = 60GB`；实际运行还需要量化元数据或框架对象、激活、KV Cache 和工作区。

## 磁盘与模型下载注意事项

初始环境配置**不需要下载模型**。第 1-10 周使用随机初始化的微型配置即可。

在第 11 周以后下载任何 checkpoint 前：

1. 阅读模型仓库文件列表和许可证/访问要求。
2. 用权重分片总大小估算实际下载量。
3. 运行 `df -h`，为下载临时文件、Hugging Face cache 和可能的转换副本预留空间。
4. 明确 cache 位置，例如大容量数据盘，而不是较小的系统盘。
5. 先下载配置、tokenizer 或权重索引等小文件做结构检查。
6. 不把 token、私有 URL、模型权重或 cache 提交到源码仓库。

可按服务器策略设置缓存位置：

```bash
export HF_HOME=/path/to/large-disk/huggingface
```

`/path/to/large-disk` 是占位符，必须替换为已获准使用且确实存在的目录。不要在初始配置阶段执行下载命令。

## 常见问题排查

### `uv` 或 Python 3.11 不可用

- 重新加载 shell，或执行 `source "$HOME/.local/bin/env"` 后检查 `uv --version`。
- 执行 `uv python install 3.11` 和 `uv python find 3.11`；受管理服务器使用管理员提供的 Python 3.11。
- 如果发行版自带的 `python3 --version` 已是兼容版本，也可用 `python3 -m venv .venv`；Ubuntu 22.04 默认 Python 3.10 不满足本项目推荐的 3.11 环境。
- 不要强行修改系统 Python 的符号链接。

### `No module named venv` 或创建环境失败

- 使用发行版 Python 创建环境时通常需要对应的 `python3-venv` 包；使用 `uv venv` 时不依赖该包。
- 删除创建失败且不完整的 `.venv` 后重新创建；确认仓库内没有要保留的数据放在其中。

### `torch.cuda.is_available()` 为 `False`

- 先运行 `nvidia-smi`。若失败，优先处理宿主机/服务器驱动或 WSL GPU 透传。
- 检查是否误装 CPU-only PyTorch：打印 `torch.__version__` 和 `torch.version.cuda`。
- 确认当前 shell 已激活正确 `.venv`，并用 `python -m pip show torch` 查看安装位置。
- 不要用安装完整 CUDA Toolkit 作为第一反应；PyTorch wheel 通常自带所需 runtime 库。

### `CUDA out of memory`

- 记录失败时的模型、batch、序列长度、dtype 和已占用显存。
- 先减小 `B/S/D/层数/专家数`，确认实现正确；不要立刻清空所有缓存掩盖峰值来源。
- 用 `nvidia-smi` 检查其他进程，用 PyTorch 内存统计区分 allocated 与 reserved。
- 30B BF16 在 8GB 或 24GB 单卡 OOM 是容量限制，不是环境安装错误。

### BF16 报错或结果异常

- 检查 `torch.cuda.is_bf16_supported()`。
- 正确性参考优先在 CPU/FP32 小张量上运行，再按硬件支持切换 dtype。
- 不假设所有算子、量化库或 GPU 对 BF16 支持相同。

### Editable install 失败

- 确认位于仓库根目录，并且后续任务已经创建 `pyproject.toml`。
- 确认使用虚拟环境内的 `python -m pip`。
- 完整保留错误输出，不要通过随意移动 `src` 文件规避包配置问题。

### SSH 中断后任务停止

- 使用 `tmux` 保持会话。
- benchmark 将结果写到明确的小型文本/JSON 文件，并记录启动命令。
- 不在共享服务器上启动无边界的下载或生成任务。
