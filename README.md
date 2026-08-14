# Qwen3 MoE 推理课程

本项目从零实现 Qwen3 MoE 推理，并通过静态交互式教程同步展示原理和累计源码。

仓库只保留三类内容：

1. `src/qwen3_moe/`：右侧源码阅读器展示的累计推理框架。
2. `lessons/`：教程正文和对应的源码 checkpoint。
3. `web/`、`scripts/`：将正文与 checkpoint 生成静态网站的代码。

## 当前内容

Step 01 完成配置读取和 Safetensors 权重目录检查：

- `src/qwen3_moe/config.py`
- `src/qwen3_moe/checkpoint.py`
- `lessons/step01-overview-config-weights.md`
- `lessons/checkpoints/step01.json`

后续步骤会沿真实推理数据流继续累计实现：

```text
文本 -> token IDs -> hidden states -> Qwen3 MoE Transformer
     -> logits -> next token -> 自回归生成
```

## 生成网站

```bash
uv sync --locked --python 3.11.15
uv run python scripts/generate_step01_assets.py
uv run python scripts/validate_course_assets.py

cd web
npm install
npm run build
```

开发预览：

```bash
cd web
npm run dev
```
