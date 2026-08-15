# Qwen3 MoE 推理课程

本项目从零实现 Qwen3 MoE 推理，并通过静态交互式教程同步展示原理和累计源码。

在线课程：<https://cypressvillage.github.io/learn-qwen3-moe/>

仓库只保留三类内容：

1. `src/qwen3_moe/`：右侧源码阅读器展示的累计推理框架。
2. `lessons/`：教程正文和对应的源码 checkpoint。
3. `web/`、`scripts/`：将正文与 checkpoint 生成静态网站的代码。

## 当前内容

Step 00 先建立完整推理地图与空文件骨架，Step 01 完成配置和权重读取，Step 02 实现 Qwen3 byte-level BPE Tokenizer，Step 03 实现 Embedding、RMSNorm 和 Linear 基础层，Step 04 用 RoPE 把位置信息写入 Query 和 Key，Step 05 实现带 QK Norm、RoPE 与 causal mask 的 GQA Attention，Step 06 实现 top-k Router 与 SwiGLU Experts 组成的 Sparse MoE，Step 07 把 RMSNorm、Attention、MoE 与两条 residual 组装成 Decoder Layer，Step 08 把完整模型串到 vocabulary logits，Step 09 再用 greedy、temperature 与随机采样选择 next token，Step 10 为每层保存并追加 Attention Key 与 Value，Step 11 让新 Query 读取缓存历史，Step 12 将缓存与绝对位置贯穿完整模型，Step 13 完成自回归生成循环：

- `lessons/step00-inference-map.md`
- `lessons/checkpoints/step00.json`
- `src/qwen3_moe/config.py`
- `src/qwen3_moe/checkpoint.py`
- `lessons/step01-overview-config-weights.md`
- `lessons/checkpoints/step01.json`
- `src/qwen3_moe/tokenizer.py`
- `lessons/step02-tokenizer.md`
- `lessons/checkpoints/step02.json`
- `src/qwen3_moe/layers.py`
- `lessons/step03-basic-layers.md`
- `lessons/checkpoints/step03.json`
- `src/qwen3_moe/rope.py`
- `lessons/step04-rope.md`
- `lessons/checkpoints/step04.json`
- `src/qwen3_moe/attention.py`
- `lessons/step05-gqa-attention.md`
- `lessons/checkpoints/step05.json`
- `src/qwen3_moe/moe.py`
- `lessons/step06-sparse-moe.md`
- `lessons/checkpoints/step06.json`
- `src/qwen3_moe/model.py`
- `lessons/step07-decoder-layer.md`
- `lessons/checkpoints/step07.json`
- `lessons/step08-causal-lm-prefill.md`
- `lessons/checkpoints/step08.json`
- `src/qwen3_moe/generation.py`
- `lessons/step09-next-token-selection.md`
- `lessons/checkpoints/step09.json`
- `src/qwen3_moe/cache.py`
- `lessons/step10-kv-cache.md`
- `lessons/checkpoints/step10.json`
- `lessons/step11-cached-attention.md`
- `lessons/checkpoints/step11.json`
- `lessons/step12-cached-decode.md`
- `lessons/checkpoints/step12.json`
- `lessons/step13-autoregressive-generation.md`
- `lessons/checkpoints/step13.json`

后续步骤会沿真实推理数据流继续累计实现：

```text
文本 -> token IDs -> hidden states -> Qwen3 MoE Transformer
     -> logits -> next token -> 自回归生成
```

## 生成网站

```bash
uv sync --locked --python 3.11.15
uv run python scripts/generate_step00_assets.py
uv run python scripts/generate_step01_assets.py
uv run python scripts/generate_step02_assets.py
uv run python scripts/generate_step03_assets.py
uv run python scripts/generate_step04_assets.py
uv run python scripts/generate_step05_assets.py
uv run python scripts/generate_step06_assets.py
uv run python scripts/generate_step07_assets.py
uv run python scripts/generate_step08_assets.py
uv run python scripts/generate_step09_assets.py
uv run python scripts/generate_step10_assets.py
uv run python scripts/generate_step11_assets.py
uv run python scripts/generate_step12_assets.py
uv run python scripts/generate_step13_assets.py
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
