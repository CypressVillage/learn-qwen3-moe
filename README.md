# Qwen3 MoE 推理课程

本项目从零实现 Qwen3 MoE 推理，并通过静态交互式教程同步展示原理和累计源码。

在线课程：<https://cypressvillage.github.io/learn-qwen3-moe/>

仓库只保留三类内容：

1. `src/qwen3_moe/`：左侧源码阅读器展示的累计推理框架。
2. `lessons/`：教程正文和对应的源码 checkpoint。
3. `web/`、`scripts/`：将正文与 checkpoint 生成静态网站的代码。

## 当前内容

Step 00 先建立完整推理地图与空文件骨架，之后沿真实推理数据流逐章实现配置与权重读取、Tokenizer、基础层、RoPE、GQA Attention、Sparse MoE、Decoder Layer、完整 Causal LM、token selection、KV Cache、Cached Attention、完整模型 decode、自回归循环，最终在 Step 14 从模型目录与 prompt 生成文本。

主线课程已经闭合完整推理链路：

```text
文本 -> token IDs -> hidden states -> Qwen3 MoE Transformer
     -> logits -> next token -> 自回归生成
```

## 生成网站

安装 [uv](https://docs.astral.sh/uv/) 和 Node.js 后，在仓库根目录运行：

```bash
uv run --locked python scripts/build_site.py
```

该命令会生成并校验全部课程资产、安装前端依赖，然后将网站构建到 `web/dist/`。

开发预览：

```bash
cd web
npm install
npm run dev
```

开发服务器通常运行在 <http://localhost:5173/>；如果该端口已被占用，请以终端输出的地址为准。
