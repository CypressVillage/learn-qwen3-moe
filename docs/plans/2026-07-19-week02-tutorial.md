# Week 2 Matmul and nn.Module Tutorial Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish a self-contained Chinese Week 2 tutorial that teaches a Week 1 graduate to reason about broadcasting, matrix multiplication, PyTorch module state, and a hand-written linear layer.

**Architecture:** One long-form Markdown tutorial owns the complete Predict-Run-Explain sequence, keeps runnable experiments beside explanations, and places hints and answers at the end. README and roadmap provide stable links; no production package, standalone exercise script, Notebook, model download, or mandatory GPU path is introduced.

**Tech Stack:** Markdown, Python 3.11, PyTorch 2.7, uv, shell-based documentation checks

---

### Task 1: Create the tutorial frame and learning contract

**Files:**
- Create: `docs/tutorials/week02-matmul-nn-module.md`
- Reference: `docs/tutorials/week01-tensors-shapes-memory.md`
- Reference: `docs/superpowers/specs/2026-07-19-week02-tutorial-design.md`

**Step 1: Add the document heading and navigation**

Create these top-level sections in order:

```markdown
# 第二周教程：矩阵乘法与 nn.Module

## 这周要学会什么
## 学习方式
## 开始前的环境检查
## 模块 1：广播复习与形状语义
## 模块 2：矩阵乘法、批量矩阵乘法与 einsum
## 模块 3：第一个 nn.Module
## 模块 4：parameter、buffer 与普通属性
## 模块 5：手写教学版线性层
## 模块 6：状态、模式、梯度与可复现性
## 综合任务：构建可保存的微型线性模块
## 最终验收
## 提示
## 参考答案
## 术语与速查表
## 下一周预告
```

Add a local table of contents and explicit HTML anchors matching the Week 1 style.

**Step 2: State the learning contract**

Document the 6-10 hour expectation, Predict-Run-Explain loop, CPU-only completion guarantee, optional CUDA/low-precision observations, no downloads, deterministic tiny inputs, and requirement to explain every axis before running code.

**Step 3: Add the locked environment check**

Use the repository workflow exactly:

```bash
uv --version
uv sync --locked --python 3.11.15
uv run python scripts/check_environment.py
uv run pytest
```

Explain that an unavailable CUDA path is a successful skip and that tutorial snippets run through `uv run python` when saved to files.

**Step 4: Verify the frame**

Run:

```bash
test -f docs/tutorials/week02-matmul-nn-module.md
```

Expected: exit code 0.

### Task 2: Write Modules 1-2

**Files:**
- Modify: `docs/tutorials/week02-matmul-nn-module.md`

**Step 1: Write Module 1, broadcasting review**

Use `hidden [B,S,D]`, feature bias `[D]`, position scale `[S,1]`, and batch offset `[B,1,1]`. Require right-aligned paper derivations before execution.

Include:

- A legal and semantically correct broadcast.
- A shape-legal but semantically wrong broadcast that demonstrates PyTorch cannot read axis names.
- An incompatible broadcast in `try/except` with printed operand shapes and real error text.
- Two exercises `M1-E1` and `M1-E2`.
- Three module acceptance questions.

**Step 2: Write Module 2, matmul variants**

Teach these exact relationships:

```text
[B,S,Din] @ [Din,Dout] -> [B,S,Dout]
[B,M,K] bmm [B,K,N] -> [B,M,N]
einsum("bsd,do->bso", hidden, weight) -> [B,S,Dout]
```

Explain `@`, `torch.matmul`, `torch.bmm`, and `torch.einsum` at first use. State that `bmm` requires rank-3 operands and does not broadcast, while `einsum` is explicit notation rather than a universal performance optimization.

**Step 3: Add controlled failures**

Include a contraction mismatch and a rank mismatch for `bmm`, both caught so the script exits successfully.

**Step 4: Add exercises and acceptance**

Create `M2-E1` and `M2-E2`, covering shape derivation, contracted dimensions, and equivalent `matmul`/`einsum` notation.

### Task 3: Write Modules 3-4

**Files:**
- Modify: `docs/tutorials/week02-matmul-nn-module.md`

**Step 1: Write Module 3, first Module**

Define a minimal parameter-free module:

```python
class AddConstant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value
```

Explain inheritance, `super().__init__()`, `forward`, `module(x)`, `training`, and why direct `forward` calls bypass parts of the framework call path.

Add `M3-E1`, `M3-E2`, a common misconception, and module acceptance questions.

**Step 2: Write Module 4, registered state**

Define one module containing:

```python
self.weight = nn.Parameter(...)
self.register_buffer("running_scale", ...)
self.label = "demo"
```

Compare the exact observable results of:

```python
named_parameters()
named_buffers()
state_dict()
module.to(dtype=torch.float64)
```

Explain that `weight` and the persistent buffer enter `state_dict`, while `label` remains a normal Python attribute and does not automatically change dtype/device.

**Step 3: Demonstrate recursive registration**

Use a parent module holding a child module and show dotted state keys. Keep this to one small example; do not introduce containers or custom serialization.

**Step 4: Add exercises and acceptance**

Create `M4-E1` and `M4-E2`, including one state-key prediction and one dtype/device migration prediction.

### Task 4: Write Module 5, ManualLinear

**Files:**
- Modify: `docs/tutorials/week02-matmul-nn-module.md`

**Step 1: Present the shape contract**

Use:

```text
input:  [B,S,Din]
weight: [Din,Dout]
bias:   [Dout]
output: [B,S,Dout]
```

Require learners to identify `Din` as the contracted dimension and explain bias broadcasting.

**Step 2: Implement the teaching layer**

Use a concise implementation with `nn.Parameter`, optional bias, fixed initialization, and an explicit last-dimension check that reports actual and expected sizes.

**Step 3: Verify deterministic forward values**

Assign hand-computable weight and bias values under `torch.no_grad()`, run a `[2,2,3]` input, print output, and assert its shape and selected values.

**Step 4: Compare with nn.Linear**

Copy parameters with:

```python
reference.weight.copy_(manual.weight.T)
reference.bias.copy_(manual.bias)
```

Compare outputs using `torch.testing.assert_close(..., atol=1e-6, rtol=1e-6)` and explain `[Dout,Din]` versus `[Din,Dout]`.

**Step 5: Add controlled shape failure**

Pass an input whose last dimension is not `Din`; print the custom error and explain why arbitrary reshape is not a valid repair.

**Step 6: Add exercises and acceptance**

Create `M5-E1` and `M5-E2`, including parameter-count and reference-weight mapping questions.

### Task 5: Write Module 6 and the capstone

**Files:**
- Modify: `docs/tutorials/week02-matmul-nn-module.md`

**Step 1: Teach state save/load**

Use `tempfile.TemporaryDirectory`, `torch.save(module.state_dict(), path)`, and `torch.load(path, weights_only=True, map_location="cpu")`. Save baseline output, modify parameters, show output changes, reload state, and assert baseline restoration.

Include one shape-incompatible `load_state_dict` demonstration inside `try/except`; do not use `strict=False` as a shape repair.

**Step 2: Teach module mode versus gradient mode**

Show:

```python
module.train()
module.eval()
with torch.no_grad():
with torch.inference_mode():
```

Explicitly demonstrate that `eval()` changes `module.training` but does not by itself force `requires_grad=False` on outputs.

**Step 3: Teach seeds and conversion**

Show that resetting `torch.manual_seed` before constructing equivalent modules reproduces initialization. Move module and input together to `float64` on CPU. Add guarded optional CUDA and low-precision observations; unsupported combinations must print a skip or recorded runtime error.

**Step 4: Add exercises and acceptance**

Create `M6-E1` and `M6-E2`, covering state contents, mode/gradient distinctions, deterministic initialization, and dtype/device agreement.

**Step 5: Write the capstone**

Build a small stateful linear module with one parameter, one bias parameter, one buffer, and one ordinary attribute. Require written predictions for shapes, state keys, parameter bytes, reference-layer mapping, and save/mutate/reload behavior before runnable assertions.

### Task 6: Add assessment, hints, answers, and references

**Files:**
- Modify: `docs/tutorials/week02-matmul-nn-module.md`

**Step 1: Add 10 concept questions**

Cover broadcasting semantics, matmul contraction, `bmm`, `einsum`, Module call behavior, parameter/buffer/property distinctions, recursive state, weight orientation, train/eval versus gradient mode, and state save/load.

**Step 2: Add 5 derivation questions**

Progress from broadcast alignment through matmul/bmm/einsum to parameter/state prediction and ManualLinear reference mapping. Use identifiers `S1` through `S5`.

**Step 3: Add scoring rules**

State concept pass `8/10`, derivation pass `4/5`, capstone oral explanation required, and optional CUDA/low-precision work excluded from passing.

**Step 4: Add hints and answers**

Provide one directional hint and one reasoned answer for every `M1-E1` through `M6-E2`, `C1` through `C10`, and `S1` through `S5`. Keep hints and answers away from the questions using stable anchors.

**Step 5: Add glossary and quick-reference tables**

Include concise references for broadcast alignment, matmul/bmm/einsum shape rules, registration categories, mode/gradient contexts, ManualLinear orientation, and state save/load checklist.

**Step 6: Add Week 3 preview**

Preview integer token IDs, embedding lookup, LM Head, logits, stable softmax, temperature, and greedy next-token selection without teaching them in detail.

### Task 7: Link the tutorial from entry points

**Files:**
- Modify: `README.md:50-77`
- Modify: `docs/roadmap.md:58-85`

**Step 1: Update the repository map and status**

Add `docs/tutorials/` to the README tree and state that Week 1 and Week 2 tutorials are available.

**Step 2: Add the README learning entry**

Keep the Week 1 link and add:

```markdown
**继续学习：[第二周教程：矩阵乘法与 nn.Module](docs/tutorials/week02-matmul-nn-module.md)**
```

**Step 3: Add the roadmap entry**

Under Week 2 add:

```markdown
完整教程：[第二周教程：矩阵乘法与 nn.Module](tutorials/week02-matmul-nn-module.md)
```

Adjust roadmap implementation wording so it points to inline tutorial implementations and does not falsely claim a production `src/` module or repository test suite is delivered this week.

**Step 4: Verify links resolve**

Run:

```bash
test -f docs/tutorials/week01-tensors-shapes-memory.md
test -f docs/tutorials/week02-matmul-nn-module.md
test -f docs/roadmap.md
test -f README.md
```

Expected: all commands exit 0.

### Task 8: Perform final verification

**Files:**
- Verify: `docs/tutorials/week02-matmul-nn-module.md`
- Verify: `README.md`
- Verify: `docs/roadmap.md`
- Reference: `docs/superpowers/specs/2026-07-19-week02-tutorial-design.md`

**Step 1: Check whitespace and placeholders**

Run:

```bash
git diff --check
rg -n 'TODO|TBD|待补充|稍后填写' docs/tutorials/week02-matmul-nn-module.md README.md docs/roadmap.md
```

Expected: whitespace check passes and ripgrep has no matches.

**Step 2: Check required content and identifier coverage**

Verify six modules, capstone, assessment, hints, answers, glossary, and Week 3 preview. Count every exercise identifier and confirm each question has a hint and answer.

**Step 3: Check Markdown fences and local links**

Confirm the number of triple-backtick fence lines is even and every relative Markdown link target exists.

**Step 4: Execute mandatory Python snippets**

Extract mandatory snippets into `/tmp/opencode/week02_tutorial_snippets.py` or equivalent temporary scripts and run:

```bash
uv run python /tmp/opencode/week02_tutorial_snippets.py
```

Expected: exit code 0 on CPU, all assertions pass, controlled failures are caught, and no repository files are generated.

**Step 5: Run repository verification**

Run:

```bash
uv run pytest
```

Expected: existing environment-check tests pass.

**Step 6: Compare with the approved design**

Audit every goal, constraint, module, error exercise, assessment, and link requirement in `docs/superpowers/specs/2026-07-19-week02-tutorial-design.md` against the final files.

**Step 7: Inspect final changes**

Run:

```bash
git status --short --branch
git diff --stat
git diff -- README.md docs/roadmap.md docs/tutorials/week02-matmul-nn-module.md docs/superpowers/specs/2026-07-19-week02-tutorial-design.md docs/plans/2026-07-19-week02-tutorial.md
```

Expected: only intended Week 2 documentation changes appear. Do not commit unless the user explicitly requests it.
