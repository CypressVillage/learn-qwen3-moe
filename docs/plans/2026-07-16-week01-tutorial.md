# Week 1 Tensor Tutorial Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish a self-contained Chinese Week 1 tutorial that teaches a Python beginner to reason about PyTorch tensors, shapes, layouts, matrix operations, and memory before starting Qwen3-MoE implementation.

**Architecture:** One long-form tutorial owns the complete learning sequence and keeps every experiment next to its explanation while moving hints and answers away from exercises. README and roadmap provide stable entry links; no Python package, standalone script, model download, or GPU dependency is introduced.

**Tech Stack:** Markdown, Python 3.11 snippets, PyTorch 2.x snippets, shell-based documentation checks

---

### Task 1: Create the tutorial frame and learning contract

**Files:**
- Create: `docs/tutorials/week01-tensors-shapes-memory.md`

**Step 1: Create the document heading and navigation**

Add these top-level sections in this order:

```markdown
# 第一周教程：张量、形状与内存

## 这周要学会什么
## 学习方式
## 开始前的环境检查
## 模块 1：第一个 Tensor
## 模块 2：读懂 Tensor 的基本属性
## 模块 3：索引、切片与形状变换
## 模块 4：stride、transpose 与 contiguous
## 模块 5：广播与矩阵乘法
## 模块 6：dtype 与内存估算
## 综合任务：走过一次微型语言模型数据流
## 最终验收
## 提示
## 参考答案
## 术语与速查表
## 下一周预告
```

Add a local table of contents linking to every top-level section.

**Step 2: State the learning contract**

Explain the predict-run-explain loop, 6-10 hour expectation, CPU-only completion guarantee, optional CUDA sections, no model download, and the rule that learners must write predictions before running code.

**Step 3: Add environment checks**

Provide commands that print Python and PyTorch versions and report CUDA availability without failing when CUDA is unavailable. Link to `../environment.md` for installation details rather than duplicating the entire environment guide.

**Step 4: Verify the tutorial frame**

Run:

```bash
test -f docs/tutorials/week01-tensors-shapes-memory.md
```

Expected: exit code 0.

**Step 5: Commit**

```bash
git add docs/tutorials/week01-tensors-shapes-memory.md
git commit -m "docs: start week one tensor tutorial"
```

### Task 2: Write Modules 1-3

**Files:**
- Modify: `docs/tutorials/week01-tensors-shapes-memory.md`

**Step 1: Write Module 1, the first Tensor**

Teach `import torch`, `torch.tensor`, values, `type`, shape, dtype, and device using deterministic tiny tensors. Explain the difference between a Python list and a Tensor only to the depth needed for this week. Connect `[B,S]` token IDs to later language-model inference.

Include:

- One predict-before-run question.
- One runnable snippet with expected observations.
- One common misconception.
- Two exercises, each with a unique identifier such as `M1-E1`.
- Two corresponding hints and answers in the later Hint/Answer sections.
- Three module acceptance questions.

**Step 2: Write Module 2, basic attributes**

Teach scalar/vector/matrix/higher-rank tensors, `ndim`, `shape`, `dtype`, `device`, `numel`, and the meaning of each dimension. Use `token_ids [B,S]`, `embedding_weight [V,D]`, and `hidden [B,S,D]` consistently.

Include hand calculations before API calls and explicitly distinguish rank from shape.

**Step 3: Write Module 3, indexing and shape transforms**

Teach indexing, slicing, `reshape`, `view`, `unsqueeze`, and `squeeze`. Explain element-count compatibility and avoid claiming that `reshape` always returns a view or always copies.

Include one intentional incompatible-reshape example inside a `try/except` block so readers can continue through the tutorial after observing the error.

**Step 4: Check Module 1-3 API explanations**

For every first occurrence of a PyTorch API, verify the prose explains its purpose and result. Check every code block states either expected output or the exact properties to observe.

**Step 5: Commit**

```bash
git add docs/tutorials/week01-tensors-shapes-memory.md
git commit -m "docs: teach tensor shapes and transforms"
```

### Task 3: Write Modules 4-6

**Files:**
- Modify: `docs/tutorials/week01-tensors-shapes-memory.md`

**Step 1: Write Module 4, layout and contiguous storage**

Teach `stride()`, `transpose`, `permute`, `is_contiguous`, and `contiguous`. Use a small 2D or 3D Tensor whose values make dimension changes visible. Explain that logical shape and storage layout are separate properties.

Include an intentional non-contiguous `view` failure inside `try/except`, then demonstrate `reshape` and `contiguous().view(...)` without claiming they always have identical copying behavior.

**Step 2: Write Module 5, broadcasting**

Teach trailing-dimension broadcast comparison with compatible and incompatible examples. Require the learner to align shapes on paper before running code.

Include examples equivalent to:

```text
[B,S,D] + [D]     -> [B,S,D]
[B,S,D] + [S,1]   -> [B,S,D]
```

Use concrete small dimensions and explain why every broadcast dimension is valid.

**Step 3: Write Module 5, matrix multiplication**

Distinguish elementwise `*` from `torch.matmul`/`@`. Derive:

```text
[B,S,D] @ [D,H] -> [B,S,H]
```

Name the contracted dimensions and preserved dimensions. Include an incompatible matmul inside `try/except` and explain the error using shapes rather than memorized fixes.

**Step 4: Write Module 6, dtype and memory**

Teach theoretical bytes as element count times bits per element divided by eight. Cover FP32, FP16, BF16, INT8, and packed INT4. Use ceiling division for odd INT4 element counts and state that normal PyTorch tensors do not expose a regular instructional INT4 dtype.

Show `tensor.numel() * tensor.element_size()` for real PyTorch dtypes. Add an optional CUDA subsection that resets peak statistics, allocates a small Tensor, synchronizes where needed, and reads allocated/reserved/peak values. It must clearly allow CPU-only learners to skip it.

**Step 5: Commit**

```bash
git add docs/tutorials/week01-tensors-shapes-memory.md
git commit -m "docs: teach tensor layout and memory"
```

### Task 4: Add the comprehensive task and assessments

**Files:**
- Modify: `docs/tutorials/week01-tensors-shapes-memory.md`

**Step 1: Write the comprehensive micro language-model task**

Use fixed, hand-computable dimensions and these logical tensors:

```text
token_ids:        [B,S]
embedding_weight: [V,D]
hidden:           [B,S,D]
projection:       [D,H]
output:           [B,S,H]
```

Require written predictions for shapes, dimension meanings, element counts, selected dtype byte counts, embedding lookup behavior, and matmul contraction before showing runnable verification code.

**Step 2: Add 10 concept questions**

Cover rank vs shape, dtype/device, numel, reshape constraints, view/copy nuance, stride/contiguous, broadcasting, matmul contraction, INT4 packing, and theoretical vs observed GPU memory.

**Step 3: Add 5 shape derivation questions**

Questions must progress from indexing and reshape to broadcast and batched projection. Avoid introducing Attention or MoE equations.

**Step 4: Add scoring and completion rules**

State:

- Concept pass: at least 8/10.
- Shape pass: at least 4/5.
- Comprehensive task: learner can explain the complete data flow without reading the answer.
- CUDA exercise is optional and does not affect passing.

**Step 5: Commit**

```bash
git add docs/tutorials/week01-tensors-shapes-memory.md
git commit -m "docs: add week one tensor assessment"
```

### Task 5: Complete hints, answers, glossary, and quick references

**Files:**
- Modify: `docs/tutorials/week01-tensors-shapes-memory.md`

**Step 1: Complete the Hint section**

Provide a hint for every exercise and assessment item that needs calculation. Keep hints directional: identify the rule or first step without giving the final result.

**Step 2: Complete the Answer section**

Provide a reference answer for every exercise, all 10 concept questions, all 5 shape questions, and the comprehensive task. Explain reasoning, not only final shapes or numbers.

**Step 3: Add glossary and quick references**

Add concise tables for:

- rank, shape, dimension, dtype, device, stride, view, contiguous, broadcast.
- trailing-dimension broadcast rules.
- common dtype bit/byte widths.
- shape and memory reasoning checklist.

**Step 4: Add Week 2 preview**

Preview `nn.Module`, parameters, buffers, and a hand-written linear layer without teaching those topics in detail.

**Step 5: Audit exercise coverage**

Assign stable identifiers to exercises and verify each identifier appears in the question, hint, and answer sections.

Run:

```bash
rg -o 'M[1-6]-E[0-9]+' docs/tutorials/week01-tensors-shapes-memory.md | sort
```

Expected: every identifier appears at least three times where a question has both hint and answer.

**Step 6: Commit**

```bash
git add docs/tutorials/week01-tensors-shapes-memory.md
git commit -m "docs: complete week one tutorial references"
```

### Task 6: Link the tutorial from repository entry points

**Files:**
- Modify: `README.md`
- Modify: `docs/roadmap.md`

**Step 1: Add the README entry**

Add a prominent link near the first-learning-session section:

```markdown
开始学习：[第一周教程：张量、形状与内存](docs/tutorials/week01-tensors-shapes-memory.md)
```

Do not claim later-week tutorials exist.

**Step 2: Add the roadmap entry**

Under `## 第 1 周：张量、形状与显存`, add:

```markdown
完整教程：[第一周教程：张量、形状与内存](tutorials/week01-tensors-shapes-memory.md)
```

**Step 3: Verify links resolve**

Run:

```bash
test -f docs/tutorials/week01-tensors-shapes-memory.md
test -f docs/roadmap.md
test -f README.md
```

Expected: all commands exit 0.

**Step 4: Commit**

```bash
git add README.md docs/roadmap.md
git commit -m "docs: link week one tensor tutorial"
```

### Task 7: Perform final documentation verification

**Files:**
- Verify: `docs/tutorials/week01-tensors-shapes-memory.md`
- Verify: `README.md`
- Verify: `docs/roadmap.md`
- Reference: `docs/superpowers/specs/2026-07-16-week01-tutorial-design.md`

**Step 1: Check whitespace and unresolved placeholders**

Run:

```bash
git diff --check HEAD~6..HEAD
rg -n 'TODO|TBD|待补充|稍后填写' docs/tutorials/week01-tensors-shapes-memory.md README.md docs/roadmap.md
```

Expected: `git diff --check` exits 0; ripgrep returns no matches.

**Step 2: Check Markdown fence balance**

Count lines beginning with triple backticks in the tutorial and verify the count is even. Also inspect language tags for shell, Python, and text blocks.

**Step 3: Check required sections**

Verify the tutorial contains all six modules, the comprehensive task, final assessment, hints, answers, glossary/quick reference, and Week 2 preview.

**Step 4: Validate snippets in a clean temporary script**

Manually extract each mandatory Python snippet in order into `/tmp/opencode/week01_tutorial_snippets.py`, preserving required imports and wrapping intentional failures exactly as documented. Run it with an environment containing PyTorch:

```bash
python /tmp/opencode/week01_tutorial_snippets.py
```

Expected: exit code 0 on CPU without downloading files. If PyTorch is unavailable in the current environment, record that snippet execution could not be performed and complete syntax/manual review instead; do not install or alter project dependencies solely for documentation verification.

**Step 5: Compare against the approved specification**

Read `docs/superpowers/specs/2026-07-16-week01-tutorial-design.md` and verify every goal, module, assessment, constraint, and link requirement is represented.

**Step 6: Check repository status**

Run:

```bash
git status --short --branch
git log --oneline -8
```

Expected: no uncommitted tutorial changes and all planned documentation commits appear.
