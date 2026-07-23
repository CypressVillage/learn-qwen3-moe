# Week 3 Token, Logits, and Causal Modeling Tutorial Implementation Plan

**Goal:** Publish a self-contained Chinese Week 3 tutorial that connects a manual micro-vocabulary, Embedding, LM Head, stable softmax, temperature, greedy next-token selection, and causal input/target shifting.

**Architecture:** One long-form Markdown tutorial owns the complete Predict-Run-Explain sequence and keeps runnable deterministic experiments beside their explanations. README and roadmap provide stable links. No production package, standalone exercise files, model downloads, mandatory GPU path, attention implementation, or full generation loop is introduced.

**Tech Stack:** Markdown, Python 3.11, PyTorch 2.7, uv, shell-based documentation checks

---

### Task 1: Create the tutorial frame

- Create `docs/tutorials/week03-token-logits-causal-lm.md` with the same navigation and explicit-anchor style as Week 1 and Week 2.
- Add goals, 6-10 hour study contract, Predict-Run-Explain method, locked environment commands, CPU-only guarantee, deterministic inputs, and no-download boundary.
- Add six modules, capstone, final assessment, hints, answers, glossary, and Week 4 preview sections.

### Task 2: Write Modules 1-2

- Teach a strict manual whitespace vocabulary boundary with reversible ID mapping and an unknown-token controlled failure.
- Teach `nn.Embedding(V,D)`, `[B,S] -> [B,S,D]`, direct weight-index equivalence, integer dtype requirements, and out-of-range IDs.
- Add two stable exercises and module acceptance checks to each module.

### Task 3: Write Modules 3-4

- Teach `nn.Linear(D,V,bias=False)` as LM Head, actual weight shape `[V,D]`, explicit `hidden @ weight.T`, and logits semantics.
- Implement stable softmax by subtracting the maximum along `V`; compare with naive exponentiation and `torch.softmax`.
- Include wrong-axis semantic failure, finite-value checks, explicit tolerances, two exercises, and acceptance checks per module.

### Task 4: Write Modules 5-6

- Teach positive temperature, distribution sharpness, stable entropy observation, last-position selection, and greedy argmax with output `[B,1]`.
- Reject non-positive temperature with an informative error.
- Teach causal input/target shifting and teacher-forcing boundaries without claiming Embedding plus LM Head is a contextual language model.
- Add two exercises and acceptance checks per module.

### Task 5: Write the capstone

- Build a deterministic micro token-to-logits module from `nn.Embedding` and bias-free `nn.Linear` with fixed weights.
- Require written predictions for tokens, shapes, dtype, axis semantics, element counts, lookup values, LM Head contraction, probabilities, greedy choice, and causal shift.
- Add CPU assertions for module/index equivalence, LM Head/matmul equivalence, stable softmax/reference equivalence, finite probabilities, sum-to-one, `[B,1]` greedy output, decoding, and shifted targets.

### Task 6: Add assessment and references

- Add 10 concept questions, 5 derivation questions, scoring rules, and reasoned hints/answers for all module and final questions.
- Add quick-reference tables for vocabulary boundaries, shape flow, Embedding, LM Head, softmax, temperature, greedy choice, and causal shifting.
- Add a narrow Week 4 preview for causal self-attention and GQA.

### Task 7: Update entry points

- Add the Week 3 tutorial link and availability status to `README.md`.
- Add the complete tutorial link under Week 3 in `docs/roadmap.md` without changing the existing Week 3 scope.

### Task 8: Verify end to end

- Run whitespace and placeholder checks.
- Check required sections, exercise identifiers, matching hints/answers, Markdown fences, anchors, and local links.
- Execute all mandatory Python snippets in a temporary file under `/tmp/opencode` and require exit code 0 on CPU.
- Run `uv run pytest`.
- Inspect the final diff and ensure only intended Week 3 documentation files and entry points changed.
