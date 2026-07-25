# Week 6 SwiGLU and Dense MLP Tutorial Implementation Plan

**Goal:** Publish a self-contained Chinese Week 6 tutorial that teaches Dense MLP, SiLU, SwiGLU gating, module state, parameter counting, and intermediate activation memory through deterministic CPU examples.

**Architecture:** One long-form Markdown tutorial contains six problem-driven modules, independently runnable Python fences, exercises, a three-path micro SwiGLU capstone, assessment, hints, answers, and quick references. README and roadmap provide stable entry points. No production source module, formal test file, complete Decoder Layer, MoE, model download, or environment change is introduced.

**Tech Stack:** Markdown, Python 3.11, PyTorch 2.7, documentation checks

---

### Task 1: Create the tutorial frame

- Create `docs/tutorials/week06-swiglu-dense-mlp.md` with the navigation and explicit-anchor style used by Weeks 3-5.
- Add the full-inference position, learning goals, Predict-Run-Explain method, notation, CPU path, and scope boundaries.
- Add six modules, capstone, final assessment, hints, answers, glossary, quick references, and Week 7 preview.

### Task 2: Teach the Dense MLP baseline and SiLU

- Explain token-wise feature transformation and prove that a Dense MLP preserves batch and sequence axes without mixing tokens.
- Implement a deterministic two-projection no-bias MLP and compare batched and token-loop paths.
- Derive `SiLU(x)=x*sigmoid(x)` and compare a handwritten implementation with `torch.nn.functional.silu`.
- Use controlled examples to compare ReLU, GELU, and SiLU without making model-quality claims.

### Task 3: Build the SwiGLU gate

- Introduce independent gate and up projections with matching `[B,S,I]` outputs.
- Implement `silu(gate(x)) * up(x)` as elementwise multiplication.
- Demonstrate gate behavior with hand-checkable tensors and reject implicit broadcasting between branches.
- Track every intermediate shape, dtype, device, and finite-value invariant.

### Task 4: Add functional and module implementations

- Implement a Tensor-weight functional path and a three-`nn.Linear` module path.
- Explain the transpose relationship between explicit `x @ W` weights and `nn.Linear.weight` storage.
- Copy identical weights into both paths and compare all observable intermediate values and final output.
- Inspect `state_dict`, parameter registration, eval mode, and inference mode.

### Task 5: Count parameters and activation memory

- Derive no-bias SwiGLU parameter count `3DI` and compare it with a no-bias two-layer MLP's `2DI`.
- Calculate theoretical parameter bytes for common dtypes without conflating them with process memory.
- Build a forward shape ledger and calculate the size of gate, up, activated gate, product, and output tensors.
- Sweep `I` to show linear parameter and activation growth while explaining simultaneous tensor lifetime.

### Task 6: Add distribution comparison and capstone

- Compare ReLU MLP, GELU MLP, and SwiGLU summary statistics under controlled dimensions, seeds, and weight scales.
- Build the fixed `B=2, S=3, D=4, I=6` capstone with token-loop, functional, and module paths.
- Verify shape, numerical equality, token independence, finite values, state keys, parameter count, theoretical bytes, `I` scaling, and controlled failures.

### Task 7: Add exercises and references

- Add two stable exercises per module, `C1-C10`, `S1-S5`, scoring rules, hints, and reasoned answers.
- Add quick-reference tables for formulas, shapes, weight layouts, parameter counts, activation bytes, and error checks.
- Add a narrow Week 7 preview for pre-norm, residual connections, and Dense Decoder composition.

### Task 8: Update entry points and verify

- Add the Week 6 tutorial link to `README.md` and `docs/roadmap.md`.
- Execute each Python fence independently when a compatible PyTorch interpreter is available.
- Verify fences, anchors, internal links, exercise counts, absence of placeholders, and relative links.
- Run `git diff --check`, inspect the final diff, and do not change the environment or commit unless explicitly requested.
