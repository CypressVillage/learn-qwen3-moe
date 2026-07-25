# Week 7 Dense Decoder Tutorial Implementation Plan

**Goal:** Publish a self-contained Chinese Week 7 tutorial that combines the previously built normalization, positional attention, GQA, and SwiGLU components into a traceable Dense Decoder and a deterministic `[B,S] -> [B,S,V]` causal LM forward.

**Architecture:** One long-form Markdown tutorial contains six boundary-driven modules, independently runnable Python fences, exercises, a two-layer micro Dense Causal LM capstone, assessment, hints, answers, and quick references. README and roadmap provide stable entry points. The main path uses independent embedding and LM Head weights. No production source module, formal test file, MoE, KV Cache, generation loop, model download, or environment change is introduced.

**Tech Stack:** Markdown, Python 3.11, PyTorch 2.7, documentation checks

---

### Task 1: Create the tutorial frame

- Create `docs/tutorials/week07-dense-decoder.md` with the navigation and explicit-anchor style used by Weeks 3-6.
- Add the full-inference position, learning goals, Predict-Run-Explain method, notation, CPU path, and scope boundaries.
- Add six modules, capstone, final assessment, hints, answers, glossary, quick references, and Week 8 preview.

### Task 2: Establish the pre-norm residual contract

- Contrast `x + F(norm(x))` with post-norm ordering using hand-checkable tensors.
- Require exact shape, dtype, and device equality before residual addition.
- Treat branch switches as gates on update terms while preserving the identity skip.
- Demonstrate that residual norms are observable but not guaranteed to increase monotonically.

### Task 3: Build the attention residual block

- Reuse the Week 5 order: Q/K/V projection, head reshape, QK Norm, split-half RoPE, causal GQA, merge, and output projection.
- Wrap attention with input RMSNorm and the first residual addition.
- Trace Q/K/V, scores, probabilities, update, and residual output contracts.
- Verify exact future-zero probabilities and identity behavior when the attention update is disabled.

### Task 4: Build the SwiGLU residual block

- Reuse the Week 6 no-bias `SiLU(gate) * up -> down` path.
- Wrap SwiGLU with post-attention RMSNorm and the second residual addition.
- Prove that the MLP residual base is the attention block output rather than the original layer input.
- Reject broadcasting and residual metadata mismatches.

### Task 5: Compose one Dense Decoder Layer

- Combine distinct input and post-attention RMSNorm instances with attention and MLP modules.
- Compare full, attention-only, MLP-only, and both-updates-disabled paths.
- Align an explicit reference composition with the module at six stable boundaries.
- Record branch-update norms without imposing a monotonicity claim.

### Task 6: Stack layers and diagnose first divergence

- Use `nn.ModuleList` with independent layer instances and parameters.
- Record cloned, execution-ordered trace values with names such as `layers.1.mlp_update`.
- Compare one-layer and multi-layer hidden states while preserving `[B,S,D]`.
- Implement a comparator that checks key order, shape, dtype, device, finite values, and numerical tolerance.

### Task 7: Build the micro Dense Causal LM capstone

- Compose token embedding, two Dense Decoder Layers, final RMSNorm, and an independent no-bias LM Head.
- Use fixed `B=2, S=4, V=11, D=8, I=12, Hq=2, Hkv=1, Dh=4, L=2` CPU FP32 inputs.
- Verify every boundary, causal probabilities, one-layer versus two-layer updates, branch switches, and equal-prefix logits.
- Deep-copy the model, identify an active layer-2 SwiGLU product channel, perturb its down projection, and require the first trace difference at `layers.1.mlp_update`.
- Keep embedding/LM Head weight tying as a non-required extension.

### Task 8: Add exercises, entry points, and verification

- Add two stable exercises per module, `C1-C10`, `S1-S5`, scoring rules, hints, and reasoned answers.
- Add quick-reference tables for layer order, shapes, masks, trace keys, residual checks, and common errors.
- Add a narrow Week 8 preview for router logits and Top-K before replacing the Dense MLP boundary.
- Add the Week 7 tutorial link to `README.md` and `docs/roadmap.md`.
- Execute each Python fence independently when a compatible PyTorch interpreter is available.
- Verify Python syntax, fences, anchors, internal links, exercise counts, absence of placeholders, relative links, and `git diff --check`.
- Do not change the environment; report unavailable runtime verification separately.
