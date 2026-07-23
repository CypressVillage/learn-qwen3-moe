# Week 5 RMSNorm, RoPE, and QK Norm Tutorial Implementation Plan

**Goal:** Publish a self-contained Chinese Week 5 tutorial that teaches RMSNorm, per-head QK Norm, split-half RoPE, position offsets, dtype effects, and their correct integration into causal GQA.

**Architecture:** One long-form Markdown tutorial contains deterministic CPU experiments, exercises, a complete micro attention capstone, assessment, hints, answers, and quick references. README and roadmap provide stable entry points. No production package, model download, KV Cache, complete Decoder Block, long-context scaling policy, or performance kernel is introduced.

**Tech Stack:** Markdown, Python 3.11, PyTorch 2.7, uv, documentation checks

---

### Task 1: Create the tutorial frame

- Create `docs/tutorials/week05-rmsnorm-rope-qk-norm.md` with the same navigation and explicit-anchor style as Weeks 1-4.
- Add goals, Predict-Run-Explain method, locked environment commands, notation, CPU guarantee, and scope boundaries.
- Add six modules, capstone, final assessment, hints, answers, glossary, and Week 6 preview.

### Task 2: Teach RMSNorm and QK Norm

- Implement deterministic RMSNorm with FP32 statistics, explicit `eps`, shape checks, and non-uniform weights.
- Compare RMSNorm with LayerNorm and explain why the final weighted output need not have RMS 1.
- Apply separate Q/K RMSNorm over `Dh` after projection reshape and before transpose/RoPE.
- Demonstrate why hidden-state RMSNorm, QK Norm, and an incorrect axis are different operations.

### Task 3: Teach RoPE from rotation to vectors

- Derive split-half `rotate_half` from a two-dimensional rotation.
- Verify position 0 identity, paired norm preservation, same-position dot-product preservation, shape preservation, and rejection of odd `Dh`.
- State that adjacent-pair layouts are possible but cannot be mixed with split-half weights and formulas.

### Task 4: Add frequencies, broadcasting, and offsets

- Build inverse frequencies, angles, cos, and sin in FP32 from integer positions.
- Apply them to Q/K `[B,H,S,Dh]` with explicit broadcast shapes.
- Verify common position offsets preserve attention scores and reject invalid position inputs.

### Task 5: Cover dtype and operation order

- Compare FP32 frequency computation with a BF16 path when supported and report maximum error.
- Keep unsupported dtype cases non-blocking and require finite outputs.
- Use non-uniform QK Norm weights to show that QK Norm before RoPE is not interchangeable with RoPE before QK Norm.

### Task 6: Integrate causal GQA

- Build the complete Week 5 data flow from hidden RMSNorm through projections, per-head QK Norm, RoPE, causal GQA, merge, and output projection.
- Preserve both physical K/V replication and logical grouping paths.
- Verify all shapes, causal probabilities, normalization, finite values, offset invariance, and physical/logical equality.

### Task 7: Add exercises and references

- Add two stable exercises per module, 10 concept questions, 5 derivation questions, scoring rules, hints, and reasoned answers.
- Add quick-reference tables for formulas, shapes, invariants, error checks, and correct operation order.
- Add a narrow Week 6 preview for SwiGLU without implementing it.

### Task 8: Update entry points and verify

- Add the Week 5 tutorial link to `README.md` and `docs/roadmap.md`.
- Execute every Python fence independently in the locked environment.
- Verify fences, anchors, internal links, question counts, absence of placeholders, and repository tests.
- Run `git diff --check`, inspect the final diff, and do not commit unless explicitly requested.
