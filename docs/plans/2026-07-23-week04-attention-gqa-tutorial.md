# Week 4 Self-Attention and GQA Tutorial Implementation Plan

**Goal:** Publish a self-contained Chinese Week 4 tutorial that progresses from hand-computable single-head scaled dot-product attention through causal masking, multi-head shape transforms, complete MHA, and equivalent physical-copy and logical-grouped GQA implementations.

**Architecture:** One long-form Markdown tutorial owns the complete Predict-Run-Explain sequence and keeps deterministic CPU experiments beside their explanations. README and roadmap provide stable links. No production package, standalone exercise files, model downloads, padding mask, RoPE, KV Cache, cross-attention, or training loop is introduced.

**Tech Stack:** Markdown, Python 3.11, PyTorch 2.7, uv, shell-based documentation checks

---

### Task 1: Create the tutorial frame

- Create `docs/tutorials/week04-self-attention-gqa.md` with the same navigation and explicit-anchor style as Weeks 1-3.
- Add goals, 6-10 hour study contract, Predict-Run-Explain method, locked environment commands, CPU-only guarantee, deterministic inputs, and no-download boundary.
- Add six modules, capstone, final assessment, hints, answers, glossary, and Week 5 preview sections.

### Task 2: Write Modules 1-2

- Teach single-head `Q @ K.T / sqrt(Dh)`, softmax over key positions, and `probabilities @ V` with hand-computable tensors.
- Teach pre-softmax upper-triangular causal masking, broadcast shape, zero future probabilities, and row-sum checks.
- Add controlled failures for mismatched Q/K head dimensions, invalid mask broadcasting, and fully masked rows.
- Add two stable exercises and module acceptance checks to each module.

### Task 3: Write Modules 3-4

- Teach `[B,S,D] -> [B,H,S,Dh] -> [B,S,D]`, divisibility, transpose, non-contiguity, and safe reshape.
- Build a complete deterministic causal MHA path from bias-free Q/K/V projections through output projection.
- Verify split/merge value recovery, output `[B,S,D]`, finite probabilities, row sums, and zero future probabilities.
- Add two exercises and module acceptance checks per module.

### Task 4: Write Modules 5-6

- Teach `G=Hq/Hkv`, divisibility, head mapping, and physical K/V replication with `repeat_interleave`.
- Teach logical grouping with query shape `[B,Hkv,G,S,Dh]` and K/V retained at `[B,Hkv,T,Dh]`.
- Compare physical and logical scores, probabilities, contexts, outputs, and K/V logical element counts.
- Add two exercises and module acceptance checks per module.

### Task 5: Write the capstone

- Build a deterministic micro causal GQA forward with `B=1`, `S=T=3`, `D=4`, `Hq=2`, `Hkv=1`, `Dh=2`, and fixed projection weights.
- Require written predictions for every key shape, at least one score, mask behavior, probability row, output shape, group mapping, and replication factor.
- Assert shape invariants, finite values, probability normalization, exact future zeros, physical/logical equivalence, final `[B,S,D]`, and `G`-fold physical K/V logical size.
- State explicitly that the capstone is an attention sublayer, not a complete Decoder Block.

### Task 6: Add assessment and references

- Add 10 concept questions, 5 derivation questions, scoring rules, and reasoned hints/answers for all module and final questions.
- Add quick-reference tables for Q/K/V, score contraction, scaling, mask, softmax, context, split/merge heads, MHA, and GQA.
- Add a narrow Week 5 preview for RoPE without implementing it.

### Task 7: Update entry points

- Add the Week 4 tutorial link and availability status to `README.md`.
- Add the complete tutorial link under Week 4 in `docs/roadmap.md`.

### Task 8: Verify documentation and repository

- Execute every Python fence independently in the locked Python 3.11/PyTorch 2.7.1 CPU environment.
- Verify fences, unique anchors, internal references, exercise counts, final-question counts, links, and absence of unfinished placeholders.
- Run `uv run pytest` or an equivalent locked CPU validation environment if the project CUDA wheel download is unavailable.
- Run `git diff --check`, inspect status and final differences, and do not commit unless explicitly requested.
