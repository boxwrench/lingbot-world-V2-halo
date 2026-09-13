# C6A verdict: REJECTED (negative result, 2026-09-12)

Branch: `experiment/c6a-helper-islands` (unmerged). Implementation commits
remain on the branch as the record; nothing from this branch merges.

## What was tried

Fused `_norm_affine` (block norm1/norm2 + affine) and compiled `_rmsnorm`
for attention norm_q/norm_k, following the accepted pure-helper pattern.
CPU pytest 2/2 vs eager (≤2e-6).

## Why it fails (measured on gfx1151, not inferred)

GPU probe (`scripts/c6a_norm_check.py` logic, live shapes/dtypes):

- eager_noAC vs eager_AC: 0.0 (autocast innocent)
- eager vs compiled (either autocast state): **0.057** (`norm_affine`),
  **0.0145** (`rmsnorm`)

Inductor-CUDA codegen for these norm+cast patterns diverges ~1e-2 from
eager CUDA kernels (consistent with lost bf16 round-trip / codegen
reduction differences; CPU parity does not transfer to this stack).

## Why that is fatal, not tunable

Every norm output feeds attention QKV and therefore the K/V cache, whose
contract gate is **bitwise 0.0** (C1/C1R). A ~0.05 norm-level delta cannot
preserve it. No threshold relaxation is available: the cache gate is the
contract. Further GPU spend (latents/browser runs) cannot change this
conclusion, so gates 2-3 were not run.

## Dispositions

- C5A candidate 3 (norm islands): REJECTED on this stack.
- C5A candidate 2 (modulation hoist): REFUTED earlier (per-block
  modulation parameters, upstream `model_fast.py:288`; hoist impossible).
- C5A candidate 1 (12→10): still declined (C3-contract change).
- Denoise phase: no acceptable candidate remains. The phase is healthy
  K-linear flash attention + tuned GEMMs at their floor.

## Artifacts on branch

- `scripts/pure_compile_helpers.py` (norm islands + qk swap + prewarm)
- `scripts/c6a_norm_check.py`, `tests/test_pure_islands_c6a.py` (CPU green)
- This verdict. Full probe numbers in `/tmp/c6a-eval-20260912/norm_check.json`.
