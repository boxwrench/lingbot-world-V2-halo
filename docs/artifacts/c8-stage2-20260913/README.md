# C8 Stage 2: bounded mechanism discovery (2026-09-13)

Branch `experiment/c8-attention`. Question: is there a specific, evidenced
gfx1151/AOTriton tuning or kernel-configuration mechanism that materially
improves the exact production shape (Q=[1,12,1008,128], K=V=[1,12,12096,128],
BF16, noncausal, no mask; control `attn_fwd.kd` ≈ 4.22 ms/call)?
No custom kernel was written. CPU-first, then minimal reversible GPU probes.

## A. Tuning-DB coverage (static, read-only)

- Shipped DB: `torch/lib/aotriton.images/amd-gfx115x/`, AOTriton
  `fe12b77d`, 396 `FONLY` forward images covering
  {bf16,fp16,fp32} x head-dims 16..256 x causal/local combos.
- For our exact feature combo (bf16, hd128, non-causal, non-local) there are
  exactly **3 baked images** (`..._F_F_{0_0,0_1,3_0}...`, 86/106/84 KB,
  AKS2 xz containers, decompressed + inspected).
- Static resource decode from embedded AMDGPU metadata (wave32, 128-thread
  groups for all three):

  | image | VGPR | SGPR | VGPR spills | SGPR spills | scratch |
  |---|---|---|---|---|---|
  | 0_0 | 217 | 52 | 48 | 0 | 196 B |
  | 0_1 | 242 | 56 | 2 | 0 | 12 B |
  | 3_0 | 226 | 107 | 124 | 11 | 500 B |

  No LDS-size field in the container; LDS comes from the dispatch trace.
- No AOTriton/Triton configuration env knobs exist in `libaotriton`
  (`optune_op_attn_fwd__Trivial` = first-valid-style internal selection).

## B. Kernel characterization (rocprofv3, stack-matched profiler)

System rocprofv3 cannot attach to this stack (HSA version skew); the venv's
own `rocprofv3` works. Dispatch trace of the unmodified production call:

- kernel `attn_fwd.kd`, grid **2048 x 12 x 1** workgroups of 128 threads
  (24576 groups; y=12 heads; x implies KV-split/persistent geometry over the
  1008 query rows, not one-group-per-row tiling);
- LDS (group segment) **16384 B**/workgroup; scratch (private) **524 B**;
- steady durations 3.65–3.95 ms under profiler (HIP-event control 4.22 ms).
- Arithmetic: ~75 GFLOP/call at ~4 ms ≈ **18 TFLOPS** with only ~80 MB
  traffic (**~19 GB/s**) — not memory-bandwidth-bound; latency/occupancy
  regime with high register pressure (see table above).

## C. Configuration sensitivity — image-isolation sweep (bounded, reversible)

With sha256-backed-up images, exactly one of the three F_F bf16 images was
left visible at a time (exact-name moves, per-state verification, restore
verified by hash — 396 files, all hashes match, staging empty):

| visible image | exact-shape result |
|---|---|
| all (baseline) | p50 3.94 |
| only 0_0 | p50 3.91, identical numerics — serves the shape at parity |
| only 0_1 | FAIL `AcceleratorError: invalid argument` — cannot serve this shape |
| only 3_0 | FAIL `AcceleratorError: invalid argument` — cannot serve this shape |

(A first attempt with shell globs mis-staged files; caught by the
per-state count check, restored from backup, redone exactly. Procedure
`../stage2-notes.md` records this.)

## Mechanism verdict vs the decision bar

- The shipped DB offers **exactly one runnable config** for this workload;
  there is no better-of-three to select and no exposed knob to turn.
- Combined with Stage 1 (no backend wins; identical kernel everywhere):
  **no evidenced mechanism clears even the 5% noise bar, let alone the
  ≥10% escalation bar.** Hypotheses (rectangular retiling, spill-free
  config) remain hypotheses — the spill-free image (0_1) demonstrably
  cannot run this shape.
- This is the second stop condition in a row. Recommendation: close C8 as
  a clean negative; custom-kernel work would be speculative, without the
  evidenced mechanism §9 requires.
