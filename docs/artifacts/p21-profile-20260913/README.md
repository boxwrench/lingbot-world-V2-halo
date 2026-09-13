# P2.1 quiet denoise profile (2026-09-13)

Code under test == tagged accepted main (`9cc986c`), branch
`experiment/p2-baseline`. Two parts (GPU serialized, quiet host):

1. Forward-level timing: `/tmp/run-p21-profile.sh`
   (`run_pure_compile_live.py`, 16 actions, accepted config) →
   `live-metrics/` (live_metrics.json). Intrusive overhead present;
   use for structure, not absolutes.
2. Module-hook attribution: `/tmp/run-p21-module.sh`
   (`run_interactive.py --profile-dit-from-chunk 18
   --profile-attention-from-chunk 18`, steady-state rolled chunks) →
   `module-metrics.json` (DecoderProfiler CUDA-event exclusive +
   DitAttentionProbe SDPA dispatch timings).

## Headline (per 3-forward denoise phase, P2.0 anchor 816 ms)

Base-RGB (847) ≈ denoise (816) + decode (~38) + host (~10). Clean-KV
(270) is serial before next-ready only — irrelevant to base-RGB.
The base-RGB critical path IS the three denoise forwards.

## Re-ranked hotspots (module-exclusive shares, 4 mature chunks)

- Self-SDPA kernel ~44% (~360 ms/phase, 4.02 ms/call, K=12096):
  K-linear fused flash (`attn_fwd.kd`); dispatch overhead only 20 ms
  over 328 calls — not a dispatch problem, a kernel-math problem.
- Tuned GEMMs ~38% (~310 ms/phase, 3627 calls): TunableOp-covered;
  residual is tuned floor unless a selection gap is found.
- Block remainder ~10% (~85 ms/phase): modulation/chunk/affine dispatch
  + camera-path norms. C6A's two envelope candidates are dead here
  (modulation hoist refuted per-block params; norm islands rejected
  Inductor divergence) — needs a NEW mechanism, not a retry.
- Norms ~4.5% (~37 ms): rejected (C6A); do not revisit without new evidence.
- Cross-attn non-SDPA ~2%, GELU ~1%, tail <1%: below individual action value.

## Change vs C4

Ranking unchanged in order; shares confirmed on mainline code with
tighter steady-state windowing (chunks 18+). New precision: SDPA is
~80% of the self-attention body (kernel, not dispatch); host gap to
next-ready is only ~10 ms. P2.2 must target SDPA-kernel alternatives,
GEMM selection gaps, or a new block-remainder mechanism.
