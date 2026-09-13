# Plan 2 P2.0 — accepted baseline, static reconstruction (2026-09-12)

Source of truth for configuration. Measurements (rolled P50/P95s, denoise,
A/A noise) come from the pending P2.0 GPU run on this exact state, not from
this file. C0B/C4 figures are historical reference only, with the C6B
adjustment noted; they are not the P2.0 measurement.

## Identity

- main SHA: `9cc986c`; tag: `lingruntime-gfx1151-b001`; prior main `add8a69`.
- Baseline record: `docs/artifacts/baseline-b001-20260912/README.md` (on main).

## Frozen configuration (unchanged from RC1/C3 contract)

- Geometry 384x672 (480x832 @ 264192 max-area px); chunk 1 latent/action
  (4 frames/action); local attention 12 frames incl. 6-frame sink
  (6048 sink + 5040 recent + 1008 current = 12096 K tokens).
- Denoise schedule 3-drop-957 → timesteps 999/899/702, 30 DiT blocks,
  12 heads, head-dim 128, bf16 autocast; TAEHV `taew2_1.pth` FP16
  (sha256 `d26151e7…`), presentation path unchanged by B001.
- Flash SDPA (`attn_fwd.kd` on gfx1151); serial execution; clean-KV
  deferred exact t=0 full DiT forward before next-ready.
- TunableOp enabled, tuning off (pinned CSV); pure-helper compilation:
  30 blocks, {modulation, affine, scaled_residual, add, silu, gelu_tanh,
  camera_update}, prewarmed; norms/attention/linears/cache eager.
- C6B `kv_write_only` dead-tail skip active (next-ready-only effect).
- Warmed-Y invariant: chunk 0 / first post-reset prepares are never
  quality or timing signals; steady-state warmed comparisons only;
  16-action quality horizon.
- Environment: Strix Halo gfx1151, ROCm (torch 2.13.0, HIP 715,
  hipBLASLt 100401, rocBLAS 5.6.0), upstream pinned `45fa406`.

## C6B confirmation (P2.0 requirement)

C6B changes next-ready only; the base-RGB path is unchanged: accepted
latents bitwise identical (screen-A1 16/16, max_abs 0.0), presentation
path byte-identical (pure-block mirror only removes post-store dead tail).
Base-RGB/denoise historical figures therefore carry over as reference;
next-ready historical figures (C0B P50 ~1259 ms) predate the ~−6 ms
clean-KV saving and must be re-measured, not reused.

## Pending P2.0 GPU measurement (queued, serialized GPU)

On tagged main, accepted rolled protocol (waterfall + A/A repeats,
15-action string, warmed-Y): rolled base-RGB P50/P95, rolled next-ready
P50/P95, three-forward denoise P50/P95 + per-forward, clean-KV result,
A/A noise envelope, environment snapshot. GPU idle at open (0%, 47 C).
