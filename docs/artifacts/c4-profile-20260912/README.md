# C4 frozen-point profile — verdict (2026-09-12)

Apparatus: browser waterfall run ×2 (ports 8781/8782, 15 actions, pooled
n=8 rolled) + intrusive action-13 profile via the pure-compile wrapper.
Branch: `experiment/c4-profile`. Frozen point == RC1 config, so this is a
confirmation capture, not a new configuration.

## Numbers (pooled medians, n=8)

- base-RGB 969.9 ms (960.8–976.9); next-ready 1297.1 (1289.7–1308.5).
- Representative forwards: denoise 327.7 / 308.7 / 304.6, clean-KV 305.2.
- Intrusive chunk-13 attribution: denoise self-SDPA 379.4,
  denoise non-SDPA 582.5, clean-KV non-SDPA 196.0, clean-KV self-SDPA 127.1.

## Comparison with C0B (same code, config, machine, ~10 h apart)

- Level: +30 / +38 ms above C0B medians (939.9 / 1258.9), outside C0B's
  observed ranges. Both C4 runs agree with each other (tight spread).
- Shape: attribution ratios identical (denoise ~941 vs ~910, clean-KV
  ~308 vs ~300, self-SDPA 379 vs 369).
- Reading: uniform environmental level shift, not a config effect. Host
  was busy (load ~11.5, GPU 58 °C idle vs 40 °C at C0B time; desktop
  session + idle LLM server resident). No code or config changed
  (main untouched at `add8a69`).
- Consequence: C0B's envelope stands as the acceptance reference
  (quieter conditions). C4 confirms the attribution structure. Any C6
  acceptance claim must re-measure medians under quiet conditions and
  show a delta beyond run-level offsets, not just beat C4's numbers.

## Measured ranking (raw-ms order; NOT opportunity order)

1. denoise non-SDPA (Linears, norms, blocks) 582.5
2. denoise self-SDPA (rolled rectangular) 379.4
3. clean-KV non-SDPA recomputation 196.0
4. clean-KV self-SDPA 127.1

Raw ms ≠ opportunity: denoise Linears are GEMM/TunableOp-covered.
C5 research decides actionability. PF1 (self-SDPA) and PF2 (clean-KV
non-SDPA) are now active inputs — C3/C4 confirmed the hotspots they
describe. C5A takes the denoise phase, C5B the clean-KV transaction.
