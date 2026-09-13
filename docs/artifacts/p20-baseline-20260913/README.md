# P2.0 accepted baseline (2026-09-13)

Code under test == tagged accepted main, no optimization changes:
`main@9cc986c` (`lingruntime-gfx1151-b001`), branch
`experiment/p2-baseline`. 15-action string (`w w j w l l s s j w d d w l a`),
warmed-Y protocol. C0B/C4 figures are historical reference only and were
not inherited.

## Runs

- `waterfall/benchmark.json`, `aa-run-1/benchmark.json`,
  `aa-run-2/benchmark.json` (15 rolled actions each, n=45 pooled).
- `environment/` (sanitized env capture: ROCm/PyTorch/UMA/host).
- `summary.json` (pooled + per-run stats, `/tmp/p20-analyze.py` procedure).
- Runner: `/tmp/run-p20-baseline.sh`. Server logs stay in
  `/tmp/p20-20260913/` (not committed).

## Pooled results (n=45)

- action → base RGB: P50 847.4, P95 963.9, mean 837.6, sd 110.6
- action → next-ready: P50 1136.5, P95 1280.3, mean 1118.1, sd 145.2
- three-forward denoise: P50 816.4, P95 932.8, mean 807.2, sd 110.8
- per-forward: fwd0 P50 274.2, fwd1 P50 271.2, fwd2 P50 272.5
  (even split — all three forwards equal-cost rectangular evaluations)
- clean-KV total: P50 269.5; clean-forward: P50 269.4
- clean-vs-denoise3: 44/45 wins, mean −5.4, sd 2.6 (C6B effect reproduced)
- decode+present proxy: P50 2.3 (negligible)
- A/A run medians: baseRGB 847.4/859.3/844.3; nextReady 1136.5/1146.1/1135.3;
  denoise 816.4/828.2/813.7 (run-to-run ~±12 ms; future wins must clear
  run noise — within-run comparisons are the sensitive metric)

## C6B verification (explicit)

- Semantically invariant: model code on main is byte-identical to the
  C6B-verified tip (`git diff 70d3220 main --
  scripts/run_interactive.py scripts/pure_compile_helpers.py` empty), so
  the screen-A1 bitwise-0.0 proof (16/16 chunks) transfers deductively.
  No re-run of an unchanged proof (wiring tests 11/11 green on merged state).
- Presentation-path invariant: merge diff touches no decoder,
  presentation, browser-protocol, or model-math files.
- Next-ready-only: only the clean-KV commit path changed; P2.0
  clean-vs-d3 reproduces the ~−6 ms effect (44/45, mean −5.4).
- Not a base-RGB optimization: no base-RGB code path changed. The P50
  847 vs C0B 940 gap is attributed to host quietness, NOT to a code win —
  do not cite it as an improvement. C0B absolutes are retired.

## Conditions

Quiet host (load ~3.0), GPU idle at start, 40 C. Full versions in
`environment/` (torch 2.13+rocm, HIP 715, gfx1151, upstream 45fa406).
