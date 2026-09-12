# C1R exact-RC1 state/cache confirmation — first execution

Date: 2026-09-12
Raw evidence: [`c1r-gpu.json`](c1r-gpu.json) (validator `--mode gpu` output).

## Verdict recorded by the validator: FAIL, driven by one assertion

15 of 16 assertions PASS. The single failure is `fresh_reset`
(correctness): after the 16-chunk rollout, a fresh `prepare_session`
bootstrap reproduced neither x0 nor layer-0 clean K/V bitwise
(positions and all-layer cursors were correct).

## What passed (exact RC1 operating point confirmed)

- Configuration: chunk size 1, 16 chunks, local 12 / sink 6 frames,
  timesteps 999/899/702 (`3-drop-957`), 384x672, BF16 DiT, seed 42,
  scripted actions w,w,j,w,l,l,s,s,j,w,d,d,w,l,a.
- 64/64 layer-0 calls recorded; all selected boundaries
  (fill, first eviction, three repeated rolls) match the independent
  cache reference exactly; clean-t0 overwrites without advancing.
- Scenario coverage labels present incl. exact conditioning revisit;
  repeated `w` pluckers bitwise identical.
- Stale-state control detected (stale x0 differs); broken-KV control
  detected (consumer delta 34.25).

## Provenance

- Validator base commit `4f92ca5`, upstream `45fa406`, device
  Radeon 8060S / gfx1151, torch 2.13.0+rocm7.15.0a20260728.
- No optimization implemented; no timing claims.

## Next step, not a conclusion

The reset FAIL is unresolved between a real rollout-dependent
persistent-state leak and an invalid bitwise-hash reset criterion.
A minimal A/B/C tensor discriminator (`--mode reset-discriminator`,
same script) is queued to decide; C2 stays blocked until it reports.
