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

## Reset discriminator result (2026-09-12): BASELINE_NONDETERMINISM_OR_STRICT_HASH

Raw evidence: [`reset-discriminator.json`](reset-discriminator.json).

Fresh bootstraps A and B on one loaded pipe were NOT bitwise equal, with
no rollout between them, so the expensive rollout was correctly skipped
and no persistent-state leakage is claimed. Per-tensor A-vs-B:

| diagnostic      | bitwise | max_abs_diff | note                          |
| --------------- | ------- | ------------ | ----------------------------- |
| noise chunk 0   | equal   | 0.0          | seeded RNG reproduces         |
| plucker chunk 0 | equal   | 0.0          | pose path deterministic       |
| text context    | equal   | n/a (list)   | T5 pipe cache hit; see quirk  |
| condition ch. 0 | differs | 0.159        | VAE recompute through live pipe |
| bootstrap x0    | differs | 0.167        | exceeds 0.02 consumer tolerance |
| layer-0 clean K | differs | 0.156        | exceeds 0.0 cache tolerance   |
| layer-0 clean V | differs | 1.478        | exceeds 0.0 cache tolerance   |

Sharpening samples: capture A is bitwise identical to the first-execution
rollout's chunk 0 (x0 `087807ff…`, K `bba30e…`, V `235b7b…`), so the
first prepare on a pipe reproduces across processes; a third prepare
(state probe) matched B, not A (pattern X, Y, Y). Eliminated by code
inspection: VAE feature-cache staleness (`encode` brackets with
`clear_cache`), T5/context drift (identical), seed/plucker drift
(identical). Leading hypothesis for follow-up, not a conclusion:
first-encode kernel selection/warmup effect on the long-lived pipe.

Comparator quirk: `comparison_ab.text_context` reads `sha_equal: false`
because list-kind frozen values carry no top-level sha; the nested item
`text_context[0]` ([38,4096] bf16) is bitwise identical. Check
`capture_a/b.frozen.text_context.items`, not the top-level flag.
