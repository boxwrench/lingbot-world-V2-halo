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

## First-encode probe result (2026-09-12): pattern within pipes, first-sample differs across pipes

Raw evidence: [`encode-probe.json`](encode-probe.json) (`--mode encode-probe`:
4 fresh prepares on pipe 1, 2 on a newly built pipe 2; prepare-only, no DiT,
no rollout).

- Condition patterns: pipe 1 `[X,Y,Y,Y]`, pipe 2 `[X,Y]` — the within-pipe
  pattern test passed.
- Strict gate verdict `FIRST_ENCODE_EFFECT_NOT_CONFIRMED`, solely because
  the first samples differ across pipes: pipe 1 X = `10888b…` (identical to
  the discriminator's capture A and the first execution's chunk 0),
  pipe 2 X' = `e50d85…`.
- Steady-state Y (`b3aee5…`, full SHA) is identical across 5 samples in
  2 processes plus the earlier discriminator run: every prepare after the
  first, on any pipe, yields Y. Transition magnitude X->Y reproduces exactly
  (0.15896); X'->Y differs (0.40841).
- Noise, plucker, text context bitwise identical across all 6 samples.

Reading: only the first encode on a fresh pipe is unstable; the post-first
steady state is deterministic and universal. This explains the original
`fresh_reset` FAIL (initial bootstrap on X vs reset check on Y) with no
rollout-dependent leak, and motivates a one-time throwaway conditioning
warmup before measured sessions, retaining a strict bitwise reset criterion.
Warmup validation in the validator is the next step; production
`run_browser.py` is untouched.

## Warmed-reset run (2026-09-12, /tmp only at commit time of the mode)

Raw evidence: [`state-cache-warmed-reset.json`](state-cache-warmed-reset.json)
(`--mode warmed-reset`: 1 throwaway prepare, strict bitwise A==B gate,
conditional 16-chunk rollout, strict C==A).

- Gate A==B STRICT PASS on all 7 tensors: the throwaway warmup works;
  post-warmup bootstraps are bitwise steady-state Y.
- Rollout 15/16, FAIL only on `fresh_reset` — and every semantic sub-check
  inside it is true (x0, layer-0 K, layer-0 V, all-layer cursors all match).
  The FAIL is a validator dict bug, not a measurement: `reset_pass`
  compares `reset_pos` (3 keys incl. `cache_capacity_tokens`) against a
  2-key literal, which is always False. Fixed in the companion code commit
  (compare the two cursor keys via `reset_cursors_at_bootstrap`).
- Check C==A FAIL: C.condition == warmup X (`10888b…`), i.e. the
  post-rollout prepare reverts to first-encode-like output. Recovery probe
  (C1, C2 captures, `recovery` classification) added to `--mode
  warmed-reset` in the companion code commit to classify transient vs
  persistent on rerun.
- Provenance note: this file was produced by the uncommitted warmed-mode
  tree on top of `123db37` (pre dict-fix, pre C1/C2); see the committing
  commit message for the dirt record.

## Warmed-reset rerun on fixed code (2026-09-12): rollout 16/16 PASS, transient single-prepare reversion, immediate recovery

Raw evidence: [`state-cache-warmed-reset-rerun.json`](state-cache-warmed-reset-rerun.json)
(provenance base `06ba33c`, clean tree — no dirt).

- Gate A==B strict PASS 7/7 (steady-state Y).
- Rollout verdict PASS, all 16 assertions green, failed list empty: with
  the cursor fix, `fresh_reset` passes on the exact RC1 point. Both
  original FAIL causes are now closed as measurement artifacts: (i) the
  dict bug (fixed), (ii) the X-vs-Y first-encode effect (characterized).
- Check C==A FAIL (`WARMED_C_MISMATCH`): C.condition == warmup X
  (`10888b…`) — the first prepare after the rollout reverts to
  first-encode-like output once.
- Recovery `RECOVERED_IMMEDIATELY`: C1==A and C2==A bitwise on all 7
  tensors; C1/C2 vs warmup differ only on condition (expected: A is Y,
  warmup is X). The reversion lasts exactly one prepare (C itself) and
  self-heals on the next prepare with no re-warmup procedure.
- No leak signature anywhere: noise/plucker/text identical across all
  captures; only the first-encode-sensitive condition path moves, and it
  returns on its own.
