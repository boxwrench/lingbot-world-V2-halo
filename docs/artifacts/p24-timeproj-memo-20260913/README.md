# P2.4/P2.5 time-projection memo: ACCEPTED **in screen-harness scope** (2026-09-13; scope corrected P2.9-A5)

Branch `experiment/p24-timeproj-memo` (from tagged main `9cc986c`).
Runner: `procedures/run-p24-onoff.sh` (recovered from `/tmp` per P2.9-A1;
back-to-back ON/OFF, same host state).

## P2.9 correction (2026-09-13, new record — history below preserved)

Gate 2's PASS (−70.4 ms) is **an untuned screen-harness effect, not a
production effect**, and the title "ACCEPTED" above is qualified to that
scope. The production adjudication (P2.6, accepted browser path, pinned
TunableOp CSV) measures a paired per-action denoise effect of **−7.33 ms**
(`../p26-20260913/paired-analysis.json`), below the P2.3 `<10 ms` stop
rule — memo REJECTED for mainline. The −70.4 ms figure stands as a
screen-harness measurement (lead-recomputed −55.9..−83.7, mean −70.43)
and remains valid evidence that the mechanism is real; it must never be
cited as expected production gain. Screen order-swap control confirming
the screen effect is not drift: `../p25c-order-20260913/`.

## Implementation

- NEW `scripts/timeproj_memo.py`: session-level exact-replay wrapper for
  `model.time_projection`, keyed by published timestep; clone-on-hit;
  passthrough without a published timestep (safe default).
- `scripts/run_interactive.py`: env-gated install in `prepare_session`
  (`LINGBOT_TIMEPROJ_MEMO=1`, mainline default-off, upstream untouched);
  timestep published in `generate_chunk` (denoise) and `commit_clean_kv`
  (clean t=0); install event + final stats in report output.
- `scripts/c2_context_screen.py`: byte-identical copy of the C2 screen
  tool (provenance: `experiment/c2-quality-screen`) so evaluation runs
  branch-local code — script-dir-first import means invoking the C2
  worktree copy would silently run stale branch modules.
- NEW `tests/test_timeproj_memo.py`: 3 passed (CPU).

## P2.5 funnel

1. Unit/helper: 3/3 green.
2. Isolated numeric: ON-vs-OFF accepted latents bitwise **0.0**, 16/16.
3. Subsystem/block: full-trajectory equivalence IS the subsystem proof
   (every chunk bitwise; cache untouched by construction and by equality).
4. State/cache: memo touches no cache code or layout (diff-scoped).
5. Full interactive: → P2.6 browser adjudication.

## P2.5 gates

- Gate 1 (correctness): PASS — bitwise 0.0 all chunks (`on-run/` vs
  `off-run/` trajectories committed here).
- Gate 2 (effect): PASS — per-chunk denoise deltas −56 to −84 ms,
  mean **−70.4 ms** (predicted ~−66 from the 22.2 ms/call microbench).
  Consistent sign on all 16 chunks; an order of magnitude above run
  noise (±12) and far above within-run sensitivity.

## Prior vacated result (recorded, not hidden)

An earlier ON/OFF pair showed ~0 effect because both sides silently ran
the C2 worktree's stale `run_interactive` (script-dir-first import).
Root-caused via missing install print; fixed by the branch-local tool
copy; pair re-run correctly. Stale-side numbers are discarded.

## Quality gate

Not required: accepted latents bitwise identical + presentation path
untouched → objective path per P2.5.
