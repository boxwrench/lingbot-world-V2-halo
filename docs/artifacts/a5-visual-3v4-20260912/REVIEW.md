# A5 blinded visual review: 3-step vs 4-step (2026-09-12)

Rater: project owner/user (human gate). The earlier agent single-rater
assessment is not the final verdict.

## How to review (blinded)

1. Open `pair1-chunk05/left.png` vs `pair1-chunk05/right.png`, then
   `pair2-chunk17/left.png` vs `pair2-chunk17/right.png`.
2. For each pair, judge: which side (if either) is sharper / more detailed /
   less artifacted? Pairs are matched by chunk (same trajectory position,
   same scene, same seed/actions/state/presentation path, TAEHV decoder).
3. Record one verdict per pair plus an overall verdict in
   `VERDICT.md` (template below): `VISUAL_PASS` (3-step confirmed — state
   which side won and on what grounds), `VISUAL_REJECT`, or
   `VISUAL_INCONCLUSIVE`. A scoped result is valid.
4. Only then open `KEY.json`.

One side of each pair is the RC1 3-step schedule (999/899/702), the other
is 4-step. Left/right assignment was randomized per pair (seed 20260912).

## Provenance

Source PNGs: committed C2 evidence on `experiment/c2-quality-screen`
(`docs/artifacts/c2-screen-20260912/`): A1 = 12+6 3-step reference,
C = 12+6 4-step. chunk05 (early) + chunk17 (late) rgb0 frames, unmodified
copies. Second-scene matched screen still pending (GPU step, runs
alongside Plan 2 profiling).

## Verdict template (rater fills in, commits on this branch)

```text
pair1-chunk05: <left|right|tie> sharper because <grounds>
pair2-chunk17: <left|right|tie> sharper because <grounds>
overall: <VISUAL_PASS|VISUAL_REJECT|VISUAL_INCONCLUSIVE>
scope: <e.g. PASS on tested scenes; universal superiority not established>
```
