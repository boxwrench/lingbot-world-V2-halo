# P2.5c order-swap control: OFF / ON / OFF screen runs (2026-09-13)

Branch `experiment/p24-timeproj-memo`. Procedure: `run-p25c-order.sh`
(recovered from `/tmp` per P2.9-A1; outputs committed here, heavy
PNG/`.pt` files excluded).

Screen harness (`scripts/c2_context_screen.py`), 16 scored chunks,
`--screen-actions "w,w,j,w,l,l,s,s,j,w,d,d,w,l,a,a"`, same host state,
back-to-back. Memo flag `LINGBOT_TIMEPROJ_MEMO` 0/1/0. Untuned GEMMs
(screen context — see P2.6 lesson 2; deltas valid within-pair only).

## Result (per-chunk `denoise_ms` vs run1-off, n=16)

- run2-on vs run1-off: mean **−70.1 ms**, range [−92.7, −49.9].
- run3-off vs run1-off: mean **−3.3 ms**, range [−27.6, +8.9].

The return-to-OFF run reproduces the first OFF run within ±~10 ms, so
the −70 ms ON effect is not run-order drift (clocks/thermals). Consistent
with the P2.5 ON/OFF pair (mean −70.4 ms).

## Memo counters (stderr, committed)

- run2-on: `{"timeproj_memo_final": {"hits": 68, "misses": 4, "keys": 4,
  "installed": 1}}`.
- run1-off / run3-off: no memo lines (memo off, as configured).

## Scope honesty

This is a **screen-harness** order swap, not a production-browser one:
it separates memo effect from run-order confounds in the untuned screen
context only. The P2.6 production adjudication rests on the back-to-back
browser ON/OFF pair plus the committed paired per-action analysis
(`../p26-20260913/paired-analysis.json`); no third production run exists.
See the P2.6 README for that scoping correction (P2.9-A4).
