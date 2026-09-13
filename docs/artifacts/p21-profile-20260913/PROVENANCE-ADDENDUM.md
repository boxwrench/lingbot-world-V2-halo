# P2.1 provenance addendum (P2.9-A1, 2026-09-13, new record)

P2.1 ran 2026-09-13 on the same host/stack as the P2.0 baseline the same
day (torch 2.13+rocm, HIP 715, gfx1151; full versions in
`../p20-baseline-20260913/environment/`). No separate environment capture
was taken for P2.1; this addendum closes that gap to the P2.0 standard
without fabricating a per-run capture.

- Code under test: tagged accepted main `9cc986c`, branch
  `experiment/p2-baseline`.
- Model dir: `models/lingbot-world-v2-1.3b-causal-fast`; upstream
  `.upstream/lingbot-world-v2 @ 45fa40673607c9acba6cf96a1f9396c95bcef25f`.
- Procedures (recovered from `/tmp`, committed here):
  `procedures/run-p21-profile.sh`, `procedures/run-p21-module.sh`.
- Profile part: `run_pure_compile_live.py`, 16 actions, accepted config
  (3-drop-957, 12+6 window, seed 42,
  `--scripted-actions 'w,w,j,w,l,l,s,s,j,w,d,d,w,l,a'`).
- Module part: `run_interactive.py --profile-dit-from-chunk 18
  --profile-attention-from-chunk 18`, seed 42, same schedule/window.
- TunableOp context: vars explicitly unset per the C4 recipe (untuned) —
  this is the context mismatch documented in P2.6 lesson 1, not an
  omission: absolute kernel times from P2.1 must not feed production
  predictions.
- GPU serialized, quiet host; windowing flags are verbatim in the runners
  (`--profile-contexts 13` for the profile part,
  `--profile-dit/attention-from-chunk 18` for the module part).
