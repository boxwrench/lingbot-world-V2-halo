# LingRuntime quality-approved semantic contract (C3)

Date: 2026-09-12 | Status: frozen | Basis: C2 screen PASS + C0B Phase 0.
Every field traces to committed evidence; no field is aspirational.

| # | Field | Frozen value | Evidence |
|---|-------|--------------|----------|
| 1 | Chunk size | 1 latent per causal action (4 frames/action) | Live path fixed (`run_live.py:564`, `run_browser.py:825`); C2 screen ran chunk-1 |
| 2 | Local attention size | 12 frames total including sink | RC1 point; 18+6 diverges post-eviction without proven benefit (`c2-screen-20260912`) |
| 3 | Sink size | 6 frames | RC1 point; sink-budget prior + C2 screen |
| 4 | Denoise schedule | `3-drop-957` → timesteps 999/899/702 | Quality-supported over 4-step (C2: 4-step visibly softer) + ~300 ms/action cheaper (C0B) |
| 5 | Resolution | 384x672 (`480*832` @ 264192 max-area px) | RC1 point, exact-geometry asserted in C1R/C2 runs |
| 6 | Decoder contract | TAEHV presentation, `taew2_1.pth` pinned, FP16; canonical MAD 0.0237 noted, not equivalence | C2 config D + prior offline compare; `run_browser.py:819-820` enforces taehv |
| 7 | Responsiveness requirement | No regression vs RC1 envelope: next-ready P50 ~1259 ms (1244.8–1272.3), base-RGB P50 ~940 ms (927.6–954.5), n=12 pooled rolled actions | C0B `phase0-20260912`; C4 must re-derive, not reuse, hotspot claims |
| 8 | Quality horizon | 16 actions from a warmed steady state | C2 screen + C1R rollout length; longer horizons need recurrence evidence (branch explicitly not triggered) |

## Standing rules carried forward

- Warmed-Y steady state is the comparison baseline; chunk 0 / first
  post-reset prepares are never quality signals (C1R).
- A/A bitwise identity holds for warmed scripted trajectories (C2):
  nonzero deltas are real effects, not noise.
- `defer_clean_kv=True` is part of the transaction boundary, not a knob.
- This contract equals the RC1 configuration C0B measured. C4 profiles
  this exact point; PF1/PF2 become candidate C5 input only through C4.
