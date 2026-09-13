# C2 bounded chunk/context screen — verdict (2026-09-12)

Design: [`PLAN.md`](PLAN.md). Runs: `A1/ A2/ B/ C/` (trajectory JSON +
x0 stream + two record PNGs each), `D-decoder/metrics.json`,
`analysis/c2_screen_analysis.json`. Full PNG sets, decoded frame tensors
(2×189 MB) and mp4s remain in `/tmp/c2-screen-20260912/`; everything
committed here regenerates or summarizes them (D re-runs offline from
`A1/accepted_latents.pt`).

## Headline

- **A/A determinism**: A1 vs A2 (separate processes, ~2.5 min apart)
  bitwise identical in x0 and RGB. The warmed scripted path is exactly
  reproducible; the noise floor is absolute zero, so every nonzero delta
  below is a real config effect.
- **Context (18+6 vs 12+6)**: chunks 2–11 bitwise IDENTICAL; divergence
  starts exactly at chunk 12 — the smaller window's first eviction
  (matches the C1R boundary map). x0 grows 0.164→1.64, RGB MAD
  0.0019→0.030 by chunk 17. Single-rater viewing (chunk17-rgb0 in each
  dir): same coherent scene, no degradation either way.
- **Denoise (4-step vs 3-step)**: differs everywhere from chunk 2
  (x0 2.1–3.5, RGB MAD 0.04→0.11). Single-rater viewing: the 4-step
  frame is visibly softer (foliage, mountains, detail) than 3-step on
  this scene. The RC1 3-step choice is quality-supported, not just
  latency-motivated. Broader-scene confirmation still open for C3.
- **Decoder (D)**: canonical-vs-TAEHV MAD 0.0237 on A1's live latents
  (61 frames), consistent with the prior offline 0.0319. Small
  presentation delta; TAEHV stands.
- All runs finite; temporal adjacent-frame stats near-identical across
  configs (0.051–0.055 mean); no drift, no boundary artifacts.

## Branch dispositions

- context-sensitive: TRIGGERED-BOUNDED by post-eviction divergence, then
  RESOLVED on the saved frames — sensitivity without degradation; C3
  carries the 12+6 vs 18+6 latency/quality tradeoff, no new node.
- Denoise direction: no DAG branch exists for it; fed to C3 as contract
  input (3-step supported). No new node.
- chunk-boundary-sensitive / decoder-specific / precision /
  recurrence-action-stress / deeper-limitation: explicitly NOT triggered
  (no boundary artifacts; decoder delta small and prior-evidenced; no
  precision variation tested and no anomaly suggesting it; 16-action runs
  stable with no drift; no unexplained gap).

## Verdict: PASS

Bounded screen completed; A/A floor quantified at zero; per-axis verdicts
above; follow-ups resolved or explicitly not-triggered with reasons.
C3 may consume this as the quality basis for freezing the contract.
