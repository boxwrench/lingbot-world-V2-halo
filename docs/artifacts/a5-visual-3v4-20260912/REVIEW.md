# A5 visual review: 3-step vs 4-step (revised 2026-09-12/13)

Rater: project owner/user (human gate). The earlier agent single-rater
assessment is not the final verdict.

## Status

- Committed still pairs (`pair1-chunk05/`, `pair2-chunk17/` + `KEY.json`):
  supporting evidence for close inspection, NOT sufficient for final
  visual acceptance.
- Primary evidence: blinded matched A/B **video clips** per scene
  (`scene1/clip.mp4`, `scene2/clip.mp4` once generated) — temporal
  judgment requires mature rollout viewing: accumulated drift, object
  persistence, texture stability, flicker, geometry change, motion
  coherence, post-eviction degradation.

## Clip protocol (executed for generation, recorded here)

- 4 serialized runs (`/tmp/run-a5-clips.sh`): scenes `examples/03`
  (seed 42) and `examples/01` (seed 42) × schedules 3-drop-957 and 4;
  identical 20-action string, geometry, decoder/presentation path,
  context policy; only the schedule differs.
- Clips built by `scripts/build_a5_clips.py` from mature chunks 10..21
  (48 frames @ 8 fps, side-by-side, same duration/rate/crop, no config
  labels); A/B randomized per scene, key sealed in `scene-KEY.json`.
  The builder refuses mismatched pairs (seed/actions/geometry/chunks).
- S1-3step doubles as a reproduction control: chunks 2..17 latents must
  match committed C2 A1 `accepted_latents.pt` bitwise.

## How to judge (after clips land)

Watch each scene clip. Judge temporal coherence, object/world
persistence, motion quality, texture stability, structural drift,
visible artifacts, overall preference. Record per scene:

```text
scene1: <A preferred | B preferred | No meaningful difference | Both problematic> because <grounds>
scene2: <A preferred | B preferred | No meaningful difference | Both problematic> because <grounds>
overall: <VISUAL_PASS | VISUAL_REJECT | VISUAL_INCONCLUSIVE>
scope: <e.g. PASS on tested scenes; universal superiority not established>
```

A valid conclusion may be: no consistent perceptual advantage found;
3-step is visually acceptable on tested mature rollouts and retains its
latency advantage. That supports the operating point without a universal
claim. Only then open the scene key.

## Verdict template (rater fills in)

```text
scene1: <>
scene2: <>
overall: <>
scope: <>
```
