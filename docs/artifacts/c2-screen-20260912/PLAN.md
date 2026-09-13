# C2 bounded chunk/context screen — plan

Date: 2026-09-12 | Branch: `experiment/c2-quality-screen` (base `main` = `add8a69`)
DAG node: C2 (ready) — mandatory first branch. Vehicle: `scripts/c2_context_screen.py`
(headless, live chunk-size-1 path, modeled on `scripts/action_response_probe.py`).

## Question

Under a matched-state comparison, does the RC1 latency-oriented point
(12+6 window, 3-step denoise, TAEHV) hold acceptable quality, and which
axis — context size, denoise steps, presentation decoder — actually moves
output quality? Follow-up DAG branches instantiate only on this screen's
evidence.

## Configurations (fresh GPU process each, serialized, one owner)

| id | window | denoise | decoder | purpose |
|----|--------|---------|---------|---------|
| A1 | 12+6 | 3-drop-957 | TAEHV | RC1 control |
| A2 | 12+6 | 3-drop-957 | TAEHV | A/A repeat: noise floor + first-encode guard |
| B | 18+6 | 3-drop-957 | TAEHV | context-size axis |
| C | 12+6 | 4-step | TAEHV | denoise-step axis |
| D | — | — | canonical vs TAEHV | offline `taehv_offline_compare.py` on A1's saved latents; zero DiT cost |

Fixed everywhere: seed 42, Great Wall prompt/image/action-path 03,
16 scripted actions `w,w,j,w,l,l,s,s,j,w,d,d,w,l,a,a`, 384x672
(`480*832` @ 264192 px), BF16 DiT, FP16 math VAE, `defer_clean_kv=True`,
sink 6 (for B), upscale off.

## First-encode guards (from C1R evidence)

- Score chunks 2..17 only; chunks 0-1 are warmup (hashes recorded, not scored).
- Never compare chunk-0/first-prepare outputs across processes as quality.
- Warmed-Y steady state is the comparison baseline.

## Captures per GPU config

`accepted_latents.pt` (scored x0 stream, accepted format), per-chunk
TAEHV RGB PNGs, `screen_trajectory.json` (config, provenance incl. git
base, warmup hashes, per-chunk x0 stats/finite/cache positions).

## Analysis (CPU, `scripts/c2_screen_compare.py`)

- x0 max_abs per chunk: B-vs-A1, C-vs-A1, ruled against the A1-vs-A2 floor.
- RGB MAD / p95 / fraction-over-0.05 per chunk pair (action-response method).
- Temporal adjacent-frame stats per run (upscale-docs method).
- Decoder MAD canonical-vs-TAEHV on A1 latents (config D).

## Verdict criteria

- PASS: all 4 GPU runs complete finite; A/A floor quantified; per-axis
  verdicts (sensitive / not-sensitive with numbers) recorded; each of the
  six follow-up branches explicitly triggered or not-triggered with reason.
- FAIL: harness or config error preventing a comparison.
- INCONCLUSIVE: metrics do not discriminate (then human viewing of the
  saved contact material decides, recorded as such).

Sensitivity ruler: axis max x0/RGB delta vs the A/A floor (ratio >> 1 with
visible structure = sensitive). The 0.02 consumer tolerance is a latent
reference, not an RGB quality bar. No blind sweeping beyond A-D.

## Cost

~1.5–3 min GPU per config (C0B: ~1.26 s/action × 18 generations + load);
4 configs ≈ 6–12 min serialized. Config D is offline.
