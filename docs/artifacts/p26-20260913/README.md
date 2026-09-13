# P2.6 memo adjudication: REJECTED (2026-09-13)

The mechanism is real and bitwise-exact, but its production value is
~12 ms/phase — inside run noise — because the prediction was measured
in an unrepresentative context. Honest negative, fully preserved.

## Production measurement (accepted browser path, back-to-back)

- OFF: baseRGB 850.0, nextReady 1137.9, denoise 819.7 (n=15)
- ON: baseRGB 836.2 (−13.8), nextReady 1123.1 (−14.8), denoise 805.7 (−14.0)
  (between-run medians)
- Paired per-action (same action order, `paired-analysis.py` →
  `paired-analysis.json`, committed P2.9-A4): base −6.66 (t ≈ −3.9),
  next −9.45 (t ≈ −5.2), denoise **−7.33 ms** (t ≈ −5.8) — below the
  P2.3 `<10 ms` stop rule. The reject basis is the paired estimate, not
  the between-run medians.
- Per-forward deltas ≈ −2 ms; clean forward −1 ms.
- Instrumented run (`LINGBOT_TIMEPROJ_TIMELOG=1`): the stderr per-call
  series was never committed and is now lost — the previously quoted
  prose counters (60 hits / 4 misses / 4 keys) and per-call costs
  (3.5–4.9 ms) are **retracted** as unevidenced (P2.9-A2).
  `timelog-benchmark.json` carries an explicit `memo_stats` null-note
  to that effect. The root-cause argument therefore rests on the
  committed TunableOp CSV row (below), not on memo counters.

## Why the prediction was wrong

The 22 ms microbench and the 27 ms P2.1 hook figure ran WITHOUT the
pinned TunableOp results (microbench: no CSV; P2.1 profile: vars
explicitly unset per C4 recipe). The production browser loads the
pinned CSV (`tunableop_results.csv`, 16 results), where the committed
row is `GemmAndBiasTunableOp_float_TN,tn_9216_1008_1536...` (**f32**,
3.80812 ms — corrected P2.9-A3; the earlier "bf16" label was wrong,
consistent with the upstream f32 autocast block and the P2.2 memo).
The memo skips a ~4 ms call, not a 22 ms call: 3 × ~4 ≈ 12 ms is the
structural upper bound; the paired observed effect is −7.3 ms.
Prediction and observation agree in magnitude once the context matches.

## Verdict

REJECT for mainline: paired denoise −7.33 ms does not clear the P2.3
`<10 ms` stop rule or the P2.6 bar ("benefit beyond normal noise", run
band ±12 ms). No semantic or quality concern — purely sub-threshold.
Implementation retained on branch for the record; mainline untouched
(env-gated default-off in any case).

## Measurement-discipline lessons (binding for future screens)

1. Microbench/profile contexts MUST match production (pinned TunableOp
   CSV loaded) or kernel-time predictions inflate ~5×.
2. Screen-harness runs currently execute untuned GEMMs — their absolute
   timings are not production-representative (within-pair deltas remain
   valid only when both sides share the context).
3. Effect claims require order-swapped replication. Correction
   (P2.9-A4): the OFF/ON/OFF order swap that separates the −70 ms screen
   effect from run-order drift is the **screen-harness** P2.5c control
   (`../p25c-order-20260913/`, committed), not a production run — the
   earlier wording wrongly implied a production OFF/ON/OFF. The
   −70 (screen) vs −7.3 (production, paired) split was caught by the
   production-context rerun (P2.6), and the production reject rests on
   the committed paired analysis, with no third production run.

## Evidence committed here

- `on-benchmark.json` (memo-ON browser), `off-benchmark.json`
  (memo-OFF browser), `timelog-benchmark.json` (instrumented run;
  memo-stats null-note per P2.9-A2).
- `paired-analysis.py` + `paired-analysis.json` (paired per-action
  estimator, P2.9-A4).
- `procedures/run-p26-browser.sh` (recovered from `/tmp`, P2.9-A1).
- Screen ON/OFF trajectories: see `../p24-timeproj-memo-20260913/`.
- Screen order-swap control: see `../p25c-order-20260913/`.
