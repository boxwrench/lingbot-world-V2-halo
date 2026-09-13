# P2.6 memo adjudication: REJECTED (2026-09-13)

The mechanism is real and bitwise-exact, but its production value is
~12 ms/phase — inside run noise — because the prediction was measured
in an unrepresentative context. Honest negative, fully preserved.

## Production measurement (accepted browser path, back-to-back)

- OFF: baseRGB 850.0, nextReady 1137.9, denoise 819.7 (n=15)
- ON: baseRGB 836.2 (−13.8), nextReady 1123.1 (−14.8), denoise 805.7 (−14.0)
- Per-forward deltas ≈ −2 ms; clean forward −1 ms.
- Instrumented run (`LINGBOT_TIMEPROJ_TIMELOG=1`): 60 hits / 4 misses /
  4 keys — the memo fires correctly. Per-call cost IN PRODUCTION:
  **3.5–4.9 ms**, not 22 ms.

## Why the prediction was wrong

The 22 ms microbench and the 27 ms P2.1 hook figure ran WITHOUT the
pinned TunableOp results (microbench: no CSV; P2.1 profile: vars
explicitly unset per C4 recipe). The production browser loads the
pinned CSV (`tunableop_results.csv`, 16 results), where
`tn_9216_1008_1536` bf16 → 3.8 ms tuned kernel. The memo skips a
4 ms call, not a 22 ms call: 3 × ~4 ≈ 12 ms ≈ observed −14 ms.
Prediction and observation agree once the context matches.

## Verdict

REJECT for mainline: ~12–14 ms (≈1.6% of baseRGB) does not clear the
P2.6 bar ("benefit beyond normal noise", run band ±12 ms) or the P2.3
stop rule. No semantic or quality concern — purely sub-threshold.
Implementation retained on branch for the record; mainline untouched
(env-gated default-off in any case).

## Measurement-discipline lessons (binding for future screens)

1. Microbench/profile contexts MUST match production (pinned TunableOp
   CSV loaded) or kernel-time predictions inflate ~5×.
2. Screen-harness runs currently execute untuned GEMMs — their absolute
   timings are not production-representative (within-pair deltas remain
   valid only when both sides share the context).
3. Effect claims require order-swapped replication (OFF/ON/OFF), which
   is what caught the −70 vs −14 split here.

## Evidence committed here

- `on-benchmark.json` (memo-ON browser), `off-benchmark.json`
  (memo-OFF browser), `timelog-benchmark.json` (instrumented run).
- Screen ON/OFF trajectories: see `docs/artifacts/p24-timeproj-memo-20260913/`.
