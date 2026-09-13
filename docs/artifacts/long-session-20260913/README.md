# Long-session regression evidence (2026-09-13)

Runs behind
[`../../long-session-regression-diagnosis-20260913.md`](../../long-session-regression-diagnosis-20260913.md).
All runs: accepted production runtime at detached `9cc986c`
(`wt-play` code + primary venv/models/upstream), pinned TunableOp CSV,
existing telemetry only. Labeled diffs per run: ports, `/tmp` output dirs,
scripted keys through the same websocket submit path.

## Contents

- `per-action-latency.csv` — 329 rows: every complete action from all runs
  (`run,idx,key,phase,action_to_base_rgb_ms,action_to_next_ready_ms,denoise_ms,
  fwd1..3_ms,clean_kv_ms,action_prepare_ms,tae_first_rgb_ms,global/local_end_index,queue_wait_ms`).
- `browser_metrics-*.json` — full server reports (startup marks, per-action
  ledger, presentation counters) for r2a (67 actions, ex-03), r2c (240 actions,
  ex-00), r4hot (12 fresh actions), r5q (queue experiment).
- `drive-*.log` — driver console transcripts (checkpoint lines).
- `health-ready-r1.json`, `health-final-r{4hot,5q}.json` — server status
  snapshots incl. the R1 startup waterfall marks.
- `drive-*.py` — exact reproduction drivers (r2b abandoned: readiness wait too
  short for the 961-frame path; kept for provenance).
- `frames/` — six representative lead (f0) frames: matched-path onset series
  `a0015/a0030/a0045/a0060` (clean → mosaic → dissolving → static) and
  mismatched-path `ex00-a0001/ex00-a0100` (clean → static).

## Runs

| Run | Path | Actions | Result |
|---|---|---|---|
| r1-startup | ex-03 | bootstrap only | wall 288 s; session prep 226.9 s (84%) |
| r2a | ex-03 | 67 (ceiling) | fill ramp then plateau ~986/1311 ms |
| r2b | ex-00 | 0 | abandoned: driver 600 s readiness wait < ~779 s prep |
| r2c | ex-00 | 240 (ceiling) | flat 15→240; turn blocks ±8 ms |
| r4hot | ex-03 | 12 | fresh process replays ramp ±15 ms |
| r5q | ex-03 | 4 serial + 8 @700 ms | queue 189–681 ms; 2 replaced; base to 1610 ms, denoise unchanged |

## Known limitations (do not re-derive without reading these)

- Only lead (f0) frames saved per action: tail frames JPEG-encode after
  `action_record` is published, so a record-gated driver misses them.
  Server-side dropped tails were 0 in all runs.
- Per-action RSS samples in early drivers matched the wrapper pid and are
  invalid (excluded from the CSV); memory verdicts rest on host `free` trends.
- Full frame sets (~20 MB) and f0 clips lived in volatile `/tmp`
  (`/tmp/r2c-long/frames`, `/tmp/r6-clips/`); regenerate with `drive-r2c.py`
  if needed. The six frames here carry the onset finding.
- r2b's empty `health-final` snapshots were curl-vs-shutdown races and were
  not preserved.
