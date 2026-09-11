# Strix Halo RC1 benchmark — 2026-09-11

This is the formal measurement record for the frozen localhost browser lane.
The run used the existing deterministic 40-action sequence from
`scripts/browser_benchmark.py`, sent the next action only after `next_ready`,
and completed 29 rolled actions after the 12-frame local KV reached capacity.
No video capture or optimization was enabled.

Configuration:

```text
384x672, 1008 tokens/frame, chunk_size=1
local_attn_size=12, sink_size=6
999 -> 899 -> 702
BF16 DiT, exact deferred t=0 clean KV
TAEHV taew2_1 FP16
16 persisted TunableOp results
pure-helper compilation + shape-matched prewarm
JPEG quality 90, nominal tail cadence 16 FPS
```

The public headline is the rolled server boundary:

```text
action -> base RGB ready: 938.4 ms P50 / 949.7 ms P95
action -> next-ready:     1255.3 ms P50 / 1265.9 ms P95
```

Browser-clock results are separate: keydown→frame received was 944.1/954.5
ms P50/P95, decoded 946.2/956.5 ms, and presented proxy 946.2/956.5 ms.
The presented proxy is the browser-side completion boundary available in the
harness; it is not a compositor, monitor, or time-to-photon measurement.

Startup was separate from action latency:

```text
process -> model initialized:       36.255 s
model initialized -> prewarm done:   2.091 s
explicit prewarm:                    1.898 s
world/session preparation:         132.792 s
runtime-ready -> bootstrap RGB0:     0.875 s
bootstrap RGB0 -> INTERACTIVE_READY: 0.219 s
process -> INTERACTIVE_READY:      173.057 s
```

The per-action server/browser records are in `actions.jsonl`; environment and
compact metrics are in `environment.json` and `summary.json`. Full raw output
is retained under `results/raw/rc1-benchmark-20260911/` and is intentionally
not part of the compact export.

The existing matched-state action-response record is referenced rather than
rerun: RGB0 was already materially responsive for strong turn and forward vs
reverse controls.

## Input policy

The separate real-browser smoke validated monotonically assigned action IDs,
latest-valid one-slot replacement (`w`, then `d`, then `s`; `d` was replaced by
`s`), and safe `q` termination. The exact clean-KV barrier remained mandatory.

## Memory note

The browser telemetry does not currently export PyTorch peak allocator or
process high-water RSS, so those fields are null in `summary.json` rather than
being inferred from instantaneous system readings. The closest accepted
same-stack reference is recorded there for context.
