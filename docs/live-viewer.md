# Keyboard live viewer

The repository now includes a bounded OpenCV viewer for the accepted
`384x672`, chunk-size-1, 3-step LingBot path with opt-in TAEHV presentation.
It reuses the existing persistent DiT/KV and causal decoder state. It does
not serialize a video by default.

## Launch

From the repository root:

```bash
bash scripts/run_live.sh
```

The default session is limited to 20 user actions and 900 seconds. Override
the bounded action count or output directory without editing files:

```bash
LINGBOT_LIVE_MAX_ACTIONS=8 bash scripts/run_live.sh
```

## Controls

```text
W/A/S/D   move
J/L       turn left/right
I/K       look up/down
SPACE     generate a no-op step
Q or ESC  stop after the current safe boundary
Ctrl-C    emergency stop from the launch terminal
```

The viewer presents the first newly decoded RGB frame as soon as TAEHV has
produced and copied it. It then completes the exact clean-KV commit and waits
for the next keypress. The status overlay shows the first-RGB timing and
next-action-ready timing separately.

## Emergency behavior

The shell wrapper runs the viewer under GNU `timeout`. If the process does not
exit after the configured limit, it sends `SIGINT` and then `SIGKILL` after 20
seconds. If a GPU call appears stuck, press `Ctrl-C` in the launch terminal;
the timeout wrapper is the second line of defense. A hard GPU/driver lockup
cannot be repaired by Python, so the viewer is bounded by default and does
not run an unbounded action loop.

The run writes only metrics to
`results/raw/live-taehv-384x672/live_metrics.json` by default. It does not
write an MP4 unless a separate experiment is run with the existing batch
runner.

The viewer path has been smoke-tested through bootstrap plus one automated
`W` action on gfx1151. That action produced finite output, advanced persistent
KV from 1008 to 2016 tokens, and measured 665 ms to first RGB and 941 ms to
next-action readiness. These are a validation sample, not a P50/P95 result.

## Semantics

The first bootstrap chunk initializes the persistent world and is not counted
as a user keypress. Each subsequent recognized key generates one new latent
chunk. Decoder selection is fixed for the fresh session; no mid-session
canonical/TAE cache migration exists. The clean t=0 KV pass remains exact and
is still required before another DiT action may begin.
