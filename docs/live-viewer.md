# Keyboard live viewer

The repository now includes a bounded Tk/Pillow viewer for the accepted
`384x672`, chunk-size-1, 3-step LingBot path with opt-in TAEHV presentation.
It reuses the existing persistent DiT/KV and causal decoder state. It does
not capture a video by default.

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

For a longer persistence test that fills and rolls the 18-frame world window
multiple times:

```bash
LINGBOT_LIVE_MAX_ACTIONS=40 \
LINGBOT_LIVE_TIMEOUT_SECONDS=900 \
bash scripts/run_live.sh
```

To save the generated RGB stream as an MP4 after the session exits, add
`--save-video`:

```bash
LINGBOT_LIVE_MAX_ACTIONS=40 \
LINGBOT_LIVE_TIMEOUT_SECONDS=900 \
bash scripts/run_live.sh --save-video
```

The capture is written to
`results/raw/live-taehv-384x672/live.mp4` (or the directory selected with
`LINGBOT_LIVE_OUTPUT_DIR`) and the path is also recorded in
`live_metrics.json`. It is a generated-frame capture at 16 FPS, not a screen
recording with the viewer window or keyboard timing. Frames are copied after
the first frame is presented and encoded only when the session ends, so MP4
encoding does not block first-visible latency. Capture mode does add a host
copy after each action for the frames that are retained.

The supplied example pose path provides up to 67 user actions; larger values
are automatically limited by the available path.

For deterministic profiling, pass a comma-separated action sequence. This
uses the same live generation path but does not wait for manual keypresses:

```bash
ACTIONS='w,w,j,w,l,l,s,s,j,w,d,d,w,l,a,a,w,j,s,l,w,w,d,j,a,s,l,w,d,d,j,w,a,a,l,s,w,j,l,w'
LINGBOT_LIVE_OUTPUT_DIR=results/raw/live-context-profile \
LINGBOT_LIVE_MAX_ACTIONS=40 \
LINGBOT_LIVE_TIMEOUT_SECONDS=900 \
bash scripts/run_live.sh --scripted-actions "$ACTIONS"
```

To attach detailed DiT/SDPA and clean-KV probes only at selected chunk IDs,
add `--profile-contexts 1,9,17,18`. Profiling output is stored in
`live_metrics.json`; those profiled wall times are attribution data, not
product latency numbers. See
[`profile-live-context-20260910.md`](profile-live-context-20260910.md) for
the recorded experiment.

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

The run writes metrics to
`results/raw/live-taehv-384x672/live_metrics.json` by default. With
`--save-video`, it also writes `live.mp4` in that same output directory.

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
