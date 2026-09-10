# Bounded pending-input control — 2026-09-10

This change adds a usability/control policy to the Tk live viewer. It does
not change LingBot generation, sampling, KV semantics, decoder scheduling, or
GPU execution.

## Policy

`PendingInputState` sits behind the existing `LiveViewer` key callback. While
one model action is in flight, it retains at most one valid action key. A new
valid key replaces the previous pending key, so the newest command wins.
Unsupported keys do nothing. Repeated keys do not create a repeat count.

Q, Escape, and window close are sticky quit requests and have priority over
ordinary movement; they are never queued as movement.

At the exact end of an action, `choose_action()` consumes a pending key before
entering its normal wait. The clean t=0 KV pass remains a hard generation
barrier: no next DiT call can begin before it completes.

The controller records per-action generation start, input observation and
storage, replacement, selection, next-action-permitted, and presentation/KV
timestamps in `live_metrics.json`.

## Tests

The model-free checks in `scripts/test_live_input_queue.py` pass:

* latest valid command replaces prior pending commands;
* `W W W` leaves one pending `W`;
* invalid input does not erase a valid pending command;
* quit is sticky and does not accept later movement;
* input already delivered before bootstrap is promoted to the pending slot.

## Real Tk/X11 validation

The viewer was run on the actual gfx1151 host with the real model and
`xdotool` key events. The control smoke completed two user actions with finite
output and KV progression from `1008` to `3024` tokens under the 12-frame
capacity.

In the timing-controlled run, external `W` was sent at Unix millisecond
timestamp `1789075860946`, then external `D` at `1789075861745`. Using the
viewer clock, the first action had:

```text
generation start:       63118.9 ms
first RGB presented:    63801.4 ms
clean KV:               63801.4–64032.5 ms
next action permitted:  64054.6 ms
```

The external `D` arrival was approximately `63903.9 ms` in that same clock,
inside the clean-KV interval. Tk delivered it at `64066.3 ms`, after the busy
action had returned to its input wait. It was selected at `64076.4 ms` and the
next generation began at `64078.4 ms`, without another keypress. In this run
the event survived through Tk's OS event buffering and entered the normal
waiting slot rather than the busy callback's pending slot.

Relative to the first-action clock, the external `D` was sent about
`174.5 ms` before the next legal generation start. The event was not lost or
executed speculatively while clean KV was incomplete.

A separate real run exercised the pending callback path directly: Tk observed
and stored a movement key at `78244.2 ms` during the current action's final
busy/update boundary; the next action selected it at `78245.6 ms` and began at
`78246.7 ms`. The recorded event had `replaced_pending=false`, and no GPU
generation crossed the clean barrier.

The real injections did not provide a reliable multi-key X11 replacement
sequence, so replacement/repeat/invalid semantics are accepted from the
model-free state-controller tests rather than claimed from physical-key
timing. This is sufficient for the bounded policy; a worker thread is not
justified by the evidence.

## Latency and state result

The queue adds no model work. The accepted rolled 12-frame product numbers
remain approximately `1033.2 ms` first-new RGB, `333.9 ms` clean commit, and
`1418.4 ms` next-action-ready. A normal early-context real smoke after the
change measured `672.95 ms` first RGB and `935.34 ms` next-ready; this is not a
filled-window benchmark and is within the previously observed early-session
variation. A separate scripted 13-action regression reached the first
12-frame rollover with finite output: chunk 12 measured `1023.7 ms` first RGB
and `1416.6 ms` next-ready, while chunk 13 measured `1032.5 ms` and
`1422.9 ms`. This confirms the ordinary rolled path remains near the accepted
baseline after the control change.

The implementation preserves finite output, monotonic global KV progress,
bounded local capacity, and the exact clean-KV barrier. It is retained as the
viewer control contract, without changing the 12/14/18 product modes.

## Artifacts

* [real event metrics, clean-arrival run](../results/raw/input-queue-real-20260910-v4/live_metrics.json)
* [real event metrics, busy-pending run](../results/raw/input-queue-real-20260910-v2/live_metrics.json)
* [normal 12-frame rollover regression](../results/raw/input-queue-normal-20260910/live_metrics.json)
* [state-controller test](../scripts/test_live_input_queue.py)
* [implementation](../scripts/run_live.py)
