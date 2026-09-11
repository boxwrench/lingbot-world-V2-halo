# Local browser serving RC1

This change wraps the frozen LingBot World v2 1.3B Strix path in a localhost
browser application. It does not change the sampler, DiT, KV cache, TAEHV,
clean-KV transaction, or pending-input policy.

## Launch

```bash
./scripts/run_browser.sh
```

The launcher binds to `127.0.0.1:8765` by default, prepares the pinned
upstream checkout, loads the validator-matched TunableOp results, installs the
accepted pure tensor-island compilation, prewarms the live helper instances,
and only then reports `INTERACTIVE_READY`. Set
`LINGBOT_BROWSER_OPEN_BROWSER=0` to suppress automatic browser launch.

## Architecture

The model owner is one worker thread and is the only owner of the GPU, LingBot
model, TAEHV stream, KV state, camera/world state, and exact clean-KV commit.
The presentation worker receives CPU RGB tensors only and converts them to
quality-90 JPEG. It performs no CUDA/HIP operation. The aiohttp event loop
serves the static page and WebSocket; it never calls inference.

RGB0 is copied to host and submitted immediately. The three remaining TAEHV
frames are submitted to a bounded queue and the browser schedules them at a
nominal 62.5 ms cadence. Arrival of a new action's RGB0 clears stale tail
frames. The model-side mailbox has one pending slot with latest-valid
replacement semantics; quit has priority.

## Transport contract

Frames use one in-memory WebSocket binary message:

```text
uint32 big-endian JSON-header length
JSON header
JPEG bytes
```

The header carries `action_id`, `frame_index`, `frame_count`, source
`width`/`height`, `bootstrap`, server timing metadata, JPEG encode time, and
JPEG byte count. No frame is written to disk for transport. JSON messages carry
startup state, action acknowledgements, action records, server events, and
browser-only telemetry. Server and browser clock domains are never subtracted
from one another.

## Validation

The model-free protocol and HTTP smoke tests pass in the repository virtual
environment. A 40-action WebSocket benchmark and a real Chrome/CDP smoke were
also run on the Strix system. Raw run logs belong under the ignored
`results/raw/browser-serving-20260910/` directory.

| Boundary | Result |
|---|---:|
| process → `INTERACTIVE_READY` | 173.493 s |
| explicit prewarm | 1.407 s |
| runtime-ready → bootstrap RGB0 | 0.833 s |
| bootstrap RGB0 → `INTERACTIVE_READY` | 0.214 s |
| rolled action → base RGB, server P50/P95 | 930.6 / 936.5 ms |
| rolled action → next ready, server P50/P95 | 1245.7 / 1252.2 ms |
| browser keydown → frame received, P50/P95 | 936.6 / 941.5 ms |
| browser keydown → frame decoded, P50/P95 | 938.3 / 943.0 ms |
| browser keydown → frame presented proxy, P50/P95 | 938.3 / 943.0 ms |

The 40-action benchmark sent the next command only after the server's legal
`next_ready` frontier, so it measured the normal no-queue-pressure path. It
completed 29 rolled actions with no frame-tail drops. The final generation
state was global KV `41328`, local capacity `12096`, sink retention `6048`,
and all recorded outputs were finite. JPEG quality was 90; encode time was
`5.7 ms` P50 / `6.9 ms` P95, with `45.9 KB` P50 / `66.3 KB` P95 packets.

A real Chrome smoke loaded the page and dispatched actual browser key events:
`w`, then `d`, then `s` while the first action was busy. The server recorded
`d` as replaced by the newer `s`; the UI displayed action `s` and its fourth
frame, with no dropped tails. The browser smoke then sent `q` and the model
owner stopped safely. Its two generated actions measured server-side
`604.5 ms` and `1108.3 ms` browser keydown-to-paint-proxy, respectively; the
second value includes the intentional pending-action wait. This smoke is
separate from the authoritative 40-action timing run.

The accepted control values remain the release reference: approximately
`943 ms` rolled first RGB and `1292 ms` rolled next-ready for the optimized
research lane. Browser-serving acceptance requires these server-side values to
remain within normal run variance, while browser keydown/receive/decode/paint
are reported separately.
