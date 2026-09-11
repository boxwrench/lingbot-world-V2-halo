# LingBot World v2 Strix Halo RC1 benchmark

Formal release-candidate measurement for commit
`f70e4e00a9b784f171330a1c4f19a1f96de2dbb7`.

## Result

The frozen localhost browser path passed the formal benchmark. The established
40-action protocol completed 40 actions and 29 rolled actions with finite
outputs, no tail drops, correct 12-frame cache rollover, and no browser-induced
engine regression relative to the qualification run.

| Rolled metric | P50 | P95 |
|---|---:|---:|
| 3× denoise | 909.6 ms | 922.5 ms |
| action → base RGB ready | 938.4 ms | 949.7 ms |
| clean KV | 298.3 ms | 299.9 ms |
| action → next-ready | 1255.3 ms | 1265.9 ms |
| browser keydown → frame received | 944.1 ms | 954.5 ms |
| browser keydown → frame decoded | 946.2 ms | 956.5 ms |
| browser keydown → presented proxy | 946.2 ms | 956.5 ms |

The presented proxy is the browser-side completion boundary currently
available in the harness. It is not a compositor-paint, monitor, or
time-to-photon measurement.

## Frozen configuration

```text
384×672, latent 48×84, 1008 tokens/frame
local_attn_size=12, sink_size=6
3 denoise: 999 → 899 → 702
BF16 DiT, exact deferred t=0 clean-KV transaction
TAEHV taew2_1 FP16
16 validator-matched TunableOp results
pure tensor-island compilation with runtime prewarm
one-slot latest-valid pending input
aiohttp/WebSocket localhost
CPU RGB→JPEG, quality 90, nominal 16 FPS tail cadence
```

## Startup

```text
process → model initialized:        36.255 s
model initialized → prewarm done:    2.091 s
explicit prewarm:                    1.898 s
world/session preparation:         132.792 s
runtime-ready → bootstrap RGB0:       874.6 ms
bootstrap RGB0 → INTERACTIVE_READY:   219.1 ms
process → INTERACTIVE_READY:        173.057 s
```

Startup is reported separately and is not included in action latency.

## Persistence and controls

```text
actions:             40
rolled actions:      29
final global KV:     41328
final local KV:      12096
final sink KV:       6048
finite output:       yes
tail drops:          0
```

The existing matched-state action-response result remains the valid reference:
RGB0 was materially action-responsive for strong turn and forward-versus-
reverse controls. A separate real Chrome smoke validated action IDs, latest
valid pending replacement (`w`, `d`, then `s`, with `d` replaced by `s`), and
safe `q` termination.

## Memory and artifacts

The browser server does not currently export PyTorch peak allocator or process
high-water RSS. The formal compact summary records these as unavailable rather
than inferring them. The closest accepted same-stack reference is 30.049 GiB
peak allocated, 42.358 GiB peak reserved, and 25.870 GiB peak RSS.

Compact machine-readable evidence is in
[`docs/artifacts/rc1-benchmark-20260911/`](artifacts/rc1-benchmark-20260911/):

```text
summary.json
environment.json
actions.jsonl
README.md
```

The full protocol and server metrics remain under the ignored
`results/raw/rc1-benchmark-20260911/` directory.
