# Closed-loop action response on Strix Halo — 2026-09-10

This was a measurement/control experiment on the accepted 384×672 LingBot
World v2 1.3B interactive path. It did not change the sampler, attention
window, renderer, clean-KV semantics, or queue behavior.

## Method

A common deterministic 20-action prefix was generated first, including the
bootstrap chunk. At the resulting rolled world state, each branch cloned the
self-attention KV state, cross-attention cache, RNG state, and decoder setup.
The matched branches were no-op versus strong right turn (`space` versus
`l`), and forward versus reverse (`w` versus `s`). Each branch used the real
three-step sampler and saved all four TAEHV RGB outputs.

A frame was classified as materially different when its pairwise mean
absolute RGB difference was at least `0.02` and at least 10% of pixels had
absolute difference above `0.05`; visual camera/geometry change remained the
primary interpretation. This threshold is diagnostic, not a perceptual
quality metric.

The probe used a virtual input observer 50 ms after RGB0. It is not a physical
keyboard injection. The current Tk viewer has no explicit application-level
input queue while GPU work is running, so this documents the timing window but
does not claim that a real key is retained.

## First action-responsive RGB

The earliest materially action-dependent output was RGB0 for both matched
control pairs at both tested window sizes.

| Window | Matched controls | RGB0 mean abs diff | Pixels > 0.05 | Earliest material frame |
|---|---|---:|---:|---|
| 12 | no-op vs strong turn | 0.0459 | 21.2% | RGB0 |
| 12 | forward vs reverse | 0.0257 | 12.9% | RGB0 |
| 14 | no-op vs strong turn | 0.0421 | 19.3% | RGB0 |
| 14 | forward vs reverse | 0.0245 | 12.2% | RGB0 |

For the tested controls, the first newly decoded frame was already a
genuinely action-responsive frame rather than merely historical temporal
content. This does not mean every weak input must produce a visible RGB0
difference.

## Timing boundaries

These are diagnostic branch timings from the rolled matched-state probe. The
probe copied each RGB tail to the host, so the accepted lightly instrumented
product numbers remain authoritative for latency.

### 12-frame window

| Branch | Denoise | x0 accepted | RGB0 host-ready proxy | Clean KV | Next ready |
|---|---:|---:|---:|---:|---:|
| no-op | 963.3 ms | 967.1 ms | 985.2 ms | 327.4 ms | 1466.5 ms |
| strong turn | 936.8 ms | 938.1 ms | 947.2 ms | 319.4 ms | 1411.1 ms |
| forward | 932.4 ms | 933.6 ms | 942.7 ms | 319.3 ms | 1406.8 ms |
| reverse | 940.0 ms | 941.2 ms | 950.1 ms | 317.9 ms | 1416.6 ms |

The accepted 12-frame rolled product baseline is approximately `1033.2 ms`
first-new RGB, `333.9 ms` clean commit, and `1418.4 ms` next-action-ready.

### 14-frame window

| Branch | Denoise | x0 accepted | RGB0 host-ready proxy | Clean KV | Next ready |
|---|---:|---:|---:|---:|---:|
| no-op | 1049.4 ms | 1053.2 ms | 1068.5 ms | 355.5 ms | 1555.1 ms |
| strong turn | 1039.2 ms | 1040.5 ms | 1049.6 ms | 351.7 ms | 1532.9 ms |
| forward | 1024.4 ms | 1025.8 ms | 1035.0 ms | 352.6 ms | 1523.7 ms |
| reverse | 1036.9 ms | 1038.3 ms | 1047.4 ms | 354.8 ms | 1529.9 ms |

The accepted 14-frame rolled product baseline is approximately `1106.7 ms`
first-new RGB, `361.3 ms` clean commit, and `1517.1 ms` next-action-ready.

Both modes therefore have the same measured action-response frame index:
RGB0. The 14-frame mode costs roughly 70–110 ms in the matched probe, but
there is no evidence here that it delays action response to a later TAE frame.

## Input arriving during deferred clean commit

For the 12-frame no-op branch, the virtual observer recorded an input arrival
at `1071.6 ms`. The exact clean pass ran from `1020.0 ms` through `1349.0 ms`,
so the arrival was inside the clean-KV interval.

The current live runner behavior is:

* no GPU generation can start before the exact clean-KV commit completes;
* there is no explicit one-slot application queue in the current viewer;
* the next command is selected only when the main loop returns to its input
  wait after the current action finishes;
* the observer therefore reports the safe barrier timing, not a guarantee that
  a real key pressed during the interval is retained.

## Readiness-gap accounting

For the accepted 12-frame rolled baseline:

```text
next-ready - first-visible - clean-KV
1418.4 - 1033.2 - 333.9 = 51.3 ms
```

Accepted TAE live measurements account for approximately 17–18 ms of
remaining GPU output work. The first-visible boundary is recorded immediately
before the Tk display update, so the remaining roughly 33–34 ms includes
presentation/assembly, remaining TAE drain, control-loop bookkeeping, and
boundary/synchronization effects. It is not yet justified as removable work.

## Product conclusion

For the tested strong and contrasting controls, `action → first new RGB` and
`action → first materially action-responsive RGB` are currently the same
frame boundary: RGB0. Continue reporting both until a weak-control case shows
a repeatable distinction.

The 12-frame mode remains the minimum-latency interactive candidate; 14-frame
mode remains a higher-quality option. The input-during-clean behavior is safe
with respect to the clean-KV barrier, but pending-input retention is not yet an
explicit product guarantee.

## Artifacts

* [12-frame raw response metrics](../results/raw/action-response-12-v3/action_response_metrics.json)
* [14-frame raw response metrics](../results/raw/action-response-14/action_response_metrics.json)
* [12-frame branch frames](../results/raw/action-response-12-v3/)
* [14-frame branch frames](../results/raw/action-response-14/)
* [12-frame branch contact sheet](../results/raw/action-response-12-v2/action-branches-contact.png)

The probe source and launcher are `scripts/action_response_probe.py` and
`scripts/run_action_response_probe.sh`.
