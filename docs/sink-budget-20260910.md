# Fixed-budget sink allocation: 6 frames versus 2 frames

## Scope

This was one bounded cache-policy A/B on the accepted 384x672, 3-step,
TAEHV, TunableOp, pure-helper-compiled Strix path. The only model setting
changed was `sink_size`; `local_attn_size` stayed at 12. The authoritative
timing runs used fresh processes, shape-matched prewarm, no video capture, the
same seed/prompt/image, and the same 40-action scripted traversal.

The invalid first launch using `max_area_pixels=258048` was stopped and
excluded because it resolved to 966 tokens/frame. The reported runs use the
known accepted `max_area_pixels=264192`, resolving to 384x672 and 1008
tokens/frame.

## Sink semantics

The pinned v2 causal implementation defines the sink as the first
`sink_size` frame blocks in the self-attention KV cache. On rollover those
tokens are copied unchanged to the front of the contiguous local cache; the
non-sink history is shifted and the new current frame is appended. Sink is a
self-KV retention policy, not a second cross-attention cache and not a
separate conditioning tensor bank.

In this path, the sink therefore represents the earliest self-KV world-history
prefix written by the bootstrap and early generated chunks. It is not six
special image-only frames. Before the first eviction it is only a logical
policy; after eviction it is physically retained at the front of each layer's
cache.

## Cache contract

At 384x672 the measured frame token count was 1008. Both lanes allocated and
attended exactly 12,096 local tokens.

| lane | sink | recent before current | current | total K |
| --- | ---: | ---: | ---: | ---: |
| control, `sink_size=6` | 6,048 | 5,040 | 1,008 | 12,096 |
| candidate, `sink_size=2` | 2,016 | 9,072 | 1,008 | 12,096 |

The final action in both 40-action sessions reached global KV position
41,328, with local K capped at 12,096. The runtime records show the expected
retained layout after rollover in every lane: the control retained 6,048
sink tokens and 5,040 recent tokens; the candidate retained 2,016 sink tokens
and 9,072 recent tokens.

## Timing

Timing is from the no-capture primary runs. `P95` is the empirical upper-tail
order statistic over the 28 rolled actions; the single full and first-rollover
rows are shown separately because they are one action each.

| state / metric | sink 6 | sink 2 | delta (2 - 6) |
| --- | ---: | ---: | ---: |
| early action first-visible | 631.1 ms | 631.8 ms | +0.7 ms |
| full, first fill action first-visible | 931.1 ms | 926.1 ms | -5.0 ms |
| first rollover first-visible | 933.7 ms | 955.5 ms | +21.8 ms |
| rolled 3x denoise P50 | 915.0 ms | 933.8 ms | +18.8 ms |
| rolled 3x denoise P95 | 920.4 ms | 938.8 ms | +18.4 ms |
| rolled first-visible P50 | 939.0 ms | 957.7 ms | +18.7 ms |
| rolled first-visible P95 | 944.7 ms | 962.8 ms | +18.1 ms |
| rolled clean KV P50 | 302.4 ms | 300.5 ms | -1.9 ms |
| rolled clean KV P95 | 304.8 ms | 301.7 ms | -3.1 ms |
| rolled next-ready P50 | 1,289.7 ms | 1,305.8 ms | +16.1 ms |
| rolled next-ready P95 | 1,295.8 ms | 1,313.3 ms | +17.5 ms |

The equal-K hypothesis held: there is no intended attention-size reduction.
The candidate was slightly slower in this matched run, while clean-KV timing
was effectively the same. This is not a latency win.

## Persistent-world quality

The same traversal exercised forward travel, turns, backward travel, abrupt
direction changes, and repeated rollovers. The early frames were identical or
near-identical because neither policy had evicted history. Divergence began at
the first rollover and increased later:

| comparison region | matched RGB MAD, sink 6 versus sink 2 |
| --- | ---: |
| early | 0.0000 |
| first-rollover region | 0.0119 |
| mid rolled | 0.0797 |
| late rolled | 0.1100 |

The contact sheet shows the concrete failure mode. The six-sink lane retained
the established shoreline/mountain/tree relationships later in the traversal.
The two-sink lane increasingly substituted or moved the background and, in
late frames, introduced a much closer dark stone/wall structure that was not
the same persistent layout. The extra recent slots did not produce a useful
recent-history continuity gain; the loss of the old anchor history was more
visible.

The framewise temporal diagnostic also worsened for the candidate:

| lane | adjacent-frame RGB MAD mean | P95 |
| --- | ---: | ---: |
| sink 6 | 0.0252 | 0.0370 |
| sink 2 | 0.0378 | 0.1004 |

These metrics are supporting diagnostics, not an automatic quality score, but
they agree with the visual result: the candidate has more late temporal
instability/scene substitution. No separate exact revisit checkpoint was
added; the prescribed traversal included reversals and return-like motion,
which was sufficient to expose the late anchoring loss.

## Correctness

Both sessions completed all 40 actions with finite latent and RGB outputs,
monotonic global KV position, repeated rollover, the expected local capacity,
and the exact deferred clean `t=0` transaction. The one-slot pending-input
policy and all inference/renderer/compiler settings were unchanged. The
per-forward finite checks remained true throughout. No direct per-layer KV
finite flag was added to the timing harness; the observed cache state and
finite attention/output path remained healthy in both sessions.

Raw metrics and logs are under the ignored
`results/raw/sink-budget-20260910/` tree. The aligned comparison video and
contact sheet are:

- `results/raw/sink-budget-20260910/comparison/sink6-vs-sink2.mp4`
- `results/raw/sink-budget-20260910/comparison/sink6-vs-sink2-contact.png`

Compact machine-readable evidence is in
`docs/artifacts/sink-budget-20260910/summary.json`.

## Decision

**REJECT — SINK HISTORY IS MATERIAL.** Keep the validated six-sink layout as
the RC1 default and retain 12-frame/sink-6 as the minimum-latency interactive
mode. Do not make sink 2 an RC1 candidate: it held the same K budget, did not
improve readiness, was slightly slower in the matched run, and produced clear
late persistent-world anchoring/continuity loss.

The next release decision is to freeze inference and move to serving or
product packaging; no further sink sweep is authorized by this result.
