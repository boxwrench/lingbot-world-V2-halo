# Live context profile — 2026-09-10

## Question

Why does the same 384×672, three-denoise, TAEHV live path move from roughly
0.7 seconds early in a session to roughly 1.2–1.25 seconds after the 18-frame
world-memory window fills?

This is a profile-only experiment. No sampler, resolution, renderer, cache
size, attention backend, or queue behavior was changed.

## Method

The experiment started from repository commit `948c672` and used the existing
`scripts/run_live.sh` path on the Strix Halo accepted configuration:

```text
384×672 actual geometry
chunk_size=1
local_attn_size=18
sink_size=6
3-drop-957: 999 → 899 → 702
BF16 DiT, FP16 TAEHV, serial execution
deferred exact clean t=0 KV commit
```

The same deterministic 40-action sequence was used for the unprofiled control
and the profiled prefix:

```text
w,w,j,w,l,l,s,s,j,w,d,d,w,l,a,a,w,j,s,l,w,w,d,j,a,s,l,w,d,d,j,w,a,a,l,s,w,j,l,w
```

The unprofiled control ran bootstrap plus 40 actions and is the source for
product boundary timings. A second run attached the existing module-tree and
SDPA probes only at chunks `1, 9, 17, 18`, covering Early, Mid, Full, and the
first Rolled action. Detailed hook timings are not used as product latency.

Reproduction:

```bash
ACTIONS='w,w,j,w,l,l,s,s,j,w,d,d,w,l,a,a,w,j,s,l,w,w,d,j,a,s,l,w,d,d,j,w,a,a,l,s,w,j,l,w'

LINGBOT_LIVE_OUTPUT_DIR=results/raw/live-context-profile-unprofiled-v2 \
LINGBOT_LIVE_MAX_ACTIONS=40 \
LINGBOT_LIVE_TIMEOUT_SECONDS=900 \
bash scripts/run_live.sh --scripted-actions "$ACTIONS"

LINGBOT_LIVE_OUTPUT_DIR=results/raw/live-context-profile-detailed-v2 \
LINGBOT_LIVE_MAX_ACTIONS=18 \
LINGBOT_LIVE_TIMEOUT_SECONDS=900 \
bash scripts/run_live.sh --scripted-actions "$ACTIONS" \
  --profile-contexts 1,9,17,18
```

Raw results are intentionally ignored generated artifacts:

* [`unprofiled live metrics`](../results/raw/live-context-profile-unprofiled-v2/live_metrics.json)
* [`detailed live metrics`](../results/raw/live-context-profile-detailed-v2/live_metrics.json)
* [`unprofiled run log`](../results/raw/live-context-profile-unprofiled-v2/run.log)
* [`detailed run log`](../results/raw/live-context-profile-detailed-v2/run.log)

## Context and product boundaries

The unprofiled control completed 41 rows including bootstrap, all finite. The
model reports a frame-sequence length of `1008` tokens and a physical local
cache capacity of `18144` tokens (`18 × 1008`). Product rows are selected as
the Early chunk, the median of Mid chunks, the Full chunk, and the median of
the Rolled chunks from 18 through 40.

| Context | Representative chunk(s) | Attended K | 3× denoise | Forward 1 / 2 / 3 | Clean KV | TAE first RGB | First visible | Next action permitted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Early | 1 | 2,016 | 713.6 ms | 235.5 / 239.8 / 235.0 ms | 229.5 ms | 21.1 ms | 735.1 ms | 1,016.2 ms |
| Mid | median 2–16 | 10,080 | 938.2 ms | 317.5 / 309.7 / 307.6 ms | 310.7 ms | 20.1 ms | 958.8 ms | 1,315.4 ms |
| Full | 17 | 18,144 | 1,192.3 ms | 400.4 / 392.1 / 396.4 ms | 394.2 ms | 20.2 ms | 1,212.9 ms | 1,655.0 ms |
| Rolled | median 18–40 | 18,144 | 1,224.9 ms | 434.8 / 394.6 / 393.8 ms | 396.1 ms | 20.2 ms | 1,245.7 ms | 1,688.0 ms |

The single first rolled forward is more variable because it performs the
cache eviction/shift; the steady rolled median is the useful product number.
TAE presentation remains nearly context-independent. The exact clean pass is
deferred, so it affects next-action readiness but not first-visible.

The measured Full−Early deltas are:

| Boundary | Increase |
|---|---:|
| 3× denoise | **478.7 ms** |
| clean KV | **164.7 ms** |
| first visible | **477.8 ms** |
| next action permitted | **638.8 ms** |

## Attention contract

The detailed SDPA probe observed 90 self-attention calls during the three
denoise forwards: 30 layers × 3 forwards. At every context:

```text
Q: [1, 1008, 12, 128], BF16, contiguous
K/V before SDPA: [1, K, 12, 128], BF16, contiguous
Q/K/V passed to SDPA: [1, 12, 1008/K, 128] views
mask: absent
is_causal: false
```

Representative strides were:

```text
Q before:     [1548288, 1536, 128, 1]
K before:     [27869184, 1536, 128, 1]
Q in SDPA:    [1548288, 128, 1536, 1]
K in SDPA:    [27869184, 128, 1536, 1]
```

The cache is already causally ordered and rolling; the self-attention call is
rectangular cached attention rather than ordinary square causal attention.
The accepted backend remains PyTorch fused SDPA, observed by the repository's
prior kernel trace as `aten::_scaled_dot_product_flash_attention` using HIP
`attn_fwd.kd`. The wrapper probe itself records the SDPA dispatch and timing;
it does not expose the HIP kernel symbol directly.

The actual K lengths are:

| Context | K tokens | Frame equivalent |
|---|---:|---:|
| Early | 2,016 | 2 |
| Mid | 10,080 | 10 |
| Full | 18,144 | 18 |
| Rolled | 18,144 | 18 |

## Operator attribution

The table below combines synchronized unprofiled action boundaries with the
intrusive module/SDPA probe. Module values are exclusive CUDA-event totals for
three denoise forwards; they explain the profile rather than replace the
unprofiled action timings.

| Context | Self SDPA | Self QKV + output projections | MLP projections | Other profiled DiT work |
|---|---:|---:|---:|---:|
| Early | 69.8 ms | 72.2 ms | 170.4 ms | 448.5 ms |
| Mid | 300.6 ms | 67.4 ms | 155.9 ms | 441.4 ms |
| Full | 552.6 ms | 68.5 ms | 154.0 ms | 441.3 ms |
| Rolled | 544.0 ms | 65.8 ms | 153.8 ms | 483.0 ms |

“Other” includes modulation, camera-conditioning linears, norms, residual and
attention-block work, cross-attention, patch/head work, and self-cache work;
it is intentionally not presented as one optimization target.

The detailed class totals at Full were:

| Profile class | Exclusive time | Share of profiled module time |
|---|---:|---:|
| self-attention module body (SDPA + non-child cache/RoPE work) | 617.3 ms | 50.8% |
| MLP projections | 154.0 ms | 12.7% |
| attention-block non-child work | 110.6 ms | 9.1% |
| other linears | 86.3 ms | 7.1% |
| camera-conditioning linears | 65.4 ms | 5.4% |
| self QKV projections | 52.9 ms | 4.4% |
| normalization | 47.4 ms | 3.9% |
| remaining recorded classes | 82.5 ms | 6.8% |

This accounts for the full profiled module tree. The individual largest fixed
linear was `time_projection.1` at approximately `80.0 ms`; it does not grow
with cached K and is therefore not the explanation for Full−Early.

## Self-attention growth and clean pass

Self SDPA totals across the 90 denoise calls were:

| Context | Self SDPA total | Per denoise forward | Cross SDPA total |
|---|---:|---:|---:|
| Early | 69.8 ms | 22.7 / 23.2 / 23.9 ms | 18.4 ms |
| Mid | 300.6 ms | 101.4 / 100.8 / 98.4 ms | 18.1 ms |
| Full | 552.6 ms | 187.4 / 184.0 / 181.3 ms | 18.4 ms |
| Rolled | 544.0 ms | 180.0 / 183.1 / 180.9 ms | 18.4 ms |

Full−Early self SDPA growth is `+482.8 ms`, versus `+478.7 ms` observed total
denoise growth. That is approximately `100.9%` of the observed increase; the
small negative residual in other work is measurement variation, not a second
growing bottleneck. Cross-attention and QKV/MLP projections are effectively
flat.

The clean pass has the same cached rectangular self-attention shape, but only
one forward. Its detailed self-SDPA event totals were approximately `23.1 ms`
(Early), `99.8 ms` (Mid), `172.7 ms` (Full), and `184.5 ms` (Rolled). The Full
clean module hook also caught one anomalous one-time layer event, so the
unprofiled `394.2 ms` Full wall time is retained as the product value. The
clean wall-time increase is therefore consistent with the same K-dependent
self-attention mechanism, not with TAE or host presentation.

## Cache and sink accounting

`18+6` is not a 24-frame attended window in this implementation. The physical
K/V tensor is `18 × 1008 = 18,144` tokens. `sink_size=6` is a retention policy
inside that capacity. Before the first eviction, the contiguous cache has not
physically split sink tokens from recent history; after rollover, the layout is
measurably:

```text
6 sink frames       = 6,048 tokens
11 recent frames    = 11,088 tokens
current frame       = 1,008 tokens
total               = 18,144 tokens
```

Global position advanced beyond capacity after chunk 18 while local K stayed
at `18,144`, confirming bounded attention. The first rolled denoise forward
was about 30–40 ms slower than the Full first forward in the unprofiled run;
the profiled self-attention module body also rose from `617.3` to `651.1 ms`
while self-SDPA itself stayed flat. This is evidence for cache eviction/shift
work at rollover, but it is not the source of the large Early→Full increase.

Allocation snapshots were stable across the run (approximately `19.1 GB`
allocated and `19.7 GB` reserved after the warm-up region; peak values include
backend warm-up). There is no observed allocator-growth explanation for the
latency curve.

## Conclusion and next experiment

The Early→Full latency increase is a healthy, expected cost of attending to
more historical tokens in fused rectangular self-attention. It is not a
backend fallback, mask materialization, cross-attention growth, or a TAE
presentation problem. Rollover adds a smaller cache-shift cost, while keeping
K length bounded.

The next justified single experiment is a shorter-history A/B at
`local_attn_size=12`, `sink_size=6`, with all other settings and the same
persistent traversal held constant. It should measure whether the projected
self-attention saving is worth the world-memory reduction. No optimization is
included in this profile commit.
