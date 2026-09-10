# Revised low-latency plan

Date: 2026-09-10  
Repository: `boxwrench/lingbot-world-V2-halo`  
Current implementation commit: `90b13640996bc8fbdffacdc380de50dc2ec6cd54`

## Decision from the TAEHV phase

The TAEHV experiment is complete. The pinned `taew2_1` decoder passed the
latent-contract, contiguous-stream, quality, and live persistent-session
gates. It is retained as an explicit opt-in presentation mode. The canonical
FP16 Wan VAE remains the default/reference decoder because TAEHV is softer in
fine detail and changes texture/edges.

The active product candidate is now:

```text
384x672
chunk_size=1
3 denoise: 999 -> 899 -> 702
exact deferred t=0 clean-KV commit
BF16 DiT with persistent rolling KV
TAEHV FP16 streaming presentation
serial execution
```

TAEHV source and weight provenance are pinned in
[`taehv-manifest.json`](taehv-manifest.json). The complete result is in
[`taehv-20260910.md`](taehv-20260910.md).

## Current measured product boundaries

The primary comparison is filled-window steady state, not the faster early
session. Values below are from the accepted 3-step, `18+6` configuration.

| Boundary | Canonical FP16 VAE | TAEHV presentation | Difference |
|---|---:|---:|---:|
| Filled-window first RGB | about 1.841 s | 1.229–1.232 s | about -0.61 s |
| Filled-window next action permitted | about 2.232 s | 1.644–1.648 s | about -0.59 s |
| DiT denoise work | about 1.202 s | about 1.21 s | unchanged |
| Presentation decoder all-output work | about 0.634 s | about 0.037 s | about -0.60 s |

The long TAE session produced 81/81 finite frames, rolled global KV to 21168
tokens while holding local KV at its 18144-token cap, and retained sink state.
The remaining gap to approximately one-second first-visible is therefore
roughly 0.23 s in the three denoising forwards, subject to measurement and
quality constraints.

## Protected and closed work

Keep `scripts/run_window_sweep.sh` untouched and unstaged. Do not reopen:

```text
4-step/3-step schedule search
FP32 or BF16 VAE defaults
fused VAE SDPA
temporal Conv3d decomposition
generic same-device overlap
CK backend search on the current gfx1151 stack
14B/GGUF work
chunk-size sweeps
weight/KV quantization
offload
custom VAE kernels
```

Do not silently change the canonical default. TAEHV remains selected with
`--display-decoder taehv` for a fresh session; decoder-cache migration is not
implemented.

## Phase 1 — Freeze and re-baseline the TAE candidate

Use one warmed persistent session at full `18+6` occupancy. Do not spend time
on more TAE decoder qualification unless a regression appears. Extract and
report separately:

```text
early first-visible
filled-window first-visible
filled-window next-action-permitted
median and P95 action cadence
3 denoise total and each forward
TAE first-RGB and all-RGB timings
exact clean-KV commit
peak allocation and RSS
```

The session must still exercise window rollover, sink retention, direction
changes, and finite output. This is a measurement freeze, not a new decoder
algorithm.

## Phase 2 — Profile the actual display-critical DiT path

Profile only the accepted TAE candidate after the window is full. Build a
waterfall with synchronized, non-overlapping boundaries:

```text
action/conditioning preparation
denoise forward 1
denoise forward 2
denoise forward 3
latent postprocessing
TAE first RGB and host copy
presentation boundary
clean-KV commit
remaining TAE output
next-action permission
```

Within the three denoise forwards, rank enough operator classes to explain at
least 90% of DiT time:

```text
fused self-attention kernel
Q/K/V and output projections
MLP
normalization/RoPE/conditioning
KV reads, writes, and layout movement
synchronization and allocation
```

Record real Q/K/V shapes, strides, attended lengths, and selected HIP kernel.
Do not infer that a large theoretical attention cost is an optimization
opportunity. The current fused SDPA kernel is the correctness baseline and
must remain available.

## Phase 3 — One bounded exact DiT intervention

Choose at most one intervention from evidence in Phase 2. The preferred order
is:

1. Remove a demonstrated materialization, copy, or synchronization that does
   not change tensor values or causal state.
2. Improve an existing KV layout/packing path if the current layout is shown
   to dominate outside the fused attention kernel.
3. Test one installed AMD-compatible attention path only if the real
   rectangular cached-attention shapes are supported on gfx1151.

Do not hunt libraries or write kernels speculatively. Keep a candidate only if
it saves at least approximately 0.2–0.3 s of whole filled-window action
latency, or a smaller exact saving with unusually low complexity.

Validate against the canonical accepted generation state, not synthetic
attention alone:

```text
single action
next action
18-frame rollover
sink retention
extended persistent traversal
```

The exact clean-KV forward remains separate and mandatory. No candidate may
permit another DiT action before that state is committed.

## Phase 4 — Re-baseline and decide whether approximation is justified

If Phase 3 produces a durable exact saving, remeasure the full TAE candidate.
If it does not, stop exact DiT work rather than accumulating weak branches.

Only then consider one new approximation experiment: a defensible two-step
denoise schedule followed by the exact clean-KV pass. It must be treated as a
quality experiment, not a default optimization. Use a matched-state local
comparison and independent long persistent rollouts with translation,
rotation, reversal, revisits, rollover, and sink retention. Reject it for
repeatable control/geometry degradation even if it reaches the one-second
latency target.

## Phase 5 — Product modes and reporting

Maintain two explicit modes:

```text
quality/reference: 3-step + canonical FP16 Wan VAE
interactive:       3-step + opt-in TAEHV presentation
```

Report the following independently in every future result:

```text
keypress -> first genuinely new RGB frame
keypress -> next action legally permitted
steady-state action cadence and P50/P95
```

Also report whether the result is early-session or filled-window. Never use
TAE decoder throughput divided by four as an application latency claim.

## Next single experiment

The next experiment is a warmed, full-window DiT/operator profile of the TAE
interactive candidate. It should answer whether the remaining approximately
1.21 s denoise path contains an exact layout/copy/synchronization opportunity.
No new sampler, decoder, kernel, or hardware branch starts before that profile
is recorded.
