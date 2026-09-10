# LingBot World v2 1.3B on Strix Halo — Interactive Latency History

Date: 2026-09-10

This document records how the Strix Halo implementation moved from a slow native 1.3B clip-generation baseline to a persistent interactive world with roughly 1.23 s filled-window first-visible latency.

The purpose is historical and experimental: preserve what was measured, which interventions mattered, which ideas failed, and how the product metric changed as the implementation became genuinely interactive.

## Product metric

The final optimization target is not offline generation throughput. It is:

```text
keypress -> first genuinely new visible RGB frame
```

Two other frontiers must remain separate:

```text
keypress -> next action can legally begin
steady-state action cadence
```

Early in the project, some runs were ordinary batch generations, so their total generation time is not directly comparable with later persistent first-visible measurements. The chronology below calls that out explicitly.

---

## Hardware and software target

Strix Halo system:

```text
APU: Ryzen AI MAX+ 395
GPU: Radeon 8060S
GPU architecture: gfx1151
ROCm/HIP: 7.15
PyTorch: 2.13.0+rocm7.15.0a20260728
```

The machine has a large unified-memory pool, so the optimization goal was latency rather than minimum memory footprint.

Model:

```text
Robbyant LingBot World v2 1.3B causal-fast
BF16 transformer
Wan-derived causal video VAE
persistent rolling self-attention KV
persistent cross-attention cache
camera control through Plucker-ray conditioning
```

---

# Executive summary

The main optimization ladder was:

| Stage | Key change | Relevant metric | Result |
|---|---|---:|---:|
| Native 1.3B batch baseline | BF16 DiT + FP32 VAE, 480x832 requested, 21 frames | warm generation | 81.216 s |
| Persistent chunk_size=1 bring-up | Long causal session and rolling KV | correctness | 81-frame session finite in batch path |
| Streaming VAE correctness | Force VAE math SDPA | persistent correctness | 81-frame streamed decode becomes finite |
| FP16 VAE | FP32 -> FP16 decoder | early first-visible | 11.581 -> 2.935 s |
| Whole-path profiling | Measure filled 18-frame context | filled-window first-visible | 4.562 s discovered |
| Clean-KV deferral | Present before exact clean-cache commit | filled-window first-visible | 4.544 -> 3.851 s |
| Lower native resolution | 464x832 -> 384x672 actual | filled-window first-visible | 3.851 -> 2.231 s |
| Three denoise steps | 4 -> 3 denoise evaluations | filled-window first-visible | 2.231 -> 1.841 s |
| TAEHV presentation decoder | Canonical Wan VAE -> taew2_1 display path | filled-window first-visible | 1.841 -> 1.229-1.232 s |

The current low-latency presentation candidate is therefore approximately:

```text
384x672 actual
chunk_size=1
18+6 world-memory configuration
3 denoise steps: 999 -> 899 -> 702
BF16 DiT
TAEHV taew2_1 streaming presentation decoder
deferred exact clean t=0 KV commit
serial execution

filled-window first-visible: 1.229-1.232 s
next-action-ready:           1.644-1.648 s
```

The canonical FP16 Wan VAE remains the quality/reference decoder.

---

# 1. Starting point — native 1.3B batch generation

The first accepted native 1.3B Strix baseline was committed at:

```text
1799dd100c835edc4c052e163547b1414efaf397
```

Configuration:

```text
ROCm/HIP 7.15
PyTorch 2.13.0+rocm7.15.0a20260728
gfx1151
PyTorch SDPA
BF16 transformer
FP32 VAE
480x832 requested
21 valid output frames
chunk_size=3
local_attn_size=18
sink_size=6
no quantization
offload disabled
FSDP disabled
Ulysses disabled
```

Measured warm run:

```text
generation:     81.216 s
VAE encode:     24.930 s
DiT:            13.328 s
VAE decode:     42.748 s
mean chunk:      6.664 s
```

This proved the native 1.3B model could run correctly on gfx1151, but it was not yet a persistent interactive implementation. The 81.216 s number is therefore a clip-generation baseline, not an action-latency measurement.

The major lesson at this point was obvious: the model ran, but the existing pipeline architecture was nowhere near a playable feedback loop.

---

# 2. Move from clip generation to persistent chunk_size=1 interaction

The next goal was to stop treating each output as an independent clip and instead maintain one causal world across player actions.

The causal-fast architecture supports a single latent query chunk with persistent history. The interactive target became:

```text
chunk_size=1
4 visible RGB frames per accepted latent
a rolling 18-frame local world-memory window
6 sink units
persistent self-attention KV
persistent cross-attention cache
persistent VAE causal feature cache
persistent camera state
```

A long chunk_size=1 batch test generated:

```text
81 visible frames
21 latent chunks
```

with finite output.

After the 18-frame world-memory window filled, DiT settled around:

```text
3.53-3.59 s / latent chunk
```

The batch VAE decode of the long result was approximately:

```text
158-159 s
```

This was not yet an interactive baseline because decoding still occurred after generation, but it established two important facts:

1. chunk_size=1 was viable for long causal progression;
2. the rolling transformer state itself could survive full-window operation.

Observed KV progression reached:

```text
global position: 31668 tokens
local state cap: 27144 tokens
```

---

# 3. Persistent streaming exposed a VAE correctness problem

The first persistent streaming runner looked healthy on a short nine-frame FP32 session, but a full 81-frame run eventually produced non-finite VAE output.

Crucially:

```text
DiT latents stayed finite
KV rollover stayed correct
VAE output diverged
```

This prevented us from falsely blaming transformer state or rolling-window logic.

A VAE-only batch-versus-incremental comparison isolated the problem to the persistent decoder attention path.

## Root cause

Fused ROCm/PyTorch SDPA in the VAE attention block diverged under long persistent incremental decoding.

Forcing the VAE attention to the math SDPA implementation restored equivalence/stability.

This became an important architectural rule:

```text
Transformer attention and VAE attention are separate problems.

The VAE requires math SDPA for the accepted persistent path.
The DiT may continue to use fused PyTorch SDPA.
```

After this change, the 81-frame FP32 streamed session remained finite through rolling-window turnover.

---

# 4. FP16 VAE — first major latency collapse

The validated persistent decoder was then compared in FP32, FP16, and BF16 with identical prompt, image, seed, resolution, 18+6 state, and math-SDPA correctness fallback.

Measured result:

| VAE path | First visible | Median action | Mean VAE/chunk | Peak allocation |
|---|---:|---:|---:|---:|
| FP32 + math SDPA | 11.581 s | 10.629 s | 7.577 s | 43.300 GB |
| FP16 + math SDPA | **2.935 s** | **3.772 s** | **0.926 s** | **37.226 GB** |
| BF16 + math SDPA | 2.947 s | 3.816 s | 0.975 s | 37.226 GB |

Both reduced-precision paths survived an 81-frame / 21-latent-chunk traversal.

FP16 was slightly faster and became the interactive canonical-VAE default.

This was the first enormous optimization win: it did not come from a new kernel or a new sampler. It came from using the correct numerical precision for the actual decoder path.

The earlier R9700 work had suggested FP16 VAE as a strong candidate, but Strix still required its own A/B. That distinction mattered because several lower-level R9700 VAE conclusions did not transfer to gfx1151.

---

# 5. Negative result — same-device VAE/DiT overlap did not help

After FP16 made the decoder much lighter, same-device overlap was retested rather than relying on the earlier FP32 failure.

Matched short run:

```text
serial:
  session        8.390 s
  first visible  3.000 s
  median action  2.932 s

overlap:
  session        8.419 s
  first visible  3.039 s
  median action  2.967 s
```

The implementation used a separate HIP stream with explicit dependencies.

Result:

```text
No useful product-level improvement.
Serial execution retained.
```

This closed an attractive but misleading idea: independent work does not automatically mean useful concurrency on one GPU.

---

# 6. Negative result — temporal Conv2d decomposition did not transfer from R9700

The R9700 had needed an exact temporal decomposition workaround for a gfx1201/MIOpen Conv3d solver cliff.

Strix gfx1151 had previously shown healthy native MIOpen behavior, so the workaround was not blindly ported. Instead, the dominant full-resolution VAE CausalConv3d was compared against an exact temporal Conv2d sum.

Matched result:

| Decoder path | Session | First visible | Median action |
|---|---:|---:|---:|
| Native Conv3d | 8.381 s | 3.003 s | 2.945 s |
| Temporal Conv2d replacement | 8.627 s | 3.021 s | 2.949 s |

Result:

```text
Native gfx1151 Conv3d wins.
Temporal split rejected as default.
```

This was another important cross-machine lesson: an exact optimization can be architecture-specific even when model semantics are identical.

---

# 7. The metric correction — filled-window latency was worse than the headline

The early FP16 first-visible result of roughly 2.94 s looked promising, but it was not the full product truth.

The model's DiT cost increases as the persistent attention history fills. A full-window profile at 18 remembered latent frames showed:

```text
DiT forward 1:       762.9 ms
DiT forward 2:       711.2 ms
DiT forward 3:       709.7 ms
DiT forward 4:       710.7 ms
clean KV forward 5:  710.9 ms
latent postprocess:    4.4 ms
VAE decode:           950.7 ms
host copy:              0.36 ms
```

Filled-window first-visible was therefore:

```text
~4.562 s
```

not ~2.94 s.

This changed the benchmarking discipline permanently. From this point forward, the repository should distinguish:

```text
early-session first-visible
filled-window steady-state first-visible
next-action-ready latency
```

A persistent world spends most of its useful lifetime at full context, so filled-window latency is the product metric that matters most.

## DiT profile

At full context, the DiT breakdown was approximately:

```text
causal self-attention:        57.7%
linear projections:           13.1%
attention-block exclusive:     7.7%
MLP projections:             ~16%
```

The fifth pass was confirmed to be a full clean-latent transformer pass required to populate exact persistent K/V. Its output is discarded, but simple reuse of the fourth noisy denoising pass is invalid.

---

# 8. Clean-KV deferral — remove required work from the display critical path

The clean t=0 KV pass is required for the next action, but that does not mean it must finish before the player sees the current action.

Old ordering:

```text
4 denoise
-> clean KV commit
-> VAE decode
-> display
```

New ordering:

```text
4 denoise
-> accepted x0
-> VAE decode
-> display
-> exact clean KV commit
-> next action permitted
```

Measured full-resolution result:

| Metric | Current order | Deferred clean-KV |
|---|---:|---:|
| Early first-visible | 3.015 s | 2.575 s |
| Filled-window first-visible | 4.544 s | **3.851 s** |
| Filled-window KV/state frontier | 3.593 s | 4.564 s |
| Next-action-ready | 4.545 s | 4.565 s |
| 81-frame session | 79.194 s | 79.206 s |

The exact clean pass was not removed. It was moved off the first-visible critical path.

Result:

```text
~0.693 s faster first-visible
no measurable cadence penalty
persistent correctness preserved
```

This was a pure scheduling optimization and one of the most important conceptual discoveries of the project:

> Required computation is not necessarily first-visible-critical computation.

---

# 9. Lower native resolution — the largest exact whole-model win

The next experiment reduced actual generation geometry from approximately:

```text
464x832
```

to:

```text
384x672
```

This changed tokens per latent frame from approximately:

```text
1508 -> 1008
```

and full local KV capacity from:

```text
27144 -> 18144 tokens
```

Because spatial resolution affects both the current query and historical K/V, it reduces several costs simultaneously:

```text
DiT projections / MLP
self-attention QK and AV work
KV footprint / movement
VAE convolution
upsampling
pointwise work
```

Measured filled-window deferred result:

```text
DiT denoise:        1.598 s
clean KV commit:    0.392 s
VAE decode:         0.632 s
first-visible:      2.231 s
next-action-ready:  2.623 s
```

Compared with the 464x832 deferred path:

```text
first-visible:      3.851 -> 2.231 s  (~42% improvement)
next-action-ready:  4.565 -> 2.623 s  (~43% improvement)
```

Output remained finite and visually coherent through sampled chunk boundaries and late window rollover.

A one-time MIOpen warm-up of roughly 6.9 s occurred for the new geometry, so selected geometries must be warmed before the world is declared interactively ready.

This experiment established 384x672 as the leading low-latency geometry.

---

# 10. Three-step sampling — remove one denoising evaluation

At 384x672, the next approximation experiment asked whether the four distilled denoising evaluations could become three while retaining the exact clean t=0 KV pass.

Actual current four-step schedule was found to be:

```text
999 -> 957 -> 899 -> 702
```

The tested three-step schedule was:

```text
999 -> 899 -> 702
```

The scheduler algebra supports the larger interval, but this remains an empirical approximation rather than checkpoint-certified training behavior.

Measured filled-window A/B:

| Metric | 4-step | 3-step | Improvement |
|---|---:|---:|---:|
| First-visible | 2230.8 ms | **1840.7 ms** | 390.1 ms |
| Next-action-ready | 2623.1 ms | **2232.3 ms** | 390.8 ms |
| Denoise total | 1594.0 ms | **1202.3 ms** | 391.8 ms |
| Clean KV | 391.9 ms | 391.2 ms | unchanged |
| VAE decode | 632.3 ms | 634.0 ms | unchanged |

The saving matched one denoising evaluation almost exactly.

Long rollout:

```text
81 visible frames / 21 latent chunks
finite throughout
KV local window remained capped correctly
no collapse
no frozen output
no obvious chunk seams
camera/world motion coherent through rollover
median action latency: 1.924 s
last-five mean: 2.210 s
session time: 41.028 s
```

Observed quality tradeoff:

```text
some additional texture/water variation
somewhat softer detail versus four-step
```

Decision:

```text
3-step = low-latency interactive candidate
4-step = quality/reference mode
```

---

# 11. TAEHV taew2_1 — presentation decode becomes almost free

With first-visible at roughly 1.84 s, the canonical Wan VAE still consumed approximately 0.634 s. Rather than building a custom prefix decoder first, the project evaluated an existing Wan 2.1 tiny streaming decoder:

```text
repository: madebyollin/taehv
commit: 011dfc2112197741c540e0bdd5b7b67bcc930771
weight: taew2_1.pth
SHA-256: d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e
```

Verified architecture:

```text
16 latent channels
temporal scale: 4
startup trim: 3 frames
StreamingTAEHV world-model-style decode API
```

## Latent contract

LingBot accepted latent:

```text
normalized model-space x0
saved sequence shape: [16, 21, 48, 84]
```

Canonical decoder input:

```text
NCTHW after Wan VAE scale/shift
```

TAE input:

```text
direct accepted x0
transposed to [1, 21, 16, 48, 84]
no additional normalization
```

This contract was established before judging output quality so that incorrect scaling would not be mistaken for model incompatibility.

## Offline result

```text
canonical warmed decode: ~639 ms / latent
TAE first RGB:             7.7-8.1 ms
TAE all RGB:              14.6-16.7 ms
output alignment:          1 + 20*4 = 81 frames
```

Both streams were finite and temporally stable.

Mean absolute image difference in [0,1]:

```text
0.0319
```

Qualitative result:

```text
same broad world and motion
TAE presentation softer
fine texture / edges altered
```

TAE was therefore treated as an approximate presentation decoder, not as a canonical reconstruction replacement.

## Live result

A full 81-frame persistent session succeeded.

Measured filled-window result:

```text
TAE first-visible:        1.229-1.232 s
canonical first-visible: ~1.841 s

TAE next-action-ready:    1.644-1.648 s
canonical next-ready:    ~2.232 s

TAE first-RGB GPU time:  ~20 ms
TAE all-output GPU time: ~37 ms
clean KV commit:          385-387 ms
```

Final state:

```text
global KV: 21168
local KV: 18144 / 18144
all frames finite through rollover
```

Memory:

```text
peak PyTorch allocation: 30.05 GB
peak reserved:           42.36 GB
peak RSS:                25.87 GB
```

Decision:

```text
Retain TAEHV as an explicit low-latency presentation decoder.
Keep canonical FP16 Wan VAE as default/reference quality decoder.
```

This eliminated roughly 0.61 s from first-visible latency without changing the generated latent world itself.

---

# 12. How the latency budget was whittled down

The most useful way to view the project is not as one optimization, but as a sequence of bottleneck changes.

## Initial native pipeline

```text
~81 s warm batch generation for 21 frames
```

The initial problem was pipeline structure plus an extremely expensive FP32 VAE.

## Persistent FP16 stage

```text
early-session first-visible ~2.94 s
```

This looked interactive, but full-window profiling revealed that the mature world-memory state was much slower.

## Honest filled-window baseline

```text
~4.56 s first-visible
```

At this point the critical path was approximately:

```text
five transformer forwards ~3.61 s
FP16 canonical VAE        ~0.95 s
```

## Defer exact clean-KV work

```text
4.54 -> 3.85 s
```

No compute removed; ~0.69 s moved after presentation.

## Reduce native geometry

```text
3.85 -> 2.23 s
```

The 384x672 geometry reduced both transformer and VAE work and preserved the 18-frame memory depth.

## Remove one denoising evaluation

```text
2.23 -> 1.84 s
```

One ~0.39 s transformer pass removed, with a documented modest quality tradeoff.

## Replace presentation decode with TAEHV

```text
1.84 -> 1.23 s
```

Canonical ~0.63 s decode was replaced for interactive presentation by a roughly 20 ms first-RGB streaming tiny decoder.

The current first-visible path is therefore dominated almost entirely by the three DiT denoising evaluations:

```text
3 denoise passes      ~1.20 s
TAE first RGB         ~0.02 s
small remaining work  ~0.01 s
-----------------------------
first visible         ~1.23 s
```

The exact clean t=0 KV commit then completes in roughly:

```text
~0.385-0.391 s
```

placing next-action readiness around:

```text
~1.64-1.65 s
```

---

# 13. What did NOT work

Several negative results were as valuable as the successful optimizations.

## Generic same-GPU overlap

Even after the VAE became FP16, separate-stream overlap did not improve the actual interaction metric. Independent tasks still contended for the same GPU resources.

Status: rejected.

## Temporal Conv3d -> Conv2d decomposition on gfx1151

This workaround was valuable on the R9700/gfx1201 solver-cliff path but slightly slower than healthy native Strix Conv3d.

Status: rejected on Strix.

## Simple fourth-forward KV reuse

The clean t=0 pass encodes accepted clean x0, whereas denoising K/V represents a noisy latent state. Exact persistent cache semantics require the clean pass.

Status: rejected.

## Immediate CK attention backend swap

The actual Strix transformer path was found to use fused PyTorch SDPA:

```text
aten::_scaled_dot_product_flash_attention
HIP kernel: attn_fwd.kd
```

Representative full-window attention used:

```text
Q: [1,1508,12,128]
K/V: [1,27144,12,128]
BF16
mask: none
is_causal: false
```

Across the measured self-attention calls, wrapper/dispatch overhead was only about 0.4%. CK SDPA was unavailable on the tested gfx1151 build, so no backend swap was justified.

Status: immediate backend hunt closed; attention remains a future optimization target if a supported better path appears.

---

# 14. Cross-machine lessons from the R9700 work

The R9700 effort helped identify hypotheses, but the Strix work repeatedly demonstrated why each architecture needed its own validation.

Transferred successfully:

```text
FP16 VAE as a high-value hypothesis
chunk_size=1 as the latency-oriented interaction mode
critical-path focus rather than throughput focus
clean-KV work can be scheduled after presentation
```

Did not transfer directly:

```text
R9700 temporal VAE Conv3d workaround
same exact backend behavior
same performance balance between DiT and VAE
```

General lesson:

> Transfer mechanisms and hypotheses across AMD architectures, not conclusions.

---

# 15. Current candidate modes

## Low-latency interactive presentation

```text
384x672 actual
chunk_size=1
18+6 context
3 denoise: 999 -> 899 -> 702
BF16 DiT
TAEHV taew2_1 presentation
exact deferred clean t=0 KV commit
serial

filled-window first-visible: 1.229-1.232 s
next-action-ready:           1.644-1.648 s
```

## Canonical quality/reference presentation

```text
384x672 actual
chunk_size=1
18+6 context
3 denoise: 999 -> 899 -> 702
BF16 DiT
FP16 canonical Wan VAE
math SDPA in VAE
native gfx1151 Conv3d
exact deferred clean t=0 KV commit
serial

filled-window first-visible: ~1.841 s
next-action-ready:           ~2.232 s
```

## Sampling quality reference

Four denoise evaluations remain available as a higher-quality reference because the three-step candidate shows somewhat softer detail and greater texture/water variation.

---

# 16. Current bottleneck and next question

TAEHV changed the optimization problem again.

The first-visible path is no longer meaningfully VAE-bound. It is almost entirely DiT-bound:

```text
~1.20 s three-step denoise
~0.02 s first TAE RGB
```

The next single experiment identified in the current plan is therefore:

> profile the warmed, filled-window three-step DiT/operator path at 384x672.

The purpose is to find where the next roughly 200-300 ms can be removed from the three denoising evaluations without prematurely spending another large quality tradeoff.

The clean-KV pass should be profiled separately because it affects next-action readiness, not current first-visible latency.

---

# 17. Optimization principles learned

The project produced several reusable lessons:

1. **Measure the product frontier, not the convenient benchmark.** Early-session and average metrics hid the full-window attention cost.
2. **Correctness comes before speed.** The persistent VAE fused-SDPA divergence would have invalidated every later benchmark if it had not been isolated first.
3. **Required work can move off the perceived-latency path.** Clean-KV deferral saved ~0.69 s without reducing total compute.
4. **Resolution is a whole-model optimization.** It reduced attention, projections, VAE work, and cache footprint simultaneously.
5. **Approximation should be introduced deliberately.** Three-step sampling produced an almost exact one-forward latency reduction with a visible but bounded quality tradeoff.
6. **Presentation does not have to use the canonical decoder.** Once decoder output was proven not to feed generation state, TAEHV could provide a much cheaper player view while leaving the latent world intact.
7. **Architecture-specific workarounds are not universal optimizations.** The R9700 Conv3d split solved a real gfx1201 problem but regressed healthy gfx1151 execution.
8. **Negative experiments should stay documented.** They prevent future agents from repeatedly rediscovering attractive dead ends.

---

# 18. Milestone timeline

Key repository milestones discussed during this optimization sequence:

```text
1799dd1  native 1.3B Strix baseline

a30adfc  native gfx1151 Conv3d retained over temporal Conv2d split

99833d8  full-window DiT/operator profiling milestone

717d098  deferred clean-KV + attention characterization + 384x672 result

4109834  three-step 999->899->702 low-latency candidate

fd8a131  TAEHV validation and revised next-phase plan
```

For exact chronological experimental details, raw paths, and individual negative-result controls, see:

```text
docs/FINDINGS.md
docs/lingbot-world-v2-1.3b-strix-halo-bringup-20260909.md
docs/taehv-20260910.md
docs/next-phase-plan-20260910.md
```

---

# Bottom line

The Strix path did not become fast because of one heroic custom kernel.

It became fast by repeatedly identifying the real current bottleneck and changing the shape of the critical path:

```text
native batch pipeline
-> persistent chunk_size=1 world
-> correct long-running VAE attention
-> FP16 canonical VAE
-> honest full-window profiling
-> defer exact housekeeping after presentation
-> lower native geometry
-> remove one denoise evaluation
-> replace canonical presentation decode with streaming TAEHV
```

That progression moved the implementation from an ~81 s warm 21-frame batch-generation baseline to a persistent world whose mature filled-window state can show the player a genuinely new model-generated frame in about:

```text
1.23 seconds
```

with the exact persistent state ready for the next action at about:

```text
1.65 seconds
```

The remaining latency problem is now primarily a roughly 1.2 s three-forward DiT problem.