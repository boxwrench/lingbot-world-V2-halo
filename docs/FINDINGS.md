# Chronological findings

This is the experiment ledger for the Strix Halo control run. Results are
recorded as observed; superseded interpretations remain in place and are
marked rather than deleted. Raw JSON and MIOpen logs are kept locally under
`results/raw/` and are intentionally ignored by Git.

## F1 — gfx1151 does not reproduce the gfx1201 Conv3d cliff (2026-09-09)

The matched synthetic FP32 test was run on the Radeon 8060S with PyTorch
`2.13.0+rocm7.15.0a20260728`, HIP `7.15.0`, native `gfx1151`, two warmup
calls, and five synchronized timed calls. All outputs were finite and had the
expected shapes.

| Case | Input | Weight | Padding | Output | First | Warm median | Effective |
|---|---|---|---|---|---:|---:|---:|
| A | `[1,96,4,480,832]` | `[96,96,3,3,3]` | 0 | `[1,96,2,478,830]` | 3.395 s | **0.2309 s** | **1.710 TFLOP/s** |
| B | `[1,96,6,482,834]` | `[96,96,3,3,3]` | 0 | `[1,96,4,480,832]` | 5.853 s | **0.5043 s** | **1.576 TFLOP/s** |
| C | `[1,96,6,482,834]` | `[3,96,3,3,3]` | 0 | `[1,3,4,480,832]` | 4.103 s | **0.2980 s** | **0.083 TFLOP/s** |
| D | `[1,96,4,480,832]` | `[96,96,3,3,3]` | 1 | `[1,96,4,480,832]` | 5.632 s | **0.4783 s** | **1.662 TFLOP/s** |

Warm ranges were A `0.2307–0.2313 s`, B `0.5024–0.5069 s`, C
`0.2963–0.2994 s`, and D `0.4765–0.4809 s`. The first call includes
kernel/setup work and is not representative of steady-state throughput.

The production-padded case B is only 2.18× slower than A while doing about
2.01× the output work. It is therefore not the approximately 17× efficiency
collapse observed on the R9700/gfx1201 control. The RGB-output case is also
about 10× faster on gfx1151 than the reference gfx1201 result.

The raw measurement is [`strix-gfx1151-default.json`](../results/raw/conv3d/strix-gfx1151-default.json)
(local only). The reproducer is [`conv3d_micro.py`](../scripts/conv3d_micro.py).

## F2 — MIOpen solver and cache evidence (2026-09-09)

MIOpen logging was enabled for one synchronized call per case. All four cases
selected:

```text
solver: GemmFwdRest
kernel: Im3d2Col
GEMM backend: rocBLAS
```

The log reports workspaces of `0x1ea5b0400` (A, approximately 7.66 GiB) and
`0x3db300000` (B/C/D, approximately 15.42 GiB). These are MIOpen-reported
workspace requirements, not claims about physical VRAM or total process
allocation. The corresponding PyTorch peak allocations in the five-iteration
lane were 8.80 GiB for A, 17.43 GiB for B, 16.32 GiB for C, and 17.15 GiB for
D; reserved memory reached 23.09 GiB for A and 24.79 GiB for B–D.

The active MIOpen library identifies itself in the log as
`3.6.0.baa93758`, loaded from the known-good environment's bundled ROCm SDK
libraries. The host package inventory separately reports system ROCm
`7.2.2` and `miopen-hip 3.5.1`; the bundled library is the one used by this
PyTorch process and is the version that should be compared for this lane.

The installed system database path `/opt/rocm-7.2.2/share/miopen/db` has no
gfx1151 entry. The process log says the bundled installed-path database is
absent. MIOpen instead read and updated:

```text
/home/keith/.config/miopen/gfx1151_20.HIP.3_6_0_baa93758.ufdb.txt
/home/keith/.cache/miopen/3.6.0.baa93758/gfx1151_20.ukdb
```

No `MIOPEN_USER_DB_PATH` or `MIOPEN_FIND_MODE` override was supplied. The
runtime log reports `MIOPEN_FIND_MODE = DYNAMIC_HYBRID(5)`. The database was
available on the second run, and the selected solver remained the same; this
experiment does not claim that database lookup explains performance.

The full logging output is [`strix-gfx1151-miopen-logging.log`](../results/raw/conv3d/strix-gfx1151-miopen-logging.log)
(local only), with its machine-readable companion
[`strix-gfx1151-miopen-logging.json`](../results/raw/conv3d/strix-gfx1151-miopen-logging.json)
(local only).

## F3 — cross-platform interpretation (2026-09-09)

The following R9700 values are copied only as the explicitly reported matched
controls from the parallel investigation; they are not remeasured here.

| Measurement | R9700 gfx1201 reference | Strix gfx1151 |
|---|---:|---:|
| ROCm / active MIOpen | ROCm 7.2.1 / reference lane | system 7.2.2 / active bundled 3.6.0.baa93758 |
| PyTorch | reference lane | 2.13.0+rocm7.15.0a20260728 |
| A warm median / TFLOP/s | 0.146 s / 2.70 | 0.2309 s / 1.710 |
| B warm median / TFLOP/s | 5.045 s / 0.157 | 0.5043 s / 1.576 |
| C warm median | 3.168 s | 0.2980 s |
| D warm median / TFLOP/s | 5.549 s / 0.143 | 0.4783 s / 1.662 |
| Selected solver | reference investigation: direct naive for B | `GemmFwdRest` for A–D |

The evidence supports an architecture/backend-dependent problem on gfx1201,
not a general failure of this mathematical shape on AMD ROCm. It does not yet
establish whether the difference is caused by GPU architecture, MIOpen build,
solver database contents, or their interaction.

## F4 — isolated real VAE control completed (2026-09-09)

The real Wan2.1 VAE was decoded independently from the LingBot transformer
using a zero latent of shape `[16,4,60,104]`, equivalent to a 480×832 decode.
The test used three synchronized decodes on gfx1151 with the VAE weights in
FP32 and instrumented every `nn.Conv3d` without changing its arguments.

All three outputs were finite and had shape `[3,13,480,832]`. The cold decode
took **30.970 s**; the two warm decodes took **26.449 s** and **26.588 s**
(warm median **26.518 s**). Conv3d accounted for **78.695 s of 84.006 s**
across the three decodes (**93.68%** of measured decode time). Peak PyTorch
allocation was **21.24 GiB** and peak reservation **27.15 GiB**; process RSS
was approximately **2.0 GiB**.

The dominant actual VAE shape was `[1,96,4,480,832]` with `[96,96,3,3,3]`
weights: 54 calls, 29.686 s aggregate, or 37.72% of Conv3d time. This is
the real VAE's depth-4 shape, not synthetic case B's explicitly pre-padded
depth-6 input. The next-largest measured shapes were the 192-channel
`[1,192,4,240,416]` convolution (28.67%) and the 384-channel
`[1,384,2,120,208]` convolution (14.27%).

Raw data is [`strix-gfx1151-480x832-aggregate.json`](../results/raw/vae/strix-gfx1151-480x832-aggregate.json)
(local only); the instrumented runner is [`vae_repro.py`](../scripts/vae_repro.py).
This establishes that the isolated VAE is numerically viable on Strix, while
also identifying Conv3d as the dominant decode cost. It does not yet justify
a VAE optimization.

## F5 — live gfx1201 solver audit: old stack (2026-09-09)

The matched reproducer was run on the live R9700 host (`nautilus`) with
`HIP_VISIBLE_DEVICES=1`, using PyTorch
`2.9.1+rocm7.2.1.gitff65f5bc`, HIP `7.2.53211-e1a6bc5663`, and active
MIOpen `3.5.1.dabb6df2b9`. This is a fresh audit of the exact four-shape
cases, not an estimate copied from the earlier reference run.

With `MIOPEN_FIND_ENFORCE=1` and `MIOPEN_FIND_MODE=NORMAL`, MIOpen reported:

| Case | `GemmFwdRest` workspace check | Successful candidates | Selected solver |
|---|---|---|---|
| A | `8,226,800,640` bytes | `ConvDirectNaiveConvFwd`, `GemmFwdRest` | `GemmFwdRest` |
| B | `16,562,257,920 > 14,574,367,538` bytes | `ConvDirectNaiveConvFwd` | `ConvDirectNaiveConvFwd` |
| C | `16,562,257,920 > 14,574,367,538` bytes | `ConvDirectNaiveConvFwd` | `ConvDirectNaiveConvFwd` |
| D | `16,562,257,920 > 14,574,367,538` bytes | `ConvDirectNaiveConvFwd` | `ConvDirectNaiveConvFwd` |

For B–D, the `GetWorkspaceSize` comparison is followed by
`GemmFwdRest: Not applicable` in both workspace and search logging. The only
successful candidate is the direct naive kernel
`naive_conv_ab_nonpacked_fwd_ncdhw_float_double_float`, with zero workspace.
The enforced-find log therefore identifies the rejection point rather than
merely showing the final slow choice.

The old-stack user FindDb contains the same outcome: GEMM for A, direct naive
for B–D. The installed ROCm database has no gfx1201 entry; the active user
database is
`/home/boxwrench/.config/miopen/gfx1201_32.HIP.3_5_1_dabb6df2b9.ufdb.txt`,
with kernel cache
`/home/boxwrench/.cache/miopen/3.5.1.dabb6df2b9/gfx1201_32.ukdb`.

The official MIOpen documentation says that `MIOPEN_DEBUG_FIND_ONLY_SOLVER`
fails when a named solver is valid but not applicable. An old-stack test with
`MIOPEN_DEBUG_FIND_ONLY_SOLVER=GemmFwdRest` and a direct
`MIOpenDriver --solution GemmFwdRest` invocation both reproduced that behavior:
the normal API could not force GEMM for B. This is a solver applicability
failure, not evidence that the solver binary is absent.

## F6 — newer gfx1201 stack removes the observed cliff (2026-09-09)

As a bounded software control, the same live R9700 was run in the separate
`comfyui-rocm714` environment: PyTorch `2.12.0+rocm7.14.0`, HIP
`7.14.60850`, active MIOpen `3.5.2.cd957402`, with an isolated user FindDb.
The microbenchmark used the same script, dtype, shapes, warmup, and timing
methodology. It generated `GemmFwdRest` records for all four cases:

| Case | First | Warm median | Effective |
|---|---:|---:|---:|
| A | 5.03898 s | 0.083286 s | 4.741 TFLOP/s |
| B | 1.85430 s | 0.171255 s | 4.642 TFLOP/s |
| C | 1.54249 s | 0.124190 s | 0.200 TFLOP/s |
| D | 1.90291 s | 0.165354 s | 4.808 TFLOP/s |

The generated user FindDb records B–D as `GemmFwdRest` with the same
16,562,257,920-byte workspace requirement. Thus the gfx1201 hardware can
execute the GEMM path for these shapes; the old ROCm 7.2.1/MIOpen 3.5.1 lane
does not select it under its available-workspace gate. This strongly elevates
the old software-stack/MIOpen selection path as the immediate cause. It does
not isolate whether the change came from MIOpen policy, rocBLAS, database
behavior, or their interaction, and it is not an architecture-only claim.

## F7 — solver-audit conclusion and boundary (2026-09-09)

The main question is reduced to a concrete, reproducible mechanism:

* `GemmFwdRest` exists in the old gfx1201 MIOpen build and works for A.
* For B/C/D, old MIOpen rejects it as not applicable because its required
  workspace exceeds the reported `14,574,367,538`-byte ceiling.
* The direct naive solver is the only remaining successful candidate and
  causes the 5.032 s / 0.158 TFLOP/s B result.
* The same gfx1201 card under the separate ROCm 7.14/MIOpen 3.5.2 stack uses
  GEMM for B/C/D and reaches 4.6–4.8 TFLOP/s.

This is enough evidence to stop before custom Conv3d decomposition or a
LingBot/VAE rewrite. The next bounded question is which old-stack control
changes the workspace ceiling or selection policy, followed by a clean
isolated VAE comparison only if that is needed for the platform study.

Detailed commands, raw-log paths, and the cross-platform table are in
[`r9700-solver-audit-20260909.md`](r9700-solver-audit-20260909.md).

## F8 — R9700 ROCm 7.14 isolated VAE succeeds without temporal split (2026-09-09)

The real Wan2.1 VAE was decoded independently on the R9700 / gfx1201 using
the separate ROCm 7.14 environment: PyTorch `2.12.0+rocm7.14.0`, HIP
`7.14.60850`, active MIOpen `3.5.2.cd957402`, and
`WAN_VAE_CONV3D_TEMPORAL_SPLIT=0`.  The test used a zero latent of
`[16,4,60,104]`, three synchronized decodes, and the same FP32 VAE weights as
the Strix control.

All outputs were finite with shape `[3,13,480,832]`.  Cold decode was **9.445
s**; warm decodes were **7.802 s** and **7.837 s** (median **7.820 s**).
Conv3d accounted for **22.109 s** across the three decodes.  Peak PyTorch
allocation was **21.24 GiB**, reservation **27.15 GiB**, and peak process RSS
was **2.52 GiB**.  The dominant `[1,96,4,480,832]` group consumed 10.042 s
(45.42% of Conv3d time) across 54 calls.

This confirms that the newer gfx1201 software stack makes the isolated VAE
viable without the old-stack temporal split.  It does not prove that the full
LingBot pipeline can keep the transformer resident at the same time; that
separate smoke test initially failed because the large VAE workspace competed
with the transformer and existing GPU users.

Raw output is
[`r9700-gfx1201-rocm714-480x832.json`](../results/raw/vae/r9700-gfx1201-rocm714-480x832.json)
(local and ignored).  The runner and exact command are in
[`vae-repro-20260909.md`](vae-repro-20260909.md).

## F9 — complete R9700 ROCm 7.14 LingBot baseline succeeds (2026-09-09)

The pinned 1.3B causal-fast model generated valid finite output on the R9700
under PyTorch `2.12.0+rocm7.14.0`, HIP `7.14.60850`, and MIOpen
`3.5.2.cd957402`. The transformer ran in native BF16 and the attention backend
was PyTorch SDPA. The run used five causal chunks at the requested 480×832
area, `local_attn_size=18`, `sink_size=6`, seed 42, and no temporal Conv3d
split. The upstream frame normalization produced 77 frames from the 81-frame
request.

The cold run took **167.341 s** after **31.390 s** model initialization; the
warm repeated run took **160.674 s**. Mean DiT chunk time was **5.440 s** cold
and **5.426 s** warm, while VAE encode took **50.981/49.695 s** and VAE
decode **83.095/82.782 s**. Effective throughput was **2.773/2.888 FPS**.
Output was `[77,464,832]`, finite, and exactly deterministic across the two
runs for this seed.

The first no-offload attempt failed at VAE encode because a roughly 14.91 GiB
MIOpen allocation could not coexist with the resident transformer. The
successful control therefore used an experiment-only transformer CPU
offload boundary around VAE encode/decode plus
`PYTORCH_NO_CUDA_MEMORY_CACHING=1`; this is explicitly not part of the
hardware-agnostic upstream PR. Peak process RSS was 25.71 GB. The observed
GPU process allocation was approximately 13.75 GB during the run, while
post-generation PyTorch allocation was zero and is not a peak statistic.

This establishes a complete newer-stack R9700 control and confirms that the
MIOpen solver fix is enough for correctness, but the 32 GiB discrete-card
capacity boundary still makes native no-offload VAE/transformer coexistence
impractical in the observed host state.

Detailed results are in
[`r9700-rocm714-baseline-20260909.md`](r9700-rocm714-baseline-20260909.md);
raw data is local under
`results/raw/lingbot/r9700-rocm714-baseline/metrics.json`.

## F10 — native Strix 1.3B 21-frame baseline completes (2026-09-09)

The native BF16 Strix lane now completes end-to-end with the released 1.3B
checkpoint. It uses one `gfx1151` device, no quantization, no CPU offload,
PyTorch SDPA, requested 480x832, 21 frames, `chunk_size=3`,
`local_attn_size=18`, `sink_size=6`, seed 42, and the fixed examples/03
prompt/image/actions. The upstream normalization produces exactly 21 frames
because six latent frames divide into two three-frame chunks.

| Measurement | Cold | Warm repeat |
|---|---:|---:|
| Model initialization | 36.115 s | same process |
| T5 encode | 2.238 s | cache hit |
| VAE encode | 31.228 s | 24.930 s |
| DiT, two chunks | 13.492 s | 13.328 s |
| VAE decode | 42.658 s | 42.748 s |
| Generation | **89.940 s** | **81.216 s** |
| Effective FPS | **0.2335** | **0.2586** |

The outputs are finite and the corrected harness writes valid 832x464 H.264
video with 21 frames. The native transformer has 1,709,502,016 BF16
parameters on `cuda:0`; the VAE is also device-resident and T5 parameters are
CPU-resident outside prompt encoding. Peak PyTorch allocation was 43.475 GB,
peak reservation 52.647 GB cold (57.363 GB warm), and peak process RSS was
25.864 GB. PyTorch reports 120.259 GB of device-visible memory; the host DRM
surfaces separately report 4 GB VRAM and 120.259 GB GTT.

The instrumented per-forward record confirms four denoising forwards at
timesteps 999/937/833/624 plus one zero-timestep cache update per chunk. The
tracked patch is `patches/0003-per-forward-metrics.patch`; setup is idempotent
even when this later patch overlaps the earlier metrics patch.

## F11 — same-seed repeat is finite but not bit-identical (2026-09-09)

The warm repeat used the same prompt, image, actions, seed, model, and causal
settings, but its raw generated tensor differed from the cold tensor:

```text
max absolute difference:  1.9584124
mean absolute difference: 0.03384074
```

This is a reproducibility caveat, not an optimization target. It is preserved
in the raw JSON and compact benchmark record. The likely source is not yet
assigned; a later bounded stage-by-stage repeatability test should separate
T5, VAE encode, DiT, and VAE decode behavior before any kernel change.

The output-shape wrapper issue found during this baseline is fixed in the
experiment harness: upstream channel-first VAE output is normalized to
`[frames,height,width,3]` and written through checked FFmpeg. This fix does
not alter model computation.

## F12 — persistent VAE divergence is in fused ROCm SDPA (2026-09-09)

The first version of the incremental comparator had two harness errors: it
omitted the VAE batch dimension and iterated over latent height instead of
latent time. Those results are superseded. The corrected internal latent
shape is `[1,16,1,58,104]`, with time taken from the public `[C,T,H,W]`
dimension.

With those errors removed, FP32 batch and one-frame incremental decode still
diverged. `conv2` outputs were bit-identical for each latent frame, and the
first divergent decoder submodule was `decoder.middle.1.proj`, the projection
following Wan's spatial `AttentionBlock`. Instrumentation showed that Q, K,
and V entering `torch.nn.functional.scaled_dot_product_attention` were
bit-identical, while the fused ROCm SDPA output differed substantially (up to
about 7.2 before projection) between the batch-slice and one-frame paths.
Repeated identical decodes were deterministic, so this was not random output
noise or an accumulating decoder-cache error.

Selecting PyTorch's explicit `SDPBackend.MATH` for the VAE attention restored
bit-exact batch/incremental output on the 3-latent `[16,3,58,104]` control;
both paths remained finite. This is an experiment-local correctness fallback:
the transformer continues to use its normal SDPA path, and the accepted
batch baseline is unchanged. The persistent runner exposes
`--vae-attention-backend default` for comparison, but its accepted streaming
default is `math` until a faster ROCm VAE SDPA path is proven equivalent.

## F13 — 1.3B persistent FP32 stream passes the long correctness gate (2026-09-09)

The real LingBot 1.3B chunk-size-1 session was rerun with the VAE-only math
SDPA fallback. It used the same pinned model, prompt, image, seed 42,
requested 480x832, 81 frames, 21 latent chunks, `local_attn_size=18`, and
`sink_size=6`. All 81 decoded frames were finite. The prior fused-SDPA run
became non-finite; this run remained finite through the attention-window
rollover and final output.

| Metric | Result |
|---|---:|
| Session time | 217.621 s |
| Time to first visible frame | 11.581 s |
| Median action latency | 10.629 s |
| VAE encode | 102.109 s |
| DiT chunk 0 / steady after rollover | 1.877 / 3.53–3.59 s |
| VAE decode chunk 0 / steady | 2.734 / 7.67–7.80 s |
| Peak PyTorch allocation | 43.300 GB |
| Peak process RSS | 25.867 GB |
| Output | `[81,464,832,3]`, finite |

KV state progressed from global/local `1508/1508` tokens to
`31668/27144`; the local cap held after chunk 17 while global position
continued advancing. This verifies the DiT/KV rollover and persistent VAE
cache across the long run. Raw evidence is local at
`results/raw/interactive-chunk1-stream-fp32-480x832-81f-math-sdpa/metrics.json`
and the corresponding `interactive.mp4`.

The correctness milestone is complete. The math SDPA fallback is currently a
latency rejection for interactive use, not an optimization result; precision
and overlap experiments remain deferred until a faster numerically valid VAE
attention path is compared against this reference.

## F14 — FP16 is the validated persistent Strix VAE default (2026-09-09)

The persistent decoder was tested in FP32, FP16, and BF16 with the same
prompt, image, seed, resolution, 18+6 causal state, and math-SDPA correctness
fallback. The 9-frame controls were finite in all three modes. FP16 and BF16
were then run through the full 81-frame / 21-latent-chunk traversal; both
remained finite through local-window rollover and preserved the same KV
progression as FP32.

| VAE path | First visible | Median action | Mean VAE/chunk | Peak allocation | Output |
|---|---:|---:|---:|---:|---|
| FP32 + math SDPA | 11.581 s | 10.629 s | 7.577 s | 43.300 GB | finite |
| FP16 + math SDPA | **2.935 s** | **3.772 s** | **0.926 s** | **37.226 GB** | finite |
| BF16 + math SDPA | 2.947 s | 3.816 s | 0.975 s | 37.226 GB | finite |

The FP16 81-frame session took 78.454 s end-to-end; the BF16 session took
79.498 s. Both had approximately 25.867 GB peak RSS and ended at global/local
KV positions `31668/27144`. Representative first and late frames were
visually stable across FP32, FP16, and BF16, with no NaN/Inf, frozen output,
or obvious causal seam observed.

FP16 is therefore the default for the experimental persistent Strix decoder.
The `--vae-dtype bf16` and `--vae-dtype fp32` controls remain available. This
does not change the accepted native batch baseline, whose VAE remains FP32;
it changes only the persistent interactive runner after the FP32 reference
was validated.

Raw evidence is local under
`results/raw/interactive-chunk1-stream-fp16-480x832-81f-math-sdpa/metrics.json`
and
`results/raw/interactive-chunk1-stream-bf16-480x832-81f-math-sdpa/metrics.json`.

## F15 — same-device VAE/DiT overlap is not retained (2026-09-09)

Because FP16 substantially reduced VAE time, one bounded queued-stream test
was run with three 480x832 latent chunks, FP16 VAE, math SDPA, and the same
seed/prompt/action inputs. The test launched VAE decode on a second HIP stream
while the main stream generated the next DiT chunk, with explicit stream
dependencies and allocator lifetime tracking. Output was finite and the KV
state remained valid.

The matched post-refactor serial run took **8.390 s**, with **3.000 s** first
visible and **2.932 s** median action latency. The overlap run took **8.419 s**,
with **3.039 s** first visible and **2.967 s** median action latency. The
small total difference is within run-to-run noise, while action latency was
slightly worse under contention. The overlap mode is retained as an explicit
`--overlap` reproducibility control, but serial scheduling remains the
accepted default; UMA does not by itself make same-device compute overlap a
benefit.

Raw controls are local at
`results/raw/interactive-test-fp16-9f-math-sdpa-serial-postoverlap/metrics.json`
and
`results/raw/interactive-test-fp16-9f-math-sdpa-overlap/metrics.json`.

## F16 — steady-state FP16 decoder profile (2026-09-09)

The real serial FP16 persistent decoder was profiled for three 480x832
chunks with math SDPA. CUDA events were recorded for the decoder module tree;
the report includes inclusive and exclusive time, first-call time, repeat
mean, shapes, dtypes, and cumulative exclusive share. The profile run was
finite, took 8.381 s for the session, and had 36.787 GB peak PyTorch
allocation.

The first 13 grouped rows below account for **92.58%** of exclusive decoder
time (2.483 s total); percentages are within the profiled decoder tree, not
the complete DiT/session wall time:

| Operator group | Calls | Aggregate | Share |
|---|---:|---:|---:|
| CausalConv3d, 96ch, 464x832 -> 96ch | 18 | 793.50 ms | 31.95% |
| CausalConv3d, 192ch, 232x416 -> 192ch | 18 | 559.82 ms | 22.54% |
| CausalConv3d, 384ch, 116x208 -> 384ch | 15 | 212.54 ms | 8.56% |
| RMS_norm, 96ch, 464x832 | 21 | 190.50 ms | 7.67% |
| Output CausalConv3d, 96ch -> 3ch, 464x832 | 3 | 131.13 ms | 5.28% |
| RMS_norm, 192ch, 232x416 | 18 | 83.57 ms | 3.37% |
| CausalConv3d, 384ch, 58x104 -> 384ch | 30 | 73.47 ms | 2.96% |
| VAE AttentionBlock, 384ch, 58x104 | 3 | 53.11 ms | 2.14% |
| ResidualBlock, 96ch, 464x832 | 9 | 48.58 ms | 1.96% |
| Upsample, 192ch, 232x416 -> 464x832 | 3 | 45.66 ms | 1.84% |
| SiLU, 96ch, 464x832 | 21 | 40.88 ms | 1.65% |
| Conv2d, 192ch, 464x832 -> 96ch | 3 | 36.70 ms | 1.48% |
| CausalConv3d, 192ch, 116x208 -> 384ch | 6 | 29.02 ms | 1.17% |

The profile makes Conv3d the first optimization target, but it does not by
itself justify temporal decomposition: the next experiment must compare the
dominant native causal shape against a numerically equivalent Conv2d split on
the real persistent path. The full raw operator table is local at
`results/raw/interactive-test-fp16-9f-math-sdpa-profile-v2/metrics.json`.

## F17 — native gfx1151 Conv3d beats the temporal split (2026-09-09)

The dominant full-resolution 96-channel causal Conv3d was replaced only for
`upsamples.12.residual.2` by the exact temporal Conv2d sum, including the
existing causal feature-cache history, temporal padding, spatial padding,
stride, dilation, groups, and bias. The real 480x832 FP16 persistent session
used math SDPA, three latent chunks, and the same seed/prompt/action inputs.

| Decoder path | Session | First visible | Median action | VAE chunks | Peak allocation | Output |
|---|---:|---:|---:|---|---:|---|
| Native Conv3d | 8.381 s | 3.003 s | 2.945 s | 0.464 / 1.144 / 0.944 s | 36.787 GB | finite |
| One-module temporal split | 8.627 s | 3.021 s | 2.949 s | 0.719 / 1.190 / 0.943 s | 36.787 GB | finite |

The split is therefore rejected on Strix: native gfx1151 Conv3d is already
slightly faster on the real causal path, and the whole-session result does
not improve. The implementation remains available as the explicit
`--vae-temporal-split-module` control so the negative result is reproducible;
it is not part of the default path.

Raw controls are local at
`results/raw/interactive-test-fp16-9f-math-sdpa-profile-v2/metrics.json` and
`results/raw/interactive-test-fp16-9f-math-sdpa-temporal-split-u12r2/metrics.json`.

## F18 — steady-state chunk-size-1 action and DiT profile (2026-09-09)

The accepted FP16/math-SDPA/native-Conv3d interactive path was run for the
full 81-frame traversal, with DiT profiling enabled only from chunk 18 after
the 18-frame local window had filled. The run remained finite and ended at
global/local KV positions `31668/27144`. The profiling control itself took
78.949 s, with first action first-visible latency **2.962 s** and median
action latency **3.801 s**; the previously accepted uninstrumented lane
remains the performance baseline at **2.935 s first-visible** and **3.772 s
median**.

For the three profiled steady-state actions (chunks 18–20), the synchronized
waterfall averaged:

| Stage | Mean |
|---|---:|
| Action/camera/conditioning preparation | 0.039 ms |
| DiT forward 1 | 762.925 ms |
| DiT forward 2 | 711.242 ms |
| DiT forward 3 | 709.672 ms |
| DiT forward 4 | 710.738 ms |
| DiT forward 5, clean-latent KV write | 710.868 ms |
| Latent postprocessing | 4.357 ms |
| Persistent FP16 VAE decode | 950.694 ms |
| RGB layout conversion | 0.078 ms |
| First-frame device-to-host copy | 0.364 ms |
| First-visible total | **4562.323 ms** |
| All-frame host copy completion | 0.368 ms after first-frame boundary |

Presentation and video serialization were not on this harness's first-visible
path; the latter remains out of band. The profiled tail is slower than the
first action because self-attention cost increases as the causal state fills,
so the 2.935 s accepted first-visible number and the 4.56 s filled-window
waterfall describe different points in the same persistent session.

The DiT profile recorded 10.773 s of exclusive module time across 15 forwards
(three chunks × five forwards). The groups below account for 94.50% of that
exclusive time; shape-grouped `Linear` rows intentionally merge projections
with the same tensor shape, while the raw module paths remain in the JSON:

| Operator group | Calls | Aggregate | Share |
|---|---:|---:|---:|
| `CausalWanSelfAttention`, `[1,1508,1536]` | 450 | 6216.9 ms | 57.71% |
| `Linear`, `[1,1508,1536] -> [1,1508,1536]` | 4560 | 1415.9 ms | 13.14% |
| `CausalWanAttentionBlock` exclusive work | 450 | 828.2 ms | 7.69% |
| `Linear`, `[1,1508,1536] -> [1,1508,8960]` | 450 | 606.1 ms | 5.63% |
| `Linear`, `[1,1508,1536] -> [1,1508,9216]` | 15 | 584.8 ms | 5.43% |
| `Linear`, `[1,1508,8960] -> [1,1508,1536]` | 450 | 528.9 ms | 4.91% |

The dominant remaining DiT cost is therefore self-attention, not VAE or host
presentation overhead. Peak PyTorch allocation was 37.226 GB, peak RSS was
25.867 GB, and PyTorch reported 120,259,084,288 bytes of device-visible
memory. Raw evidence is local at
`results/raw/interactive-chunk1-dit-profile-81f/metrics.json`.

## F19 — fifth forward is a clean-latent state write, not cache bookkeeping (2026-09-09)

The five-forward trace records a 1508-token query on every forward. At the
filled-window point, the local cache remains capped at 27,144 tokens while
the global position advances by 1,508 tokens per action. The fifth forward
has the exact call form:

```text
pipe.model(x=[x0], t=0, cross_attn_first_call=False, ...)
```

where `x0` is the final clean latent from denoising and the model output is
discarded. It is required because the causal self-attention implementation
writes K/V for every transformer call: the four denoising calls write state
for intermediate noisy latents, while this final call overwrites the current
chunk's K/V with state derived from the clean latent that persists into the
next chunk. Cross-attention K/V is already initialized and reused.

The fifth call averaged **710.868 ms**, within the same range as denoising
forwards 2–4. Forward 4 cannot safely supply its K/V because it consumes a
different latent; the clean-latent K/V depends on the transformer block
sequence applied to `x0`. No fifth-forward removal or approximation was
accepted. A partial state-producing subgraph would need to preserve the
block-by-block dependencies and is not yet a correctness-preserving bounded
optimization.

The instrumentation and exact per-forward/cache/memory records are in commit
`d18b819` and the raw profile above. The next evidence-backed intervention is
therefore a narrowly scoped analysis of whether the fifth call can omit only
provably output-independent work; broad attention or kernel changes are
deferred until that dependency question is resolved.

## F20 — transactional clean-KV commit after first display (2026-09-09)

The exact clean-latent KV forward was moved after the first decoded RGB frame
was copied to the host. No model math was removed: the next chunk was not
allowed to begin until the clean K/V state had been committed. The current
serial runner remains the state-machine boundary, so an input arriving during
the deferred interval cannot launch speculative GPU work; only the completed
commit can make the next action eligible.

The 9-frame control first verified finite output and matching aggregate output
statistics. The full 81-frame A/B then exercised the filled 18-frame window
and multiple rollovers:

| Metric | Current-order A | Deferred clean-KV B |
|---|---:|---:|
| Early first-visible | 3.015 s | 2.575 s |
| Filled-window first-visible, chunks 18–20 mean | 4.544 s | **3.851 s** |
| Filled-window state-ready, chunks 18–20 mean | 3.593 s | 4.564 s |
| Filled-window next-action-ready, chunks 18–20 mean | 4.545 s | 4.565 s |
| Filled-window clean-KV cost | 0.705 s | 0.713 s |
| 81-frame session | 79.194 s | 79.206 s |

The deferred lane therefore removes approximately **0.69 s** from the
display-critical path, close to the full clean-forward cost, while preserving
the complete action cadence. Its output summary was exactly equal to the
current-order lane (`[81,464,832,3]`, finite, identical aggregate mean/std and
adjacent-frame delta), and both ended at global/local KV positions
`31668/27144`. The state machine is intentionally conservative: one queued
input may be parsed on the CPU in a future UI integration, but GPU generation
must wait for `state_ready`.

The waterfall now labels the clean pass separately as either
`clean_kv_on_first_visible_path` or `clean_kv_after_first_visible`. This is a
real scheduling candidate, not an approximation, and is retained for further
interactive testing. Raw A/B evidence is local at
`results/raw/clean-kv-order-serial-81f/metrics.json` and
`results/raw/clean-kv-order-deferred-81f/metrics.json`; the runner change is
in commit `67ba075` plus the critical-path labeling follow-up.

## F21 — filled-window DiT attention is real ROCm flash SDPA (2026-09-09)

The exact application attention path was probed for three filled-window
chunks (18–20), covering 450 self-attention and 450 cross-attention calls.
Self-attention dispatch receives contiguous BF16 tensors with
`Q=[1,1508,12,128]` and `K=V=[1,27144,12,128]`; the SDPA layout is
`[1,12,1508,128]` and `[1,12,27144,128]`. There is no attention mask, no
`is_causal` flag, and no length-mask argument: the rolling causal semantics
are represented by the explicit cache slice and its sink/window placement.

| Attention path | Calls | Dispatcher | SDPA | Dispatcher overhead |
|---|---:|---:|---:|---:|
| Self, 18+6 window | 450 | 5403.962 ms | 5382.036 ms | 21.926 ms |
| Cross, 512-token prompt cache | 450 | 126.214 ms | 123.543 ms | 2.671 ms |

The self-attention dispatcher overhead is only about 0.4% of its measured
time; this is not primarily a transpose/contiguous or Python-dispatch
problem. A same-shape BF16 SDPA trace with the installed ROCm/PyTorch stack
identified `aten::_scaled_dot_product_flash_attention` and the HIP kernel
`attn_fwd.kd` (about 12.2 ms for one `[1,12,1508,128]` ×
`[1,12,27144,128]` operation). The application probe's exact shapes and
dispatcher path match that operation. PyTorch reports all SDPA families
enabled, but CK SDPA is unavailable on this gfx1151 build; no backend switch
was made.

This closes the first backend question: the dominant 57.7% DiT category is
healthy fused ROCm flash SDPA arithmetic, not a hidden dense score tensor or
an obvious layout fallback. A broad attention-library swap is therefore
deferred. A future alternative would need to beat this real rectangular
flash path while preserving the explicit rolling cache semantics. Raw
application layout/timing evidence is local at
`results/raw/attention-probe-full-81f/metrics.json`.

## F22 — one lower-area persistent session scales below 480x832 (2026-09-09)

The single bounded lower-resolution test used a custom Wan-compatible area of
`264192` pixels, which resolved to actual geometry **384x672**. It used a
fresh persistent cache, chunk size 1, local/sink window 18+6, BF16 DiT,
FP16 VAE, math VAE SDPA, native Conv3d, and the deferred clean-KV schedule.
The output was finite for all 81 frames and ended with local KV capped at
`18144` tokens while global position advanced to `21168`.

| Filled-window metric, chunks 18–20 | 464x832 reference | 384x672 |
|---|---:|---:|
| Tokens/frame | 1508 | 1008 |
| Local KV capacity | 27144 | 18144 |
| DiT total | 2896.6 ms | **1598.1 ms** |
| Denoise 1 / 2 / 3 / 4 | 756–767 / 705–710 / 705–716 / 704–717 ms | **424.0 / 390.0 / 389.8 / 390.2 ms** |
| Clean KV forward | 713.1 ms | **392.0 ms** |
| VAE decode | 954.2 ms | **632.3 ms** |
| First-visible | 3851.3 ms | **2230.8 ms** |
| State-ready | 4564.4 ms | **2622.8 ms** |
| Next-action-ready | 4565.0 ms | **2623.1 ms** |

The first run's first post-bootstrap VAE decode took 6.918 s while MIOpen
initialized the new geometry; subsequent decodes were about 0.632 s. A
second identical run after that cache warm-up reached 1.726 s early
first-visible and 2.252 s median action, confirming that the large first
decode is startup behavior rather than steady-state geometry cost.

Peak PyTorch allocation was 30.049 GB, peak reservation 42.358 GB, and peak
RSS 25.870 GB. Sampled frames at the beginning, direction changes, and late
rollover remained visually coherent: lake/tree geometry and camera motion
continued without an obvious seam or frozen output. This is a serious
candidate for interactive use because filled-window first-visible improves by
about **42%** and next-action-ready by about **42%** relative to the
480x832 deferred lane. It is not a blanket resolution recommendation until
the low-resolution attention component is recorded and the quality tradeoff
is reviewed with the same traversal.

Raw metrics and the representative video are local at
`results/raw/interactive-custom-area-384x672-deferred-81f/metrics.json` and
`results/raw/interactive-custom-area-384x672-deferred-81f-video/interactive.mp4`.

## F23 — lower-area attention scales with the same fused backend (2026-09-09)

The custom `384x672` geometry was separately profiled for chunks 18–20 with
the current-order path so all five forwards, including clean-KV, were in the
attention capture. It recorded the same 450 self and 450 cross calls and the
same `aten::_scaled_dot_product_flash_attention` / `attn_fwd.kd` backend as
the reference geometry:

| Self-attention quantity | 464x832 | 384x672 |
|---|---:|---:|
| Q shape before transpose | `[1,1508,12,128]` | `[1,1008,12,128]` |
| K/V shape before transpose | `[1,27144,12,128]` | `[1,18144,12,128]` |
| Calls | 450 | 450 |
| SDPA aggregate | 5382.036 ms | **2686.682 ms** |
| Dispatcher overhead | 21.926 ms | 20.926 ms |

The no-defer component control measured approximately 2.02 s total DiT for
the three steady chunks, including clean-KV; the deferred end-to-end lane
measured 1.598 s denoise-only plus a 0.392 s clean commit after first
display. This supports the conclusion that the lower-area improvement comes
from shorter Q/K histories and lower attention arithmetic, with additional
scaling in projections/MLP and VAE—not from changing the backend.

Raw evidence is local at
`results/raw/attention-probe-384x672-81f/metrics.json`. No second resolution
was started; the bounded resolution experiment stops here pending a product
quality decision.

## F24 — the causal-fast sampler is a four-point flow update, not a scheduler step loop (2026-09-09)

The current checkout's actual causal-fast schedule is selected by hard-coded
indices `[0,179,358,679]` after `set_timesteps(..., shift=5.0)`. On this
machine those indices produce timestep values `[999,957,899,702]`; the
previous approximate values `[999,967,908,768]` do not describe this pinned
source/model pair.

The causal loop does not call the scheduler's ordinary `step()` method. For a
model flow prediction `v`, current latent `x_t`, and scheduler sigma
`sigma_t`, upstream converts the prediction to

```text
x0 = x_t - sigma_t * v
```

and, between denoising points, calls `scheduler.add_noise(x0, noise,
next_timestep)`. The final accepted `x0` is followed by the exact separate
zero-timestep transformer pass that writes the persistent self-attention KV
state. That clean pass remains mandatory in the approximation experiment.

This makes a larger interval algebraically representable by the existing
flow-matching update code, but it does not prove that the released checkpoint
was distilled for that interval. The single controlled candidate therefore
drops only the second hard-coded point, retaining the first endpoint, the
`899` lower-noise stage, and the final `702` stage:

```text
four-step: 999 -> 957 -> 899 -> 702
three-step: 999 -> 899 -> 702
clean KV:  0 (exact separate pass)
```

The runner records both the schedule name and resolved timestep values in each
metrics file. No timestep sweep was performed.

## F25 — dropping the 957 denoising point buys approximately 390 ms (2026-09-09)

The candidate was run independently at actual `384x672`, chunk size 1, local
window/sink `18+6`, BF16 transformer, FP16 VAE, math VAE SDPA, native Conv3d,
deferred exact clean-KV commit, seed 42, and the same prompt/image/action
sequence as the four-step reference. The short warmed control confirmed three
denoising records plus one clean-KV record; the long run independently evolved
its own persistent state for 81 visible frames and 21 latent chunks.

### Matched warmed control

The 9-frame controls started from the same fresh model/session configuration.
The first action is affected by the session's initial state, so the filled-
window measurements below are the primary A/B metric. The warmed short control
still showed the expected reduction in per-action work:

| Schedule | Denoising forwards | Short-control median action | Raw metrics |
|---|---:|---:|---|
| 4 point | 4 + exact clean pass | 1910.7 ms | `results/raw/interactive-384x672-4step-deferred-9f-warm/metrics.json` |
| 3 point | 3 + exact clean pass | 1669.9 ms | `results/raw/interactive-384x672-3step-deferred-9f/metrics.json` |

Both short outputs were finite and retained the expected cache progression.

### Filled-window performance

The 3-step long run completed with finite output shape `[81,384,672,3]`,
output range `[-1,1]`, and no non-finite action/forward result. The final
window remained capped at `18144` local KV tokens while global position
advanced through the full 21-chunk session. Comparing chunks 18–20 against
the existing four-step 384x672 reference:

| Filled-window metric | 4 point | 3 point | Change |
|---|---:|---:|---:|
| Denoise total | 1594.0 ms | 1202.3 ms | -391.8 ms |
| Denoise forward 1 | 424.0 ms | 421.8 ms | -2.2 ms |
| Denoise forward 2 | 390.0 ms | 390.6 ms | +0.6 ms |
| Denoise forward 3 | 389.8 ms | 389.9 ms | +0.0 ms |
| Denoise forward 4 | 390.2 ms | — | — |
| Exact clean-KV pass | 391.9 ms | 391.2 ms | -0.7 ms |
| VAE decode | 632.3 ms | 634.0 ms | +1.7 ms |
| First-visible | 2230.8 ms | **1840.7 ms** | **-390.1 ms** |
| Next-action-ready | 2623.1 ms | **2232.3 ms** | **-390.8 ms** |

The result is close to the cost of one removed denoising evaluation, with no
measurable VAE or clean-KV regression. The 21-chunk candidate session took
41.028 s; this total includes session setup and is not used as the primary
steady-state comparison. Its median action latency was 1924.1 ms and the
last-five mean was 2209.6 ms.

### Persistent-world behavior

The independent 81-frame candidate stayed finite through multiple local-window
rollovers. The recorded KV lengths remained bounded at `18144` in the filled
window, and the output statistics were:

```text
mean: 0.28545
std:  0.58217
mean adjacent-frame delta: 0.15863
```

Manual inspection of sampled beginning, middle, direction-change, and late
frames found a coherent lake/tree scene with continued camera motion, no
catastrophic collapse, no frozen output, and no obvious chunk seam. The
three-step frames show somewhat more texture/water variation and softer detail
than the four-step reference; this is a qualitative observation, not a formal
image-quality score. The run is therefore a successful persistent-world
control, but longer and broader human quality evaluation would still be
appropriate before treating the approximation as universally preferable.

The candidate video and raw metrics are local at
`results/raw/interactive-384x672-3step-deferred-81f-video/interactive.mp4` and
`results/raw/interactive-384x672-3step-deferred-81f-video/metrics.json`.

### Recommendation

Retain `3-drop-957` as a validated low-latency candidate for Strix interactive
use. It removes one exact denoising evaluation while preserving the separate
clean-KV commit, improves filled-window first-visible from about `2.23 s` to
`1.84 s`, and improves next-action-ready from about `2.62 s` to `2.23 s`.
Keep the four-point schedule as the quality/reference mode until a deliberate
human or task-level evaluation decides whether the observed texture variation
is acceptable for the intended application.

## F26 — TAEHV latent-contract correction (2026-09-10)

The pinned `madebyollin/taehv` checkout is commit
`011dfc2112197741c540e0bdd5b7b67bcc930771`, with `taew2_1.pth` SHA-256
`d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e`.
The loaded architecture has 16 latent channels, temporal down/upscale 4, and
startup trim of 3 frames.

The first offline candidate incorrectly applied the canonical LingBot Wan VAE
mean/std transform before TAE. That produced visibly oversaturated output and
was rejected as a latent-contract error, not as evidence against TAEHV. The
pinned Wan 2.1 Diffusers wrapper uses identity latent mean/std, so the corrected
path feeds the accepted LingBot model-space `x0` directly after only
`NCTHW`/`NTCHW` conversion. At `384x672`, the saved stream is
`[1,16,21,48,84]` for canonical input and `[1,21,16,48,84]` for TAE input.

## F27 — TAEHV is a viable fast presentation decoder (2026-09-10)

The corrected offline comparison decoded the same contiguous 21-latent stream
through canonical FP16 Wan VAE and TAEHV. Both outputs were finite and had
exactly 81 frames: one frame from the first latent after startup trim and four
from each later latent. Warm canonical filled-latent decode was approximately
`639 ms`; TAE emitted its first frame in `7.7–8.1 ms` and drained a filled
latent in `14.6–16.7 ms`. Mean absolute frame difference was `0.0319` in
`[0,1]`. Human inspection found the same broad lake/tree world and motion, with
TAE softer fine detail and changed texture/edges.

The opt-in live mode then completed an independent 81-frame, 21-chunk
persistent session on gfx1151. All outputs were finite. At full `18+6`
occupancy, TAE first RGB was `1229–1232 ms` after action acceptance and all
TAE output was complete in `36.9–37.6 ms` GPU time. The exact deferred clean-KV
commit remained `385–387 ms`, the final local KV length was capped at `18144`
tokens while global position reached `21168`, and next-action readiness was
`1644–1648 ms`.

Against the accepted canonical 3-step path (`~1841 ms` filled-window
first-visible and `~2232 ms` next-action-ready), TAE saves about `0.6 s` on
both product boundaries. The live long-run peak PyTorch allocation was
`30.049 GB`, peak reserved allocation `42.358 GB`, and peak RSS `25.870 GB`.
These are whole-pipeline UMA accounting values, not TAE-only memory.

TAE is therefore retained as an explicit opt-in presentation mode, while the
canonical FP16 decoder remains the default/reference path for quality. The
offline report and raw metrics are in
`docs/taehv-20260910.md` and
`results/raw/taehv-offline-384x672-3step-81f/metrics.json`; the live raw result
is in
`results/raw/interactive-taehv-384x672-3step-deferred-81f-video/metrics.json`.

## F28 — bounded keyboard viewer reaches a real persistent action (2026-09-10)

The live viewer initially exposed three ordinary integration issues: the
preparation script assumed `rg` was installed, the wrapper did not pass the
pinned upstream checkout on `PYTHONPATH`, and the viewer omitted the accepted
BF16 DiT autocast context. These were fixed independently in commits
`f1adf80`, `7319e17`, and `f5db488`.

The final wrapper was validated with a real Tk window and an automated `W`
keypress. Bootstrap plus one user action completed successfully with finite
TAE output. The action measured `665.3 ms` to first RGB and `941.3 ms` to the
next-action-ready boundary; the exact clean-KV pass took `219.3 ms`. KV
progressed from `1008` to `2016` tokens with capacity `18144`, and the action
output remained finite. This confirms that the handoff command exercises the
persistent model rather than merely opening a video player.

The bounded launch path is documented in `docs/live-viewer.md`. It uses a
20-action default limit, a configurable wall-clock timeout, Q/ESC in the
window, and Ctrl-C from the terminal. At this milestone it wrote metrics only
by default; the later opt-in generated-frame capture is documented separately
and does not change the default path.

## F29 — context occupancy profile identifies self-attention growth (2026-09-10)

The real live viewer path was extended with opt-in deterministic scripted
actions and per-chunk module/SDPA probes; accepted defaults were unchanged.
An unprofiled bootstrap-plus-40-action control and a detailed prefix at chunks
`1, 9, 17, 18` covered Early, Mid, Full, and Rolled occupancy. All 41 control
rows were finite. At 384×672, each latent frame contributes `1008` tokens and
the physical local cache is `18144` tokens, not 24 frames: `sink_size=6` is a
retention policy inside the 18-frame capacity.

| Regime | K tokens | 3× denoise | clean KV | first visible | next action |
|---|---:|---:|---:|---:|---:|
| Early | 2,016 | 713.6 ms | 229.5 ms | 735.1 ms | 1,016.2 ms |
| Mid | 10,080 | 938.2 ms | 310.7 ms | 958.8 ms | 1,315.4 ms |
| Full | 18,144 | 1,192.3 ms | 394.2 ms | 1,212.9 ms | 1,655.0 ms |
| Rolled median | 18,144 | 1,224.9 ms | 396.1 ms | 1,245.7 ms | 1,688.0 ms |

Full minus Early denoise growth was `478.7 ms`. The intrusive SDPA probe measured
self-attention growth of `482.8 ms` across the 90 self-attention calls, while
cross-attention stayed at about `18.4 ms` total and self QKV/MLP projections
were effectively flat. Thus self-attention arithmetic accounts for essentially
all of the context-dependent denoise increase. The first rolled action adds a
smaller cache-shift cost while keeping K bounded at `18144`.

The current fused path remains PyTorch SDPA, previously traced to
`aten::_scaled_dot_product_flash_attention` / HIP `attn_fwd.kd`; the current
probe confirmed rectangular Q/K/V shapes, contiguous pre-SDPA tensors,
transpose views into SDPA, no explicit mask, and `is_causal=False`. The exact
sink/history/current accounting after rollover is 6,048 + 11,088 + 1,008
tokens. The detailed report and raw ignored metrics are in
[`profile-live-context-20260910.md`](profile-live-context-20260910.md).

This closes the profile-only phase. The next justified single experiment is a
controlled `local_attn_size=12`, `sink_size=6` persistent A/B; no such change
is included here.

## F30 — shorter attended history is an interactive candidate (2026-09-10)

The bounded 18-vs-12 persistent A/B kept the accepted 384×672 geometry,
chunk_size 1, 3-step `999 → 899 → 702` sampler, sink size 6, BF16 DiT,
FP16 TAEHV presentation, deferred exact clean-KV pass, and serial execution
unchanged. The only variable was physical local attention capacity:

```text
18-frame control: 18,144 tokens = 6,048 sink + 11,088 recent + 1,008 current
12-frame candidate: 12,096 tokens = 6,048 sink + 5,040 recent + 1,008 current
```

Both matched 40-action sessions completed finitely through repeated rollovers.
Global position reached `41,328` tokens while local K stayed at the configured
capacity. The 12-frame candidate's rolled median first-visible latency was
`1,024.8 ms` versus `1,254.8 ms` for the 18-frame control, a saving of
`229.9 ms`. Rolled next-action-ready was `1,406.8 ms` versus `1,706.6 ms`, a
saving of `299.7 ms`. Full-capacity first-visible was `1,015.3 ms` versus
`1,212.4 ms`; clean KV improved by about `65 ms`.

The detailed candidate probe confirmed the same contiguous BF16 fused SDPA
path with Q `[1,1008,12,128]` and K/V `[1,12096,12,128]`; no mask or backend
fallback was introduced. The candidate therefore recovers latency for the
expected reason: less historical self-attention arithmetic. Matched captures
remained coherent and finite through the tested forward/turn/reversal path,
but late scenes visibly diverged as expected when older recent history was
forgotten. Classification: **ACCEPT INTERACTIVE**. Keep 18 frames as the
quality/reference default and expose 12 frames only as an opt-in low-latency
mode. The detailed report and local ignored artifacts are in
`docs/window-18-vs-12-20260910.md`.

## F31 — 14-frame window adds visible persistence at bounded cost (2026-09-10)

The bounded 12-vs-14 A/B changed only physical local attention capacity. Both
runs used 384×672, chunk_size 1, the 3-step `999 → 899 → 702` sampler, sink
size 6, BF16 DiT, FP16 TAEHV presentation, deferred exact clean-KV, and serial
execution. Runtime accounting verified:

```text
12 frames: 6,048 sink + 5,040 recent + 1,008 current = 12,096 K tokens
14 frames: 6,048 sink + 7,056 recent + 1,008 current = 14,112 K tokens
```

Both 40-action sessions were finite through repeated rollovers and reached
global position `41,328`. Rolled 14-frame latency was `1,106.7 ms`
first-visible and `1,517.1 ms` next-action-ready, versus `1,033.2 ms` and
`1,418.4 ms` for 12 frames. The added cost was `73.5 ms` for first RGB and
`98.7 ms` for readiness; detailed SDPA probes confirmed the expected increase
in historical self-attention.

Matched 161-frame captures showed both paths preserving the main lake/tree/
mountain world through the scripted turns and reversals. After the shorter
window forgot more recent history, 14 frames more consistently retained the
right-hand shoreline/structure and tree-to-background relationship, while 12
showed more late texture/edge substitution. No collapse, frozen output, or
catastrophic drift was observed. Classification: **BOTH MODES JUSTIFIED**.
Keep 12 as the minimum-latency interactive option, expose 14 as a higher-
quality interactive option, and retain 18 as quality/reference. The dedicated
report is `docs/window-12-vs-14-20260910.md`.

## F32 — first new RGB is already action-responsive (2026-09-10)

A matched-state closed-loop probe compared no-op versus strong turn and
forward versus reverse after the local window had filled. Each branch cloned
the persistent self-KV, cross-attention cache, RNG state, and TAE setup, so
the intended independent variable was the action. At both 12 and 14 frames,
RGB0 was already materially different for both control pairs: 12-frame MAD
was `0.0459` / `0.0257`, and 14-frame MAD was `0.0421` / `0.0245`, with
12.2–21.2% of pixels above the supporting threshold. This indicates that the
first newly displayed frame is action-responsive for the tested controls;
there was no measured RGB1-or-later response delay.

A virtual input observer also placed an arrival at `1071.6 ms` inside the
12-frame no-op branch's exact clean-KV interval (`1020.0–1349.0 ms`). The
current viewer has no explicit application queue, so this proves only that
the clean barrier protects GPU generation; it does not promise retention of
a physical key pressed during that interval. The next justified technical
experiment is a bounded one-slot pending-input queue.

The accepted rolled 12-frame product baseline remains approximately
`1033.2 ms` first-new RGB, `333.9 ms` clean commit, and `1418.4 ms` next
action. Its residual gap is `51.3 ms`; known TAE tail work is about 17–18 ms,
with the remainder not yet separately isolated because the first-visible
boundary is immediately before Tk presentation. Detailed branch metrics and
artifacts are in `docs/action-response-20260910.md` and
`results/raw/action-response-12-v3/` / `results/raw/action-response-14/`.

## F33 — bounded pending input survives the busy action (2026-09-10)

The live viewer now has a one-slot input controller. While model work is in
flight, the newest valid movement/look/no-op replaces any older pending key;
invalid keys are ignored and quit remains sticky. At the legal end of the
current action, the pending key is consumed before normal input waiting. The
exact clean t=0 KV pass remains mandatory before the next DiT generation.

Model-free tests pass for replacement, same-key repeats, invalid input,
pre-bootstrap promotion, and quit priority. A real X11 run sent `D` during a
measured clean interval (`63801.4–64032.5 ms`). Tk delivered it at `64066.3
ms`, selected it at `64076.4 ms`, and started the next generation at `64078.4
ms`, with no second keypress. A separate real run observed and stored a
movement key at `78244.2 ms` on the busy update boundary; it was selected at
`78245.6 ms` and generation began at `78246.7 ms`.

The first run demonstrates survival through Tk/X11 buffering; the second
demonstrates the explicit busy-time pending slot. Real multi-key X11 timing
was not treated as proof of replacement semantics; those semantics are
covered by the deterministic controller tests. The change adds no model work
and leaves the accepted rolled 12-frame product baseline (`~1033.2 ms`
first-new RGB, `~1418.4 ms` next-action-ready) unchanged. Details and raw
metrics are in `docs/pending-input-20260910.md`.

## F34 — TunableOp improves recurring gfx1151 GEMM dispatch (2026-09-10)

The installed PyTorch ROCm build exposes TunableOp and validator-bound offline
tuning. A real 384×672, 12-frame, three-step LingBot live run captured 16
unique GEMM/BGEMM signatures, including the BF16 MLP/conditioning projections
and the float `time_projection.1` operation. All 16 signatures were tuned in
a separate 125.314-second process; 13 selected hipBLASLt implementations and
three retained `Default`.

In matched fresh-process 40-action runs with online tuning disabled, rolled
median first-visible latency improved from `1033.762 ms` to `970.854 ms`
(`−62.908 ms`). Rolled median next-action-ready improved from `1418.512 ms`
to `1329.256 ms` (`−89.256 ms`), with similar P95 deltas. Both lanes were
finite through global KV position `41,328`, with the 12,096-token local cap
and 6,048 sink tokens retained. A 14-frame transfer loaded the same file
without retuning and remained finite.

The detailed CUDA-event profile shows the observed gain is GEMM-side: denoise
linear-module exclusive time fell `402.511 → 322.846 ms`, clean-KV linear
time fell `132.756 → 107.974 ms`, and `time_projection.1` fell `78.925 →
10.953 ms` across the three denoise passes. Fused self-attention remained
approximately unchanged (`357.335 → 358.311 ms`). This is retained as an
opt-in, validator-specific gfx1151 artifact; accepted defaults remain
unchanged. Full CSVs and a compact summary are in
`docs/artifacts/tunable-op-20260910/`, with detailed methodology in
`docs/tunable-op-20260910.md` and large raw process outputs under the ignored
`results/raw/tunable-op-20260910/` directory.

## F35 — direct regional compilation of the stateful causal block is rejected (2026-09-10)

The first bounded compiler probe held the accepted TunableOp configuration
constant and wrapped only one real `CausalWanAttentionBlock` with PyTorch
Inductor (`backend="inductor"`, `mode="default"`, `fullgraph=False`). The
validator-matched file loaded all 16 persisted TunableOp results with online
tuning and untuned recording disabled. No model, sampler, renderer, window, or
clean-KV behavior was changed.

The stateful block did not form one reusable graph. The run produced 27 unique
graphs from 32 Dynamo frames, 23 graph-break events, and 124 logged
recompilation events. Breaks and guards repeatedly centered on
`local_end_index.item()`, `global_end_index.item()`, rollover comparisons, and
in-place cache-index updates in `model_fast.py`. The captured regions also
contained `aten.addmm` signatures, so the persisted TunableOp hipBLASLt
solutions could not be verified as preserved inside Inductor-generated code.

Correctness failed before a performance A/B was legitimate: bootstrap output
was finite, but the first subsequent action produced non-finite latent and RGB
output, with an invalid-value cast warning. The first bootstrap transformer
interval was `8,581.99 ms` because lazy compilation occurred there, versus
`779.17 ms` in the prior TunableOp-only profile. A long or 30-block benchmark
was therefore stopped. The accepted TunableOp-only path remains unchanged.

Classification: **REJECT** direct compilation of the full causal block on this
build. The detailed report is [`compile-20260910.md`](compile-20260910.md),
with the compact committed metrics at
`docs/artifacts/compile-20260910/`; raw logs remain under the ignored
`results/raw/compile-20260910/probe-block-1b/` directory. If compilation is
revisited, the only justified next target is a newly isolated state-free
pointwise/FFN helper, leaving SDPA and all KV/cache mutation eager.

## F36 — pure state-free tensor islands are a small opt-in win (2026-09-10)

The follow-up compiler probe left all stateful and GEMM-bearing work eager and
compiled only seven pure tensor helper families with Inductor: modulation,
affine transforms, scaled residuals, tensor addition, SiLU, tanh-approximate
GELU, and camera tensor update arithmetic. Self- and cross-attention, every
Linear, KV reads/writes, cache dictionaries, rollover/index logic, and SDPA
remain outside the compiled regions. The same helper bank was installed on all
30 repeated blocks; no accepted default was changed.

On the validator-matched gfx1151 TunableOp stack, the candidate produced eight
stable graphs, 23 captured helper calls, no logged graph breaks, and no logged
recompiles across a 40-action persistent run. Compiler diagnostics contained no
`aten.addmm`, `aten.mm`, or `aten.bmm` entries. TunableOp loaded all 16 persisted
results with the expected PT/HIP/hipBLASLt/gfx1151/rocBLAS validators. A separate
real-shape helper check was finite: simple helpers were bitwise equal, affine
and residual max error was `9.54e-7`, and camera-update max error was `0.0625`
at BF16-scale values (the expected fused-rounding difference).

Against the exact TunableOp-only 12-frame control, the lightly instrumented
candidate's rolled median improved from `950.627` to `922.412 ms` for the
three denoise passes, from `970.854` to `943.247 ms` first-visible, from
`310.164` to `300.973 ms` for clean KV, and from `1329.256` to `1291.867 ms`
next-action-ready. P95 improved similarly. The detailed profile attributes
the change to state-free block-local work: self-SDPA and eager Linear/GEMM
time were effectively unchanged, while the block exclusive remainder fell
about `25 ms` across the three denoise passes and `8 ms` in clean KV; GELU
module work disappeared into the compiled helper. These profile timings are
attribution evidence, not the product numbers.

All 40 actions reached global position `41,328` with the local KV cap at
`12,096` tokens and `6,048` sink tokens retained; latents and RGB stayed finite.
A small 14-frame transfer run reused the same eight graphs, reached a
`14,112`-token local cap with `6,048` sink tokens, and stayed finite through
rollover without a second tuning campaign.

Classification: **RETAIN AS OPT-IN**. The gain clears the approximately 20 ms
whole-path threshold, but compilation remains opt-in because the first
bootstrap invocation carries roughly `1.9 s` of lazy compile/setup overhead
even when the process sees populated FX/Inductor cache entries. The accepted
TunableOp-only lane remains the default. Raw logs and metrics are under the
ignored `results/raw/compile-20260910/` tree; the dedicated report is
[`pure-helper-compile-20260910.md`](pure-helper-compile-20260910.md).

## F37 — host-side KV cursor removes scalar syncs but is not a release win (2026-09-10)

The causal self-attention path reads `global_end_index` and
`local_end_index` tensors with `.item()` for rollover and cache-slice control.
Static tracing and live checks showed these are host-determined counters, not
independent device-computed state: all 30 layers had equal cursor values at
early startup, full capacity, first eviction, repeated rollover, and the exact
clean t=0 pass. A rolled action contains 360 model-side cursor reads; harness
and reporting reads are separate.

An opt-in `--host-kv-cursor` path now carries per-layer Python integer cursors
as the control-flow authority while retaining the existing tensor cursors as
`fill_`-updated compatibility mirrors. It does not consolidate layers into a
single shared cursor and does not change local history, sink retention, RoPE
positioning, attention K length, or clean-KV semantics.

The mechanism-only profiler showed the expected change: model-side scalar
reads fell from 360 to zero (total `aten::item` calls in the selected trace
fell `401 → 41`), and the associated `hipStreamSynchronize` calls fell
`389 → 29`. This trace is intrusive and is not a product-latency result.

Fresh matched 40-action runs were fully finite and exact. In 29 rolled rows,
host cursors changed first-visible P50 `936.809 → 928.553 ms` (`−8.256 ms`,
P95 `942.659 → 935.103 ms`) and next-ready P50 `1282.925 → 1272.936 ms`
(`−9.989 ms`, P95 `1293.757 → 1280.307 ms`). The equivalence harness found
bitwise-equal x0, denoise K/V, clean K/V, and cursor mirrors across 16 chunks;
the live runs reached global `41328`, local `12096`, sink `6048`, with finite
RGB. The result is below the release optimization threshold, so defaults are
unchanged and no 14-frame extension was run.

Classification: **REJECT AS RELEASE OPTIMIZATION; RETAIN OPT-IN FOR
DIAGNOSTICS**. Details and raw artifact paths are in
[`kv-cursor-20260910.md`](kv-cursor-20260910.md) and
`docs/artifacts/kv-cursor-20260910/`; ignored profiler traces remain under
`results/raw/kv-cursor-20260910/`. The next authorized experiment is
shape-matched compile prewarm, which was not started here.

## F38 — shape-matched pure-helper prewarm removes the first-bootstrap compile stall (2026-09-10)

The accepted pure-helper lane was unchanged: seven state-free tensor-helper
families, eight Inductor graphs, eager stateful attention/KV/cache operations,
eager Linear/GEMM boundaries for the 16-result validator-matched TunableOp
file, and TAEHV presentation. A new opt-in prewarm call now invokes the exact
`PureTensorIslands` instance installed on the live 30-block model, using
zero-filled tensors with the observed `[1008, 1536]`/`[1008, 8960]` BF16 and
FP32 contracts and live modulation/view strides. It does not pass data through
model blocks, touch persistent state, or consume RNG; CPU and GPU RNG states
were unchanged.

The cold lazy control created eight graphs during first use and paid a
`7553.1 ms` runtime-ready-to-bootstrap-transformer interval. The cold explicit
prewarm lane paid `4315.1 ms` before `Runtime ready`, then reduced the same
bootstrap interval to `754.0 ms`. A fresh process using the populated compiler
cache still required `1777.7 ms` explicit prewarm but had a `754.3 ms`
bootstrap interval. Thus disk cache hits reduce prewarm work but do not replace
runtime prewarm. Both 40-action lanes had eight graphs, 23 captured calls, no
graph breaks, no recompilations, and no `aten.addmm/mm/bmm` in the compiled
regions.

The cold-prewarm 40-action run stayed finite through repeated rollovers and
ended at global `41328`, local capacity `12096`, and sink retention `6048`.
Its rolled medians were `943.2 ms` first-visible and `1292.4 ms` next-ready,
versus `935.3 ms` and `1284.5 ms` for the fresh cold-lazy control; this small
spread is not treated as a steady-state regression. A small 14-frame transfer
reused the eight graphs, reached `14112` local tokens with `6048` sink tokens,
and stayed finite.

Classification: **RETAIN AS OPT-IN RELEASE PREPARATION**. The prewarm path
moves meaningful first-use work before an explicit runtime-ready boundary and
keeps the existing non-prewarmed lane as the default pending startup UX and
cache packaging work. Detailed methodology and startup tables are in
[`prewarm-20260910.md`](prewarm-20260910.md); compact evidence is in
`docs/artifacts/prewarm-20260910/`, with raw process logs under the ignored
`results/raw/prewarm-20260910/` tree.

## F39 — RealESRGAN x2 improves offline detail but is too expensive in the live process (2026-09-10)

The official Real-ESRGAN repository was pinned at `a4abfb2979a7bbff3f69f58f58ae324608821e27`,
the BasicSR RRDBNet source at `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a`, and
the official `v0.2.1` `RealESRGAN_x2plus.pth` checkpoint was recorded with
SHA-256 `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb`.
The dependency-light runner uses the official x2plus architecture without
altering the accepted ROCm environment. Input is the actual TAE RGB output;
only HWC/CHW and host/device conversion is performed.

On 32 real TAE frames, isolated FP16 RealESRGAN produced 768x1344 output in
`119.5 ms` warmed network time and `145.9 ms` to host-ready, versus `283.1`
and `300.2 ms` in FP32. Its peak isolated allocation was `1.53 GB` and the
learned output was visibly sharper than Lanczos around tree branches,
mountains, and water. Adjacent-frame statistics were close to the TAE/Lanczos
references (`0.05252` mean absolute difference, `0.09212` P95), with no gross
temporal instability visible in the bounded sample; the model necessarily
invents some plausible high-frequency texture.

The opt-in live mode displayed TAE RGB first, then ran FP16 RealESRGAN on that
frame before the exact clean-KV commit. In the co-resident LingBot process the
refiner network was `312.6 ms` P50 and frame-ready was `313.6 ms` P50, not the
isolated `119.5/145.9 ms`. At rolled state, base first RGB remained `927.8 ms`
P50, refined RGB presentation was `1297.9 ms`, clean KV was `293.1 ms`, and
next-ready was `1648.7 ms` P50. The run stayed finite through 20 actions and
ended with global KV `21168`, local capacity `12096`, and sink `6048`.

Classification: **REJECT FOR LIVE USE, KEEP OFFLINE OPTION**. The current
serial refinement adds about `357 ms` to next-action readiness even though the
base sub-second image is preserved. The new mode remains opt-in; no accepted
LingBot default, sampler, VAE, TAE, TunableOp, compiler, or cache behavior was
changed. Detailed measurements and artifact paths are in
[`spatial-upscale-20260910.md`](spatial-upscale-20260910.md); raw images/videos
and live metrics are under the ignored `results/raw/spatial-upscale-20260910/`
tree.

## F40 — official SPAN x2 is fast but does not clear the visual gate (2026-09-10)

The official SPAN repository was pinned at
`c77a5917759f09e66fbc7124220c5afc5ee221e5`. Its official `span.zip` checkpoint
archive contains `spanx2_ch48.pth`, whose extracted checkpoint SHA-256 is
`561fd5cf419a23d4de1231ce258180f61aee4aa8caa1aaaa783769c7301847bc`. The
official architecture is six SPAB blocks with 48 feature channels and native
x2 pixel-shuffle output; the strict `params_ema` state has 2,221,140
parameters. A dependency-light runner loads only the official architecture
file, bypassing incompatible legacy BasicSR imports and leaving the accepted
ROCm environment unchanged.

On the same 32 real TAE frames used for the RealESRGAN reference, SPAN FP16
produced 768x1344 frames in `21.0 ms` warm network P50 and `48.1 ms` to
frame-ready P50 (`23.9`/`56.3 ms` P95). Its peak isolated allocation was
`0.516 GB`, with `4.5 MB` of model-weight allocation. FP32 was `49.7 ms`
network and `67.8 ms` frame-ready P50. These are substantially below the
isolated RealESRGAN FP16 reference of `119.5`/`145.9 ms`.

The speed gate passed, but the visual gate did not. SPAN was broadly
Lanczos-like with mild sharpening rather than a clear scene-detail benefit;
the higher gradient statistic (`0.03229` versus Lanczos `0.02729`) is not
evidence of correct detail recovery. SPAN's adjacent-frame statistic was
`0.05481` mean / `0.09805` P95 versus Lanczos `0.05171` / `0.09547`, and no
gross temporal failure was visible in the bounded sample. Because no clear
visual benefit was established, no co-resident live run was justified and the
candidate's live slowdown/next-ready impact remain unmeasured.

Classification: **REJECT** for RC1. Do not add SPAN to the live viewer and stop
the learned-upscaler search for this release. The detailed report is
[`span-upscale-20260910.md`](span-upscale-20260910.md), compact provenance is
under `docs/artifacts/span-upscale-20260910/`, and raw frames/video/metrics are
under the ignored `results/raw/span-upscale-20260910/` tree. The next
authorized experiment, not started here, is the fixed-budget `sink_size 6 → 2`
test.

## F41 — Fixed 12-frame budget: six sink frames beat two sink frames (2026-09-10)

The accepted 384x672 / 1008-token frame / 12-frame local-cache path was run
with exactly one change: `sink_size=6` versus `sink_size=2`. Both lanes kept
the same 12,096-token physical and attended K budget. Runtime accounting
confirmed the expected layouts after rollover: six sink retained 6,048 sink +
5,040 recent + 1,008 current tokens; two sink retained 2,016 sink + 9,072
recent + 1,008 current tokens. Both completed the same 40-action traversal at
global KV position 41,328 with finite outputs and repeated rollover.

The candidate did not produce a latency benefit. Rolled first-visible was
`939.0 ms` P50 for six sink versus `957.7 ms` for two sink; next-ready was
`1289.7 ms` versus `1305.8 ms`. Clean-KV timing was effectively equal. The
result therefore does not represent an attention-size optimization.

Quality was the deciding result. Early frames matched, but divergence grew
after the first eviction. The two-sink capture showed late scene/layout
substitution and weaker shoreline/mountain/tree anchoring, with supporting
matched RGB MAD rising from `0.0119` in the first-rollover region to `0.1100`
late. Its adjacent-frame MAD also increased to `0.0378` mean / `0.1004` P95,
versus `0.0252` / `0.0370` for six sink. The aligned video and contact sheet
are in the ignored `results/raw/sink-budget-20260910/comparison/` directory;
compact evidence is in `docs/artifacts/sink-budget-20260910/`.

Classification: **REJECT — SINK HISTORY IS MATERIAL**. Keep `sink_size=6` as
the RC1 default and do not run a sink sweep. The detailed report is
[`sink-budget-20260910.md`](sink-budget-20260910.md).

## F42 — localhost browser serving (2026-09-10)

The frozen optimized 384x672 / 12-frame / 6-sink lane now has an opt-in
browser frontend at `scripts/run_browser.sh`. One model-owner context owns
GPU/model/TAEHV/KV/camera state; an independent CPU-only presentation worker
JPEG-encodes host RGB; aiohttp serves a plain HTML/CSS/JavaScript client over
an in-memory WebSocket. RGB0 is sent as soon as it is host-ready, while tails
are bounded and scheduled by the browser at a nominal 16 FPS. New action RGB0
invalidates stale tails.

The browser mailbox preserves the accepted one-slot latest-valid pending-input
policy. Action IDs and server/browser telemetry are retained in per-action
records. The exact clean t=0 KV transaction remains the barrier before a next
model action can begin. The launcher stays localhost-only and keeps the Tk
viewer available as a fallback. Static assets are served `no-store` so a fresh
launch cannot silently retain an older frontend after an RC update.

Protocol and implementation details are in
[`browser-serving-20260910.md`](browser-serving-20260910.md). Model-free
protocol/server smoke tests pass. The 40-action Strix run completed 29 rolled
actions with server-side action→RGB `930.6/936.5 ms` P50/P95 and
action→next-ready `1245.7/1252.2 ms` P50/P95, within normal variation of the
accepted optimized lane. The browser protocol client measured keydown→frame
received/decoded/presented at `936.6/938.3/938.3 ms` P50 and
`941.5/943.0/943.0 ms` P95. JPEG encode was `5.7/6.9 ms` P50/P95 and no
presentation tails were dropped. Final state was global KV `41328`, local
`12096`, sink `6048`, finite throughout. A real Chrome/CDP smoke also loaded
the page and proved actual key events plus latest-input replacement (`w`,
`d`, then `s`, with `d` replaced by `s`); it ended through `q` safely.

Classification: **RETAIN BROWSER AS RC1 DEFAULT UX**. The optimized model path
is unchanged; the browser is a localhost-only presentation/control layer.
Compact evidence is in
`docs/artifacts/browser-serving-20260910/summary.json`; full raw metrics are
under the ignored `results/raw/browser-serving-20260910/` tree.

## F43 — formal Strix Halo RC1 browser benchmark (2026-09-11)

The frozen RC1 browser lane was benchmarked with the established 40-action
script. The run completed 40 actions, including 29 rolled actions after the
12-frame local KV reached capacity. Rolled server-side action→base RGB was
`938.4 ms` P50 / `949.7 ms` P95; action→next-ready was `1255.3/1265.9 ms`
P50/P95. Denoise was `909.6/922.5 ms` P50/P95 and clean KV was
`298.3/299.9 ms`. These reproduce the qualification run's general range with
no browser-induced engine regression.

The browser-clock protocol measurements were keydown→frame received
`944.1/954.5 ms`, decoded `946.2/956.5 ms`, and presented proxy
`946.2/956.5 ms` P50/P95. JPEG encode was `6.9/9.4 ms`; packet size was
`45.9/66.3 KiB`; server-reported tail drops were zero. The presented proxy is
not a monitor or time-to-photon measurement. Browser telemetry does not yet
export per-run PyTorch peak allocation or RSS, so those fields are explicitly
null in the formal summary; the accepted same-stack reference remains
`30.049 GiB` allocated, `42.358 GiB` reserved, and `25.870 GiB` RSS.

The run remained finite and ended at global KV `41328`, local KV `12096`, and
sink `6048`. The existing matched-state action-response evidence is retained:
RGB0 was the first materially responsive frame for strong-turn and
forward-versus-reverse probes. A real Chrome/CDP smoke separately validated
action IDs, latest-valid pending replacement, and safe quit. Compact export
artifacts are in [`rc1-benchmark-20260911/`](artifacts/rc1-benchmark-20260911/)
and the detailed record is [`rc1-benchmark-20260911.md`](rc1-benchmark-20260911.md).

Classification: **RC1 BENCHMARK PASS**.
