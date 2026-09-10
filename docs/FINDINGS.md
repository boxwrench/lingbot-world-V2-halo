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
