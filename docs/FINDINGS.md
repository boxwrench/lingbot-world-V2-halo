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
