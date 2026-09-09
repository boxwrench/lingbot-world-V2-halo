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

## F4 — next control: isolated real VAE

The synthetic control is decisive enough to justify the next bounded test:
decode a 480×832-equivalent latent with the real Wan2.1 VAE, instrument every
Conv3d call, and separate first/repeated decode time. No LingBot kernel or VAE
optimization is introduced until that measurement is complete.

