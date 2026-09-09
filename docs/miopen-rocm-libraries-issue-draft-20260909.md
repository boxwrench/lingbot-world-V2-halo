# Draft issue: gfx1201 Conv3d GEMM solver rejected by old MIOpen workspace gate

Target tracker: [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries/issues)

## Title

`[MIOpen] gfx1201 Conv3d GemmFwdRest rejected by workspace check on 3.5.1, same GPU succeeds on 3.5.2`

## Body

### Summary

On the same Radeon AI PRO R9700 (`gfx1201`), MIOpen 3.5.1 rejects
`GemmFwdRest` for the production-shaped FP32 Conv3d below because its
workspace requirement exceeds the available-workspace ceiling.  The fallback
is `ConvDirectNaiveConvFwd`, which is approximately 29--33x slower than the
GEMM implementation.  The same GPU and operation select `GemmFwdRest` and run
normally with MIOpen 3.5.2 under ROCm 7.14.

This is not an architecture-only limitation.  The exact old/new same-GPU
control is the main reason for reporting it separately from any LingBot issue.

### Reproducer

The following is the complete operation.  Inputs and weights are synthetic,
contiguous, FP32 tensors; stride, dilation, and groups are all 1; bias is
present with 96 elements; every measurement synchronizes before and after the
operation.

```python
import torch
import torch.nn.functional as F
import time

torch.cuda.set_device(0)
x = torch.randn(1, 96, 6, 482, 834, device="cuda", dtype=torch.float32)
w = torch.randn(96, 96, 3, 3, 3, device="cuda", dtype=torch.float32)
b = torch.randn(96, device="cuda", dtype=torch.float32)
for _ in range(2):
    y = F.conv3d(x, w, b, padding=0)
torch.cuda.synchronize()
t0 = time.perf_counter()
y = F.conv3d(x, w, b, padding=0)
torch.cuda.synchronize()
print(time.perf_counter() - t0, y.shape)
```

The matched four-case harness and raw JSON are available in the companion
experiment repository:

```text
https://github.com/boxwrench/lingbot-world-V2-halo
scripts/conv3d_micro.py
```

### Old stack: gfx1201 / ROCm 7.2.1 / MIOpen 3.5.1

```text
GPU:      Radeon AI PRO R9700, gfx1201
PyTorch:  2.9.1+rocm7.2.1.gitff65f5bc
HIP:      7.2.53211-e1a6bc5663
MIOpen:   3.5.1.dabb6df2b9
```

For B:

```text
input  [1,96,6,482,834]
weight [96,96,3,3,3]
output [1,96,4,480,832]

GemmFwdRest workspace: 16,562,257,920 bytes
available workspace:   14,574,367,538 bytes
MIOpen result:          GemmFwdRest -> Not applicable
fallback:               ConvDirectNaiveConvFwd
warm median:            5.032319 s
throughput:             0.157976 TFLOP/s
```

With `MIOPEN_FIND_ENFORCE=1` and `MIOPEN_FIND_MODE=NORMAL`, the candidate
logging records the workspace rejection followed by `Not applicable`; the
direct naive kernel is the only successful candidate.  The same pattern is
seen for the RGB-output and padding controls:

| Case | Input | Weight | Output | Warm | Effective |
|---|---|---|---|---:|---:|
| A | `[1,96,4,480,832]` | `[96,96,3,3,3]` | `[1,96,2,478,830]` | 0.144361 s | 2.735 TFLOP/s |
| B | `[1,96,6,482,834]` | `[96,96,3,3,3]` | `[1,96,4,480,832]` | 5.032319 s | 0.158 TFLOP/s |
| C | `[1,96,6,482,834]` | `[3,96,3,3,3]` | `[1,3,4,480,832]` | 3.147488 s | 0.008 TFLOP/s |
| D | `[1,96,4,480,832]`, padding=1 | `[96,96,3,3,3]` | `[1,96,4,480,832]` | 5.437341 s | 0.146 TFLOP/s |

`GemmFwdRest` is present and applicable for A on this old stack, so the
solver is not absent from the build.  Forcing it with
`MIOPEN_DEBUG_FIND_ONLY_SOLVER=GemmFwdRest` does not make B applicable.  A
direct `MIOpenDriver --solution GemmFwdRest` invocation reaches the solver but
returns:

```text
The supplied solution id: GemmFwdRest is not applicable to the current problem
```

`MIOPEN_FIND_MODE=FAST` also does not resolve the old-stack B/C/D selection;
it leaves the old direct-naive result rather than making the oversized GEMM
workspace applicable.

The ROCm 7.2.1 installation has no matching shipped gfx1201 MIOpen DB.  The
old process used this user DB and kernel cache:

```text
/home/boxwrench/.config/miopen/gfx1201_32.HIP.3_5_1_dabb6df2b9.ufdb.txt
/home/boxwrench/.cache/miopen/3.5.1.dabb6df2b9/gfx1201_32.ukdb
```

### Newer stack: same gfx1201 GPU

```text
GPU:      Radeon AI PRO R9700, gfx1201
PyTorch:  2.12.0+rocm7.14.0
HIP:      7.14.60850
MIOpen:   3.5.2.cd957402
```

Using a separate environment and isolated user FindDb, the same four cases
select `GemmFwdRest` with the same B/C/D workspace requirement:

| Case | Warm median | Effective |
|---|---:|---:|
| A | 0.083286 s | 4.741 TFLOP/s |
| B | 0.171255 s | 4.642 TFLOP/s |
| C | 0.124190 s | 0.200 TFLOP/s |
| D | 0.165354 s | 4.808 TFLOP/s |

The newer user DB records `GemmFwdRest` for all four cases with the
`miopenConvolutionFwdAlgoGEMM` backend.  This is a bounded software control;
the environments were not merged and the ROCm 7.2.1 result is preserved.

### Cross-architecture control

On a Radeon 8060S (`gfx1151`) with PyTorch
`2.13.0+rocm7.15.0a20260728` and active MIOpen
`3.6.0.baa93758`, all four matched cases select:

```text
GemmFwdRest -> Im3d2Col -> rocBLAS
```

The production B case measures 0.504324 s / 1.576 TFLOP/s.  This supports a
solver/backend or stack interaction rather than a general Conv3d limitation.

### Questions

1. Why does MIOpen 3.5.1 report `GemmFwdRest` as not applicable when its
   required workspace is 16,562,257,920 bytes and the reported ceiling is
   14,574,367,538 bytes, while 3.5.2 accepts the same solver and workspace on
   the same gfx1201 GPU?
2. Was the workspace accounting, solver applicability test, or gfx1201
   database/heuristic changed between these builds?
3. Is there a supported way to select an equivalent lower-workspace GEMM or
   im2col implementation for this shape on the old build?
4. Can the relevant solver selection/rejection reason be exposed more clearly
   in MIOpen logging?

Related reports include [MIOpen #3957](https://github.com/ROCm/MIOpen/issues/3957)
and [rocm-libraries #4071](https://github.com/ROCm/rocm-libraries/issues/4071),
but this report differs by providing an exact old/new same-GPU control and the
observed solver records.

### Attachments

I can provide the complete MIOpen find logs, user FindDb files, and JSON timing
records if useful.  The old-stack log specifically contains the B/C/D
`GemmFwdRest` workspace rejection and the final direct-naive selection.
