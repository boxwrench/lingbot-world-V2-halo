# Phase 0 independent cross-check — RC1 environment gate and warmed rolled waterfall

Date: 2026-09-12
Repo HEAD: `5f858c53d88f0462de7ff44aae8d3ea70e0b9be8` (one docs-only commit past
the formal RC1 benchmark commit `f70e4e00a9b784f171330a1c4f19a1f96de2dbb7`;
no code delta between them)

Status: **measurement only. No optimization implemented.**

This is a second, independently-derived pass over the same Phase 0
question, written before discovering that
[`docs/artifacts/phase0-20260912/README.md`](artifacts/phase0-20260912/README.md)
(commit `add8a69`, "Record gfx1151 RC1 Phase 0 baseline") already answers it
from the same raw runs. That document is the primary Phase 0 record; treat
this one as a corroborating cross-check with its own independent framing,
critical-path accounting, and opportunity-ranking method — the headline
numbers agree closely with it (see §6), which is itself useful evidence
that neither derivation made a boundary-selection mistake. One item this
pass surfaced that the primary record does not call out: `next_ready_ms` in
`scripts/run_browser.py` is set from the identical timestamp as
`all_rgb_ready_ms` (§3-4), so it is an alias, not an independently verified
"next action is legal" boundary.

This document answers the LingRuntime preflight's two immediate
recommendations — environment/GPU gate on `gfx1151` and one warmed RC1
`12+6` waterfall — and nothing else. It does not modify the RC1 generation
path.

## 0. What this is and is not

This is a Phase 0 baseline: verify the environment, resolve the running
runtime's actual arguments, and measure where a warmed rolled action's
latency goes. It does not implement attention, KV, fusion, graph-capture, or
ROCr changes, and it does not tune the system to reproduce the historical
number — the historical RC1 anchor is used only as a comparison point.

## 1. Environment/GPU gate

Gate result: **PASS**. Full raw dump is in
`results/raw/phase0-20260912/environment-gate/` (gitignored; paths/hashes
for the other raw runs are in "Required artifacts" below); the compact
manifest is
[`docs/artifacts/phase0-20260912/environment.json`](artifacts/phase0-20260912/environment.json).
Captured with the existing `scripts/env_report.sh` (unmodified) plus the
repo/upstream/model provenance already tracked by this project's other
dated reports.

```text
repo:      main @ 5f858c53d88f0462de7ff44aae8d3ea70e0b9be8
           pre-existing unrelated dirty file: scripts/run_window_sweep.sh
           (frame-count edit for the window-eviction sweep; not part of
           this Phase 0 work, left untouched)
upstream:  lingbot-world-v2 @ 45fa40673607c9acba6cf96a1f9396c95bcef25f
           patched: wan/image2video.py, wan/modules/model_fast.py
model:     robbyant/lingbot-world-v2-1.3b-causal-fast
           @ 7e36a5f919f86cb4255cc9bfc30adb44963fbde1

GPU:       AMD Radeon 8060S Graphics (device 0, the only CUDA/HIP device
           torch.cuda enumerates)
gfx:       gfx1151 (torch gcnArchName and rocminfo both agree)
APU:       AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
kernel:    6.17.0-35-generic
ROCm/HIP:  torch.version.hip 7.15.0; system hipconfig 7.2.53211;
           system rocm-libraries 7.2.2.70202-86~24.04
PyTorch:   2.13.0+rocm7.15.0a20260728
Triton:    3.8.0+git4cff872c.rocm7.15.0a20260728
BF16:      supported; bf16 matmul finite on-device
attention: PyTorch flash SDPA (aten::_scaled_dot_product_flash_attention),
           HIP kernel attn_fwd.kd (confirmed by the chunk-13 dispatch probe)
TunableOp: enabled, tuning disabled (frozen), 16 validator-matched results
           loaded from docs/artifacts/tunable-op-20260910/tunableop_results.csv;
           validators pin PT_VERSION 2.13.0 / HIP_VERSION 715 /
           HIPBLASLT_VERSION 100401-baa93758 / GCN_ARCH_NAME gfx1151 /
           ROCBLAS_VERSION 5.6.0.baa93758
decoder:   TAEHV taew2_1, FP16, weight sha256 d26151e7...c469c797e
memory:    120259084288 B unified GTT visible to the device (APU UMA, no
           discrete VRAM aperture); 130459455488 B system RAM
env vars:  PYTORCH_ROCM_ARCH=gfx1151, HIP_VISIBLE_DEVICES=0,
           CUDA_VISIBLE_DEVICES=0
power:     ROCm performance_level "auto" (no manual override observed)
```

Device detection was unambiguous: exactly one CUDA/HIP device is visible,
its name and `gcnArchName` both match `Radeon 8060S` / `gfx1151`, and
`HIP_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0` agree. There is no CPU or
alternate-GPU fallback in this run. **Gate answer: yes, execution is on
Radeon 8060S / gfx1151.**

## 2. Runtime arguments resolved from execution

Read from the loaded runtime and the rolled action-13 probe, not copied from
docs. Full manifest:
[`docs/artifacts/phase0-20260912/runtime-config.json`](artifacts/phase0-20260912/runtime-config.json).

```text
width x height:            672 x 384
session latent [f,h,w]:    15 x 48 x 84
accepted chunk latent:     [1, 16, 1, 48, 84]
chunk_size:                1
tokens/frame:               1008
local_attn_size:            12 frames
sink_size:                  6 frames
kv_capacity_tokens:         12096   (= (12+6) frames x 1008 tokens/frame;
                                       sink tokens are INSIDE this capacity,
                                       not additional to it)
sink_tokens:                6048
denoise timesteps:          999 -> 899 -> 702  (3 steps)
DiT blocks:                 30
attention heads:            12
head dimension:             128
DiT precision:               torch.bfloat16 (observed Q/K/V + autocast)
decoder:                     TAEHV taew2_1, torch.float16
attention backend:           aten::_scaled_dot_product_flash_attention /
                              attn_fwd.kd
serial execution:            yes (no overlap stream)
clean-KV:                    deferred exact t=0 full DiT forward, after
                              base-RGB presentation, before next-ready
```

The rolled chunk-13 self-attention dispatch was directly observed as
`Q [1,12,1008,128]`, `K/V [1,12,12096,128]`, all BF16, `is_causal=False`,
no mask — i.e. local capacity already includes sink tokens in the single
attended K/V tensor, confirmed by shape rather than assumed from the plan.

## 3-4. Warmed rolled waterfall (representative action)

Source run: `results/raw/phase0-20260912/waterfall-run/` (15 scripted
actions against the unmodified `scripts/run_browser.py` RC1 server; 4 rolled
actions past first eviction). The representative action (`action_id 15`) is
the rolled action whose base-RGB and next-ready times are jointly nearest
the pooled A/A medians from §5, so it is not cherry-picked for a favorable
number. Full record:
[`docs/artifacts/phase0-20260912/waterfall.json`](artifacts/phase0-20260912/waterfall.json).

Server-side boundaries are `perf_counter()` timestamps taken immediately
after a `torch.cuda.synchronize()` at each transition (an existing,
already-committed instrumentation pattern in `wan/image2video.py`, not
newly added for this task), so each interval below is genuine GPU-inclusive
elapsed time, not an unsynchronized CPU-side guess. **Profiling was
disabled for every number in this section** — the module/attention hooks
used in §7's attribution profile were not active during this run.

### Wall-clock waterfall

All timestamps below are cumulative milliseconds since `input` (t=0); the
`+` value on each line is that step's own duration.

```text
input                                                          t=0.00 ms
  +0.14 ms  input selection                        -> t=0.14 ms
  +2.70 ms  action/camera preparation               -> t=2.85 ms   [generation start]
  +314.33 ms  denoise 1 (t=999)                     -> t=317.17 ms
  +297.35 ms  denoise 2 (t=899)                     -> t=614.53 ms
  +298.67 ms  denoise 3 (t=702)                     -> t=913.19 ms
  +6.76 ms   latent postprocess + accept x0         -> t=919.95 ms   [accepted x0]
  +21.21 ms  TAEHV first-RGB decode + host copy     -> t=941.17 ms   [BASE RGB READY]
  +0.20 ms   gap before clean-KV starts             -> t=941.37 ms
  +300.61 ms exact clean-KV forward (t=0)           -> t=1241.97 ms  [clean KV complete]
  +19.35 ms  TAEHV remaining decode + copies        -> t=1261.32 ms  [NEXT-READY]
```

The three denoise `elapsed_ms` values sum to 910.35 ms; `transformer_ms` for
the whole denoise phase was measured at 915.14 ms (the 4.79 ms difference is
per-forward Python/dispatch time between the three
`torch.cuda.synchronize()` points, not a fourth forward). The two totals
that matter — 941.17 ms to base RGB and 1261.32 ms to next-ready — are each
read directly from the server ledger (`input_to_base_rgb` /
`input_to_next_ready` in `waterfall.json`), not summed by hand from the
steps above, so they are not exposed to accumulated rounding error.

### Critical-path analysis

```text
strictly serialized before base RGB:
  input selection, action preparation, denoise 1, denoise 2, denoise 3
  (each denoise forward depends on the previous accepted/renoised latent —
  no reordering is possible), TAEHV first-RGB decode, first host copy

strictly serialized before next-ready (in addition to the above):
  exact t=0 clean-KV DiT forward (depends on accepted x0, must complete
  before the persistent KV state is legal for the next action)
  TAEHV remaining-frame decode and copies

overlapping / asynchronous:
  CPU JPEG encoding and WebSocket transport of the base frame run on a
  separate presentation worker after base-frame submission; they do not
  block clean-KV or next-ready

presentation-only (not generation-state critical):
  JPEG encoding, WebSocket transport, browser decode/paint

generation-state critical:
  all three denoise forwards, the clean-KV forward, and the KV
  read/write/roll bookkeeping folded into each forward
```

**Caveat on `next_ready_ms`:** in `scripts/run_browser.py` this field is set
from the same timestamp as `all_rgb_ready_ms` (both are stamped by the same
`self.now_ms()` call right after the TAEHV remaining-frame decode/copy
loop). It marks "all RGB for this action has been produced and copied to
host," not a separately verified "next keypress is now legal" gate. In this
runtime the two happen to coincide because the clean-KV commit finishes
before the remaining-frame decode loop starts, so the alias is not
misleading here, but it should not be read as an independently instrumented
permission boundary.

Base-RGB latency (941.17 ms) is on the critical path in full: three
serialized DiT forwards (910.35 ms, 96.7% of it) plus a ~21 ms decode/copy
tail. Next-ready adds the clean-KV forward and its own short decode tail
(320 ms, 25.4% of the 1261.32 ms total) — this is required compute that
was already deliberately moved off the base-RGB path by an earlier
optimization (`docs/interactive-latency-history-20260910.md` §8); it cannot
be summed with base-RGB latency as if it were still blocking presentation,
and it is not being proposed as a new optimization target here. Nothing in
this waterfall was double-counted: the three denoise intervals, the clean-KV
interval, and the two decode/copy gaps are disjoint, sequential wall-clock
spans that sum exactly to the two reported action-level totals.

## 5. Small A/A noise check

Three independent fresh rollouts, 15 scripted actions each (12 total rolled
actions, 4 per rollout), unprofiled. Full rows:
[`docs/artifacts/phase0-20260912/timings.jsonl`](artifacts/phase0-20260912/timings.jsonl);
summary:
[`docs/artifacts/phase0-20260912/metrics-summary.json`](artifacts/phase0-20260912/metrics-summary.json).

| Metric | Median | Range (min-max) |
|---|---:|---:|
| action -> base RGB | 939.9 ms | 927.6 - 954.5 ms |
| action -> next-ready | 1258.9 ms | 1244.8 - 1272.3 ms |
| denoise (3x) | 909.8 ms | 899.5 - 926.5 ms |
| transformer total | 913.5 ms | 903.4 - 930.5 ms |
| clean KV | 298.2 ms | 295.1 - 304.5 ms |
| TAEHV first-RGB (GPU) | 20.0 ms | 19.6 - 20.2 ms |
| TAEHV remaining (GPU) | 17.8 ms | 17.6 - 18.6 ms |
| TAEHV total (GPU) | 37.8 ms | 37.5 - 38.8 ms |

Per-rollout rolled P50/P95 (4 actions each) land within about 9 ms of each
other across all three rollouts for both headline metrics, so a same-config
delta smaller than roughly 15-20 ms should be treated as noise rather than a
real effect at this sample size. This is a small-N noise estimate, not a
formal confidence interval — it is sufficient to judge the comparison in
§6, not to accept or reject a future optimization on its own.

## 6. Comparison to historical RC1

| | Historical RC1 (P50, `f70e4e0`) | New measurement |
|---|---:|---:|
| action -> base RGB | 938.4 ms | 939.9 ms median (927.6-954.5 ms pooled); 941.2 ms representative |
| action -> next-ready | 1255.3 ms | 1258.9 ms median (1244.8-1272.3 ms pooled); 1261.3 ms representative |

Delta vs. historical P50: **+1.5 ms (0.16%) base RGB, +3.6 ms (0.29%)
next-ready** — both well inside the ±15-20 ms A/A noise band measured in
§5.

**Classification: reproduced within expected variance.** No environment or
runtime difference was identified: HEAD is one docs-only commit past the
commit that produced the historical number, upstream/model revisions and
the entire resolved runtime configuration in §2 match the frozen RC1
description, and the delta is smaller than the measured noise floor. There
is no unexplained difference to investigate.

## 7. Ranked opportunities (measured, not implemented)

Absolute-ms attribution below comes from a separate **intrusive**
module/attention-hook profile of one matched rolled action (chunk 13, same
12+6 rolled regime, `docs/artifacts/phase0-20260912/profile-summary.json`).
That run's own wall time (944.7 ms for denoise, 311.5 ms for clean) is
inflated by hook overhead versus the unprofiled §3-6 numbers and **is used
here only for relative attribution, never as an acceptance latency**. Ranked
ms below are the measured relative share of each class applied to the
unprofiled §5 median transformer/clean-KV totals (913.5 ms / 298.2 ms), so
the "measured critical-path ms" column is an attribution estimate, not a
second, independent stopwatch measurement.

| Component | Measured critical-path ms (attribution) | Recoverable fraction estimate | Confidence | Candidate mechanism |
| --- | ---: | ---: | --- | --- |
| Self-attention (fused SDPA + QKV/O overhead) | ~577 ms combined (denoise 446.1/931.9 x 913.5 + clean 144.8/308.8 x 298.2 ms) | low-moderate, and only on the non-kernel share; the fused SDPA kernel itself (`attn_fwd.kd`) is already ~40% of total DiT time and has no demonstrated faster gfx1151 backend | low | The raw `aten::_scaled_dot_product_flash_attention` call accounts for ~479 ms of that ~577 ms (83%); the remaining ~98 ms is QKV/output-projection and per-block Python/dispatch overhead (the separate `CausalWanAttentionBlock` wrapper class totals ~113 ms across all forwards). That ~98 ms non-kernel remainder, not the kernel itself, is where a launch/layout intervention could plausibly land — a minority of it, not the full ~577 ms attention total. |
| Linear (projections/MLP/modulation, GEMM-bound) | ~429 ms combined (denoise 327.7/931.9 x 913.5 + clean 111.6/308.8 x 298.2 ms) | unknown until checked | low | Only 16 TunableOp results are loaded and pinned to this exact software stack (`docs/artifacts/tunable-op-20260910/tunableop_results.csv`), but that set was validated against the earlier 18+6 lane's GEMM shapes, not confirmed shape-for-shape against the 12+6 RC1 lane measured here. Cross-checking dispatched shapes against tuned coverage is a source-inspection task, not an optimization, and would tell us whether any GEMM is silently running an untuned/default kernel. |
| Exact t=0 clean-KV forward | 298.2 ms (24% of next-ready) | none identified | n/a — not an opportunity | Already investigated and rejected: it is a required full clean-latent transformer pass for persistent-cache correctness, and reusing the prior noisy-latent forward's KV was already tried and rejected (`docs/interactive-latency-history-20260910.md` §13). Listed here only to show why it is excluded from the two candidates above, not as a third recoverable item. |
| TAEHV decode + host-copy tail (base-RGB + remaining) | ~38 ms total (20.0 + 17.8 ms GPU, §5 median) | very low in absolute terms | high (small number, well measured) | Already the outcome of a prior large optimization (canonical VAE -> TAEHV); at ~3% of next-ready total there is little left to recover here regardless of mechanism. |

**Top two independent, trace-supported opportunities and plausible
recovery:**

1. **Non-kernel self-attention/attention-block overhead** (QKV/output
   projection reshape/dispatch and the `CausalWanAttentionBlock` wrapper,
   distinct from the fused SDPA kernel itself) — bounded above by the ~98 ms
   non-kernel remainder inside the ~577 ms self-attention attribution, so
   plausibly a minority of that (order 10-30 ms per warmed rolled action),
   not the full ~577 ms. The dominant ~479 ms SDPA kernel call is not itself
   a target (no faster supported gfx1151 backend is known; CK SDPA was
   already confirmed unavailable). Confidence: low, because the profiling
   run that attributes this cost is intrusive and its own overhead has not
   been separately isolated from the real non-kernel cost.
2. **TunableOp GEMM coverage for the actual 12+6 RC1 dispatched shapes** —
   currently unquantified; the recoverable ms depend entirely on whether any
   recurring shape in this lane is running without a tuned kernel, which has
   not yet been checked shape-by-shape. If none are missing, recoverable ms
   is ~0; if some are missing, the ceiling is bounded by that GEMM's share of
   the ~429 ms Linear attribution above. Confidence: low until the shape
   cross-check is done.

No custom attention, ring KV, fusion, graph capture, PM4, or custom ROCr
work was started or is being proposed by this document; the next step for
either candidate above is further **measurement** (shape/coverage
inspection, then a small isolated timing check), not implementation.

## Answers to the exit condition

1. **Is the RC1 environment correctly reproduced on gfx1151?** Yes — gate
   PASS, single unambiguous Radeon 8060S/gfx1151 device, matching
   HIP/PyTorch/Triton/TunableOp stack and matching repo/upstream/model
   revisions (§1).
2. **Current warmed rolled base-RGB latency?** ~940 ms median (927.6-954.5
   ms pooled range, n=12 rolled actions); 941.2 ms for the single
   representative waterfall action (§3, §5).
3. **Current warmed rolled next-ready latency?** ~1259 ms median
   (1244.8-1272.3 ms pooled range); 1261.3 ms for the representative
   action (§3, §5).
4. **Where is that time spent?** ~73% (910 ms) in the three serialized DiT
   denoise forwards, ~2% (21 ms) in TAEHV first-RGB decode/copy to base RGB,
   then ~24% (301 ms) in the mandatory exact clean-KV forward and ~2%
   (19 ms) in the remaining TAEHV decode/copies to next-ready (§3-4). Within
   the DiT forwards, attribution profiling assigns roughly 48% to
   self-attention (of which the fused SDPA kernel itself is ~83%) and 35%
   to linear/GEMM projections (§7).
5-6. **Top two measured optimization opportunities and plausible ms?** See
   the ranked table above: (1) non-kernel self-attention/attention-block
   overhead, plausibly ~10-30 ms/action (bounded by the ~98 ms non-kernel
   remainder), low confidence; (2) TunableOp GEMM shape-coverage gap for the
   12+6 lane, plausibly 0 ms up to a bound set by the ~429 ms Linear
   attribution, low confidence until the shape check is done. Neither has
   been implemented.

## Reproduction commands and artifacts

The authoritative commands are recorded in
[`docs/artifacts/phase0-20260912/commands.md`](artifacts/phase0-20260912/commands.md)
(actual `env`-var-driven invocations of `scripts/run_browser.sh`,
`scripts/browser_benchmark.py`, and `scripts/run_pure_compile_live.py` for
the intrusive profile) — that supersedes any reconstruction attempted here.
The full artifact set (`environment.json`, `runtime-config.json`,
`timings.jsonl`, `waterfall.json`, `metrics-summary.json`,
`profile-summary.json`, `raw-artifacts.json`) and the `build_phase0_evidence.py`
generator are indexed in
[`docs/artifacts/phase0-20260912/README.md`](artifacts/phase0-20260912/README.md).
This cross-check reads those same committed files; it adds no new raw runs,
scripts, or artifacts of its own.

## Repository state

This document is the only change made in the course of writing this
cross-check. It does not modify `scripts/run_browser.py`, add
`scripts/build_phase0_evidence.py`, or add anything under
`docs/artifacts/phase0-20260912/` — all of that was already committed
(`add8a69`, "Record gfx1151 RC1 Phase 0 baseline") before this cross-check
was finished, from the same underlying raw runs this document analyzes.

`scripts/run_window_sweep.sh` still carries a pre-existing, unrelated dirty
diff (frame count for the window-eviction sweep) that predates this work
and was already flagged as pre-existing in the environment gate in both
this document and the committed one (§1); it remains untouched.
