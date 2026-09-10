# Pure state-free tensor-island compilation on gfx1151 — 2026-09-10

## Result

Retain the pure-helper compiler candidate as an **opt-in** path. It compiles
only state-free, non-GEMM tensor arithmetic around the causal transformer
block, while leaving attention, all Linear/GEMM modules, KV/cache state, and
rollover logic eager. A matched 40-action run was finite through repeated
rollovers and improved the rolled median first-visible latency by `27.607 ms`
(`970.854 → 943.247 ms`) over the exact validator-matched TunableOp-only
control. Next-action-ready improved by `37.389 ms` (`1329.256 → 1291.867 ms`).

This is an exact execution optimization apart from normal fused BF16 rounding;
it does not change the sampler, resolution, attention backend, renderer,
window, sink, clean-KV transaction, or pending-input policy.

## Repository and environment

At the start of the experiment, `main` and `origin/main` both pointed to
`9b5080bf1f98d8d20ff5f664c824bd856e96e906`. The protected working-tree change
to `scripts/run_window_sweep.sh` was left untouched and unstaged.

| Item | Value |
|---|---|
| GPU | AMD Radeon 8060S Graphics |
| Architecture | `gfx1151` |
| PyTorch | `2.13.0+rocm7.15.0a20260728` |
| HIP | `7.15.0` |
| TunableOp | enabled, online tuning disabled, untuned recording disabled |
| TunableOp results | 16 persisted entries |
| TunableOp validators | PT 2.13.0; HIP 715; hipBLASLt 100401-baa93758; gfx1151; rocBLAS 5.6.0.baa93758 |
| Model lane | 384x672, 1008 tokens/frame, 12-frame local window, 6 sink frames |
| Sampling | three denoise evaluations, `999 → 899 → 702` |
| Renderer | TAEHV `taew2_1`, FP16 |

The API audit found `torch.compile`, `torch.compiler.compile`, and
`torch.compiler.nested_compile_region`. The installed Inductor modes were
`default`, `lite`, `reduce-overhead`, `max-autotune-no-cudagraphs`, and
`max-autotune`. The candidate used only `backend="inductor"`, `mode="default"`,
and `fullgraph=True` for each pure helper.

## Compiled-region contract

The same `PureTensorIslands` bank was used for all 30 blocks. It contains:

```text
modulation
affine
scaled_residual (separate BF16 and FP32 wrappers)
add
silu
gelu_tanh
camera_update
```

These remain eager:

```text
self-attention and SDPA
cross-attention and SDPA
all Linear/GEMM calls, including Q/K/V/O, FFN, camera, and time projection
KV reads/writes and cache dictionaries
rollover/index decisions and in-place cache movement
```

The implementation is in
[`scripts/pure_compile_helpers.py`](../scripts/pure_compile_helpers.py), and
the opt-in live wrapper is
[`scripts/run_pure_compile_live.py`](../scripts/run_pure_compile_live.py).
The wrapper loads the existing TunableOp file before model construction and
restores the original block methods on exit.

## Compiler feasibility and cache behavior

The all-30 candidate completed a fresh process with 41 recorded actions
(bootstrap plus the requested 40-action script). The compiler summary was:

| Measurement | Result |
|---|---:|
| compiled blocks | 30 / 30 |
| compiled helper names | 7 |
| unique graphs | 8 |
| Dynamo calls captured | 23 |
| logged graph breaks | 0 |
| logged recompiles | 0 |
| helper installation setup | 0.183 s |
| Dynamo `_compile.compile_inner` total | about 1.921 s across 8 entries |
| first bootstrap transformer interval | 2715.5 ms |
| first post-bootstrap action transformer interval | 588.6 ms |

The first bootstrap interval includes lazy compilation and should not be
called an inference improvement. A second fresh 14-frame process saw the same
eight graph structure, `fxgraph_cache_hit=8`, and `0.150 s` helper setup; its
first bootstrap transformer interval was still `2652.0 ms`, so a populated
compiler cache did not eliminate all first-use startup cost. Once past
bootstrap, rolled actions did not show recurring graph recompilation as global
KV position advanced.

The `TORCH_LOGS=graph_breaks,recompiles` session log contains no graph-break or
recompile records. A scan of the compiler diagnostics and compile summary
found no `aten.addmm`, `aten.mm`, or `aten.bmm` records. Together with the
source-level eager boundary, this is the evidence that the pure graphs did not
capture the model's Linear/GEMM operations. The run did not emit per-signature
hipBLASLt runtime selections, so preservation is established by eager dispatch
and absence of captured GEMMs rather than by a verbose kernel-selection dump.

## Numerical helper check

[`scripts/check_pure_compile_helpers.py`](../scripts/check_pure_compile_helpers.py)
compared compiled and eager helpers on real gfx1151 tensors with the accepted
1008-token/1536-hidden BF16 shape contract. The output is retained at
`results/raw/compile-20260910/pure-helper-check/numerical.json`.

| Helper | Result |
|---|---|
| modulation | finite, max absolute error 0 |
| affine | finite, max absolute error `9.54e-7`, mean `2.26e-8` |
| scaled residual, BF16 and FP32 variants | finite, max absolute error `9.54e-7`, mean `1.24e-8` |
| add / SiLU / GELU-tanh | finite, max absolute error 0 |
| camera update, `x` output | finite, max absolute error `0.0625`, mean `0.00198` |
| camera update, hidden output | finite, max absolute error 0 |

The camera-update difference is a fused BF16 operation-order difference. Its
reference output range in this check was `-16.875..14.5`, so the maximum
absolute difference is at BF16 rounding scale; it is not a non-finite or
stateful discrepancy. The live long-run result below is the application-level
correctness check.

## Matched rolled performance

The control is the existing exact TunableOp-only 40-action run at
`results/raw/tunable-op-20260910/tuned-long-12/live_metrics.json`. The
candidate uses the same action string, prompt, seed, geometry, model, tuning
file, and live runner at
`results/raw/compile-20260910/pure-30-candidate-matched/live_metrics.json`.
P50/P95 use the 29 actions after the local window entered `rolled` state;
P95 is the nearest-rank 95th percentile. Product timing is lightly
instrumented runner timing, not compiler/profile timing.

| Rolled metric | Tunable eager | + pure compile | Delta |
|---|---:|---:|---:|
| 3x denoise P50 | 950.627 ms | 922.412 ms | **-28.215 ms** |
| 3x denoise P95 | 958.368 ms | 930.170 ms | **-28.198 ms** |
| first-visible P50 | 970.854 ms | 943.247 ms | **-27.607 ms** |
| first-visible P95 | 978.568 ms | 951.103 ms | **-27.465 ms** |
| clean KV P50 | 310.164 ms | 300.973 ms | **-9.191 ms** |
| clean KV P95 | 312.594 ms | 302.667 ms | **-9.927 ms** |
| next-ready P50 | 1329.256 ms | 1291.867 ms | **-37.389 ms** |
| next-ready P95 | 1339.962 ms | 1300.821 ms | **-39.141 ms** |

The candidate action count was 41 including bootstrap, with total elapsed
`229.042 s`, versus `225.634 s` for the control. This total includes startup
and is not the product latency comparison.

## Attribution

A separate candidate profile at rolled chunk 18 is retained at
`results/raw/compile-20260910/pure-profile-12/live_metrics.json`; the matched
TunableOp profile is
`results/raw/tunable-op-20260910/tuned-profile-12/live_metrics.json`.
These profiles are attribution evidence and have hook/event overhead.

| Denoise category, 3 forwards | Tunable profile | Compiled profile | Interpretation |
|---|---:|---:|---|
| self-attention module exclusive | 438.274 ms | 438.206 ms | unchanged |
| eager Linear exclusive | 322.846 ms | 323.445 ms | unchanged within run variance |
| attention probe self SDPA | 358.311 ms | 363.921 ms | no SDPA win; profile variance |
| block exclusive remainder | 112.037 ms | 87.024 ms | about 25 ms less |
| RMSNorm | 35.266 ms | 35.806 ms | unchanged |
| cross-attention | 20.070 ms | 20.278 ms | unchanged |
| LayerNorm | 11.480 ms | 11.840 ms | unchanged |
| GELU module | 10.278 ms | 0.036 ms | moved into pure helper |

For clean KV, self-attention was `142.751 → 144.482 ms`, eager Linear was
`107.974 → 108.363 ms`, and block exclusive remainder was `37.057 → 29.012
ms`; the GELU module was `3.467 → 0.012 ms`. Thus the observed whole-path
gain is consistent with fewer/faster state-free pointwise regions while
TunableOp GEMMs and fused SDPA remain the same mechanism. HIP kernel launch
count was not directly measured; the reliable compiler evidence is eight
stable graph artifacts and the module/profile attribution above.

## Correctness and persistence

The 40-action candidate was finite throughout. At the final rolled action:

```text
global KV position: 41,328 tokens
local KV capacity:  12,096 tokens
sink retained:       6,048 tokens
recent accounting:    5,040 tokens
current frame:        1,008 tokens
```

The three denoise forwards plus exact clean t=0 forward remained present in
every action. No compiled helper owns cache state, so the accepted clean-KV
transaction and pending-input policy were untouched. This run did not include
a captured video, so visual-quality claims beyond finite output, motion/state
continuity, and KV invariants are intentionally limited.

The 14-frame transfer regression at
`results/raw/compile-20260910/pure-14-regression/live_metrics.json` reused the
same compiled helper bank and did not retune. It completed 17 recorded actions
with finite output; its last rolled state was:

```text
global KV position: 17,136 tokens
local KV capacity:  14,112 tokens
sink retained:       6,048 tokens
rolled first-visible P50: 1006.247 ms (3 rolled observations)
rolled clean KV P50:       318.557 ms
rolled next-ready P50:    1371.096 ms
```

## Reproduction

The exact all-30 candidate command was:

```bash
ROOT="$PWD"
OUT="$ROOT/results/raw/compile-20260910/pure-30-candidate-matched"
PYTHONPATH="$ROOT/.upstream/lingbot-world-v2:$ROOT/scripts" \
TORCH_LOGS=graph_breaks,recompiles \
PYTHONUNBUFFERED=1 \
env -u PYTORCH_TUNABLEOP_ENABLED \
    -u PYTORCH_TUNABLEOP_TUNING \
    -u PYTORCH_TUNABLEOP_RECORD_UNTUNED \
    -u PYTORCH_TUNABLEOP_FILENAME \
    "$ROOT/.venv/bin/python" "$ROOT/scripts/run_pure_compile_live.py" \
    --pure-compile-block-count 30 \
    --tunableop-results "$ROOT/docs/artifacts/tunable-op-20260910/tunableop_results.csv" \
    --upstream-dir "$ROOT/.upstream/lingbot-world-v2" \
    --model-dir "$ROOT/models/lingbot-world-v2-1.3b-causal-fast" \
    --output-dir "$OUT" --size '480*832' --max-area-pixels 264192 \
    --frames 165 --max-actions 40 --denoise-schedule 3-drop-957 \
    --local-attn-size 12 --sink-size 6 --seed 42 \
    --prompt "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky, while the camera glides smoothly forward." \
    --image "$ROOT/.upstream/lingbot-world-v2/examples/03/image.jpg" \
    --action-path "$ROOT/.upstream/lingbot-world-v2/examples/03" \
    --vae-attention-backend math --vae-dtype fp16 \
    --display-decoder taehv --taehv-dir "$ROOT/.upstream/taehv" \
    --defer-clean-kv --max-seconds 900 \
    --scripted-actions 'w,w,d,w,a,s,l,l,w,j,s,d,w,w,d,w,a,s,l,l,w,d,s,a,w,j,s,d,w,w,d,w,a,s,l,l,w,d,s,a'
```

The helper-only numerical check is:

```bash
PYTHONPATH="$PWD/scripts" \
  "$PWD/.venv/bin/python" scripts/check_pure_compile_helpers.py \
  --output results/raw/compile-20260910/pure-helper-check/numerical.json
```

## Decision

**Retain as opt-in.** The candidate is stable enough and clears the requested
whole-path threshold, but its initial bootstrap remains substantially slower
because helper graphs compile lazily. Do not change the accepted default or
claim the compile setup as an inference speedup. The next experiment should be
chosen only after reviewing whether a cached/prewarmed opt-in launcher is
worth the startup cost; do not begin another compiler or kernel branch from
this result alone.
