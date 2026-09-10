# PyTorch ROCm TunableOp on Strix Halo — 2026-09-10

## Result

PyTorch TunableOp successfully captured the actual LingBot 1.3B 12-frame
workload, tuned the captured GEMMs offline, and loaded the validator-checked
results in a fresh process with online tuning disabled. On the warmed rolled
12-frame path it reduced median first-visible latency by **62.9 ms** and
median next-action-ready latency by **89.3 ms**. The accepted model defaults
are unchanged; this is an opt-in, machine/software-version-specific result.

The primary mechanism is GEMM dispatch, not attention: the profiled fused
self-attention time was effectively unchanged, while the largest observed
linear improvement was `time_projection.1`.

## Environment

| Item | Measured value |
|---|---|
| Device | AMD Radeon 8060S Graphics |
| GFX target | `gfx1151` |
| Device-visible memory | 120,259,084,288 bytes (`114688 MB` reported by PyTorch) |
| PyTorch | `2.13.0+rocm7.15.0a20260728` |
| HIP | `7.15.0` |
| TunableOp validators | PT 2.13.0; HIP 715; hipBLASLt `100401-baa93758`; rocBLAS `5.6.0.baa93758` |
| `torch.cuda.is_available()` | `True` |
| GPU count | 1 |

The validator values are the compatibility identity for the tuning file. A
separate `rocminfo` run identified the GPU agent as `gfx1151`, with marketing
name `AMD Radeon Graphics`, and the CPU/APU as `AMD RYZEN AI MAX+ 395 w/
Radeon 8060S`. The contemporaneous `rocm-smi` snapshot reported a 120,259,084,288
byte GTT total and 48,109,821,952 bytes used; this is an UMA/GTT observability
value, not a claim that all of it belongs to this process.

## Configuration held constant

All model runs used the accepted path:

```text
384×672, 1008 tokens/frame
chunk_size=1, local_attn_size=12, sink_size=6
three denoise forwards: 999 → 899 → 702
exact deferred clean t=0 KV pass
BF16 DiT, PyTorch fused SDPA
TAEHV taew2_1 FP16 presentation
serial execution
```

The control used ordinary PyTorch dispatch. The candidate enabled TunableOp,
loaded the persisted file below, disabled tuning, and disabled untuned
recording before LingBot initialization.

## Phase 1 — coverage

Collection used the real live path through bootstrap, full occupancy, and
multiple rollovers. The collection process had tuning disabled and untuned
recording enabled:

```bash
ROOT="$PWD"
OUT="$ROOT/results/raw/tunable-op-20260910/collection-12"
mkdir -p "$OUT"
(
  cd "$OUT"
  PYTORCH_TUNABLEOP_ENABLED=1 \
  PYTORCH_TUNABLEOP_TUNING=0 \
  PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 \
  LINGBOT_LIVE_MAX_ACTIONS=16 \
  LINGBOT_LIVE_TIMEOUT_SECONDS=900 \
  LINGBOT_LIVE_OUTPUT_DIR="$OUT" \
  bash "$ROOT/scripts/run_live.sh" \
    --local-attn-size 12 --max-actions 16 \
    --scripted-actions 'w,w,d,w,a,s,l,l,w,d,s,a,w,j,s,d'
)
```

PyTorch emitted `tunableop_untuned0.csv` in the collection
working directory. It contains 16 unique signatures (the format de-duplicates
signatures; it does not contain per-call counts).

Captured workload classes included:

* BF16 MLP projections: `1536↔8960` with 1008-token current-frame inputs.
* BF16 same-width projections used by Q/K/V, output, and conditioning paths.
* BF16 text/conditioning projections involving 512 and 4096 dimensions.
* Float `time_projection.1`: `9216×1008×1536`.
* BF16 batched attention-related GEMMs and the float batched auxiliary path.

The full collection is preserved as
[`tunableop_untuned0.csv`](artifacts/tunable-op-20260910/tunableop_untuned0.csv).

## Phase 2 — offline tuning

The 16 collected signatures were tuned on the same GPU in a separate process;
the setup took 125.314 seconds. This setup cost is not included in inference
latency. The tuning file contains five validator lines and 16 selected
entries: 13 selected hipBLASLt implementations and three retained `Default`
implementations.

The reproducible helper is
[`tune_gemm_offline.py`](../scripts/tune_gemm_offline.py):

```bash
.venv/bin/python scripts/tune_gemm_offline.py \
  --untuned results/raw/tunable-op-20260910/collection-12/tunableop_untuned0.csv \
  --results results/raw/tunable-op-20260910/tuning-12/tunableop_results.csv
```

The exact validator-bound result file is also tracked in
[`tunableop_results.csv`](artifacts/tunable-op-20260910/tunableop_results.csv).
The per-entry times in that CSV are TunableOp's offline candidate measurements,
not end-to-end application timings.

## Phase 3 — warmed control/candidate

Each lane used a fresh process and the same deterministic 40-action sequence
(`w,w,d,w,a,s,l,l,w,d,s,a,w,j,s,d` repeated as needed). No video capture was
enabled. The table uses the 29 rolled actions from bootstrap + 40 user actions;
P95 is the inclusive 95th percentile of those rolled rows.

| Rolled metric | Ordinary dispatch | TunableOp file | Tuned − control |
|---|---:|---:|---:|
| 3× denoise median | 1013.039 ms | 950.627 ms | **−62.412 ms** |
| first-visible median | 1033.762 ms | 970.854 ms | **−62.908 ms** |
| clean KV median | 336.675 ms | 310.164 ms | **−26.511 ms** |
| next-action-ready median | 1418.512 ms | 1329.256 ms | **−89.256 ms** |
| first-visible P95 | 1041.984 ms | 978.439 ms | **−63.545 ms** |
| next-action-ready P95 | 1427.741 ms | 1338.038 ms | **−89.703 ms** |

The tuned 12-frame lane completed all 41 rows finitely, reached global
position 41,328 tokens, and held local K at 12,096 tokens with 6,048 sink
tokens retained. The ordinary lane had the same finite/cache result.

## Runtime dispatch evidence and profile

The tuned live wrapper printed, before model construction:

```text
enabled=True, tuning_enabled=False, record_untuned=False
loaded_results=16
validators=PT 2.13.0 / HIP 715 / hipBLASLt 100401-baa93758 /
            gfx1151 / rocBLAS 5.6.0.baa93758
```

The result file records the selected implementation for every captured
signature. Representative selected entries are:

| Workload signature | Selected solution |
|---|---|
| float `tn_9216_1008_1536` with bias | `Gemm_Hipblaslt_2875` |
| BF16 `tn_8960_1008_1536` with bias | `Gemm_Hipblaslt_1179` |
| BF16 `tn_1536_1008_8960` with bias | `Gemm_Hipblaslt_860` |
| BF16 `tn_4096_512_10240` | `Gemm_Hipblaslt_1164` |
| BF16 `tn_10240_512_4096` | `Gemm_Hipblaslt_862` |
| BF16 `tn_1536_1008_1536` with bias | `Default` |

One detailed chunk-12 CUDA-event profile was collected per lane. These module
events are a linear/GEMM timing proxy, not a direct per-kernel rocBLAS timer:

| Profiled exclusive linear work | Ordinary | TunableOp |
|---|---:|---:|
| 3 denoise linear modules | 402.511 ms | 322.846 ms |
| clean-KV linear modules | 132.756 ms | 107.974 ms |
| `time_projection.1`, denoise ×3 | 78.925 ms | 10.953 ms |
| MLP projections, denoise ×3 | 153.571 ms | 142.922 ms |

The same profile measured self-attention SDPA at 357.335 ms control versus
358.311 ms tuned for the three denoise passes, and 120.039 versus 118.733 ms
for clean KV. This shows the observed product gain is from GEMM dispatch,
especially the float time projection and smaller MLP changes, rather than a
hidden attention change.

The complete raw process outputs are intentionally ignored under
`results/raw/` because they are large. On the experiment host they are:

```text
results/raw/tunable-op-20260910/collection-12/
results/raw/tunable-op-20260910/tuning-12/
results/raw/tunable-op-20260910/control-long-12/
results/raw/tunable-op-20260910/tuned-long-12/
results/raw/tunable-op-20260910/control-profile-12/
results/raw/tunable-op-20260910/tuned-profile-12/
results/raw/tunable-op-20260910/tuned-14/
```

A compact machine-readable summary is in
[`metrics-summary.json`](artifacts/tunable-op-20260910/metrics-summary.json).

## 14-frame transfer

The same 12-frame tuning file loaded without retuning for the small 14-frame
regression. It remained finite through the 14,112-token local cap and three
rolled actions. Its rolled medians were 1,034.64 ms first-visible and
1,413.88 ms next-ready. This verifies file applicability to the higher-quality
mode; it was not a same-process control/candidate comparison and is not used
as a claimed speedup.

## Correctness and memory

No model math, sampler, attention backend, cache size, renderer, or pending
input policy was changed. Both long lanes were finite, retained the 6,048
sink tokens, held the 12,096-token physical capacity, and advanced global KV
position to 41,328. The tuned 14-frame transfer also remained finite. No
visual/state regression was observed in the bounded traversal metrics.

The long-run tuned lane observed approximately:

```text
PyTorch allocation at final row: 18.011 GB
PyTorch reserved at final row:   18.635 GB
PyTorch peak allocation:         30.425 GB
PyTorch peak reserved:           40.836 GB
process RSS:                     3.572 GB
PyTorch device-visible total:   120.259 GB
```

These values are UMA observations, not a claim that the entire device-visible
pool is physical VRAM or dedicated to this process.

## Decision

**Retain as an opt-in optimization.** The result exceeds the acceptance
threshold by a wide margin, is repeatable over 29 rolled samples, survives a
41-row persistent run, and is loadable in a fresh process with tuning disabled.
Do not make it the repository's implicit default until deployment ergonomics
are decided; users must keep the result file matched to the validator stack.

The protected `scripts/run_window_sweep.sh` edit was not staged or changed.

## Next one recommendation

The next justified experiment is block-local `torch.compile`/fusion profiling
on the transformer with TunableOp held constant, focused on the remaining
context-independent linear/modulation work. It should be a separate opt-in
lane; the current self-attention path remains the correctness and performance
reference.
