# C8 Probe 0/1: executing-backend identity + single-call bakeoff (2026-09-13)

Branch `experiment/c8-attention` (from tagged main `9cc986c`). Production
venv `models` host `.venv`; upstream `.upstream/lingbot-world-v2 @ 45fa406`;
torch 2.13.0+rocm7.15.0a20260728; GPU idle, ~50 C. Exact production shape:
Q=[1,12,1008,128], K=V=[1,12,12096,128], BF16, no mask, non-causal,
dropout 0, scale 1/sqrt(128).

## Probe 0 — backend identity (runtime observation, not assumption)

- `flash_attn` package present but has **no compiled extension** in either
  venv (`import flash_attn` → `No module named 'flash_attn_2_cuda'`;
  `aiter` and `flash_attn_interface` also absent in the production venv).
  The upstream wrapper therefore takes the `torch SDPA` fallback branch.
- `preferred_rocm_fa_library()` = **AOTriton**; `is_ck_sdpa_available()` =
  false (explicit unsupported-architecture warning).
- Profiler pass on the unmodified production call
  (`attention(roped_q, k_cache, v_cache)`): executing kernel **`attn_fwd.kd`**.
- Stage-1 brief premise corrected: the executing path is torch SDPA →
  AOTriton flash, NOT FA2 varlen. (`identify.json`.)

## Probe 1 — single-call microbenchmark (fresh process per arm, 10 warmup + 100 timed, HIP events)

| arm | HIP p50 | HIP p95 | max abs diff vs fp32 math |
|---|---|---|---|
| control (production wrapper) | 4.22 | 4.74 | 2.6e-4 |
| flash (explicit FLASH_ATTENTION) | 4.48 | 4.87 | 2.6e-4 |
| mem_efficient | 4.00 | 4.56 | 2.6e-4 |
| math (reference) | 90.17 | 90.84 | — (is the reference) |

## Probe 1b — alternating control vs mem_efficient (4 blocks x 25, one process)

- Profiler: **both arms execute `attn_fwd.kd`** — the allowlist does not
  change the kernel on this stack. The separate-process −5% "win" was noise.
- Block means interleave (control 4.04/3.96/4.02/4.14 vs memeff
  4.14/4.08/3.99/4.17). No difference. (`probe01b.json`.)

## Probe 3 — Triton-AMD launch + timing (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`)

- Launches; `FLASH_ATTN_2_AVAILABLE` becomes true; varlen branch taken
  (cu_seqlens scan visible); numerics pass (max abs 2.6e-4, finite).
- Timing n=30: p50 **4.33** vs control 4.22 (+2.6%, wrong direction, inside
  noise). No median win → not promoted. Stability/correctness pass,
  performance reject.

## Decision reading (for the C8 stop rule)

All production-available backends execute the identical `attn_fwd.kd`
kernel or score equal/slower at single call with passing numerics.
A 90-call phase rerun (Probe 4) was skipped as non-discriminating: it
cannot separate candidates that run the same kernel. This is the §12
"all available backends are equal/slower" stop condition; custom-kernel
escalation needs a plausible mechanism + a strategic call (see DAG C8).
