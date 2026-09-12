# Strix Halo RC1 Phase 0 baseline — 2026-09-12

## Decision

**PASS: the current RC1 environment is reproduced on the Radeon 8060S / gfx1151.**

Three independent, unprofiled 15-action sessions each reached four rolled
actions beyond the first local-KV eviction. Across the 12 rolled actions:

| Boundary | Median | Range |
|---|---:|---:|
| action → base RGB | **939.9 ms** | 927.6–954.5 ms |
| action → next-ready | **1258.9 ms** | 1244.8–1272.3 ms |
| 3× denoise | 909.8 ms | 899.5–926.5 ms |
| transformer total | 913.5 ms | 903.4–930.5 ms |
| exact clean KV | 298.2 ms | 295.1–304.5 ms |
| TAEHV first RGB, GPU | 20.0 ms | 19.6–20.2 ms |
| TAEHV remaining RGB, GPU | 17.8 ms | 17.6–18.6 ms |

These are final acceptance timings with module/SDPA profiling disabled. Every
measured output and latent was finite. The three per-session rolled medians
were 937.5–946.0 ms to base RGB and 1258.2–1270.0 ms to next-ready.

## Historical RC1 comparison

| Boundary | 2026-09-11 historical P50 | Current pooled median | Delta |
|---|---:|---:|---:|
| action → base RGB | 938.4 ms | 939.9 ms | +1.5 ms (+0.16%) |
| action → next-ready | 1255.3 ms | 1258.9 ms | +3.6 ms (+0.29%) |
| 3× denoise | 909.6 ms | 909.8 ms | +0.2 ms |
| exact clean KV | 298.3 ms | 298.2 ms | −0.1 ms |

Classification: **reproduced within expected variance**. No tuning was done to
force agreement.

## Environment gate

The gate saw exactly one HIP device. PyTorch reported `AMD Radeon 8060S
Graphics`, `gcnArchName=gfx1151`, 120,259,084,288 bytes of unified device
memory, native BF16 support, and a finite BF16 matrix multiplication. No CPU or
second-GPU fallback was possible under `HIP_VISIBLE_DEVICES=0` and
`CUDA_VISIBLE_DEVICES=0`.

Key provenance:

| Item | Resolved value |
|---|---|
| project HEAD / branch | `5f858c53d88f0462de7ff44aae8d3ea70e0b9be8` / `main` |
| upstream LingBot | `45fa40673607c9acba6cf96a1f9396c95bcef25f` plus the existing two RC1 patches |
| model revision | `robbyant/lingbot-world-v2-1.3b-causal-fast@7e36a5f919f86cb4255cc9bfc30adb44963fbde1` |
| GPU | Radeon 8060S / `gfx1151` |
| PyTorch / bundled HIP | `2.13.0+rocm7.15.0a20260728` / `7.15.0` |
| system ROCm libraries | `7.2.2.70202-86~24.04` |
| Triton | `3.8.0+git4cff872c.rocm7.15.0a20260728` |
| kernel | `6.17.0-35-generic` |
| power profile | `auto` |
| TunableOp | enabled, online tuning off, 16 validator-matched results |

The project used its unusual but pre-existing Git layout: metadata is in
`.git-experiment/.git` and the project root is the worktree. Before Phase 0,
`scripts/run_window_sweep.sh` already had an unrelated local modification; it
was not changed by this work. The upstream checkout already had the expected
`wan/image2video.py` and `wan/modules/model_fast.py` RC1 patches.

## Runtime configuration resolved from execution

The loaded runtime reported 384×672 output and a 48×84 latent grid. Each real
action issued a 1,008-token query and advanced global KV by 1,008 tokens, so
the accepted chunk is one latent frame (`[1,16,1,48,84]`). The real rolled
attention probe observed:

```text
chunk_size=1
local_attn_size=12
sink_size=6
physical local capacity=12,096 tokens (includes the sink)
sink reserve=6,048 tokens
recent non-sink history=5,040 tokens plus the 1,008-token current frame
timesteps=999 → 899 → 702
30 DiT blocks, 12 heads, head dimension 128
BF16 DiT Q/K/V
FP16 TAEHV taew2_1 decoder
```

The current exact-shape dispatch check used the observed BF16
`Q=[1,12,1008,128]`, `K/V=[1,12,12096,128]`, strides, no mask, and
`is_causal=false`. It selected
`aten::_scaled_dot_product_flash_attention` and HIP kernel `attn_fwd.kd`.

## Representative wall-clock waterfall

Unprofiled waterfall action 15 was selected because it was the rolled action
nearest both pooled A/A medians.

```text
input received
  ├─ 0.142 ms → input selected
  ├─ 2.704 ms → generation start (camera/action preparation)
  ├─ 314.329 ms → denoise 1, t=999
  ├─ 297.352 ms → denoise 2, t=899
  ├─ 298.668 ms → denoise 3, t=702
  ├─ inter-forward latent postprocess: 3.424 ms total
  ├─ generation start → accepted x0: 917.109 ms wall
  ├─ 21.213 ms → base RGB host-ready
  │  action → base RGB: 941.167 ms
  ├─ 0.198 ms → clean-KV start
  ├─ 300.606 ms → exact t=0 clean-KV complete
  ├─ 19.353 ms → remaining RGB work and copies
  └─ action → next-ready: 1261.325 ms
```

The DiT forward numbers use the runner's synchronization boundaries; they do
not infer asynchronous GPU duration from unsynchronized CPU wall time. The
TAEHV first and remaining GPU portions were 20.028 and 17.500 ms.

## Critical path

Before base RGB, input selection, camera preparation, all three denoise
forwards, inter-forward latent work, first-RGB decode, and first-frame copy are
strictly serialized. Base RGB is submitted to the CPU presentation worker
before the exact clean-KV transaction starts.

Before next-ready, the exact `t=0` DiT forward, remaining TAEHV decode, and
remaining frame copies are also serialized. The next action cannot safely
begin until that clean-KV transaction commits.

JPEG encoding and WebSocket/browser presentation run on the presentation/CPU
side after frame submission. They may overlap clean-KV work and are not part
of generation-state readiness. Their durations are therefore not added to
the recoverable generation critical path.

## Measured opportunity ranking

The mechanism trace was a separate intrusive chunk-13 run and is used only
for attribution. Potential savings are planning estimates, not measured
speedups, and rows are not automatically additive.

| Component | Measured critical-path ms | Recoverable fraction estimate | Confidence | Candidate mechanism |
|---|---:|---:|---|---|
| rectangular self-SDPA | 369.3 before RGB; +121.2 in clean pass | 10–25% = **37–92 ms base RGB**, **49–123 ms next-ready** | medium-low | benchmark exact-shape alternatives to the current flash `attn_fwd.kd` path |
| clean-KV non-SDPA remainder | about 177.0 after RGB (`298.2 − 121.2`) | 25–50% = **44–89 ms next-ready** | low | reduce exact `t=0` non-attention recomputation while preserving identical KV state |
| Linear/GEMM modules | 327.7 before RGB; +111.6 in clean pass | 5–15% = 16–49 ms base RGB, 22–66 ms next-ready | low | verify real-shape TunableOp coverage and only benchmark uncovered recurring GEMMs |
| self-attention non-SDPA body/KV/RoPE | 76.8 before RGB; +23.6 in clean pass | 10–30% = 8–23 ms base RGB, 10–30 ms next-ready | low | isolate cache roll/read/write and layout costs before proposing a mechanism |
| TAEHV decode | 37.8 total; about 20.0 before RGB | 10–25% = 2–5 ms base RGB, 4–9 ms next-ready | medium | decoder-only work; too small to rank above DiT work |
| host gaps/copies around decoder | about 3 ms total in representative action | less than 50% = <2 ms | medium | copy/synchronization cleanup only if a trace proves avoidable work |

The top two independent measured opportunities are therefore:

1. **Rolled rectangular self-SDPA in the three denoise passes:** plausibly
   37–92 ms from base-RGB latency (and 49–123 ms total from next-ready if the
   clean pass benefits too).
2. **The non-SDPA portion of the exact clean-KV recomputation:** plausibly
   44–89 ms from next-ready, with no base-RGB benefit because it is correctly
   deferred until after base RGB.

These targets are disjoint as stated. Eliminating the entire clean-KV pass
would overlap the SDPA estimate and must not be summed with it.

## Artifacts

- `environment.json`: machine-readable gate and provenance.
- `runtime-config.json`: values resolved from the loaded runtime and trace.
- `timings.jsonl`: all 12 unprofiled A/A rolled records.
- `metrics-summary.json`: pooled and per-rollout timing summaries.
- `waterfall.json`: representative action boundaries and critical-path tags.
- `profile-summary.json`: compact intrusive operator/SDPA attribution.
- `raw-artifacts.json`: paths, sizes, and SHA-256 hashes for ignored raw data.
- `commands.md`: exact reproduction commands and boundary notes.

No attention, KV, GEMM, decoder, graph, PM4, ROCr, or other optimization was
implemented. The functioning RC1 execution semantics remain unchanged.
