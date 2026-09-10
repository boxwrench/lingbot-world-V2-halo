# LingBot World v2 1.3B Strix Halo bring-up — 2026-09-09

## Summary

The released `robbyant/lingbot-world-v2-1.3b-causal-fast` checkpoint runs
end-to-end on the Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`) using native
BF16 transformer weights, one HIP device, and the PyTorch SDPA attention
fallback. The requested 480x832, 21-frame run produced finite output with
shape `[21,464,832,3]`, encoded as a valid 832x464 MP4.

The measured 21-frame lane took 89.940 s for generation cold and 81.216 s for
an in-process warm repeat. Including the 36.115 s model initialization, cold
end-to-end time was 126.055 s. The main costs were VAE encode/decode rather
than DiT: warm VAE encode was 24.930 s, DiT was 13.328 s, and VAE decode was
42.748 s.

Both repeats were finite, but they were not bit-identical at the tensor level
(`max_abs_diff=1.9584`, `mean_abs_diff=0.03384`). This is retained as a
reproducibility caveat. No optimization or determinism workaround has been
introduced.

## Hardware

| Item | Value |
|---|---|
| APU | AMD Ryzen AI MAX+ 395, 16C/32T |
| GPU | AMD Radeon 8060S Graphics |
| GFX target | `gfx1151` |
| Visible system RAM | approximately 121 GiB |
| OS | Linux Mint 22.3 |
| Kernel | `6.17.0-35-generic` |

## Software

| Item | Value |
|---|---|
| Python | 3.12.3 |
| PyTorch | `2.13.0+rocm7.15.0a20260728` |
| HIP | `7.15.0` |
| torchvision | `0.28.0+rocm7.15.0a20260728` |
| transformers | `4.51.3` |
| diffusers | `0.31.0` |
| active MIOpen | `3.6.0.baa93758` |
| attention | `torch.nn.functional.scaled_dot_product_attention` |
| FlashAttention 2/3 | unavailable / unavailable |

The isolated environment inherits the machine's known-good ROCm Torch stack;
it does not install CUDA Torch or a CUDA-only FlashAttention package. Relevant
run variables were `HIP_VISIBLE_DEVICES=0`, `CUDA_VISIBLE_DEVICES=0`,
`PYTORCH_ROCM_ARCH=gfx1151`, and
`TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_keith`.

## Upstream and model

| Input | Pin |
|---|---|
| LingBot source | `robbyant/lingbot-world-v2` |
| Upstream SHA | `45fa40673607c9acba6cf96a1f9396c95bcef25f` |
| 1.3B model revision | `7e36a5f919f86cb4255cc9bfc30adb44963fbde1` |
| Shared auxiliary assets | `5c33dd40b213598c418fd25bff30fdbd23fd38a7` |

## Architecture and AMD compatibility

The used path constructs one T5 encoder, one Wan 2.1 VAE, and one
`WanModelFast` transformer. Causal-fast uses four denoising timesteps per
chunk (`999`, `937`, `833`, `624` in this run), followed by one extra zero-
timestep transformer forward to update the self-attention KV cache. The
baseline retains the upstream self-cache rolling and sink behavior with
`local_attn_size=18`, `sink_size=6`, and `chunk_size=3`.

The only model compatibility patch is the tracked
[`0001-strix-halo-sdpa-cross-attention.patch`](../patches/0001-strix-halo-sdpa-cross-attention.patch):
fast cross-attention is routed through upstream's generic `attention()`
dispatcher so SDPA is selected when FlashAttention is unavailable. Self-
attention and its KV-cache/local-window semantics are unchanged. The tracked
metrics patches add measurement only.

The harness also fixes the output wrapper's channel handling. Upstream VAE
output is channel-first; the harness now preserves it and writes a checked RGB
MP4 instead of accidentally selecting the first color channel.

## Baseline configuration

```text
checkpoint:       lingbot-world-v2-1.3b-causal-fast
requested size:   480x832
actual output:    21 frames, 464x832 model tensor, RGB HWC summary
dtype:            BF16 transformer; VAE parameters/output path are FP32
quantization:     none
device:           one gfx1151 HIP device
CPU offload:      false
FSDP/Ulysses:     false
chunk_size:       3
local_attn_size:  18
sink_size:        6
steps/chunk:      4 denoising + 1 cache-update forward
seed:             42
prompt/image:     fixed prompt and examples/03 image
actions:          examples/03 action files
```

The upstream frame normalization makes this request exact: 21 source frames
become six latent frames and two latent chunks at chunk size 3. This avoids
the 13-frame truncation that would result from requesting 21 with chunk size 4.

## Baseline timings

| Measurement | Cold | Warm repeat |
|---|---:|---:|
| Model initialization | 36.115 s | same process |
| T5 encode | 2.238 s | cache hit |
| VAE encode | 31.228 s | 24.930 s |
| DiT, two chunks | 13.492 s | 13.328 s |
| VAE decode | 42.658 s | 42.748 s |
| Generation total | **89.940 s** | **81.216 s** |
| Cold end-to-end incl. initialization | **126.055 s** | — |
| Effective output rate | **0.2335 FPS** | **0.2586 FPS** |

The first chunk elapsed time was 40.105 s cold and 31.265 s warm. It includes
VAE encode plus the first chunk; the subsequent chunk took 7.107 s cold and
7.131 s warm.

### Per-chunk and per-forward timings

Each chunk contains four denoising forwards and one cache-update forward. The
individual values below are milliseconds, in execution order:

| Run | Chunk | Denoise 999 | Denoise 937 | Denoise 833 | Denoise 624 | Cache update 0 | Chunk total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold | 0 | 1323.2 | 1247.0 | 1244.2 | 1248.4 | 1239.0 | 6384.9 |
| Cold | 1 | 1403.7 | 1427.2 | 1423.2 | 1423.3 | 1423.5 | 7106.7 |
| Warm | 0 | 1246.2 | 1235.6 | 1238.3 | 1237.9 | 1234.3 | 6197.2 |
| Warm | 1 | 1428.9 | 1415.5 | 1429.8 | 1427.5 | 1424.5 | 7130.9 |

The per-forward values are collected with synchronization around the measured
model call and are intended as component measurements, not as a separate
optimized performance lane.

## Memory and UMA accounting

PyTorch reports `120,259,084,288` bytes of total device-visible memory
(114688 MiB). The DRM observables report 4,294,967,296 bytes of VRAM and
120,259,084,288 bytes of GTT. On this APU, these are distinct reporting
surfaces; the experiment does not call the entire GTT/UMA pool dedicated VRAM.

| Measurement | Cold | Warm |
|---|---:|---:|
| Peak PyTorch allocated | 43.475 GB | 43.470 GB |
| Peak PyTorch reserved | 52.647 GB | 57.363 GB |
| Current allocated after generation | 15.495 GB | 15.495 GB |
| Current reserved after generation | 49.088 GB | 57.363 GB |
| Peak process RSS | 25.864 GB | 25.864 GB |
| System MemAvailable at final snapshot | 25.509 GB | 17.149 GB |
| ROCm SMI GPU busy at final snapshot | 84% | 86% |
| ROCm SMI VRAM percentage | 3% | 4% |

The 1.3B self-KV accounting for this lane is:

```text
2 * 30 layers * 27,144 KV tokens * 12 heads * 128 head_dim * 2 BF16 bytes
= 5,003,182,080 bytes = 5.003 GB decimal
```

The cross-attention cache at 512 text tokens is approximately 94.372 MB by
the same accounting. The measured 43.475 GB device allocation therefore
includes activation/workspace and other runtime allocations in addition to KV.

## Correctness and reproducibility

Both baseline outputs were finite, in the expected `[-1,1]` source range, and
had positive temporal frame deltas. FFmpeg/ffprobe verified the first output
as H.264, 832x464, 21 frames, `yuv420p`.

The same-seed in-process repeat was not bit-identical:

```text
repeat max absolute difference:  1.9584124
repeat mean absolute difference: 0.03384074
```

The difference is recorded in the raw metrics rather than hidden behind a
claim of determinism. A separate determinism diagnosis—such as isolating
VAE encode, DiT, and VAE decode repeat behavior—remains a correctness follow-
up, not an optimization result.

## Comparison with R9700

The available R9700 control used the same 480x832 area, prompt, image, seed,
18+6 window, but requested 81 frames with chunk size 4, yielding 77 output
frames. It also used the newer ROCm 7.14 stack and an experiment-only CPU
offload boundary around VAE work. It is therefore a useful component control,
not an equal-frame/equal-chunk apples-to-apples timing.

| Measurement | Strix gfx1151, 21 frames/chunk 3 | R9700 gfx1201, 77 frames/chunk 4 |
|---|---:|---:|
| ROCm/PyTorch | ROCm 7.15 / PyTorch 2.13 | ROCm 7.14 / PyTorch 2.12 |
| Attention | SDPA | SDPA |
| Cold generation | 89.940 s | 167.341 s |
| Warm generation | 81.216 s | 160.674 s |
| Cold/warm FPS | 0.2335 / 0.2586 | 2.773 / 2.888 |
| T5 encode | 2.238 s | 3.702 s |
| VAE encode | 31.228 / 24.930 s | 50.981 / 49.695 s |
| DiT total | 13.492 / 13.328 s | 27.198 / 27.129 s |
| VAE decode | 42.658 / 42.748 s | 83.095 / 82.782 s |
| Mean DiT chunk | 6.777 / 6.674 s | 5.440 / 5.426 s |
| Output | finite, 21x464x832x3 | finite, 77x464x832 |

The Strix run is shorter because it has fewer frames and smaller chunks, so
total FPS and total times should not be interpreted as a
hardware speedup claim. The useful observations are that gfx1151 completes the
native no-offload path with ample UMA capacity, while the R9700 control needed
an experiment-only VAE offload boundary and has the stronger per-chunk DiT
control under its different stack/configuration.

## Reproduction

From the repository root, after the pinned assets are prepared:

```bash
bash scripts/setup_env.sh
bash scripts/prepare_upstream.sh
bash scripts/prepare_model.sh
bash scripts/run_smoke.sh
bash scripts/run_baseline.sh
```

The canonical baseline script now runs 480x832, 21 frames, chunk size 3,
18+6 local/sink causal memory, two in-process repetitions, native BF16, and
`--offload-model false`. It writes the ignored raw report to
`results/raw/baseline-480x832-21f/metrics.json` and outputs to the same
directory. The tracked compact record is
[`benchmarks/lingbot-world-v2-1.3b-480x832-21f.json`](../benchmarks/lingbot-world-v2-1.3b-480x832-21f.json).

The exact model/source pins and host provenance are in
[`docs/bringup.md`](bringup.md) and `results/raw/provenance-20260909/`.

## Media and raw evidence

Local ignored artifacts:

- [`generated-1.mp4`](../results/raw/baseline-480x832-21f/generated-1.mp4)
- [`generated-2.mp4`](../results/raw/baseline-480x832-21f/generated-2.mp4)
- [`representative-frame-11.png`](../results/raw/baseline-480x832-21f/representative-frame-11.png)
- [`metrics.json`](../results/raw/baseline-480x832-21f/metrics.json)

## Conclusions and bounded follow-up

The minimum success criterion is met: native BF16 LingBot World v2 1.3B
causal-fast generates valid output on gfx1151. The standard 480x832/21-frame
lane is operationally usable for batch experimentation, but VAE encode/decode
dominate the 80.9 s warm generation time. The observed same-seed difference
is the primary correctness caveat.

No optimization is started by this report. Evidence-backed follow-ups are
limited to: isolate the repeatability difference by stage, then profile the
already measured VAE and DiT paths before changing kernels, memory placement,
or scheduling. The closed 14B experiment remains documented separately as a
capacity success but an approximately 0.10 FPS interactive-latency rejection.
