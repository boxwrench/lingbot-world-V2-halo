# LingBot World v2 14B on Strix Halo

This is a separate experiment area for the **14B causal-fast** LingBot World
v2 model. It does not change or replace the native-BF16 1.3B experiment in
the repository root.

The initial reference path is the community
[RealRebelAI ComfyUI/GGUF node](https://github.com/RealRebelAI/Rebels_LingBot-World-V2_GGUF_ComfyUI)
and its published quantized assets. That path is intentionally kept as a
separate lane because its model loader, quantization, and ComfyUI integration
are materially different from upstream's safetensor 1.3B path.

## Scope

- Hardware: Ryzen AI MAX+ 395 / Radeon 8060S / `gfx1151`
- Platform: Linux, ROCm, one GPU
- Target: `lingbot-world-v2-14b-causal-fast`
- First model lane: merged `Q4_K_M`
- Text encoder: community quantized UMT5 `Q4_K_S`
- VAE: `Wan2.1_VAE.pth`
- Initial generation target: 480x832, valid short causal run, `local_attn_size=6`,
  `sink_size=2`

The objective is to measure whether the current GGUF implementation is correct
and useful on gfx1151, then characterize the precision/window/long-generation
tradeoffs that the large UMA pool enables. CUDA-only extensions and silent
dequantization back to a full-precision 14B model are outside the baseline.

## Pinned community inputs

| Input | Revision | Role |
|---|---|---|
| RealRebelAI node | `f310acc1109e7af447d5c6efb94f29fe746408a1` | GGUF loader and causal sampler |
| HF asset repository | `79be5d5f30d63327ac6020ced7f653cdbc301f49` | Published GGUF/VAE files |
| ComfyUI-GGUF | `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` | GGUF dequantization path |

The asset filenames, byte sizes, and Hub LFS SHA-256 object IDs are recorded
in [`asset-manifest.json`](asset-manifest.json). Weights are downloaded into
the ignored `models/lingbot-world-v2-14b-community/` directory and are never
committed.

## Current result

The initial Q4_K_M lane succeeds on gfx1151 in native BF16 compute. The
21-frame requested-480x832 run uses `local_attn_size=6`, `sink_size=2`, and
finishes in 215.674 s cold / 216.380 s warm (0.09737 / 0.09705 fps). Peak
PyTorch allocation is 31.40 GB and peak process RSS is 28.74 GB. The output
is finite, repeatable for the same seed, and the raw JSON plus MP4s are under
`results/raw/14b/q4-480x832-6plus2/` locally. See
[`benchmarks/q4-480x832-6plus2.json`](benchmarks/q4-480x832-6plus2.json) and
[`FINDINGS.md`](FINDINGS.md) for the exact lane and limitations.

The bounded `18+6` window also completes denoising, but retaining generation
state into VAE decode causes severe memory pressure. A cleanup-only boundary
lane, with the latent and all generation inputs unchanged, releases the
transformer/KV/runtime state before the unchanged VAE and succeeds in 210.362
s, including a 51.740 s VAE decode. See
[`benchmarks/q4-480x832-18plus6-phase-memory.json`](benchmarks/q4-480x832-18plus6-phase-memory.json)
and the chronological interpretation in [`FINDINGS.md`](FINDINGS.md).

## What the reference implementation does

The node reads GGUF tensors through `gguf.GGUFReader`. The quantized linear
weights remain CPU-side and share the reader's file-backed mapping; each linear
layer moves its quantized bytes to the active device, dequantizes with the
ComfyUI-GGUF PyTorch implementation, performs the BF16 linear, and releases
the temporary dequantized weight. Non-linear quantized tensors are
dequantized once during load. A bounded `pin_gb` option may copy part of the
quantized data into pinned host memory.

The text encoder is not the upstream 11 GB `.pth`: ComfyUI-GGUF's
`CLIPLoaderGGUF` supplies UMT5 embeddings to the node's T5 shim. The VAE is
the normal Wan VAE. Causal-fast retains four distilled denoising steps per
chunk, and the model's self-attention cache uses `local_attn_size` plus
`sink_size` for world memory. The first baseline will verify these claims
with runtime instrumentation before any window or precision conclusions are
drawn.

## Reproduce the asset/provenance setup

```bash
bash experiments/lingbot-world-v2-14b/scripts/download_q4_assets.sh
bash experiments/lingbot-world-v2-14b/scripts/env_report.sh \
  results/raw/14b/provenance
```

The execution script is deliberately separate from `scripts/run_baseline.sh`:

```bash
bash experiments/lingbot-world-v2-14b/scripts/run_q4_baseline.sh
```

It will use the machine's existing ROCm PyTorch stack and will not install
CUDA Torch. Runtime logs belong in `logs/`; machine-readable results belong
in `benchmarks/`. Large model files and generated media are ignored.

## Findings

Chronological conclusions and corrections are in [`FINDINGS.md`](FINDINGS.md).
No 14B result is considered a baseline until the exact quant file, encoder,
window, seed, frame count, and runtime stack are captured alongside it.
