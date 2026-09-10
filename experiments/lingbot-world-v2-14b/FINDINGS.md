# 14B chronological findings

This log is append-only during the experiment. Superseded interpretations
remain visible and are corrected explicitly.

## F1 — Community implementation and execution model inspected (2026-09-09)

Pinned source `RealRebelAI/Rebels_LingBot-World-V2_GGUF_ComfyUI` at
`f310acc1109e7af447d5c6efb94f29fe746408a1` was inspected before installation.
The node is a ComfyUI custom node and requires `ComfyUI-GGUF` for its GGUF
reader/dequant path. The DiT loader uses `gguf.GGUFReader`, keeps quantized
linear data CPU-side and file-backed, and dequantizes one linear weight per
forward on the active device. Quantized non-linear tensors are dequantized
once at load. The node's T5 shim consumes embeddings produced by ComfyUI's
GGUF UMT5 loader; it does not load the upstream 11 GB T5 checkpoint.

The loader defaults to `local_attn_size=6`, `sink_size=2`, and `pin_gb=2` for
the low-memory community workflow. The sampler defaults to chunk size 3 and
four causal-fast denoising steps per chunk. It invokes the existing
`WanI2V.generate(..., offload_model=True)` loop and moves the DiT/VAE to the
active device for sampling. The reference code contains CUDA-named PyTorch
APIs, but its dequantizer is ordinary PyTorch tensor code; no NVIDIA-specific
binary extension is present in the pinned node itself.

The initial experiment will still verify the ComfyUI-GGUF and ComfyUI runtime
on ROCm rather than treating source inspection as evidence of execution.

## F2 — Q4 asset set pinned (2026-09-09)

The first lane uses the merged causal-fast Q4_K_M DiT, quantized UMT5 Q4_K_S,
and Wan VAE from the pinned Hub snapshot. Exact sizes and LFS object hashes
are in `asset-manifest.json`. No 14B weights were present on the host before
this lane was added.

## F3 — Q4_K_M baseline succeeds on gfx1151 (2026-09-09)

The first complete run used the pinned merged Q4_K_M DiT, UMT5 Q4_K_S,
and Wan VAE at requested 480x832, 21 frames, chunk size 3,
`local_attn_size=6`, `sink_size=2`, `pin_gb=2`, seed 42, and the community
sampler's four causal-fast denoising steps per chunk. It ran on one
`gfx1151` device with BF16 compute and no CUDA binary extension.

The run succeeded twice in one process. Both outputs were finite and the
same-seed repeat had max/mean absolute difference 0.0/0.0. The encoded video
has 21 frames at 832x464; 480x832 is the requested model resolution and the
height reduction is the community VAE/output path's valid latent crop.

| measurement | run 1 (cold generation) | run 2 (warm repeat) |
|---|---:|---:|
| total generation, including VAE encode/decode | 215.674 s | 216.380 s |
| effective output rate | 0.09737 fps | 0.09705 fps |
| causal chunks | 2 | 2 |
| causal progress time, from tqdm | ~125 s | ~125 s |
| output | finite, 21x464x832x3 | finite, 21x464x832x3 |

The run-level instrumentation measured UMT5 GGUF load at 15.295 s, prompt
encoding at 7.577 s, and DiT/VAE object construction at 1.304 s. It did not
yet split individual DiT forwards from VAE encode/decode; the console trace
does show two causal chunks at roughly 60--65 s each, with the remaining
run time spent in the VAE path and surrounding work.

At load, the node attached 1,421 tensors with zero unmatched or meta tensors.
The Q4 linear weights remained file-backed/CPU-side and the first active
linear reported `x.device=cuda:0`, `x.dtype=torch.bfloat16`,
`qdata.device=cpu`. The loader pinned 2.16 GB of quantized data and the
resident GGUF mapping reached 12.04 GB. Peak PyTorch device allocation was
31.40 GB, peak reserved was 45.30 GB, and peak process RSS was 28.74 GB.
The final device allocation fell back to 89 MB after the community sampler's
model/VAE offload cleanup. `rocm-smi` reported 99% GPU busy during DiT work;
its VRAM percentage is not used as a UMA capacity measurement here.

The required ROCm compatibility patch is tracked at
`patches/14b-rocm-sdpa.patch`: only fast cross-attention changes from a direct
FlashAttention call to the existing generic `attention()` dispatcher. This
selects PyTorch SDPA when FlashAttention is absent. Causal self-attention,
KV-cache updates, local-window rolling, and sink handling were not changed.

The raw report is `results/raw/14b/q4-480x832-6plus2/metrics.json` and the
representative outputs are `generated-1.mp4` and `generated-2.mp4` in that
directory. These files remain local/ignored because model and media artifacts
are not committed.
