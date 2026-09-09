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

## F3 — Q4 execution

Pending download and runtime validation.

