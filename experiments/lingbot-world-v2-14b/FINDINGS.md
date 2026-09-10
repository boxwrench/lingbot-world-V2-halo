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

## F4 — 18+6 denoising completes, but the unmodified VAE boundary is not viable (2026-09-09)

The Q4_K_M lane was then held constant while increasing the world-memory
window to `local_attn_size=18`, `sink_size=6`. The request remained 480x832,
21 frames, chunk size 3, seed 42, the same prompt and `strix-example` image,
and the same Q4/K_S/VAE assets. The community pre-flight accounting was
retained exactly:

```text
2 * 40 layers * (local + sink) * tokens_per_frame * 5120 channels * 2 bytes
tokens_per_frame = 1508
estimated self-KV = 29,648,486,400 bytes = 29.648 GB decimal
```

Both chunks of denoising completed in about 125 seconds. The original
unmodified lane then remained in the VAE/output phase for more than 24 minutes
without producing metrics or a video and was terminated with its console log
preserved at `results/raw/14b/q4-480x832-18plus6/console.log`. A phase-hooked
no-cleanup repeat reached the same boundary and was terminated after recording
`before_vae` at
`results/raw/14b/q4-480x832-18plus6-phase-baseline/phase-events.json`.

The boundary sample shows substantial live generation state overlapping the
decode working set:

| before VAE measurement | value |
|---|---:|
| PyTorch allocated / reserved | 23.495 / 25.560 GB |
| generation peak allocated / reserved | 26.578 / 26.965 GB |
| process RSS | 27.807 GB |
| system MemAvailable | 38.212 GB |
| runtime self-KV tensors | 22.236 GB |
| cross-KV tensors | 0.419 GB |
| resident Q4 GGUF mapping | 12.054 GB |
| quantized data tracked by model | 11.567 GB |
| final latent | 2.316 MB, `[16,6,58,104]`, float32 |

The runtime self-KV tensor total is smaller than the pre-flight 29.648 GB
estimate because that formula includes `(local + sink)`, while the observed
runtime cache at this boundary reports the local rolling cache. Both numbers
are retained; they are not treated as interchangeable.

This is evidence of a generation/VAE working-set overlap problem, not proof
that the VAE convolution itself is intrinsically unable to run at this
window. No transformer optimization or temporal Conv3d decomposition was
introduced.

## F5 — Releasing dead generation state makes 18+6 VAE decode complete (2026-09-09)

A second lane used the exact same latent-producing computation and then, only
at the final-latent/VAE boundary, released the transformer, self/cross KV
caches, prompt/context conditioning, GGUF mapping references, and sampler
runtime state. It synchronized, collected Python references, and emptied the
unused PyTorch allocator cache before calling the unchanged VAE decode. The
latent, prompt, seed, image, actions, quantization, frame count, chunk size,
VAE, and window were not changed.

The cleanup reduced current device allocation from 23.495 GB to 0.795 GB,
process RSS from 27.807 GB to 15.298 GB, and raised MemAvailable from 38.212 GB
to 63.483 GB. The retained latent was still only 2.316 MB. The VAE then
completed in 51.740 s; its VAE-only reset peak was 23.530 GB allocated and
37.287 GB reserved. The complete 21-frame output was finite and valid at
`[21,464,832,3]`, encoded as 832x464 H.264. The complete cleanup-lane sample
took 210.362 s (0.099828 fps) and is recorded in
`benchmarks/q4-480x832-18plus6-phase-memory.json`.

The result establishes that Q4_K_M 480x832 `18+6` fits cleanly when the
generation and VAE working sets do not unnecessarily overlap. It does not
claim that the VAE is low-memory: VAE decode itself still temporarily drives
the device allocator to about 23.5 GB and system MemAvailable to 26.746 GB.

### Phase-memory table

Values are decimal GB unless stated otherwise. “Peak” is the PyTorch peak
since the relevant reset; current allocation is shown where useful. The
`model ready` row uses the recorded post-loader device snapshot; the separate
`after T5` row is captured before DiT/VAE construction.

| phase | device allocated / reserved | process RSS | system MemAvailable | live-state observation |
|---|---:|---:|---:|---|
| model ready | 4.817 / 5.908 GB | not sampled | not sampled | post-loader snapshot; model parameters CPU-side |
| after T5 | 4.817 / 5.908 GB | 7.271 GB | 67.151 GB | positive and negative conditioning 8.389 MB each |
| during denoising peak | peak 26.578 / 26.965 GB | max 27.805 GB | 38.212 GB at boundary | 18+6 KV and GGUF state live |
| final latent / before cleanup | 23.495 / 25.560 GB | 27.807 GB | 38.212 GB | latent 2.316 MB; self-KV 22.236 GB; cross-KV 0.419 GB |
| after cleanup | 0.795 / 0.914 GB | 15.298 GB | 63.483 GB | KV, conditioning, transformer state, and GGUF map released |
| VAE peak | peak 23.530 / 37.287 GB | max 27.805 GB | 26.746 GB at completion | unchanged VAE; 51.740 s |
| output complete | 0.080 / 0.080 GB | 15.873 GB | 62.719 GB | finite 21-frame 832x464 video |

The machine-readable event stream is kept locally at
`results/raw/14b/q4-480x832-18plus6-phase-cleanup/phase-events.json`; the
tracked summary is the benchmark JSON named above. The generated output and a
representative still are available locally as
`results/raw/14b/q4-480x832-18plus6-phase-cleanup/generated-1.mp4` and
`representative-frame-11.png`.

Q6/Q8 asset downloads may proceed, but their performance lanes remain held
until this 18+6 boundary result is incorporated into the next experiment plan.

## F6 — 14B conclusion: viable capacity, rejected interactive latency (2026-09-09)

The 14B Q4 characterization is complete for this phase. The cleanup lane
demonstrates that Strix Halo can sustain the larger causal world state and
successfully decode a valid 21-frame output at 480x832-equivalent dimensions.
The result is a capacity success, but its approximately 0.10 FPS throughput
is too slow for the intended interactive world-model use:

```text
14B causal-fast Q4_K_M, 480x832 request, 18+6, chunk 3, 21 frames
210.362 s total, 51.740 s VAE decode, 0.0998 FPS, finite output
```

No Q6 or Q8 performance benchmarks will be run in this branch. Their completed
downloads, if present, are retained for possible future work, but this
experiment now returns to the native BF16 1.3B baseline. The phase-cleanup
technique and the distinction between the 29.648 GB formula estimate and the
22.236 GB observed runtime self-KV total remain required findings for any
future higher-precision or longer-context test.
