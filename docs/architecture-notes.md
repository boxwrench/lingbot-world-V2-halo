# Upstream architecture notes

Pinned source: `robbyant/lingbot-world-v2` at
`45fa40673607c9acba6cf96a1f9396c95bcef25f`.

## Actual causal-fast path

`generate.py` validates `i2v-A14B`, loads `WanI2VCausal`, and defaults to
`infer_mode=causal_fast`. On a single process, distributed flags are disabled;
the pipeline creates one T5 encoder, one Wan 2.1 VAE, and one `WanModelFast`.
The 1.3B model is the same fast class with a smaller tensor configuration.

The 1.3B transformer configuration reconstructed from the safetensor headers
and public Wan architecture references is:

```text
patch_size=(1,2,2)     in_dim=36       out_dim=16
dim=1536               ffn_dim=8960   num_heads=12
num_layers=30          text_len=512   text_dim=4096
freq_dim=256           qk_norm=True   cross_attn_norm=True
```

The 36 input channels are the 16-channel noisy latent concatenated with the
20-channel image/video conditioning tensor. The output is a 16-channel latent
that is decoded by the VAE.

## Causal generation sequence

For a valid requested frame count, upstream rounds to `4n+1` source frames and
uses latent temporal chunks. The default CLI chunk size is 4, while the
pipeline’s Python default is 3; the experiment records the actual value passed
to each run. The fast model uses four denoising timesteps per chunk
(`timesteps_index=[0,179,358,679]`) and then performs one extra model forward
at zero timestep to update the KV cache with the final clean latent.

Each transformer block has self-attention with a reusable KV cache, cross
attention with a prompt KV cache, camera Plücker conditioning, and an FFN.
When `local_attn_size > -1`, the self KV cache is allocated for
`frame_seqlen * local_attn_size` tokens and old non-sink tokens are rolled out
as new chunks arrive. The `sink_size` is expressed in frames. The attention
operation receives the already-windowed cache; the SDPA adaptation therefore
does not remove the upstream windowing behavior.

## Placement

The upstream single-process path places the VAE and transformer on
`cuda:0`/HIP device 0. T5 is initialized from CPU weights and moved to the
device for prompt encoding, then returned to CPU when `offload_model=True`.
The CLI default for one process is `offload_model=True`, but that flag does not
move the transformer to CPU between chunks; it mainly clears caches, moves T5
back after encoding, and moves the transformer back after the generation.
For this experiment the baseline explicitly passes `--offload_model false` to
avoid interpreting a discrete-GPU memory workaround as necessary on UMA.

## CUDA assumptions searched

The pinned source contains CUDA-named PyTorch APIs because HIP exposes the
CUDA-compatible PyTorch namespace: `torch.cuda`, CUDA autocast, synchronization,
and device objects. It also contains distributed machinery for the 14B path:
NCCL initialization, FSDP, and Ulysses sequence parallelism. None of those are
used by the single-device 1.3B baseline.

FlashAttention is listed in `requirements.txt`. The fast self-attention uses
the generic `attention()` dispatcher, which already has an SDPA fallback. Fast
cross-attention instead called `flash_attention()` directly; this is the one
source compatibility patch applied for the no-FlashAttention ROCm baseline.

