# Draft PR: ROCm-compatible single-GPU inference and released 1.3B layout

Target: `Robbyant/lingbot-world-v2`

Branch: `boxwrench/lingbot-world-v2:rocm-single-gpu-1.3b`

Commit: `4351d3f`

## Title

Make FlashAttention optional and support single-GPU ROCm inference

## Body

This PR makes the inference path usable on platforms where the CUDA
FlashAttention extension is unavailable, while preserving the existing causal
sampling behavior.

### Changes

- make standard `flash-attn` optional instead of a mandatory dependency;
- route fast cross-attention through the existing generic `attention()`
  dispatcher;
- provide a PyTorch SDPA fallback that preserves padding lengths, causal
  bottom-right alignment, local windows, GQA/MQA head expansion, scaling, and
  BF16 conversion;
- keep causal self-attention's KV-cache, sink-token, and local-window path
  unchanged;
- add the released `i2v-1.3B` configuration and root-level sharded safetensors
  loader required by `lingbot-world-v2-1.3b-causal-fast`;
- document single-GPU ROCm usage and optional FlashAttention installation;
- add focused CPU tests for the SDPA fallback.

The implementation is hardware-agnostic. It was validated against native
BF16 ROCm/PyTorch environments on AMD `gfx1201` and `gfx1151`. No MIOpen
solver workarounds, Conv3d decomposition, profiling instrumentation, or
hardware-specific branches are included.

### Validation

```text
python -m pytest -q tests/test_attention.py
2 passed

Released six-shard 1.3B checkpoint:
WanModelFast, 1,709,502,016 parameters, torch.bfloat16
```

The 1.3B Hugging Face repository currently contains the model shards and
index but no Diffusers `config.json`; the loader reconstructs the model from
the published tensor layout and checks that every model key is present.

Existing 14B subfolder checkpoints continue through the normal
`ModelMixin.from_pretrained` path.

### Scope

This PR intentionally does not change VAE implementation, causal
self-attention semantics, distributed inference, or backend-specific solver
selection.
