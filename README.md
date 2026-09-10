# StrixHalo-LingBot

Reproducible bring-up and characterization of **LingBot World v2
`robbyant/lingbot-world-v2-1.3b-causal-fast`** on an AMD Ryzen AI MAX+ 395
Strix Halo APU with Radeon 8060S (`gfx1151`).

The name is deliberately platform-plus-model specific: this repository is a
single experiment about the 1.3B LingBot World v2 causal-fast checkpoint on
Strix Halo, rather than a general-purpose AMD benchmark collection.

## Current result

The native 1.3B BF16 path succeeds end-to-end on one Radeon 8060S
(`gfx1151`). The 480x832 / 21-frame / 18+6 / chunk-3 baseline produces finite
valid video in 89.940 s cold and 81.216 s warm (0.2335 / 0.2586 FPS), with
43.475 GB peak PyTorch allocation and 25.864 GB peak process RSS. The same-seed
repeat is finite but not bit-identical; that caveat is preserved in the dated
report and benchmark JSON.

An opt-in `taew2_1` TAEHV presentation decoder is also validated at the
lower-latency `384x672` / chunk-1 / 3-step configuration. It reduced filled-
window first-visible latency from about 1.84 s to 1.23 s and next-action
readiness from about 2.23 s to 1.65 s in an 81-frame persistent run. The
canonical FP16 VAE remains the default/reference decoder because TAEHV is
softer in fine detail. See [`docs/taehv-20260910.md`](docs/taehv-20260910.md)
for the pinned source, latent contract, raw paths, and quality caveats.

To try the model as a bounded keyboard-driven live viewer rather than an MP4
run, use [`scripts/run_live.sh`](scripts/run_live.sh). It opens a Tk/Pillow
window using the validated TAEHV presentation path. `Q`/`ESC` quits, `Ctrl-C`
in the launch terminal is an emergency stop, and GNU `timeout` supplies a
second hard limit. Details and controls are in
[`docs/live-viewer.md`](docs/live-viewer.md).

## Target configuration

- CPU/APU: AMD Ryzen AI MAX+ 395, 16 cores / 32 threads
- GPU: Radeon 8060S, `gfx1151`
- Memory: approximately 121 GiB visible system RAM, unified APU memory
- OS/kernel: Linux Mint 22.3, Linux 6.17.x class
- Intended precision: native BF16
- Intended mode: one GPU, no quantization, no CPU offload, no distributed
  inference

## Reproduce

The scripts use an isolated virtual environment layered over the host’s
known-working ROCm PyTorch environment. They never install CUDA Torch or
standard CUDA-only FlashAttention.

```bash
git clone https://github.com/boxwrench/lingbot-world-V2-halo.git
cd lingbot-world-V2-halo

# Capture provenance before changing the environment.
bash scripts/env_report.sh results/raw/provenance

# Create the isolated environment, fetch pinned source/assets, and make the
# 1.3B model layout expected by upstream Diffusers code.
bash scripts/setup_env.sh
bash scripts/prepare_model.sh

# Minimal import/device checks, then a short valid video.
bash scripts/run_smoke.sh

# Clean native BF16 baseline at 480x832, followed by bounded sweeps.
bash scripts/run_baseline.sh
bash scripts/run_window_sweep.sh
bash scripts/run_resolution_sweep.sh
```

The model assets are several gigabytes and are ignored by Git. The exact
checkpoint revisions and commands are recorded by the scripts in
`results/raw/` and in the dated bring-up report.

## Compatibility patch

The repository applies a small correctness-first patch to the pinned upstream
checkout. Upstream’s fast cross-attention calls `flash_attention()` directly;
when FlashAttention is unavailable, that function asserts instead of reaching
the existing PyTorch SDPA fallback. The patch routes that call through the
existing `attention()` dispatcher. Self-attention still uses the upstream KV
cache and bounded local-window logic, so this does not turn local attention
into global attention.

The missing 1.3B Diffusers configuration and shared auxiliary assets are
reconstructed by `scripts/prepare_model.sh` and are not model-weight patches.

## Reports and evidence

- [`docs/bringup.md`](docs/bringup.md): phase checklist, command ledger, and
  reproduction notes.
- [`docs/architecture-notes.md`](docs/architecture-notes.md): traced causal
  fast pipeline, cache/window behavior, and upstream-vs-used code.
- [`docs/r9700-solver-audit-20260909.md`](docs/r9700-solver-audit-20260909.md):
  live gfx1201 MIOpen solver audit and newer-stack control.
- [`docs/vae-repro-20260909.md`](docs/vae-repro-20260909.md): isolated real
  Wan VAE decode measurement on Strix.
- [`docs/results.md`](docs/results.md): compact results table updated as runs
  complete.
- [`docs/lingbot-world-v2-1.3b-strix-halo-bringup-20260909.md`](docs/lingbot-world-v2-1.3b-strix-halo-bringup-20260909.md): dated final report.

Model weights and large generated videos are not committed. Small media or
external artifact references may be added only when they materially support a
finding.
