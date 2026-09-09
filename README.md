# StrixHalo-LingBot

Reproducible bring-up and characterization of **LingBot World v2
`robbyant/lingbot-world-v2-1.3b-causal-fast`** on an AMD Ryzen AI MAX+ 395
Strix Halo APU with Radeon 8060S (`gfx1151`).

The name is deliberately platform-plus-model specific: this repository is a
single experiment about the 1.3B LingBot World v2 causal-fast checkpoint on
Strix Halo, rather than a general-purpose AMD benchmark collection.

## Current result

The experiment is in progress. The host-level ROCm probe succeeds in native
BF16: PyTorch sees one Radeon 8060S (`gfx1151`), reports BF16 support, and a
BF16 matmul completes. The final model result and timings are recorded in the
dated report under [`docs/`](docs/).

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
git clone https://github.com/<account>/StrixHalo-LingBot.git
cd StrixHalo-LingBot

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
