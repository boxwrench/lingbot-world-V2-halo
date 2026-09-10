# Bring-up log

This document is the working ledger for the Strix Halo experiment. The dated
report is the concise handoff; this file keeps operational detail that helps a
second owner reproduce or audit a result.

## Experimental boundary

The baseline is one `gfx1151` device, native BF16, the released 1.3B
causal-fast checkpoint, no quantization, no CPU offload, no FSDP, no Ulysses,
and upstream causal chunking/window semantics. Any change to those conditions
gets a separate result directory and is not compared as if it were baseline.

## Pinned inputs

| Input | Pin |
|---|---|
| Upstream repository | `https://github.com/robbyant/lingbot-world-v2` |
| Upstream commit | `45fa40673607c9acba6cf96a1f9396c95bcef25f` |
| 1.3B model revision | `7e36a5f919f86cb4255cc9bfc30adb44963fbde1` |
| Shared-assets revision | `5c33dd40b213598c418fd25bff30fdbd23fd38a7` |
| 1.3B transformer | 30 blocks, width 1536, FFN width 8960, 12 heads |

## Command ledger

The authoritative command lines are the executable scripts in `scripts/`.
Each run writes its expanded environment and JSON metrics to `results/raw/`.
Do not hand-edit a result JSON; add interpretation to the dated report.

## Phase status

- [x] Host provenance captured before experiment files were added.
- [x] Upstream source pinned and causal-fast path traced.
- [x] 1.3B checkpoint availability and model revision verified.
- [x] Isolated environment installed.
- [x] Import/device smoke test.
- [x] Minimal valid video.
- [x] 480x832 BF16 baseline, cold and warm.
- [x] Finite-output check; same-seed repeatability caveat recorded.
- [x] UMA allocation/RSS characterization.
- [ ] Window sweep.
- [ ] Resolution sweep.
- [ ] Long run.
- [ ] Baseline profile.

## Interpretation rules

PyTorch’s `total_memory`, DRM `mem_info_vram_total`, KFD APU pool, PyTorch
allocated/reserved bytes, process RSS, and system used memory are different
observables. Report them separately. The experiment will not label the 112
GiB APU pool as dedicated VRAM without evidence.

## Current low-latency phase

The validated TAEHV result is recorded in
[`next-phase-plan-20260910.md`](next-phase-plan-20260910.md) and
[`taehv-20260910.md`](taehv-20260910.md). TAEHV is an explicit opt-in
presentation decoder; canonical FP16 Wan VAE remains the default/reference
path. At full `18+6` occupancy and `384x672`, TAEHV reduced first-visible from
about `1.841 s` to `1.229–1.232 s` and next-action permission from about
`2.232 s` to `1.644–1.648 s` in a finite 81-frame persistent session.

The next bounded task is to profile the remaining three-step DiT path in that
TAE candidate. No new decoder, sampler, attention library, or kernel branch is
authorized until that profile identifies a measured opportunity.
