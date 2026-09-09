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
- [ ] Isolated environment installed.
- [ ] Import/device smoke test.
- [ ] Minimal valid video.
- [ ] 480x832 BF16 baseline, cold and warm.
- [ ] Determinism/finite-output check.
- [ ] UMA allocation/RSS characterization.
- [ ] Window sweep.
- [ ] Resolution sweep.
- [ ] Long run.
- [ ] Baseline profile.

## Interpretation rules

PyTorch’s `total_memory`, DRM `mem_info_vram_total`, KFD APU pool, PyTorch
allocated/reserved bytes, process RSS, and system used memory are different
observables. Report them separately. The experiment will not label the 112
GiB APU pool as dedicated VRAM without evidence.

