# Post-TAE Real-ESRGAN x2 qualification (2026-09-10)

## Decision

**REJECT FOR LIVE USE, KEEP OFFLINE OPTION.** RealESRGAN_x2plus produces a
visibly sharper 768x1344 presentation than a simple Lanczos resize, and the
sampled sequence did not show gross additional temporal instability. However,
its isolated FP16 speed does not survive co-resident execution with LingBot:
the warmed network is about 119.5 ms in isolation but about 312.6 ms in the
live process. The safe serial integration therefore adds about 357 ms to
next-action readiness. The accepted sub-second TAE base frame remains first
and the canonical LingBot path/defaults are unchanged.

This is a LingBot-inspired post-RGB experiment, not a reproduction of the
official LingBot deployment refiner. The repository starting point was
`425367ce15ac126a9972f3511e96739ff31735df`; the protected unstaged edit to
`scripts/run_window_sweep.sh` was not touched.

## Pinned model and source

| Item | Value |
| --- | --- |
| Real-ESRGAN source | `https://github.com/xinntao/Real-ESRGAN` |
| Real-ESRGAN revision | `a4abfb2979a7bbff3f69f58f58ae324608821e27` |
| BasicSR architecture source | `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a` |
| checkpoint | `RealESRGAN_x2plus.pth`, release `v0.2.1` |
| checkpoint SHA-256 | `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb` |
| architecture | `RRDBNet(3, 3, scale=2, num_feat=64, num_block=23, num_grow_ch=32)` |
| checkpoint size | 67,061,725 bytes |

The official high-level Python package was not installed. The qualification
runner contains the inference subset of the pinned BasicSR RRDBNet definition,
so the known-good ROCm LingBot environment was not upgraded or downgraded.
The model is loaded from the official checkpoint with strict state-dict
matching.

## TAE input contract

The saved accepted TAE sequence is `[32,384,672,3]`, `torch.float32`, CPU,
RGB, and `[0,1]`. In the live path the first TAE frame is an HWC RGB tensor.
The adapter converts only to contiguous NCHW RGB float on the GPU. It does not
perform BGR conversion, latent normalization, or an intermediate uint8 round
trip. The exact output is `[1,3,768,1344]`, presented as RGB after a clamp to
`[0,1]`.

## Isolated performance

Measurements used 32 real contiguous TAE frames, with the first three frames
excluded from warmed statistics.

| Metric | FP32 | FP16 |
| --- | ---: | ---: |
| model load | 1326 ms | 659 ms |
| first network inference | 811.2 ms | 1444.3 ms |
| warmed network P50 | 283.1 ms | 119.5 ms |
| warmed network P95 | 285.1 ms | 122.1 ms |
| warmed frame-ready P50 | 300.2 ms | 145.9 ms |
| warmed frame-ready P95 | 304.1 ms | 151.8 ms |
| mean input conversion | 6.9 ms | 16.6 ms |
| mean output conversion | 9.8 ms | 9.7 ms |

FP16 is the intended isolated candidate. Its first inference is a one-time
backend warm-up and is not the steady product number.

## Memory

| Measurement | Isolated FP16 | Live co-resident FP16 |
| --- | ---: | ---: |
| weight allocation/delta | 33.6 MB | included in process |
| peak PyTorch allocated | 1.53 GB | 30.10 GB |
| peak PyTorch reserved | 1.62 GB | 42.44 GB |
| peak process RSS | 2.37 GB | 3.62 GB |

The live peak allocation remained in the same broad range as the accepted
LingBot run, so Strix UMA capacity is not the blocker. The reserved/workspace
footprint increased, and the co-resident kernel timing changed substantially;
this experiment does not claim a specific solver cause for that difference.

## Visual and temporal qualification

The source, Lanczos, and learned outputs are available in the raw artifact
tree. The contact sheet shows a consistent learned-detail benefit over
Lanczos on mountain contours, tree branches, foliage, and water structure.
The learned result also invents plausible high-frequency texture; it must not
be treated as ground-truth recovery. No large halos or catastrophic color
failure were apparent in the sampled contact sheet.

For the 32-frame sequence, adjacent-frame absolute differences were:

| Sequence | Mean | P95 |
| --- | ---: | ---: |
| TAE source | 0.05199 | 0.09573 |
| Lanczos 2x | 0.05171 | 0.09547 |
| RealESRGAN FP16 | 0.05252 | 0.09212 |

These simple temporal statistics and visual inspection found no gross extra
boiling/flicker in this bounded sample. They are not a substitute for a
longer human quality campaign, which is not justified after the live latency
result.

Artifacts:

```text
results/raw/spatial-upscale-20260910/fp16/contact-sheet.png
results/raw/spatial-upscale-20260910/fp16/realesrgan-x2.mp4
results/raw/spatial-upscale-20260910/fp16/lanczos-x2.mp4
```

## Live integration

The opt-in mode used the accepted 384x672 / 12-frame / 3-step / TunableOp /
pure-helper-prewarm path. It displayed the base TAE RGB first, then ran one
FP16 RealESRGAN refinement on that RGB before the exact clean KV commit. The
canonical decoder was not run in parallel and no generation math changed.

The run included bootstrap plus 20 actions and reached the rolled state.
There were nine rolled actions in the measured tail:

| Rolled metric | Result |
| --- | ---: |
| base first RGB P50 | 927.8 ms |
| refined RGB presented P50 | 1297.9 ms |
| refiner network P50 | 312.6 ms |
| refiner frame-ready P50 | 313.6 ms |
| clean KV P50 | 293.1 ms |
| next action ready P50 | 1648.7 ms |
| base first RGB P95 | 933.1 ms |
| refined RGB presented P95 | 1308.0 ms |
| next action ready P95 | 1657.5 ms |

The refined presentation boundary is about 56 ms after GPU frame-ready time,
attributable to high-resolution host/Tk image construction and presentation
in this runner. Relative to the accepted approximately 1292 ms next-ready
lane, the current safe serial integration adds about 357 ms. Base RGB latency
itself remains intact at approximately 0.93 s.

The final state was finite with global KV `21168`, local capacity `12096`,
and sink retention `6048`. The learned decoder consumed no generation state;
this was presentation-only.

## Release recommendation

Keep `scripts/spatial_upscale.py` and the pinned artifact record as an offline
qualification option. Keep `--upscale-x2` opt-in and do not change the live
default. A future live candidate should be substantially lighter (for example,
a compact SRVGG/SPAN-style family) and must repeat the same offline gate before
integration. Do not start that follow-up in this milestone.
