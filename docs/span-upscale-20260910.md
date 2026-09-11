# SPAN x2 post-TAE qualification (2026-09-10)

## Decision

**REJECT.** The official SPAN x2 checkpoint is exceptionally light and fast on
gfx1151, but on the LingBot frames it is visually too close to ordinary
Lanczos resizing to justify another live presentation mode. It adds mild edge
sharpening and a small increase in frame-to-frame variation, but no clear
recovery of detail over Lanczos in the representative tree, mountain, and
water comparisons. No live integration was performed because the Stage-1
quality gate was not met. The accepted native 384x672 TAE presentation is
unchanged.

This is a qualification of a LingBot-inspired post-RGB upscaler, not a
reproduction of LingBot's unpublished deployment refiner. The starting
repository HEAD and the protected edit to `scripts/run_window_sweep.sh` were
left untouched.

## Provenance

| Item | Value |
| --- | --- |
| official source | <https://github.com/hongyuanyu/SPAN> |
| pinned revision | `c77a5917759f09e66fbc7124220c5afc5ee221e5` |
| architecture source | `basicsr/archs/span_arch.py` at the pinned revision |
| checkpoint | `spanx2_ch48.pth` |
| checkpoint source | official `span.zip` link in the SPAN README |
| archive entry | `spanx2_ch48.pth` |
| checkpoint SHA-256 | `561fd5cf419a23d4de1231ce258180f61aee4aa8caa1aaaa783769c7301847bc` |
| license | Apache-2.0 |
| architecture | `SPAN(3, 3, feature_channels=48, 6 SPAB blocks, upscale=2)` |
| parameters | 2,221,140 |

The official source checkpoint link resolves to `span.zip`, from which the
`spanx2_ch48.pth` archive entry was extracted and verified. The runner loads
the official architecture file directly while bypassing BasicSR's package-wide
legacy imports; this avoided changing the known-good ROCm environment. The
checkpoint's `params_ema` state was loaded strictly.

## Input contract

The same 32 real contiguous TAE frames used for the RealESRGAN reference were
used here:

```text
source: results/raw/taehv-offline-384x672-3step-81f/taehv_frames_01.pt
shape:  [32, 384, 672, 3]
dtype:  torch.float32
layout: THWC, RGB
range:  [0, 1]
```

The adapter performs only the required HWC-to-NCHW conversion and device/dtype
copy. It does not perform a BGR conversion, PNG round trip, or additional
normalization. The official SPAN forward applies its own input transform. The
output is `[1, 3, 768, 1344]`; saved comparison frames are RGB after clamping
to `[0,1]`.

## Isolated performance

The first three frames were excluded from warmed statistics. Measurements used
the accepted PyTorch 2.13.0 ROCm 7.15 environment on the Radeon 8060S.

| Metric | FP32 reference | FP16 candidate |
| --- | ---: | ---: |
| model load | 534.2 ms | 679.7 ms |
| first network inference | 1369.0 ms | 627.1 ms |
| warm network P50 | 49.7 ms | 21.0 ms |
| warm network P95 | 50.7 ms | 23.9 ms |
| warm frame-ready P50 | 67.8 ms | 48.1 ms |
| warm frame-ready P95 | 70.9 ms | 56.3 ms |
| peak PyTorch allocated | 1.279 GB | 0.516 GB |
| peak PyTorch reserved | 1.483 GB | 0.581 GB |
| peak process RSS | 2.869 GB | 2.477 GB |
| parameter/weight allocation | 8.9 MB | 4.5 MB |

FP16 is the only plausible release precision tested. Its first call includes
one-time backend warm-up and is not a live-frame estimate. The warmed network
is substantially faster than the isolated RealESRGAN FP16 reference (`119.5
ms` network, `145.9 ms` frame-ready).

## Quality and temporal result

Four-way artifacts use the same source frames and the existing RealESRGAN
reference:

```text
results/raw/span-upscale-20260910/fp16/contact-sheet.png
results/raw/span-upscale-20260910/fp16/span-x2-fp16.mp4
results/raw/span-upscale-20260910/fp16/source/
results/raw/span-upscale-20260910/fp16/lanczos/
results/raw/span-upscale-20260910/fp16/span-fp16/
```

Visual inspection found:

* SPAN is broadly similar to Lanczos, with mild sharpening rather than a
  clearly more detailed reconstruction.
* RealESRGAN remains visibly more aggressive around branches, foliage,
  mountain contours, and water, but its earlier co-resident latency penalty
  already rejected it for live use.
* SPAN did not show gross halos, color failure, or obvious catastrophic
  temporal instability in this 32-frame sample. Its small temporal statistic
  increase means this is not evidence of superior temporal stability.

| Sequence | Mean adjacent absolute difference | P95 |
| --- | ---: | ---: |
| TAE source | 0.05199 | 0.09573 |
| Lanczos 2x | 0.05171 | 0.09547 |
| RealESRGAN FP16 | 0.05252 | 0.09212 |
| SPAN FP16 | 0.05481 | 0.09805 |

Across the 32 frames, SPAN differed from Lanczos by mean absolute RGB error
`0.00758` after output conversion, with approximately `9.99%` of pixels
exceeding a per-pixel mean difference of `0.02`. The higher gradient statistic
(`0.03229` for SPAN versus `0.02729` for Lanczos) confirms that it sharpens the
image, but not that it restores correct scene detail. The contact sheet and
individual crops did not establish the required clear benefit over Lanczos.

## Live gate

No co-resident live run was justified. The candidate passed the speed side of
the gate but failed the required visual side: the isolated output was largely
Lanczos-like. Because the base TAE frame must remain immediately available,
there was no reason to put an additional live mode into the viewer without a
clear presentation benefit. RealESRGAN's measured co-resident slowdown is kept
as the relevant heavy-model reference:

```text
isolated FP16 network: 119.5 ms
co-resident network:   312.6 ms
```

Therefore SPAN's co-resident slowdown ratio and next-ready impact are
**not measured**. They must not be inferred from the isolated result.

## Release conclusion

Decision: **REJECT** for RC1 live use and do not add a SPAN viewer mode. Keep
the qualification script and provenance in the development history, but ship
the accepted native TAE presentation. Stop the learned-upscaler search for
RC1 rather than immediately trying another architecture. The next authorized
experiment, after this report is reviewed, is the fixed-budget `sink_size 6 →
2` memory test.

