# Results summary

This table is intentionally sparse until measurements exist. Values must come
from a saved `metrics.json`, not visual estimation from a progress bar.

| Run | Resolution | Frames | BF16 | Steps | Window/sink | Cold | Warm | ms/chunk | FPS | Peak alloc | Peak RSS |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| Strix gfx1151 native baseline | 480×832 → 464×832 | 21 | yes | 4 + cache update | 18 / 6 | 89.940 s | 81.216 s | 6746 / 6664 ms | 0.233 / 0.259 | 43.475 GB | 25.864 GB |
| R9700 gfx1201 ROCm 7.14 control | 480×832 → 464×832 | 77 | yes | 4 | 18 / 6 | 167.341 s | 160.674 s | 5.440 / 5.426 s | 2.773 / 2.888 | see report | 25.71 GB |

The Strix and R9700 rows use different requested frame/chunk counts and
software stacks; compare component timings with that limitation. Window and
resolution sweep rows remain paused until the baseline repeatability caveat
has a bounded diagnosis. Failures are recorded separately rather than omitted.
