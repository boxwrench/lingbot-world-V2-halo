# Results summary

This table is intentionally sparse until measurements exist. Values must come
from a saved `metrics.json`, not visual estimation from a progress bar.

| Run | Resolution | Frames | BF16 | Steps | Window/sink | Cold | Warm | ms/chunk | FPS | GPU alloc | RSS |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| R9700 ROCm 7.14 control | 480×832 | 77 | yes | 4 | 18 / 6 | 167.341 s | 160.674 s | 5.426 s | 2.888 | see report | 20.3 GiB |

Window and resolution sweep rows will be appended after the default baseline
is valid. Failures are recorded separately rather than omitted.
