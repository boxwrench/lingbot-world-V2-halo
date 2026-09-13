#!/usr/bin/env python3
"""P2.3 microbench: time_projection replica cost (kernel time, not values).

Replicates upstream time_projection = Sequential(SiLU, Linear(1536, 9216))
in fp32 on [1, 1008, 1536] — the exact P2.1-measured slice. If the call costs
X ms, caching it across chunks/forwards recovers ~3X ms per 3-forward phase
(12 measured calls over the 4-chunk P2.1 window).
"""
import torch

torch.cuda.init()
device = torch.device('cuda:0')
mod = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(1536, 9216)).to(device).float()
x = torch.randn(1, 1008, 1536, device=device, dtype=torch.float32)
mod.eval()

with torch.no_grad():
    for _ in range(20):
        _ = mod(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iters = 100
    start.record()
    for _ in range(iters):
        _ = mod(x)
    end.record()
    torch.cuda.synchronize()
    total = start.elapsed_time(end)
print(f'per-call ms: {total / iters:.3f} (total {total:.1f} over {iters})')
