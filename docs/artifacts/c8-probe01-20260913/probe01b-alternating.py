#!/usr/bin/env python3
"""C8 Probe 1b: within-process alternating control vs mem_efficient A/B
plus per-arm profiler kernel identity. Cancels thermal/drift confounds
the separate-process Probe 1 cannot. 4 blocks x 25 calls per arm."""
import json
import statistics
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, "/home/keith/Desktop/github/lingbot-world-V2-halo/.upstream/lingbot-world-v2")
from wan.modules.attention import attention  # noqa: E402


def calls(fn, q, k, v, n):
    ms = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(q, k, v)
        e.record()
        torch.cuda.synchronize()
        ms.append(s.elapsed_time(e))
    return ms


def summarize(xs):
    xs = sorted(xs)
    n = len(xs)
    return {"n": n, "min": xs[0], "p50": statistics.median(xs),
            "p95": xs[min(n - 1, int(0.95 * n))], "max": xs[-1],
            "mean": sum(xs) / n}


def kernel_name(fn, q, k, v):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            fn(q, k, v)
    torch.cuda.synchronize()
    best, best_t = None, -1.0
    for e in p.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            t = getattr(e, "cuda_time_total", None) or e.cpu_time_total
            if t > best_t:
                best, best_t = e.key, t
    return best


def main():
    torch.manual_seed(0)
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn((1, 1008, 12, 128), dtype=torch.bfloat16, device="cuda", generator=g)
    k = torch.randn((1, 12096, 12, 128), dtype=torch.bfloat16, device="cuda", generator=g)
    v = torch.randn((1, 12096, 12, 128), dtype=torch.bfloat16, device="cuda", generator=g)

    def control(qq, kk, vv):
        return attention(qq, kk, vv)

    def memeff(qq, kk, vv):
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            return torch.nn.functional.scaled_dot_product_attention(
                qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2),
                attn_mask=None, is_causal=False).transpose(1, 2).contiguous()

    with torch.no_grad():
        control(q, k, v)
        memeff(q, k, v)
        torch.cuda.synchronize()
        out = {"kernels": {"control": kernel_name(control, q, k, v),
                           "mem_efficient": kernel_name(memeff, q, k, v)},
               "blocks": []}
        for b in range(4):
            for name, fn in (("control", control), ("mem_efficient", memeff)):
                ms = calls(fn, q, k, v, 25)
                out["blocks"].append({"block": b, "arm": name,
                                      "summary": summarize(ms)})
        for name in ("control", "mem_efficient"):
            pooled = [x for blk in out["blocks"] if blk["arm"] == name
                      for x in [blk["summary"]["p50"]]]
            allx = []
            for blk in out["blocks"]:
                if blk["arm"] == name:
                    allx.append(blk["summary"]["mean"])
            out[name + "_block_means"] = allx
    with open("probe01b.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"kernels": out["kernels"],
                      "control_means": out["control_block_means"],
                      "memeff_means": out["mem_efficient_block_means"]}, indent=1))


if __name__ == "__main__":
    main()
