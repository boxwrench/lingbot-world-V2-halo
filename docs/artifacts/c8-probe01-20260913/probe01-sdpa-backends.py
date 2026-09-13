#!/usr/bin/env python3
"""C8 Probe 0c/1: executing-backend identity + production-equivalence
single-call microbenchmark for the exact production self-attention shape.

Control = unmodified upstream attention() wrapper (the production call).
Arms change exactly one knob: SDPA backend allowlist. Each arm runs in a
fresh process (see runner), 10 untimed warmups + 100 timed calls, HIP-event
timing with wall-clock cross-check. Backend identity via TORCH_LOGS=+sdpa
(stderr, separate from timed region).
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.environ.get("LINGBOT_UPSTREAM_DIR", ""))
from wan.modules.attention import attention  # noqa: E402

SHAPE_Q = (1, 1008, 12, 128)
SHAPE_KV = (1, 12096, 12, 128)


def make_inputs(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(SHAPE_Q, dtype=torch.bfloat16, device="cuda", generator=g)
    k = torch.randn(SHAPE_KV, dtype=torch.bfloat16, device="cuda", generator=g)
    v = torch.randn(SHAPE_KV, dtype=torch.bfloat16, device="cuda", generator=g)
    return q, k, v


def bench(fn, q, k, v, warmup, reps):
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()
    hip_ms, wall_ms = [], []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        s.record()
        out = fn(q, k, v)
        e.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        hip_ms.append(s.elapsed_time(e))
        wall_ms.append((t1 - t0) * 1e3)
    return out, hip_ms, wall_ms


def summarize(xs):
    xs = sorted(xs)
    n = len(xs)
    return {"n": n, "min": xs[0], "p50": statistics.median(xs),
            "p95": xs[min(n - 1, int(0.95 * n))], "max": xs[-1],
            "mean": sum(xs) / n}


def identify(q, k, v):
    flags = {
        "preferred": str(torch.backends.cuda.preferred_rocm_fa_library()),
        "ck_avail": bool(torch.backends.cuda.is_ck_sdpa_available()),
    }
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            attention(q, k, v)
    torch.cuda.synchronize()
    kernels = {}
    cuda_seen = 0
    for e in p.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            cuda_seen += 1
            t = getattr(e, "cuda_time_total", None) or e.cpu_time_total
            kernels[e.key] = kernels.get(e.key, 0.0) + t
    top = sorted(kernels.items(), key=lambda kv: -kv[1])[:8]
    return {"backend_flags": flags, "cuda_events_seen": cuda_seen,
            "top_cuda_kernels_us": [[k, round(t, 1)] for k, t in top]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["control", "flash", "mem_efficient", "math",
                             "identify"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.manual_seed(a.seed)
    q, k, v = make_inputs(a.seed)
    if a.arm == "identify":
        with open(a.out, "w") as f:
            json.dump(identify(q, k, v), f, indent=1)
        print(open(a.out).read())
        return
    backends = {
        "control": [],
        "flash": [SDPBackend.FLASH_ATTENTION],
        "mem_efficient": [SDPBackend.EFFICIENT_ATTENTION],
        "math": [SDPBackend.MATH],
    }[a.arm]
    qt = q.transpose(1, 2).contiguous()
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()

    def control_fn(qq, kk, vv):
        return attention(qq, kk, vv)

    def sdpa_fn(qq, kk, vv):
        return torch.nn.functional.scaled_dot_product_attention(
            qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2),
            attn_mask=None, is_causal=False).transpose(1, 2).contiguous()

    fn = control_fn if a.arm == "control" else sdpa_fn
    failed = None
    try:
        with sdpa_kernel(backends) if backends else torch.no_grad():
            with torch.no_grad():
                out, hip_ms, wall_ms = bench(fn, q, k, v, a.warmup, a.reps)
    except Exception as e:  # noqa: BLE001 - failure mode is the datum
        failed = f"{type(e).__name__}: {str(e)[:200]}"
        out, hip_ms, wall_ms = None, [], []

    row = {"arm": a.arm, "seed": a.seed, "warmup": a.warmup,
           "backends": [str(b) for b in backends],
           "shapes": {"q": list(SHAPE_Q), "kv": list(SHAPE_KV)},
           "dtype": "bf16", "causal": False, "mask": None,
           "failed": failed}
    if not failed:
        row["hip_ms"] = summarize(hip_ms)
        row["wall_ms"] = summarize(wall_ms)
        # Numerical diff vs fp32 math reference, fixed seed.
        with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
            ref = torch.nn.functional.scaled_dot_product_attention(
                qt.float(), kt.float(), vt.float(),
                attn_mask=None, is_causal=False).transpose(1, 2).contiguous()
        diff = (out.float() - ref).abs()
        rel = diff / ref.abs().clamp_min(1e-3)
        row["diff_vs_fp32_math"] = {
            "max_abs": diff.max().item(), "mean_abs": diff.mean().item(),
            "max_rel": rel.max().item(),
            "nonfinite_out": int((~torch.isfinite(out)).sum()),
        }
    with open(a.out, "w") as f:
        json.dump(row, f, indent=1)
    print(json.dumps({k: row[k] for k in
                      ("arm", "failed", "hip_ms", "wall_ms",
                       "diff_vs_fp32_math") if k in row}, indent=1))


if __name__ == "__main__":
    main()
