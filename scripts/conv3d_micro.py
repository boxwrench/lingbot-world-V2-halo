#!/usr/bin/env python3
"""Matched Wan VAE Conv3d microbenchmark for AMD architecture comparison.

The cases intentionally preserve the odd dimensions used by the R9700/gfx1201
investigation.  Case B is the tensor that a causal VAE convolution passes to
``F.conv3d`` after its explicit temporal/spatial pre-padding.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


CASES = {
    "A_prepad_fast_control": {
        "input": (1, 96, 4, 480, 832),
        "weight": (96, 96, 3, 3, 3),
        "padding": (0, 0, 0),
    },
    "B_production_padded": {
        "input": (1, 96, 6, 482, 834),
        "weight": (96, 96, 3, 3, 3),
        "padding": (0, 0, 0),
    },
    "C_production_rgb_out": {
        "input": (1, 96, 6, 482, 834),
        "weight": (3, 96, 3, 3, 3),
        "padding": (0, 0, 0),
    },
    "D_padding_argument": {
        "input": (1, 96, 4, 480, 832),
        "weight": (96, 96, 3, 3, 3),
        "padding": (1, 1, 1),
    },
}


def sync() -> None:
    torch.cuda.synchronize()


def rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def mem_snapshot() -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def output_shape(input_shape: tuple[int, ...], weight_shape: tuple[int, ...], padding: tuple[int, ...]) -> tuple[int, ...]:
    # stride=1, dilation=1, groups=1 for all cases in this reproducer.
    return (
        input_shape[0],
        weight_shape[0],
        input_shape[2] + 2 * padding[0] - weight_shape[2] + 1,
        input_shape[3] + 2 * padding[1] - weight_shape[3] + 1,
        input_shape[4] + 2 * padding[2] - weight_shape[4] + 1,
    )


def gflop_count(weight_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> float:
    # Count multiply-add as two floating-point operations.
    return 2.0 * weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3] * weight_shape[4] * out_shape[0] * out_shape[2] * out_shape[3] * out_shape[4] / 1e9


def bench_case(name: str, spec: dict[str, tuple[int, ...]], warmup: int, iters: int, dtype: torch.dtype) -> dict[str, object]:
    input_shape = tuple(spec["input"])
    weight_shape = tuple(spec["weight"])
    padding = tuple(spec["padding"])
    expected = output_shape(input_shape, weight_shape, padding)
    x = torch.randn(*input_shape, device="cuda", dtype=dtype)
    weight = torch.randn(*weight_shape, device="cuda", dtype=dtype)
    bias = torch.randn(weight_shape[0], device="cuda", dtype=dtype)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    record: dict[str, object] = {
        "name": name,
        "input_shape": list(input_shape),
        "weight_shape": list(weight_shape),
        "padding": list(padding),
        "dtype": str(dtype),
        "input_strides": list(x.stride()),
        "input_contiguous": bool(x.is_contiguous()),
        "expected_output_shape": list(expected),
        "rss_before_bytes": rss_bytes(),
    }

    try:
        sync()
        t0 = time.perf_counter()
        y = F.conv3d(x, weight, bias, padding=padding)
        sync()
        first_seconds = time.perf_counter() - t0

        warmup_seconds: list[float] = []
        for _ in range(warmup):
            sync()
            t0 = time.perf_counter()
            y = F.conv3d(x, weight, bias, padding=padding)
            sync()
            warmup_seconds.append(time.perf_counter() - t0)

        samples: list[float] = []
        for _ in range(iters):
            sync()
            t0 = time.perf_counter()
            y = F.conv3d(x, weight, bias, padding=padding)
            sync()
            samples.append(time.perf_counter() - t0)

        actual = tuple(y.shape)
        gflops = gflop_count(weight_shape, actual)
        median = statistics.median(samples)
        record.update({
            "first_seconds": first_seconds,
            "warmup_seconds": warmup_seconds,
            "warm_seconds": samples,
            "warm_median_seconds": median,
            "warm_min_seconds": min(samples),
            "warm_max_seconds": max(samples),
            "warm_p10_seconds": sorted(samples)[max(0, int(len(samples) * 0.1) - 1)],
            "warm_p90_seconds": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.9))],
            "output_shape": list(actual),
            "output_finite": bool(torch.isfinite(y).all()),
            "output_sum": float(y.float().sum()),
            "gflop": gflops,
            "tflops": gflops / 1000.0 / median,
            "memory": mem_snapshot(),
            "rss_after_bytes": rss_bytes(),
        })
        print(
            f"{name:25s} first={first_seconds:.4f}s "
            f"warm_median={median:.4f}s "
            f"TFLOP/s={record['tflops']:.3f} "
            f"out={list(actual)}",
            flush=True,
        )
    except Exception as exc:
        record.update({
            "error": f"{type(exc).__name__}: {exc}",
            "memory": mem_snapshot(),
            "rss_after_bytes": rss_bytes(),
        })
        print(f"{name:25s} FAILED {record['error']}", flush=True)

    del x, weight, bias
    if "y" in locals():
        del y
    torch.cuda.empty_cache()
    return record


def runtime_info() -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device_name": torch.cuda.get_device_name(0),
        "gcnArchName": props.gcnArchName,
        "device_properties": str(props),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "miopen_immediate": str(getattr(getattr(torch.backends, "miopen", None), "immediate", "unavailable")),
        "torch_device_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("MIOPEN", "ROCM", "HIP", "HSA", "ROCR", "PYTORCH", "CUDA"))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", default="strix-gfx1151")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    torch.cuda.set_device(0)
    rec: dict[str, object] = {
        "lane": args.lane,
        "warmup": args.warmup,
        "iters": args.iters,
        "runtime": runtime_info(),
        "cases": [],
    }
    print(json.dumps({"lane": args.lane, "runtime": rec["runtime"]}, indent=2), flush=True)
    for name in args.cases.split(","):
        if name not in CASES:
            raise SystemExit(f"unknown case {name!r}; choose from {', '.join(CASES)}")
        rec["cases"].append(bench_case(name, CASES[name], args.warmup, args.iters, dtype))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
