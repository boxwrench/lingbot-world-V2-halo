#!/usr/bin/env python3
"""Isolated real Wan2.1 VAE decode measurement for the Strix Halo.

This intentionally bypasses the LingBot transformer.  A latent of
``[16, 4, 60, 104]`` is the 480x832-equivalent control used by the parallel
gfx1201 investigation.  Conv3d modules are wrapped only for timing; their
inputs, weights, padding, and outputs are otherwise untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


def sync() -> None:
    torch.cuda.synchronize()


def rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def memory() -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def instrument(model: nn.Module, table: OrderedDict) -> None:
    """Wrap every nn.Conv3d and record actual F.conv3d input shapes."""
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Conv3d):
            continue
        original = module.forward

        def make_wrapper(original=original, module=module, module_name=module_name):
            def wrapped(x, *args, **kwargs):
                key = (
                    module_name,
                    tuple(x.shape),
                    tuple(module.weight.shape),
                    tuple(module.stride),
                    tuple(module.padding),
                    str(x.dtype),
                )
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                sync()
                t0 = time.perf_counter()
                out = original(x, *args, **kwargs)
                sync()
                seconds = time.perf_counter() - t0
                entry = table.setdefault(key, {"times": [], "transient_bytes": 0})
                entry["times"].append(seconds)
                entry["transient_bytes"] = max(
                    int(entry["transient_bytes"]),
                    int(torch.cuda.max_memory_allocated()) - int(before),
                )
                return out

            return wrapped

        module.forward = make_wrapper()


def runtime_info() -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device_name": torch.cuda.get_device_name(0),
        "gcnArchName": props.gcnArchName,
        "device_properties": str(props),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "miopen_immediate": str(getattr(getattr(torch.backends, "miopen", None), "immediate", "unavailable")),
        "device_total_memory_bytes": int(props.total_memory),
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("MIOPEN", "ROCM", "HIP", "HSA", "ROCR", "PYTORCH", "CUDA"))
        },
    }


def shape_records(table: OrderedDict, decode_count: int) -> list[dict[str, object]]:
    grouped: OrderedDict = OrderedDict()
    for key, entry in table.items():
        module_name, input_shape, weight_shape, stride, padding, dtype = key
        shape_key = (input_shape, weight_shape, stride, padding, dtype)
        group = grouped.setdefault(shape_key, {
            "times": [],
            "transient_bytes": 0,
            "modules": [],
        })
        group["times"].extend(entry["times"])
        group["transient_bytes"] = max(
            int(group["transient_bytes"]), int(entry["transient_bytes"])
        )
        if module_name not in group["modules"]:
            group["modules"].append(module_name)

    total_seconds = sum(sum(group["times"]) for group in grouped.values())
    records = []
    for key, group in grouped.items():
        input_shape, weight_shape, stride, padding, dtype = key
        times = list(group["times"])
        records.append({
            "modules": group["modules"],
            "input_shape": list(input_shape),
            "weight_shape": list(weight_shape),
            "stride": list(stride),
            "padding": list(padding),
            "dtype": dtype,
            "calls": len(times),
            "first_seconds": times[0],
            "repeat_mean_seconds": statistics.mean(times[1:]) if len(times) > 1 else None,
            "total_seconds": sum(times),
            "transient_bytes_proxy": int(group["transient_bytes"]),
            "share_of_conv_seconds": sum(times) / total_seconds,
            "decode_count": decode_count,
        })
    return sorted(records, key=lambda item: -float(item["total_seconds"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--latent-frames", type=int, default=4)
    parser.add_argument("--latent-height", type=int, default=60)
    parser.add_argument("--latent-width", type=int, default=104)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    vae_path = args.model_dir / "Wan2.1_VAE.pth"
    if not vae_path.is_file():
        raise SystemExit(f"missing VAE weights: {vae_path}")

    # Import after the argument/path checks so a missing asset fails clearly.
    from wan.modules.vae2_1 import Wan2_1_VAE

    record: dict[str, object] = {
        "status": "running",
        "runtime": runtime_info(),
        "vae_path": str(vae_path),
        "latent_shape": [16, args.latent_frames, args.latent_height, args.latent_width],
        "runs_requested": args.runs,
    }
    print(json.dumps(record, indent=2), flush=True)

    init_t0 = time.perf_counter()
    vae = Wan2_1_VAE(vae_pth=str(vae_path), device="cuda")
    sync()
    record["initialization_seconds"] = time.perf_counter() - init_t0
    record["parameter_count"] = sum(parameter.numel() for parameter in vae.model.parameters())
    record["parameter_dtype"] = str(next(vae.model.parameters()).dtype)
    record["parameter_device"] = str(next(vae.model.parameters()).device)

    table: OrderedDict = OrderedDict()
    instrument(vae.model, table)
    z = torch.zeros(
        16,
        args.latent_frames,
        args.latent_height,
        args.latent_width,
        device="cuda",
        dtype=torch.float32,
    )
    runs: list[dict[str, object]] = []
    first_output = None
    try:
        for index in range(args.runs):
            torch.cuda.reset_peak_memory_stats()
            sync()
            t0 = time.perf_counter()
            output = vae.decode([z])[0]
            sync()
            seconds = time.perf_counter() - t0
            if first_output is None:
                first_output = output.float().cpu()
            current = {
                "index": index,
                "seconds": seconds,
                "memory": memory(),
                "rss_bytes": rss_bytes(),
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
                "output_finite": bool(torch.isfinite(output).all()),
            }
            runs.append(current)
            print(
                f"decode {index}: {seconds:.3f}s "
                f"alloc={current['memory']['peak_allocated_bytes'] / 2**30:.2f} GiB "
                f"reserved={current['memory']['peak_reserved_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )
            del output

        output_bytes = first_output.numpy().tobytes()
        record["output"] = {
            "shape": list(first_output.shape),
            "dtype": str(first_output.dtype),
            "finite": bool(torch.isfinite(first_output).all()),
            "min": float(first_output.min()),
            "max": float(first_output.max()),
            "mean": float(first_output.mean()),
            "std": float(first_output.std()),
            "sha256_prefix": hashlib.sha256(output_bytes).hexdigest()[:32],
        }
        record["runs"] = runs
        record["cold_seconds"] = runs[0]["seconds"]
        record["warm_mean_seconds"] = statistics.mean(run["seconds"] for run in runs[1:]) if len(runs) > 1 else None
        record["warm_median_seconds"] = statistics.median(run["seconds"] for run in runs[1:]) if len(runs) > 1 else None
        records = shape_records(table, len(runs))
        record["conv3d_unique_shapes"] = len(records)
        record["conv3d_total_seconds"] = sum(sum(entry["times"]) for entry in table.values())
        record["conv3d_shapes"] = records
        record["status"] = "success"
    except Exception as exc:
        record["status"] = "failure"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["runs"] = runs
        record["conv3d_shapes"] = shape_records(table, len(runs)) if table else []
        raise
    finally:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {args.out_json}", flush=True)

    print(json.dumps({
        "status": record["status"],
        "cold_seconds": record.get("cold_seconds"),
        "warm_median_seconds": record.get("warm_median_seconds"),
        "conv3d_total_seconds": record.get("conv3d_total_seconds"),
        "unique_shapes": record.get("conv3d_unique_shapes"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
