#!/usr/bin/env python3
"""Numerically compare the pure compiled tensor islands with eager references.

The shapes and dtypes are the accepted 384x672 LingBot transformer shapes.  No
model modules are loaded here: this is a small deterministic check that the
compiled helpers preserve the eager tensor operations they replace.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from pure_compile_helpers import (
    PureTensorIslands,
    _add,
    _affine,
    _camera_update,
    _gelu_tanh,
    _modulation,
    _scaled_residual_bf16,
    _scaled_residual_float32,
    _silu,
)


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    delta = (actual.float() - expected.float()).abs()
    expected_float = expected.float()
    # Avoid meaningless relative blow-ups around zero; this is a normalized
    # error against a unit scale, useful for BF16 rounding comparisons.
    relative = delta / expected_float.abs().clamp_min(1.0)
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "finite": bool(torch.isfinite(actual).all().item()),
        "expected_min": float(expected_float.min().item()),
        "expected_max": float(expected_float.max().item()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "max_relative": float(relative.max().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("HIP/CUDA device is required")
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(1234)
    b, tokens, dim = 1, 1008, 1536
    f32 = lambda *shape: torch.randn(*shape, device=device, dtype=torch.float32, generator=g)
    bf16 = lambda *shape: torch.randn(*shape, device=device, dtype=torch.bfloat16, generator=g)

    modulation = f32(1, 6, dim)
    e = f32(b, tokens, 6, dim)
    x = bf16(b, tokens, dim)
    y = bf16(b, tokens, dim)
    scale = f32(b, tokens, dim)
    bias = f32(b, tokens, dim)
    hidden = bf16(b, tokens, dim)
    plucker = bf16(b, tokens, dim)
    cam_scale = bf16(b, tokens, dim)
    cam_shift = bf16(b, tokens, dim)

    islands = PureTensorIslands.compile()
    # Force each graph to compile before collecting the comparison values.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compiled = {
        "modulation": islands.modulation(modulation, e),
        "affine": islands.affine(x.float(), scale, bias),
        "scaled_residual_bf16": islands.scaled_residual(x, y, scale),
        "scaled_residual_float32": islands.scaled_residual(x.float(), y.float(), scale),
        "add": islands.add(x, y),
        "silu": islands.silu(x),
        "gelu_tanh": islands.gelu_tanh(x),
        "camera_update": islands.camera_update(x, hidden, plucker, cam_scale, cam_shift),
    }
    torch.cuda.synchronize()
    compile_and_first_call_seconds = time.perf_counter() - t0

    refs = {
        "modulation": _modulation(modulation, e),
        "affine": _affine(x.float(), scale, bias),
        "scaled_residual_bf16": _scaled_residual_bf16(x, y, scale),
        "scaled_residual_float32": _scaled_residual_float32(x.float(), y.float(), scale),
        "add": _add(x, y),
        "silu": _silu(x),
        "gelu_tanh": _gelu_tanh(x),
        "camera_update": _camera_update(x, hidden, plucker, cam_scale, cam_shift),
    }
    torch.cuda.synchronize()

    checks: dict[str, object] = {}
    for name in refs:
        actual = compiled[name]
        expected = refs[name]
        if isinstance(actual, tuple):
            checks[name] = {
                "parts": [_error(a, e) for a, e in zip(actual, expected)],
            }
        else:
            checks[name] = _error(actual, expected)

    report = {
        "device": str(torch.cuda.get_device_name(0)),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "shape_contract": {
            "tokens": tokens,
            "hidden_dim": dim,
            "modulation": [1, 6, dim],
            "conditioning": [b, tokens, 6, dim],
        },
        "compile_and_first_call_seconds": compile_and_first_call_seconds,
        "checks": checks,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
