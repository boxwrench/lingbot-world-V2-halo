#!/usr/bin/env python3
"""GPU numerical check for the C6A norm islands on live shapes/dtypes.

Compares compiled helpers against eager references under the same bf16
autocast the live path runs in. Gate: max_abs <= 2e-6 (fixture order).
Run on gfx1151 before any latency claim.
"""

from __future__ import annotations

import argparse
import json

import torch

from check_pure_compile_helpers import _error
from pure_compile_helpers import PureTensorIslands, _norm_affine, _rmsnorm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("HIP/CUDA device is required")
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(20260912)
    b, tokens, dim = 1, 1008, 1536
    x = torch.randn(b, tokens, dim, device=device, dtype=torch.bfloat16, generator=g)
    scale = torch.randn(b, tokens, dim, device=device, dtype=torch.float32, generator=g)
    bias = torch.randn(b, tokens, dim, device=device, dtype=torch.float32, generator=g)
    weight = torch.ones(dim, device=device, dtype=torch.float32)

    islands = PureTensorIslands.compile()
    checks: dict[str, object] = {}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        torch.cuda.synchronize()
        got_affine = islands.norm_affine(x, scale, bias)
        got_rms = islands.rmsnorm(x, weight, 1e-6)
        torch.cuda.synchronize()
        ref_affine = _norm_affine(x, scale, bias)
        ref_rms = _rmsnorm(x, weight, 1e-6)
        torch.cuda.synchronize()
    checks["norm_affine"] = _error(got_affine, ref_affine)
    checks["rmsnorm"] = _error(got_rms, ref_rms)
    gate = all(c["max_abs"] <= 2e-6 for c in checks.values())
    report = {"gate_pass": bool(gate), "threshold_max_abs": 2e-6, "checks": checks}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
