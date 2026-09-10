#!/usr/bin/env python3
"""Compare Wan's upstream full causal decode with incremental feature-cache decode."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def sync() -> None:
    torch.cuda.synchronize()


def stats(tensor: torch.Tensor) -> dict[str, object]:
    finite = bool(torch.isfinite(tensor).all())
    # Keep the diagnostic useful for non-finite tensors too.  Calling min/max
    # on a tensor containing NaN would otherwise hide the range completely.
    finite_values = tensor[torch.isfinite(tensor)]
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
        "absmax": float(tensor.abs().max()) if finite else None,
        "min": float(finite_values.min()) if finite_values.numel() else None,
        "max": float(finite_values.max()) if finite_values.numel() else None,
        "mean": float(tensor.float().mean()) if finite else None,
        "std": float(tensor.float().std()) if finite else None,
    }


def cache_stats(model) -> list[dict[str, object]]:
    """Describe decoder feature-cache state without copying its contents."""
    result: list[dict[str, object]] = []
    for index, value in enumerate(model._feat_map):
        if value is None or isinstance(value, str):
            result.append({"index": index, "kind": value or "none"})
            continue
        row: dict[str, object] = {
            "index": index,
            "kind": "tensor",
            "data_ptr": int(value.data_ptr()),
        }
        row.update(stats(value))
        result.append(row)
    return result


def reference_slice(full: torch.Tensor, index: int) -> torch.Tensor:
    """Return the batch decoder's frames corresponding to one latent frame."""
    start = 0 if index == 0 else 1 + 4 * (index - 1)
    stop = 1 if index == 0 else start + 4
    return full[:, start:stop]


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    row: dict[str, object] = {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "same_shape": tuple(actual.shape) == tuple(expected.shape),
        "actual": stats(actual),
        "expected": stats(expected),
    }
    if row["same_shape"]:
        delta = (actual - expected).abs()
        denom = expected.abs().clamp_min(1e-6)
        row.update({
            "finite_difference": bool(torch.isfinite(delta).all()),
            "max_abs_diff": float(delta.max()) if torch.isfinite(delta).all() else None,
            "mean_abs_diff": float(delta.mean()) if torch.isfinite(delta).all() else None,
            "max_relative_diff": float((delta / denom).max()) if torch.isfinite(delta).all() else None,
        })
    return row


def attention_context(backend: str):
    return sdpa_kernel(SDPBackend.MATH) if backend == "math" else nullcontext()


def configure_dtype(vae, name: str) -> torch.dtype:
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]
    vae.model.to(dtype=dtype)
    vae.mean = vae.mean.to(dtype=dtype)
    vae.std = vae.std.to(dtype=dtype)
    vae.scale = [vae.mean, 1.0 / vae.std]
    return dtype


def incremental_decode(
    vae,
    z: torch.Tensor,
    full: torch.Tensor,
    full_conv2: torch.Tensor,
    attention_backend: str,
) -> tuple[torch.Tensor, float, list[dict[str, object]]]:
    model = vae.model
    model.clear_cache()
    outputs: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    sync()
    t0 = time.perf_counter()
    with attention_context(attention_backend), torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
        z_dim = model.z_dim
        compute_dtype = next(model.parameters()).dtype
        for index in range(z.shape[1]):
            # z is [C,T,H,W] in the public Wan VAE API.  The internal model
            # requires [B,C,T,H,W]; omitting this unsqueeze silently makes the
            # Conv3d interpret channels/time/spatial axes incorrectly.
            frame = z[:, index:index + 1, :, :].unsqueeze(0)
            frame = frame.to(compute_dtype) / vae.scale[1].view(1, z_dim, 1, 1, 1)
            frame = frame + vae.scale[0].view(1, z_dim, 1, 1, 1)
            x = model.conv2(frame)
            model._conv_idx = [0]
            out = model.decoder(x, feat_cache=model._feat_map, feat_idx=model._conv_idx)
            sync()
            out = out.float().clamp_(-1, 1).squeeze(0).cpu()
            outputs.append(out)
            rows.append({
                "index": index,
                "input_latent": stats(frame),
                "output": stats(out),
                "conv2": stats(x),
                "conv2_reference_comparison": compare(
                    x.cpu(), full_conv2[:, :, index:index + 1]
                ),
                "cache": cache_stats(model),
                "reference_comparison": compare(out, reference_slice(full, index)),
            })
    return torch.cat(outputs, dim=1), time.perf_counter() - t0, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--latent-frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--sdpa-backend", choices=("math", "default"), default="math")
    parser.add_argument("--vae-dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    from wan.modules.vae2_1 import Wan2_1_VAE

    torch.cuda.set_device(0)
    vae = Wan2_1_VAE(vae_pth=str(args.model_dir / "Wan2.1_VAE.pth"), device="cuda")
    compute_dtype = configure_dtype(vae, args.vae_dtype)
    z = (torch.randn(16, args.latent_frames, 58, 104, device="cuda", dtype=torch.float32) * args.scale).to(compute_dtype)

    sync()
    t0 = time.perf_counter()
    with attention_context(args.sdpa_backend), torch.no_grad():
        full = vae.decode([z])[0].float().cpu()
    sync()
    full_seconds = time.perf_counter() - t0

    # The public wrapper uses legacy autocast(dtype=torch.float32).  Also run
    # the same model.decode loop with autocast explicitly disabled so the
    # comparison distinguishes autocast behavior from cache behavior.
    vae.model.clear_cache()
    direct_t0 = time.perf_counter()
    with attention_context(args.sdpa_backend), torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
        direct = vae.model.decode(z.unsqueeze(0), vae.scale).float().clamp_(-1, 1).squeeze(0).cpu()
    sync()
    direct_seconds = time.perf_counter() - direct_t0

    # Recompute the batch conv2 result independently.  This separates a
    # possible shape-dependent Conv3d numerical difference from the decoder
    # feature-cache path itself.
    vae.model.clear_cache()
    with attention_context(args.sdpa_backend), torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
        normalized = z.to(compute_dtype) / vae.scale[1].view(1, 16, 1, 1, 1)
        normalized = normalized + vae.scale[0].view(1, 16, 1, 1, 1)
        full_conv2 = vae.model.conv2(normalized).float().cpu()
    sync()

    incremental, incremental_seconds, rows = incremental_decode(
        vae, z, direct, full_conv2, args.sdpa_backend
    )
    same_shape = tuple(full.shape) == tuple(incremental.shape)
    delta = (full - incremental).abs() if same_shape else None
    direct_delta = (direct - incremental).abs() if tuple(direct.shape) == tuple(incremental.shape) else None
    first_divergence = None
    for row in rows:
        comparison = row["reference_comparison"]
        if not comparison["same_shape"] or not comparison["actual"]["finite"] or not comparison["expected"]["finite"]:
            first_divergence = {"index": row["index"], "reason": "shape_or_nonfinite", "row": row}
            break
        # This is deliberately a reporting threshold, not an acceptance
        # criterion.  The raw per-step errors remain in the JSON.
        if comparison.get("max_abs_diff") is not None and comparison["max_abs_diff"] > 1e-3:
            first_divergence = {"index": row["index"], "reason": "max_abs_diff_gt_1e-3", "row": row}
            break
    result = {
        "status": "success",
        "latent_shape": list(z.shape),
        "latent_scale": args.scale,
        "sdpa_backend": args.sdpa_backend,
        "vae_dtype": str(compute_dtype),
        "full_seconds": full_seconds,
        "direct_reference_seconds": direct_seconds,
        "incremental_seconds": incremental_seconds,
        "full": stats(full),
        "direct_reference": stats(direct),
        "incremental": stats(incremental),
        "same_shape": same_shape,
        "first_divergence": first_divergence,
        "max_abs_diff": float(delta.max()) if delta is not None and torch.isfinite(delta).all() else None,
        "mean_abs_diff": float(delta.mean()) if delta is not None and torch.isfinite(delta).all() else None,
        "direct_reference_max_abs_diff": float(direct_delta.max()) if direct_delta is not None and torch.isfinite(direct_delta).all() else None,
        "direct_reference_mean_abs_diff": float(direct_delta.mean()) if direct_delta is not None and torch.isfinite(direct_delta).all() else None,
        "per_latent": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
