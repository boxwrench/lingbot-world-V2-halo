#!/usr/bin/env python3
"""Compare canonical causal Wan VAE decode with pinned StreamingTAEHV.

This script intentionally consumes an already accepted LingBot x0 latent
stream. It does not run the DiT or modify any generation state. The canonical
decoder receives the model-space latents exactly as the interactive runner
does; TAEHV receives the corresponding unnormalized Wan VAE latents in
NTCHW order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from run_experiment import current_rss_bytes, host_memory, normalize_video, write_video
from run_interactive import IncrementalCausalDecoder, sync
from taehv import StreamingTAEHV, TAEHV
from wan.modules.vae2_1 import Wan2_1_VAE


TAEHV_COMMIT = "011dfc2112197741c540e0bdd5b7b67bcc930771"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--vae-path", type=Path, required=True)
    p.add_argument("--taehv-dir", type=Path, required=True)
    p.add_argument("--taehv-weights", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    return p


def load_latents(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        latents = payload["latents"]
        metadata = {key: value for key, value in payload.items() if key != "latents"}
    else:
        latents = payload
        metadata = {}
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"latent artifact does not contain a tensor: {type(latents)!r}")
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    if latents.ndim != 5 or latents.shape[1] != 16:
        raise ValueError(f"expected NCTHW latents with 16 channels, got {tuple(latents.shape)}")
    if not torch.isfinite(latents).all():
        raise ValueError("accepted latent artifact contains non-finite values")
    return latents.contiguous(), metadata


def frame_stats(video_01: torch.Tensor) -> dict[str, object]:
    finite = bool(torch.isfinite(video_01).all())
    result: dict[str, object] = {
        "shape": list(video_01.shape),
        "dtype": str(video_01.dtype),
        "finite": finite,
        "min": float(video_01.min()) if finite else None,
        "max": float(video_01.max()) if finite else None,
        "mean": float(video_01.mean()) if finite else None,
        "std": float(video_01.std()) if finite else None,
    }
    if finite and video_01.shape[0] > 1:
        delta = (video_01[1:] - video_01[:-1]).abs().mean(dim=(1, 2, 3))
        result["mean_adjacent_frame_delta"] = float(delta.mean())
        result["max_adjacent_frame_delta"] = float(delta.max())
    return result


def canonical_decode(
    latents: torch.Tensor,
    vae_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, object]]:
    init_t0 = time.perf_counter()
    vae = Wan2_1_VAE(
        z_dim=16,
        vae_pth=str(vae_path),
        dtype=torch.float32,
        device=str(device),
    )
    decoder = IncrementalCausalDecoder(
        vae,
        attention_backend="math",
        dtype_name="fp16" if dtype == torch.float16 else "fp32",
    )
    sync(device)
    init_ms = (time.perf_counter() - init_t0) * 1000.0

    frames: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    for index in range(latents.shape[2]):
        sync(device)
        t0 = time.perf_counter()
        decoded = decoder.decode_latent(
            latents[:, :, index:index + 1].to(device),
            device,
            synchronize=False,
        )
        sync(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        video = normalize_video(decoded)
        frames.append(((video + 1.0) * 0.5).clamp(0, 1))
        rows.append({
            "latent_index": index,
            "decode_ms": elapsed_ms,
            "frames": int(video.shape[0]),
            "finite": bool(torch.isfinite(video).all()),
            "min": float(video.min()),
            "max": float(video.max()),
        })
    result = torch.cat(frames, dim=0)
    decoder.clear()
    del decoder, vae
    return result, {
        "model_init_ms": init_ms,
        "rows": rows,
        "summary": frame_stats(result),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def tae_decode(
    latents: torch.Tensor,
    vae: Wan2_1_VAE,
    weights_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, object]]:
    init_t0 = time.perf_counter()
    tae = TAEHV(str(weights_path)).to(device=device, dtype=dtype).eval()
    streaming = StreamingTAEHV(tae)
    sync(device)
    init_ms = (time.perf_counter() - init_t0) * 1000.0

    # LingBot stores normalized Wan VAE latents as NCTHW. TAEHV's Wan 2.1
    # checkpoint consumes the raw Wan VAE latent in NTCHW, with no additional
    # TAE-specific scale or shift.
    raw = latents.float() / vae.scale[1].float().view(1, 16, 1, 1, 1)
    raw = raw + vae.scale[0].float().view(1, 16, 1, 1, 1)
    raw = raw.to(device=device, dtype=dtype).permute(0, 2, 1, 3, 4).contiguous()

    frames: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    for index in range(raw.shape[1]):
        sync(device)
        t0 = time.perf_counter()
        pending = streaming.decode(raw[:, index:index + 1])
        sync(device)
        first_rgb_ms = (time.perf_counter() - t0) * 1000.0
        output_count = 0
        while pending is not None:
            # StreamingTAEHV returns N1CHW in [0, 1].
            frame = pending[:, 0].permute(0, 2, 3, 1).float().clamp(0, 1).cpu()
            frames.append(frame)
            output_count += int(frame.shape[0])
            sync(device)
            pending = streaming.decode()
        all_rgb_ms = (time.perf_counter() - t0) * 1000.0
        rows.append({
            "latent_index": index,
            "first_rgb_ms": first_rgb_ms,
            "all_rgb_ms": all_rgb_ms,
            "frames": output_count,
            "state_ready": pending is None,
        })
    result = torch.cat(frames, dim=0)
    return result, {
        "model_init_ms": init_ms,
        "architecture": {
            "arch_name": tae.arch_name,
            "patch_size": tae.patch_size,
            "latent_channels": tae.latent_channels,
            "t_downscale": tae.t_downscale,
            "t_upscale": tae.t_upscale,
            "frames_to_trim": tae.frames_to_trim,
        },
        "rows": rows,
        "summary": frame_stats(result),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def main() -> int:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    weights = args.taehv_weights or (args.taehv_dir / "taew2_1.pth")
    if not weights.is_file():
        raise FileNotFoundError(weights)

    latents, latent_metadata = load_latents(args.latent_path)
    torch.cuda.reset_peak_memory_stats(device)
    # Run the two decoders in separate fresh model/state instances. The TAE
    # comparison is presentation-only and never mutates DiT/KV state.
    canonical, canonical_metrics = canonical_decode(latents, args.vae_path, device, dtype)
    vae_for_contract = Wan2_1_VAE(
        z_dim=16,
        vae_pth=str(args.vae_path),
        dtype=torch.float32,
        device=str(device),
    )
    sync(device)
    tae, tae_metrics = tae_decode(latents, vae_for_contract, weights, device, dtype)
    sync(device)

    if canonical.shape != tae.shape:
        raise RuntimeError(f"canonical/TAE frame shape mismatch: {canonical.shape} vs {tae.shape}")
    difference = (canonical - tae).abs()
    per_frame = difference.mean(dim=(1, 2, 3))
    canonical_video = canonical * 2.0 - 1.0
    tae_video = tae * 2.0 - 1.0
    write_video(canonical_video, args.output_dir / "canonical.mp4")
    write_video(tae_video, args.output_dir / "taehv.mp4")
    torch.save(canonical, args.output_dir / "canonical_frames_01.pt")
    torch.save(tae, args.output_dir / "taehv_frames_01.pt")

    report = {
        "status": "success",
        "latent_path": str(args.latent_path),
        "latent_metadata": latent_metadata,
        "latent_contract": {
            "input_shape": list(latents.shape),
            "input_layout": "NCTHW",
            "input_dtype": str(latents.dtype),
            "model_space": "LingBot accepted x0 / normalized Wan VAE latent",
            "canonical_transform": "z = x0 / vae.scale[1] + vae.scale[0]",
            "taehv_transform": "same canonical z, then NCTHW -> NTCHW; no additional TAE scale/shift",
            "taehv_input_shape": [1, int(latents.shape[2]), 16, int(latents.shape[3]), int(latents.shape[4])],
            "spatial_latent": [int(latents.shape[3]), int(latents.shape[4])],
        },
        "software": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(device),
        },
        "taehv": {
            "repository": "https://github.com/madebyollin/taehv",
            "directory": str(args.taehv_dir),
            "commit": git_head(args.taehv_dir),
            "expected_commit": TAEHV_COMMIT,
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
        },
        "canonical": canonical_metrics,
        "taehv_streaming": tae_metrics,
        "comparison": {
            "shape": list(canonical.shape),
            "finite_canonical": bool(torch.isfinite(canonical).all()),
            "finite_taehv": bool(torch.isfinite(tae).all()),
            "mean_abs_difference_01": float(difference.mean()),
            "max_abs_difference_01": float(difference.max()),
            "mean_abs_difference_by_frame": [float(value) for value in per_frame],
            "canonical_temporal_delta_01": float((canonical[1:] - canonical[:-1]).abs().mean()),
            "taehv_temporal_delta_01": float((tae[1:] - tae[:-1]).abs().mean()),
        },
        "memory": {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "rss_bytes": current_rss_bytes(),
            "host_memory": host_memory(),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "frames": report["comparison"]["shape"][0],
        "canonical_mean_abs_difference": report["comparison"]["mean_abs_difference_01"],
        "canonical_model_init_ms": report["canonical"]["model_init_ms"],
        "taehv_model_init_ms": report["taehv_streaming"]["model_init_ms"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
