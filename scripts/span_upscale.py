#!/usr/bin/env python3
"""Offline qualification of the official SPAN x2 checkpoint.

This intentionally loads only the official ``basicsr/archs/span_arch.py``
file from a pinned SPAN checkout.  Importing the historical BasicSR package
would pull in obsolete torchvision APIs that are unrelated to SPAN inference
and would contaminate the accepted LingBot environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


SPAN_REVISION = "c77a5917759f09e66fbc7124220c5afc5ee221e5"
SPAN_CHECKPOINT_SHA256 = "561fd5cf419a23d4de1231ce258180f61aee4aa8caa1aaaa783769c7301847bc"
SPAN_CHECKPOINT_SOURCE = (
    "https://drive.google.com/file/d/1iYUA2TzKuxI0vzmA-UXr_nB43XgPOXUg/view?usp=sharing"
)
SPAN_CHECKPOINT_ARCHIVE_ENTRY = "spanx2_ch48.pth"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def load_official_span_class(repo_dir: Path):
    """Load the official architecture without importing BasicSR datasets."""
    source = repo_dir / "basicsr" / "archs" / "span_arch.py"
    if not source.is_file():
        raise FileNotFoundError(f"missing official SPAN architecture: {source}")

    # span_arch.py needs only ARCH_REGISTRY for its decorator.  BasicSR's
    # package __init__ imports legacy datasets and is deliberately bypassed.
    registry_module = types.ModuleType("basicsr.utils.registry")

    class Registry:
        def register(self, obj=None, suffix=None):
            del suffix

            def decorator(value):
                return value

            return decorator if obj is None else obj

    registry_module.ARCH_REGISTRY = Registry()
    utils_module = types.ModuleType("basicsr.utils")
    utils_module.registry = registry_module
    basicsr_module = types.ModuleType("basicsr")
    basicsr_module.utils = utils_module
    sys.modules["basicsr"] = basicsr_module
    sys.modules["basicsr.utils"] = utils_module
    sys.modules["basicsr.utils.registry"] = registry_module

    spec = importlib.util.spec_from_file_location("span_arch_official", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official SPAN source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["span_arch_official"] = module
    spec.loader.exec_module(module)
    return module.SPAN


def load_model(repo_dir: Path, weight: Path, device: torch.device, dtype: torch.dtype):
    if sha256(weight) != SPAN_CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint SHA256 does not match official artifact: {weight}")
    span_class = load_official_span_class(repo_dir)
    model = span_class(3, 3, feature_channels=48, upscale=2)
    checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
    state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    model.load_state_dict(state, strict=True)
    model.eval().to(device=device, dtype=dtype)
    return model


def to_numpy_rgb(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().float().clamp(0, 1).cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected RGB frame, got {array.shape}")
    return np.ascontiguousarray((array * 255.0).round().astype(np.uint8))


def frame_to_input(frame: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if frame.ndim != 3 or tuple(frame.shape[-1:]) != (3,):
        raise ValueError(f"expected HWC RGB frame, got {tuple(frame.shape)}")
    return frame.permute(2, 0, 1).unsqueeze(0).contiguous().to(device=device, dtype=dtype)


def lanczos_2x(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    return np.asarray(
        Image.fromarray(array, mode="RGB").resize(
            (width * 2, height * 2), Image.Resampling.LANCZOS
        ),
        dtype=np.uint8,
    )


def save_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array, mode="RGB").save(path)


def load_png_sequence(directory: Path, limit: int) -> list[np.ndarray]:
    paths = sorted(directory.glob("frame-*.png"))[:limit]
    return [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths]


def temporal_stats(frames: list[np.ndarray]) -> dict[str, float | None]:
    if len(frames) < 2:
        return {"mean_adjacent_abs": None, "p95_adjacent_abs": None}
    values = [
        float(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)).mean() / 255.0)
        for i in range(1, len(frames))
    ]
    return {
        "mean_adjacent_abs": float(np.mean(values)),
        "p95_adjacent_abs": float(np.percentile(values, 95)),
    }


def write_contact_sheet(
    source: list[np.ndarray],
    lanczos: list[np.ndarray],
    realesrgan: list[np.ndarray] | None,
    span: list[np.ndarray],
    path: Path,
) -> None:
    indices = np.linspace(0, len(source) - 1, min(6, len(source)), dtype=int).tolist()
    labels = ["TAE RGB", "Lanczos 2x", "RealESRGAN x2", "SPAN x2"]
    columns = [source, lanczos, realesrgan, span]
    if realesrgan is None:
        labels = ["TAE RGB", "Lanczos 2x", "SPAN x2"]
        columns = [source, lanczos, span]
    thumb_w, thumb_h, label_h = 336, 192, 24
    sheet = Image.new(
        "RGB", (thumb_w * len(columns), (thumb_h + label_h) * len(indices)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for row, index in enumerate(indices):
        for col, (label, frames) in enumerate(zip(labels, columns)):
            image = Image.fromarray(frames[index], mode="RGB").resize(
                (thumb_w, thumb_h), Image.Resampling.LANCZOS
            )
            x = col * thumb_w
            y = row * (thumb_h + label_h)
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + thumb_h + 3), f"{label}  frame {index}", fill="black")
    sheet.save(path)


def write_video(frame_dir: Path, path: Path) -> None:
    if not list(frame_dir.glob("frame-*.png")):
        return
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", "15", "-i", str(frame_dir / "frame-%04d.png"),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def stats(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="HWC RGB float .pt tensor")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path(".upstream/SPAN"))
    parser.add_argument("--weight", type=Path, default=Path("models/upscalers/spanx2_ch48.pth"))
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("results/raw/spatial-upscale-20260910/fp16/realesrgan-fp16"),
        help="existing RealESRGAN x2 PNGs for the four-way contact sheet",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 4 or args.warmup < 1 or args.warmup >= args.limit:
        raise SystemExit("require limit >= 4 and 1 <= warmup < limit")
    if not args.input.is_file() or not args.weight.is_file():
        raise SystemExit("input frame tensor and official SPAN weight file are required")
    actual_revision = git_revision(args.repo_dir)
    if actual_revision != SPAN_REVISION:
        raise SystemExit(f"SPAN checkout is not pinned: expected {SPAN_REVISION}, got {actual_revision}")

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    frames = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise SystemExit(f"expected [T,H,W,3] tensor, got {type(frames).__name__} {getattr(frames, 'shape', None)}")
    frames = frames[: args.limit].float().contiguous()
    limit = int(frames.shape[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    span_dir = args.output_dir / f"span-{args.dtype}"
    lanczos_dir = args.output_dir / "lanczos"
    source_dir = args.output_dir / "source"
    for directory in (span_dir, lanczos_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)
    reference_frames = load_png_sequence(args.reference_dir, limit) if args.reference_dir.is_dir() else None
    if reference_frames is not None and len(reference_frames) != limit:
        reference_frames = None

    model_load_start = time.perf_counter()
    allocated_before = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    model = load_model(args.repo_dir, args.weight, device, dtype)
    sync(device)
    model_load_seconds = time.perf_counter() - model_load_start
    allocated_after_load = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    source_arrays: list[np.ndarray] = []
    lanczos_arrays: list[np.ndarray] = []
    span_arrays: list[np.ndarray] = []
    timing_rows: list[dict[str, float | int]] = []
    raw_ranges: list[dict[str, float]] = []
    for index, frame in enumerate(frames):
        source = to_numpy_rgb(frame)
        lanczos = lanczos_2x(source)
        source_arrays.append(source)
        lanczos_arrays.append(lanczos)
        save_png(source, source_dir / f"frame-{index:04d}.png")
        save_png(lanczos, lanczos_dir / f"frame-{index:04d}.png")

        sync(device)
        conversion_start = time.perf_counter()
        model_input = frame_to_input(frame, device, dtype)
        sync(device)
        input_conversion_ms = (time.perf_counter() - conversion_start) * 1000.0

        sync(device)
        network_start = time.perf_counter()
        with torch.inference_mode():
            model_output = model(model_input)
        sync(device)
        network_ms = (time.perf_counter() - network_start) * 1000.0
        raw_ranges.append({"min": float(model_output.min()), "max": float(model_output.max())})

        output_start = time.perf_counter()
        span_frame = to_numpy_rgb(model_output)
        sync(device)
        output_conversion_ms = (time.perf_counter() - output_start) * 1000.0
        span_arrays.append(span_frame)
        save_png(span_frame, span_dir / f"frame-{index:04d}.png")
        timing_rows.append(
            {
                "frame": index,
                "input_conversion_ms": input_conversion_ms,
                "network_ms": network_ms,
                "output_conversion_ms": output_conversion_ms,
                "frame_ready_ms": input_conversion_ms + network_ms + output_conversion_ms,
            }
        )

    write_contact_sheet(
        source_arrays,
        lanczos_arrays,
        reference_frames,
        span_arrays,
        args.output_dir / "contact-sheet.png",
    )
    write_video(span_dir, args.output_dir / f"span-x2-{args.dtype}.mp4")
    warm_rows = timing_rows[args.warmup :]
    network_values = [float(row["network_ms"]) for row in warm_rows]
    frame_ready_values = [float(row["frame_ready_ms"]) for row in warm_rows]
    output = {
        "provenance": {
            "repository": "https://github.com/hongyuanyu/SPAN",
            "revision": actual_revision,
            "expected_revision": SPAN_REVISION,
            "architecture_source": str(args.repo_dir / "basicsr/archs/span_arch.py"),
            "architecture": "SPAN(feature_channels=48, blocks=6, upscale=2)",
            "checkpoint": str(args.weight),
            "checkpoint_sha256": sha256(args.weight),
            "checkpoint_source": SPAN_CHECKPOINT_SOURCE,
            "checkpoint_archive_entry": SPAN_CHECKPOINT_ARCHIVE_ENTRY,
            "license": "Apache-2.0",
        },
        "environment": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": args.dtype,
        },
        "input_contract": {
            "source": str(args.input),
            "shape": list(frames.shape),
            "dtype": "torch.float32",
            "layout": "THWC",
            "color": "RGB",
            "range": [float(frames.min()), float(frames.max())],
            "network_input": "NCHW RGB float on device; official SPAN normalization in forward; no uint8/BGR round trip",
            "output_shape": [1, 3, int(frames.shape[1] * 2), int(frames.shape[2] * 2)],
        },
        "model": {
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "parameter_bytes": int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())),
            "raw_output_range": raw_ranges,
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "first_inference": timing_rows[0],
            "warmup_frames_excluded": args.warmup,
            "warm_network": stats(network_values),
            "warm_frame_ready": stats(frame_ready_values),
            "warm_mean_input_conversion_ms": float(np.mean([row["input_conversion_ms"] for row in warm_rows])),
            "warm_mean_output_conversion_ms": float(np.mean([row["output_conversion_ms"] for row in warm_rows])),
            "all_rows": timing_rows,
        },
        "memory": {
            "allocated_before_model_bytes": allocated_before,
            "allocated_after_model_load_bytes": allocated_after_load,
            "weight_load_delta_bytes": allocated_after_load - allocated_before,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            "peak_process_rss_bytes": rss_bytes(),
        },
        "temporal": {
            "source": temporal_stats(source_arrays),
            "lanczos": temporal_stats(lanczos_arrays),
            "realesrgan_reference": temporal_stats(reference_frames) if reference_frames else None,
            "span": temporal_stats(span_arrays),
        },
        "artifacts": {
            "source_dir": str(source_dir),
            "lanczos_dir": str(lanczos_dir),
            "reference_dir": str(args.reference_dir) if reference_frames else None,
            "span_dir": str(span_dir),
            "contact_sheet": str(args.output_dir / "contact-sheet.png"),
            "video": str(args.output_dir / f"span-x2-{args.dtype}.mp4"),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": "success", "metrics": str(args.output_dir / "metrics.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
