#!/usr/bin/env python3
"""Offline RealESRGAN_x2plus qualification on captured LingBot RGB frames.

The network definition is the small inference subset of BasicSR's official
``RRDBNet`` at the pinned BasicSR revision recorded in the output manifest.
The Real-ESRGAN repository and checkpoint are kept as external, pinned
artifacts; this script deliberately does not import the historical BasicSR
package so it can run on the Python 3.12 ROCm environment without changing
the accepted LingBot dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """BasicSR arch_util.default_init_weights (unused after checkpoint load)."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for submodule in module.modules():
            if isinstance(submodule, nn.Conv2d):
                nn.init.kaiming_normal_(submodule.weight, **kwargs)
                submodule.weight.data *= scale
                if submodule.bias is not None:
                    submodule.bias.data.fill_(bias_fill)


def make_layer(block, count, **kwargs):
    return nn.Sequential(*(block(**kwargs) for _ in range(count)))


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    batch, channels, height, width = x.shape
    assert height % scale == 0 and width % scale == 0
    out_height, out_width = height // scale, width // scale
    viewed = x.view(batch, channels, out_height, scale, out_width, scale)
    return viewed.permute(0, 1, 3, 5, 2, 4).reshape(
        batch, channels * scale**2, out_height, out_width
    )


class ResidualDenseBlock(nn.Module):
    """BasicSR RRDB residual dense block, copied from the pinned source."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """BasicSR RRDBNet x2plus architecture."""

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB; this is a process high-water mark.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def git_revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_numpy_rgb(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().float().clamp(0, 1).cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB frame, got {array.shape}")
    return np.ascontiguousarray((array * 255.0).round().astype(np.uint8))


def frame_to_input(frame: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if frame.ndim != 3 or tuple(frame.shape[-1:]) != (3,):
        raise ValueError(f"expected HWC RGB frame, got {tuple(frame.shape)}")
    return frame.permute(2, 0, 1).unsqueeze(0).contiguous().to(device=device, dtype=dtype)


def save_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array, mode="RGB").save(path)


def lanczos_2x(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    image = Image.fromarray(array, mode="RGB")
    return np.asarray(
        image.resize((width * 2, height * 2), Image.Resampling.LANCZOS), dtype=np.uint8
    )


def temporal_stats(frames: list[np.ndarray]) -> dict[str, float | None]:
    if len(frames) < 2:
        return {"mean_adjacent_abs": None, "p95_adjacent_abs": None}
    deltas = [
        np.abs(frames[index].astype(np.float32) - frames[index - 1].astype(np.float32)).mean() / 255.0
        for index in range(1, len(frames))
    ]
    return {
        "mean_adjacent_abs": float(np.mean(deltas)),
        "p95_adjacent_abs": float(np.percentile(deltas, 95)),
    }


def write_contact_sheet(
    source: list[np.ndarray],
    lanczos: list[np.ndarray],
    learned: list[np.ndarray],
    path: Path,
) -> None:
    indices = np.linspace(0, len(source) - 1, min(6, len(source)), dtype=int).tolist()
    thumb_w, thumb_h, label_h = 336, 192, 24
    sheet = Image.new("RGB", (thumb_w * 3, (thumb_h + label_h) * len(indices)), "white")
    draw = ImageDraw.Draw(sheet)
    labels = ("TAE RGB", "Lanczos 2x", "RealESRGAN x2")
    for row, index in enumerate(indices):
        for col, (label, frames) in enumerate(zip(labels, (source, lanczos, learned))):
            image = Image.fromarray(frames[index], mode="RGB").resize(
                (thumb_w, thumb_h), Image.Resampling.LANCZOS
            )
            x = col * thumb_w
            y = row * (thumb_h + label_h)
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + thumb_h + 3), f"{label}  frame {index}", fill="black")
    sheet.save(path)


def load_model(weight: Path, device: torch.device, dtype: torch.dtype) -> nn.Module:
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    checkpoint = torch.load(weight, map_location="cpu")
    if "params_ema" in checkpoint:
        state = checkpoint["params_ema"]
    elif "params" in checkpoint:
        state = checkpoint["params"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    if dtype == torch.float16:
        model.half()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="saved HWC RGB float frame tensor (.pt)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight", type=Path, default=Path("models/upscalers/RealESRGAN_x2plus.pth"))
    parser.add_argument("--repo-dir", type=Path, default=Path(".upstream/Real-ESRGAN"))
    parser.add_argument("--basicsr-dir", type=Path, default=Path(".upstream/BasicSR"))
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
        raise SystemExit("input frame tensor and official weight file are required")
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    frames = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise SystemExit(f"expected [T,H,W,3] tensor, got {type(frames).__name__} {getattr(frames, 'shape', None)}")
    frames = frames[: args.limit].float().contiguous()
    if frames.shape[0] < args.limit:
        args.limit = int(frames.shape[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    learned_dir = args.output_dir / f"realesrgan-{args.dtype}"
    lanczos_dir = args.output_dir / "lanczos"
    source_dir = args.output_dir / "source"
    for directory in (learned_dir, lanczos_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_load_start = time.perf_counter()
    allocated_before = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    model = load_model(args.weight, device, dtype)
    sync(device)
    model_load_seconds = time.perf_counter() - model_load_start
    allocated_after_load = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    source_arrays: list[np.ndarray] = []
    lanczos_arrays: list[np.ndarray] = []
    learned_arrays: list[np.ndarray] = []
    timing_rows: list[dict[str, float | int]] = []
    for index, frame in enumerate(frames):
        source = to_numpy_rgb(frame)
        source_arrays.append(source)
        lanczos = lanczos_2x(source)
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

        output_start = time.perf_counter()
        learned = to_numpy_rgb(model_output)
        sync(device)
        output_conversion_ms = (time.perf_counter() - output_start) * 1000.0
        learned_arrays.append(learned)
        save_png(learned, learned_dir / f"frame-{index:04d}.png")
        timing_rows.append(
            {
                "frame": index,
                "input_conversion_ms": input_conversion_ms,
                "network_ms": network_ms,
                "output_conversion_ms": output_conversion_ms,
                "frame_ready_ms": input_conversion_ms + network_ms + output_conversion_ms,
            }
        )

    write_contact_sheet(source_arrays, lanczos_arrays, learned_arrays, args.output_dir / "contact-sheet.png")
    warm_rows = timing_rows[args.warmup :]
    network_values = [float(row["network_ms"]) for row in warm_rows]
    frame_ready_values = [float(row["frame_ready_ms"]) for row in warm_rows]

    def stats(values: list[float]) -> dict[str, float]:
        return {
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
        }

    output = {
        "provenance": {
            "realesrgan_repo": str(args.repo_dir),
            "realesrgan_revision": git_revision(args.repo_dir),
            "basicsr_revision": git_revision(args.basicsr_dir),
            "weight": str(args.weight),
            "weight_sha256": sha256(args.weight),
            "weight_source": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "architecture": "RRDBNet(num_in_ch=3,num_out_ch=3,num_feat=64,num_block=23,num_grow_ch=32,scale=2)",
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
            "network_input": "NCHW RGB float on device; no uint8 or BGR conversion",
            "output_shape": [1, 3, int(frames.shape[1] * 2), int(frames.shape[2] * 2)],
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
            "learned": temporal_stats(learned_arrays),
        },
        "artifacts": {
            "source_dir": str(source_dir),
            "lanczos_dir": str(lanczos_dir),
            "learned_dir": str(learned_dir),
            "contact_sheet": str(args.output_dir / "contact-sheet.png"),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": "success", "metrics": str(args.output_dir / "metrics.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
