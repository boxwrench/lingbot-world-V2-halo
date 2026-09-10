#!/usr/bin/env python3
"""Run one instrumented, single-device LingBot World v2 experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

from wan import WanI2VCausal
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.modules import attention as attention_module


def rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def current_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return rss_bytes()


def host_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def cuda_memory(device: torch.device) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, fn in (
        ("allocated_bytes", torch.cuda.memory_allocated),
        ("reserved_bytes", torch.cuda.memory_reserved),
        ("max_allocated_bytes", torch.cuda.max_memory_allocated),
        ("max_reserved_bytes", torch.cuda.max_memory_reserved),
    ):
        try:
            result[name] = int(fn(device))
        except Exception as exc:
            result[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    try:
        free, total = torch.cuda.mem_get_info(device)
        result["mem_free_bytes"] = int(free)
        result["mem_total_bytes"] = int(total)
    except Exception as exc:
        result["mem_get_info_error"] = f"{type(exc).__name__}: {exc}"
    return result


def rocm_smi_snapshot() -> str:
    try:
        proc = subprocess.run(
            ["rocm-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return proc.stdout + proc.stderr
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def normalize_video(video: torch.Tensor) -> torch.Tensor:
    """Convert upstream VAE output to [frames, height, width, channels]."""
    cpu = video.detach().float().cpu()
    if cpu.ndim == 4 and cpu.shape[0] in (1, 3) and cpu.shape[-1] not in (1, 3):
        cpu = cpu.permute(1, 2, 3, 0)
    elif cpu.ndim == 3:
        cpu = cpu.unsqueeze(-1)
    if cpu.ndim != 4 or cpu.shape[-1] not in (1, 3):
        raise ValueError(f"unexpected generated video shape: {tuple(cpu.shape)}")
    if cpu.shape[-1] == 1:
        cpu = cpu.expand(-1, -1, -1, 3)
    return cpu.contiguous()


def write_video(video: torch.Tensor, path: Path, fps: int = 16) -> None:
    """Write a checked RGB MP4 without relying on upstream's shape assumptions."""
    frames = ((video.clamp(-1, 1) + 1.0) * 127.5).round().byte().numpy()
    height, width = frames.shape[1:3]
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(path)],
        input=frames.tobytes(), check=True,
    )


def tensor_summary(video: torch.Tensor) -> dict[str, object]:
    cpu = normalize_video(video)
    finite = bool(torch.isfinite(cpu).all())
    summary: dict[str, object] = {
        "shape": list(cpu.shape),
        "dtype": str(video.dtype),
        "finite": finite,
        "min": float(cpu.min()),
        "max": float(cpu.max()),
        "mean": float(cpu.mean()),
        "std": float(cpu.std()),
    }
    if cpu.shape[0] > 1:
        frame_delta = (cpu[1:] - cpu[:-1]).abs().mean(dim=(1, 2, 3))
        summary["mean_adjacent_frame_delta"] = float(frame_delta.mean())
        summary["max_adjacent_frame_delta"] = float(frame_delta.max())
    return {"summary": summary, "cpu_tensor": cpu}


def parameter_report(pipe: WanI2VCausal) -> dict[str, object]:
    report: dict[str, object] = {}
    model = pipe.model
    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
        dtype_counts[str(parameter.dtype)] = dtype_counts.get(str(parameter.dtype), 0) + parameter.numel()
        device_counts[str(parameter.device)] = device_counts.get(str(parameter.device), 0) + parameter.numel()
    report["transformer_parameter_count"] = total
    report["transformer_parameter_dtype_counts"] = dtype_counts
    report["transformer_parameter_device_counts"] = device_counts
    report["vae_parameter_device"] = str(next(pipe.vae.model.parameters()).device)
    report["t5_parameter_device"] = str(next(pipe.text_encoder.model.parameters()).device)
    report["transformer_class"] = type(model).__name__
    return report


def run_one(
    pipe: WanI2VCausal,
    args: argparse.Namespace,
    metrics: dict[str, object],
    run_index: int,
    result_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    pipe.metrics = metrics
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    generation_t0 = time.perf_counter()
    generated = pipe.generate(
        args.prompt,
        Image.open(args.image).convert("RGB"),
        action_path=args.action_path,
        chunk_size=args.chunk_size,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frames,
        seed=args.seed,
        offload_model=args.offload_model,
    )
    if isinstance(generated, (list, tuple)):
        if len(generated) != 1:
            raise ValueError(f"unexpected generated result container: {len(generated)} items")
        generated = generated[0]
    video = generated
    torch.cuda.synchronize(device)
    generation_ms = (time.perf_counter() - generation_t0) * 1000.0

    tensor_info = tensor_summary(video)
    video_cpu = tensor_info.pop("cpu_tensor")
    metrics["generation_ms"] = generation_ms
    metrics["current_rss_bytes"] = current_rss_bytes()
    metrics["max_rss_bytes"] = rss_bytes()
    metrics["host_memory"] = host_memory()
    metrics["cuda_memory_after_generation"] = cuda_memory(device)
    metrics["output"] = tensor_info["summary"]
    metrics["reported_rocm_smi_after_generation"] = rocm_smi_snapshot()
    metrics["run_index"] = run_index
    metrics["effective_fps"] = float(video_cpu.shape[0] / (generation_ms / 1000.0))
    chunks = metrics.get("chunks", [])
    if chunks:
        chunk_times = [float(chunk["chunk_ms"]) for chunk in chunks]
        transformer_times = [float(chunk["transformer_ms"]) for chunk in chunks]
        metrics["first_chunk_ms"] = float(chunks[0]["elapsed_ms"])
        metrics["mean_chunk_ms"] = sum(chunk_times) / len(chunk_times)
        metrics["median_chunk_ms"] = sorted(chunk_times)[len(chunk_times) // 2]
        metrics["mean_transformer_ms"] = sum(transformer_times) / len(transformer_times)
        metrics["chunk_ms"] = chunk_times
        metrics["transformer_ms"] = transformer_times

    if args.save_video:
        output_path = result_dir / f"generated-{run_index}.mp4"
        write_video(video_cpu, output_path)
        metrics["video_path"] = str(output_path)

    return {"metrics": copy.deepcopy(metrics), "video_cpu": video_cpu}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", choices=sorted(MAX_AREA_CONFIGS), required=True)
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--local-attn-size", type=int, default=-1)
    parser.add_argument("--sink-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--action-path", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--offload-model", type=lambda x: x.lower() == "true", default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    report: dict[str, object] = {
        "status": "running",
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "environment": {
                key: value
                for key, value in sorted(os.environ.items())
                if key.startswith(("HIP", "ROCR", "HSA", "PYTORCH", "TORCH", "CUDA", "ROCM", "HF_"))
            },
        },
        "arguments": vars(args),
        "torch": {
            "version": torch.__version__,
            "hip": torch.version.hip,
            "device_name": torch.cuda.get_device_name(device),
            "properties": str(torch.cuda.get_device_properties(device)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "attention_backend": (
                "FlashAttention"
                if attention_module.FLASH_ATTN_2_AVAILABLE or attention_module.FLASH_ATTN_3_AVAILABLE
                else "torch.nn.functional.scaled_dot_product_attention"
            ),
            "flash_attn_2_available": attention_module.FLASH_ATTN_2_AVAILABLE,
            "flash_attn_3_available": attention_module.FLASH_ATTN_3_AVAILABLE,
        },
        "initial_cuda_memory": cuda_memory(device),
    }

    init_t0 = time.perf_counter()
    try:
        pipe = WanI2VCausal(
            config=WAN_CONFIGS["i2v-A14B"],
            checkpoint_dir=args.model_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=False,
            local_attn_size=args.local_attn_size,
            sink_size=args.sink_size,
            infer_mode="causal_fast",
            metrics={},
        )
        torch.cuda.synchronize(device)
        report["model_initialization_ms"] = (time.perf_counter() - init_t0) * 1000.0
        report["parameter_report"] = parameter_report(pipe)
        report["cuda_memory_after_initialization"] = cuda_memory(device)

        runs: list[dict[str, object]] = []
        previous_video: torch.Tensor | None = None
        for run_index in range(1, args.repeat + 1):
            current_metrics: dict[str, object] = {
                "resolution_request": args.size,
                "dtype": "torch.bfloat16",
                "steps_per_chunk": 4,
            }
            current = run_one(pipe, args, current_metrics, run_index, result_dir, device)
            current_video = current["video_cpu"]
            if previous_video is not None:
                delta = (current_video - previous_video).abs()
                current["metrics"]["determinism_max_abs_diff"] = float(delta.max())
                current["metrics"]["determinism_mean_abs_diff"] = float(delta.mean())
            previous_video = current_video
            runs.append(current["metrics"])
            del current_video
            del current
            torch.cuda.empty_cache()

        report["runs"] = runs
        if runs:
            report["cold_generation_ms"] = runs[0].get("generation_ms")
            report["cold_total_ms"] = report["model_initialization_ms"] + runs[0].get("generation_ms", 0)
            if len(runs) > 1:
                report["warm_generation_ms"] = runs[1].get("generation_ms")
        report["status"] = "success"
    except Exception as exc:
        report["status"] = "failure"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        (result_dir / "failure.txt").write_text(report["traceback"])
        (result_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        raise

    (result_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
