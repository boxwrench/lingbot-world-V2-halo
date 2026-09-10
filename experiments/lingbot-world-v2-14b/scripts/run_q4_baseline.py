#!/usr/bin/env python3
"""Headless Q4_K_M run through the pinned RealRebelAI/ComfyUI stack.

This is an experiment harness, not a replacement model implementation. It
loads the community custom node, uses ComfyUI-GGUF for the quantized UMT5,
then calls the node's loader and sampler with a fixed 480x832/6+2 request.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


PROMPT = (
    "A sweeping cinematic journey along the Great Wall of China, winding "
    "through golden autumn hills under a brilliant blue sky, while the camera "
    "glides smoothly forward."
)


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
    out: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        out[key] = int(value.split()[0]) * 1024
    return out


def cuda_memory(torch, device) -> dict[str, int]:
    out = {}
    for name, fn in (
        ("allocated_bytes", torch.cuda.memory_allocated),
        ("reserved_bytes", torch.cuda.memory_reserved),
        ("max_allocated_bytes", torch.cuda.max_memory_allocated),
        ("max_reserved_bytes", torch.cuda.max_memory_reserved),
    ):
        out[name] = int(fn(device))
    free, total = torch.cuda.mem_get_info(device)
    out["mem_free_bytes"] = int(free)
    out["mem_total_bytes"] = int(total)
    return out


def file_mapping(path: Path) -> dict[str, int]:
    """Return resident file-map bytes for the GGUF, when psutil can see it."""
    try:
        import psutil

        path = path.resolve()
        values = {"rss_bytes": 0, "private_clean_bytes": 0, "shared_clean_bytes": 0}
        for mapping in psutil.Process().memory_maps(grouped=False):
            if Path(mapping.path).resolve() == path:
                for key in values:
                    values[key] += int(getattr(mapping, key.removesuffix("_bytes"), 0))
        return values
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def tensor_bytes(torch, value) -> int:
    """Best-effort recursive byte total for phase-state accounting."""
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_bytes(torch, item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(torch, item) for item in value)
    return 0


def model_runtime_state(torch, model) -> dict[str, object]:
    if model is None:
        return {"attached": False}
    registered_bytes = sum(
        int(parameter.numel() * parameter.element_size())
        for parameter in model.parameters()
    )
    qdata_bytes = 0
    qdata_tensors = 0
    for module in model.modules():
        qdata = getattr(module, "qdata", None)
        if torch.is_tensor(qdata):
            qdata_bytes += int(qdata.numel() * qdata.element_size())
            qdata_tensors += 1
    return {
        "attached": True,
        "registered_parameter_bytes": registered_bytes,
        "quantized_data_bytes": qdata_bytes,
        "quantized_data_tensors": qdata_tensors,
        "device_set": sorted({str(parameter.device) for parameter in model.parameters()}),
    }


def load_module(name: str, path: Path, package: bool = False):
    kwargs = {"submodule_search_locations": [str(path)]} if package else {}
    source = path / "__init__.py" if package else path
    spec = importlib.util.spec_from_file_location(name, source, **kwargs)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_video(video, path: Path):
    import numpy as np

    source = video.detach().float().cpu()
    finite = bool(source.isfinite().all())
    source_min = float(source.min())
    source_max = float(source.max())
    frames = (source.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    height, width = frames.shape[1:3]
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", "16", "-i", "-", "-an", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(path)],
        input=frames.tobytes(), check=True,
    )
    return {
        "shape": list(frames.shape),
        "dtype": str(video.dtype),
        "finite": finite,
        "source_min": source_min,
        "source_max": source_max,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-dir", required=True)
    parser.add_argument("--community-dir", required=True)
    parser.add_argument("--gguf-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--local-attn-size", type=int, default=6)
    parser.add_argument("--sink-size", type=int, default=2)
    parser.add_argument("--pin-gb", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase-memory", action="store_true",
                        help="record the final-latent/VAE memory boundary")
    parser.add_argument("--cleanup-before-vae", action="store_true",
                        help="release transformer state before VAE (boundary runtime only)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comfy_dir = Path(args.comfy_dir).resolve()
    community_dir = Path(args.community_dir).resolve()
    gguf_dir = Path(args.gguf_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    dit_path = model_dir / "LingBot-World-14B-causal-fast_merged-Q4_K_M.gguf"
    clip_path = model_dir / "umt5-xxl-encoder-Q4_K_S.gguf"
    vae_path = model_dir / "Wan2.1_VAE.pth"

    # ComfyUI parses process argv during import. Keep its parser from seeing
    # this harness's options.
    runner_argv = sys.argv
    sys.argv = [runner_argv[0]]
    sys.path.insert(0, str(comfy_dir))
    sys.path.insert(0, str(community_dir))
    import torch

    import folder_paths  # noqa: F401
    import comfy.model_management  # noqa: F401
    import nodes

    # ComfyUI-GGUF also contains a nodes.py, but it uses package-relative
    # imports and must not shadow ComfyUI's top-level nodes module above.
    sys.path.insert(0, str(gguf_dir))
    gguf_nodes = load_module("ComfyUI_GGUF_pinned", gguf_dir, package=True)
    gguf_runtime = importlib.import_module("ComfyUI_GGUF_pinned.nodes")
    from ComfyUI_Rebels_LingBotWorld.lbworld_nodes import LBWorldLoader, LBWorldSampler
    from wan.configs.wan_i2v_A14B import i2v_A14B

    sys.argv = runner_argv
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is unavailable")
    device = torch.device("cuda:0")
    if torch.cuda.get_device_properties(device).gcnArchName != "gfx1151":
        raise RuntimeError(f"unexpected device: {torch.cuda.get_device_properties(device)}")

    report: dict[str, object] = {
        "status": "running",
        "host": {"platform": platform.platform(), "python": sys.version, "environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith(("HIP", "ROCR", "HSA", "PYTORCH", "TORCH", "CUDA", "MIOPEN"))
        }},
        "source": {
            "community_dir": str(community_dir),
            "community_sha": os.popen(f"git -C {community_dir} rev-parse HEAD").read().strip(),
            "gguf_sha": os.popen(f"git -C {gguf_dir} rev-parse HEAD").read().strip(),
        },
        "arguments": vars(args),
        "assets": {
            "dit": {"path": str(dit_path), "bytes": dit_path.stat().st_size},
            "clip": {"path": str(clip_path), "bytes": clip_path.stat().st_size},
            "vae": {"path": str(vae_path), "bytes": vae_path.stat().st_size},
        },
        "torch": {
            "version": torch.__version__,
            "hip": torch.version.hip,
            "device_name": torch.cuda.get_device_name(device),
            "properties": str(torch.cuda.get_device_properties(device)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "gguf_dequant_module": str((gguf_dir / "dequant.py").resolve()),
        },
        "initial_cuda_memory": cuda_memory(torch, device),
        "file_mapping_before": file_mapping(dit_path),
        "timings_ms": {},
        "runs": [],
    }

    phase_events: list[dict[str, object]] = []

    # UMT5 is loaded by the actual ComfyUI-GGUF CLIP loader, not by upstream's
    # full-precision T5 path.
    t0 = time.perf_counter()
    clip, = gguf_runtime.CLIPLoaderGGUF().load_clip(clip_path.name, type="wan")
    report["timings_ms"]["clip_load"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    positive, = nodes.CLIPTextEncode().encode(clip, PROMPT)
    negative, = nodes.CLIPTextEncode().encode(clip, i2v_A14B.sample_neg_prompt)
    report["timings_ms"]["clip_encode"] = (time.perf_counter() - t0) * 1000
    if args.phase_memory:
        # This is deliberately a separate sample from "model ready": it
        # captures the memory state after UMT5 load and prompt encoding, but
        # before the DiT/VAE objects are constructed.
        torch.cuda.synchronize(device)
        host = host_memory()
        phase_events.append({
            "phase": "after_t5",
            "cuda_memory": cuda_memory(torch, device),
            "process_rss_bytes": current_rss_bytes(),
            "max_process_rss_bytes": rss_bytes(),
            "system_memory": {
                key: host.get(key, 0)
                for key in ("MemTotal", "MemAvailable", "MemFree", "SwapFree",
                            "Mapped", "AnonPages", "Shmem")
            },
            "positive_conditioning_bytes": tensor_bytes(torch, positive),
            "negative_conditioning_bytes": tensor_bytes(torch, negative),
            "text_store_bytes": tensor_bytes(torch, getattr(clip, "_store", {})),
        })
        (out_dir / "phase-events.json").write_text(
            json.dumps(phase_events, indent=2, ensure_ascii=False) + "\n"
        )

    t0 = time.perf_counter()
    lb_pipe, = LBWorldLoader().load(
        dit_path.name, vae_path.name, args.local_attn_size, args.sink_size, args.pin_gb
    )
    report["timings_ms"]["dit_vae_load"] = (time.perf_counter() - t0) * 1000
    report["model_state"] = {
        "local_attn_size": args.local_attn_size,
        "sink_size": args.sink_size,
        "pin_gb": args.pin_gb,
        "model_class": type(lb_pipe["model"]).__name__,
        "model_parameter_count": sum(p.numel() for p in lb_pipe["model"].parameters()),
        "model_parameter_devices": sorted({str(p.device) for p in lb_pipe["model"].parameters()}),
        "vae_device": str(next(lb_pipe["pipe"].vae.model.parameters()).device),
        "text_encoder_path": str(clip_path),
    }
    report["cuda_memory_after_load"] = cuda_memory(torch, device)
    if args.cleanup_before_vae and not args.phase_memory:
        raise ValueError("--cleanup-before-vae requires --phase-memory")
    pipe = lb_pipe["pipe"]
    if args.phase_memory:
        def phase_hook(phase, latent, self_kv_cache, cross_kv_cache, context, conditioning):
            torch.cuda.synchronize(device)
            host = host_memory()
            phase_events.append({
                "phase": phase,
                "cuda_memory": cuda_memory(torch, device),
                "process_rss_bytes": current_rss_bytes(),
                "max_process_rss_bytes": rss_bytes(),
                "system_memory": {
                    key: host.get(key, 0)
                    for key in ("MemTotal", "MemAvailable", "MemFree", "SwapFree",
                                "Mapped", "AnonPages", "Shmem")
                },
                "latent": {
                    "shape": list(latent.shape) if torch.is_tensor(latent) else None,
                    "dtype": str(latent.dtype) if torch.is_tensor(latent) else None,
                    "bytes": tensor_bytes(torch, latent),
                },
                "self_kv_cache_bytes": tensor_bytes(torch, self_kv_cache),
                "cross_kv_cache_bytes": tensor_bytes(torch, cross_kv_cache),
                "context_bytes": tensor_bytes(torch, context),
                "conditioning_bytes": tensor_bytes(torch, conditioning),
                "cached_conditioning_bytes": tensor_bytes(torch, getattr(pipe, "_t5_cache", {})),
                "text_store_bytes": tensor_bytes(
                    torch, getattr(getattr(pipe, "text_encoder", None), "_store", {})
                ),
                "transformer_state": model_runtime_state(torch, getattr(pipe, "model", None)),
                "gguf_file_mapping": file_mapping(dit_path),
            })
            (out_dir / "phase-events.json").write_text(
                json.dumps(phase_events, indent=2, ensure_ascii=False) + "\n"
            )

        pipe._experiment_phase_hook = phase_hook
        if args.cleanup_before_vae:
            # The boundary runtime wraps this marker to clear the sampler's
            # model references before invoking the hook in image2video.py.
            pipe._experiment_before_vae_cleanup = lambda: None

        original_vae_decode = pipe.vae.decode

        def timed_vae_decode(*decode_args, **decode_kwargs):
            torch.cuda.reset_peak_memory_stats(device)
            vae_start = time.perf_counter()
            phase_events.append({
                "phase": "vae_start",
                "cuda_memory": cuda_memory(torch, device),
                "process_rss_bytes": current_rss_bytes(),
                "max_process_rss_bytes": rss_bytes(),
                "system_memory": {
                    key: host_memory().get(key, 0)
                    for key in ("MemTotal", "MemAvailable", "MemFree", "SwapFree",
                                "Mapped", "AnonPages", "Shmem")
                },
            })
            (out_dir / "phase-events.json").write_text(
                json.dumps(phase_events, indent=2, ensure_ascii=False) + "\n"
            )
            decoded = original_vae_decode(*decode_args, **decode_kwargs)
            torch.cuda.synchronize(device)
            phase_events.append({
                "phase": "vae_complete",
                "elapsed_ms": (time.perf_counter() - vae_start) * 1000,
                "cuda_memory": cuda_memory(torch, device),
                "process_rss_bytes": current_rss_bytes(),
                "max_process_rss_bytes": rss_bytes(),
                "system_memory": {
                    key: host_memory().get(key, 0)
                    for key in ("MemTotal", "MemAvailable", "MemFree", "SwapFree",
                                "Mapped", "AnonPages", "Shmem")
                },
            })
            (out_dir / "phase-events.json").write_text(
                json.dumps(phase_events, indent=2, ensure_ascii=False) + "\n"
            )
            return decoded

        pipe.vae.decode = timed_vae_decode
    report["phase_memory"] = phase_events
    sampler = LBWorldSampler()
    image_path = Path(folder_paths.get_input_directory()) / "lingbot_actions" / "strix-example" / "image.jpg"
    if not image_path.exists():
        image_path = Path(__file__).resolve().parents[3] / ".upstream/lingbot-world-v2/examples/03/image.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"input image not found: {image_path}")
    # The node expects a ComfyUI IMAGE tensor in [0,1], [B,H,W,C].
    from PIL import Image
    import numpy as np
    image = torch.from_numpy(np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0)[None]

    # Match the community sampler's pre-flight accounting exactly. This is an
    # estimate of the self-attention K/V tensors, not a claim about physical
    # VRAM placement on the UMA device.
    max_area = 480 * 832
    aspect_ratio = image.shape[1] / image.shape[2]
    lat_h = round(math.sqrt(max_area * aspect_ratio) // 8 // 2 * 2)
    lat_w = round(math.sqrt(max_area / aspect_ratio) // 8 // 2 * 2)
    tokens_per_frame = max(int(lat_h * lat_w // 4), 1)
    kv_bytes = 2 * 40 * (args.local_attn_size + args.sink_size) * tokens_per_frame * 5120 * 2
    report["kv_accounting"] = {
        "formula": "2 * 40 layers * (local + sink) * tokens_per_frame * 5120 channels * 2 bytes",
        "local_attn_size": args.local_attn_size,
        "sink_size": args.sink_size,
        "latent_grid": [lat_h, lat_w],
        "tokens_per_frame": tokens_per_frame,
        "estimated_self_kv_bytes": kv_bytes,
        "estimated_self_kv_gb_decimal": kv_bytes / 1e9,
    }

    previous = None
    for run_index in range(1, args.repeat + 1):
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        result, = sampler.sample(
            lb_pipe, positive, negative, image, "strix-example", PROMPT,
            "480x832 (needs tiny window)", args.frames, args.chunk_size, 3.0,
            args.seed, None,
        )
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result_path = out_dir / f"generated-{run_index}.mp4"
        summary = save_video(result, result_path)
        current = {
            "run_index": run_index,
            "elapsed_ms": elapsed_ms,
            "effective_fps": float(result.shape[0] / (elapsed_ms / 1000)),
            "output": summary,
            "video_path": str(result_path),
            "cuda_memory": cuda_memory(torch, device),
            "process_rss_bytes": current_rss_bytes(),
            "max_process_rss_bytes": rss_bytes(),
            "host_memory": host_memory(),
            "gguf_file_mapping": file_mapping(dit_path),
        }
        if previous is not None:
            delta = (result.float() - previous.float()).abs()
            current["determinism_max_abs_diff"] = float(delta.max())
            current["determinism_mean_abs_diff"] = float(delta.mean())
        previous = result.detach().cpu()
        report["runs"].append(current)
        del result
        torch.cuda.empty_cache()

    report["status"] = "success"
    report["file_mapping_after"] = file_mapping(dit_path)
    report["final_cuda_memory"] = cuda_memory(torch, device)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
