#!/usr/bin/env python3
"""Persistent chunk-size-1 LingBot session with incremental causal VAE decode.

This is an experiment harness, not an upstream implementation change.  It
reuses the causal-fast DiT path and its KV caches, but exposes each latent
chunk to the existing causal VAE feature cache immediately.  That makes the
interactive metric meaningful: the action timer ends when the first new
decoded frame has been synchronized and copied to host memory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel

from run_experiment import (
    cuda_memory,
    current_rss_bytes,
    host_memory,
    normalize_video,
    parameter_report,
    rocm_smi_snapshot,
    rss_bytes,
    write_video,
)
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.image2video import (
    WanI2VCausal,
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def cache_positions(self_kv_cache: list[dict[str, torch.Tensor]]) -> dict[str, int]:
    """Read layer-zero cache indices after a completed cache-writing forward."""
    first = self_kv_cache[0]
    return {
        "global_end_index": int(first["global_end_index"].item()),
        "local_end_index": int(first["local_end_index"].item()),
        "cache_capacity_tokens": int(first["k"].shape[1]),
    }


def causal_conv2d_temporal_split(module, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
    """Evaluate one causal Conv3d by summing spatial Conv2d slices.

    This preserves CausalConv3d's existing cache and padding contract: cache_x
    is prepended exactly as in the upstream module, and only the current
    temporal outputs are returned.  It is an opt-in experiment, not a general
    replacement for Conv3d.
    """
    padding = list(module._padding)
    if cache_x is not None and padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding)
    kernel_t, kernel_h, kernel_w = module.kernel_size
    stride_t, stride_h, stride_w = module.stride
    dilation_t, dilation_h, dilation_w = module.dilation
    output_t = (x.shape[2] - dilation_t * (kernel_t - 1) - 1) // stride_t + 1
    outputs = []
    for output_index in range(output_t):
        terms = []
        for kernel_index in range(kernel_t):
            frame = x[:, :, output_index * stride_t + kernel_index * dilation_t]
            terms.append(F.conv2d(
                frame,
                module.weight[:, :, kernel_index],
                bias=None,
                stride=(stride_h, stride_w),
                padding=0,
                dilation=(dilation_h, dilation_w),
                groups=module.groups,
            ))
        output = terms[0]
        for term in terms[1:]:
            output = output + term
        if module.bias is not None:
            output = output + module.bias.view(1, -1, 1, 1)
        outputs.append(output.unsqueeze(2))
    return torch.cat(outputs, dim=2)


def install_temporal_split(decoder, module_name: str) -> None:
    """Patch exactly one named decoder CausalConv3d for an A/B run."""
    from wan.modules.vae2_1 import CausalConv3d

    modules = dict(decoder.named_modules())
    if module_name not in modules:
        raise ValueError(f"decoder module not found: {module_name}")
    module = modules[module_name]
    if not isinstance(module, CausalConv3d):
        raise TypeError(f"decoder module is not CausalConv3d: {module_name}")

    def split_forward(x, cache_x=None):
        return causal_conv2d_temporal_split(module, x, cache_x)

    module.forward = split_forward


class IncrementalCausalDecoder:
    """Decode one latent frame while retaining Wan's causal feature cache."""

    def __init__(
        self,
        vae,
        attention_backend: str = "math",
        dtype_name: str = "fp32",
        profile: bool = False,
        temporal_split_module: str | None = None,
    ) -> None:
        self.vae = vae
        self.model = vae.model
        self.attention_backend = attention_backend
        self.dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[dtype_name]
        self.dtype_name = dtype_name
        self.model.to(dtype=self.dtype)
        self.vae.mean = self.vae.mean.to(dtype=self.dtype)
        self.vae.std = self.vae.std.to(dtype=self.dtype)
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
        self.model.clear_cache()
        if temporal_split_module:
            install_temporal_split(self.model.decoder, temporal_split_module)
        self.last_stats: dict[str, object] = {}
        self.profiler = DecoderProfiler(self.model.decoder) if profile else None

    @torch.no_grad()
    def decode_latent(
        self,
        latent: torch.Tensor,
        device: torch.device,
        synchronize: bool = True,
    ) -> torch.Tensor:
        # WanVAE_.decode() performs this normalization once, then calls
        # decoder() once per latent frame while retaining _feat_map.  conv2 is
        # a temporal 1x1x1 projection, so applying it to one frame at a time
        # preserves the upstream operation order.
        # The upstream VAE wrapper uses an FP32 autocast lane nested inside
        # the DiT BF16 context.  Disable the outer autocast explicitly here;
        # otherwise long persistent sessions can become non-finite even when
        # the corresponding batch decode remains finite.
        attention_context = (
            sdpa_kernel(SDPBackend.MATH)
            if self.attention_backend == "math"
            else nullcontext()
        )
        with attention_context, torch.autocast(device_type="cuda", enabled=False):
            z_dim = self.model.z_dim
            z = latent.float() / self.vae.scale[1].float().view(1, z_dim, 1, 1, 1)
            z = z + self.vae.scale[0].view(1, z_dim, 1, 1, 1)
            z = z.to(self.dtype)
            self.last_stats["normalized_latent_finite"] = bool(torch.isfinite(z).all())
            self.last_stats["normalized_latent_absmax"] = float(z.abs().max())
            x = self.model.conv2(z)
            self.last_stats["conv2_finite"] = bool(torch.isfinite(x).all())
            self.last_stats["conv2_absmax"] = float(x.abs().max())
            self.model._conv_idx = [0]
            output = self.model.decoder(
                x,
                feat_cache=self.model._feat_map,
                feat_idx=self.model._conv_idx,
            )
        self.last_stats["attention_backend"] = self.attention_backend
        self.last_stats["vae_dtype"] = str(self.dtype)
        if synchronize:
            sync(device)
        return output.float().clamp_(-1, 1).squeeze(0)

    def clear(self) -> None:
        self.model.clear_cache()

    def profile_report(self) -> dict[str, object] | None:
        return self.profiler.report() if self.profiler is not None else None


class StreamingTAEHVDecoder:
    """Pinned TAEW2.1 presentation decoder with one-latent streaming output."""

    def __init__(self, taehv_dir: str, device: torch.device, weights_path: str | None = None) -> None:
        taehv_path = Path(taehv_dir).resolve()
        if str(taehv_path) not in sys.path:
            sys.path.insert(0, str(taehv_path))
        from taehv import StreamingTAEHV, TAEHV

        weights = Path(weights_path) if weights_path else taehv_path / "taew2_1.pth"
        self.tae = TAEHV(str(weights)).to(device=device, dtype=torch.float16).eval()
        self.streaming = StreamingTAEHV(self.tae)
        self.weights_path = str(weights)
        self.last_stats: dict[str, object] = {
            "display_decoder": "taehv",
            "arch_name": self.tae.arch_name,
            "latent_channels": self.tae.latent_channels,
            "t_upscale": self.tae.t_upscale,
            "frames_to_trim": self.tae.frames_to_trim,
        }

    @torch.inference_mode()
    def begin_latent(self, latent: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, float, float]:
        # The pinned TAEW2.1 Diffusers wrapper uses identity latent mean/std.
        # LingBot x0 is already in that model-space; only NCTHW -> NTCHW is
        # required by StreamingTAEHV.
        tae_latent = latent.to(device=device, dtype=torch.float16).permute(0, 2, 1, 3, 4).contiguous()
        t0 = time.perf_counter()
        pending = self.streaming.decode(tae_latent)
        sync(device)
        first_rgb_ms = (time.perf_counter() - t0) * 1000.0
        if pending is None:
            raise RuntimeError("TAEHV did not produce an RGB frame for an accepted latent")
        return pending, first_rgb_ms, t0

    @torch.inference_mode()
    def drain_latent(
        self,
        first_pending: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, float]:
        t0 = time.perf_counter()
        frames = [first_pending[:, 0]]
        while True:
            pending = self.streaming.decode()
            sync(device)
            if pending is None:
                break
            frames.append(pending[:, 0])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Return the same [C,T,H,W] convention used by the canonical decoder,
        # but keep TAE's native [0,1] range until the presentation boundary.
        decoded = torch.cat(frames, dim=0).permute(1, 0, 2, 3).contiguous()
        self.last_stats["output_finite"] = bool(torch.isfinite(decoded).all())
        self.last_stats["output_min"] = float(decoded.min())
        self.last_stats["output_max"] = float(decoded.max())
        return decoded, elapsed_ms

    def clear(self) -> None:
        self.streaming.reset()

    def profile_report(self) -> dict[str, object] | None:
        return None


class DecoderProfiler:
    """CUDA-event profile of the real persistent decoder call tree."""

    def __init__(self, decoder) -> None:
        self.calls: list[dict[str, object]] = []
        self.active: list[dict[str, object]] = []
        self.handles = []
        for name, module in decoder.named_modules():
            if not name:
                continue
            self.handles.append(module.register_forward_pre_hook(self._pre(name, module)))
            self.handles.append(module.register_forward_hook(self._post))

    def _pre(self, name: str, module):
        def hook(_module, inputs):
            start = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream())
            call = {
                "name": name,
                "class": type(module).__name__,
                "start": start,
                "input_shape": list(inputs[0].shape) if inputs and isinstance(inputs[0], torch.Tensor) else None,
                "input_dtype": str(inputs[0].dtype) if inputs and isinstance(inputs[0], torch.Tensor) else None,
                "parent": self.active[-1] if self.active else None,
                "children": [],
            }
            if self.active:
                self.active[-1]["children"].append(call)
            self.active.append(call)
        return hook

    def _post(self, _module, _inputs, output):
        call = self.active.pop()
        end = torch.cuda.Event(enable_timing=True)
        end.record(torch.cuda.current_stream())
        call["end"] = end
        call["output_shape"] = list(output.shape) if isinstance(output, torch.Tensor) else None
        call["output_dtype"] = str(output.dtype) if isinstance(output, torch.Tensor) else None
        self.calls.append(call)

    def report(self) -> dict[str, object]:
        for handle in self.handles:
            handle.remove()
        for call in self.calls:
            call["inclusive_ms"] = float(call["start"].elapsed_time(call["end"]))
            child_ms = sum(float(child.get("inclusive_ms", 0.0)) for child in call["children"])
            call["exclusive_ms"] = max(0.0, float(call["inclusive_ms"]) - child_ms)

        groups: dict[str, dict[str, object]] = {}
        for call in self.calls:
            group = groups.setdefault(call["name"], {
                "module": call["name"],
                "class": call["class"],
                "calls": 0,
                "inclusive_ms": 0.0,
                "exclusive_ms": 0.0,
                "input_shape": call["input_shape"],
                "output_shape": call["output_shape"],
                "dtype": call["input_dtype"],
                "output_dtype": call["output_dtype"],
                "exclusive_times_ms": [],
            })
            group["calls"] += 1
            group["inclusive_ms"] += float(call["inclusive_ms"])
            group["exclusive_ms"] += float(call["exclusive_ms"])
            group["exclusive_times_ms"].append(float(call["exclusive_ms"]))
        rows = sorted(groups.values(), key=lambda row: -float(row["exclusive_ms"]))
        total = sum(float(row["exclusive_ms"]) for row in rows)
        cumulative = 0.0
        for row in rows:
            times = row.pop("exclusive_times_ms")
            row["first_exclusive_ms"] = times[0] if times else None
            row["repeat_mean_exclusive_ms"] = (
                sum(times[1:]) / len(times[1:]) if len(times) > 1 else None
            )
            row["min_exclusive_ms"] = min(times) if times else None
            row["max_exclusive_ms"] = max(times) if times else None
            row["share_of_exclusive_ms"] = float(row["exclusive_ms"]) / total if total else 0.0
            cumulative += float(row["exclusive_ms"])
            row["cumulative_exclusive_share"] = cumulative / total if total else 0.0
        return {
            "calls_recorded": len(self.calls),
            "exclusive_total_ms": total,
            "operators": rows,
        }


class DitAttentionProbe:
    """Record real DiT attention dispatch, SDPA, and tensor layouts."""

    def __init__(self) -> None:
        from wan.modules import model_fast

        self.model_fast = model_fast
        self.original_dispatch = model_fast.attention
        self.original_sdpa = F.scaled_dot_product_attention
        self.calls: list[dict[str, object]] = []
        self.active: dict[str, object] | None = None
        self.backend_flags = {
            "flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            "cudnn_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
            "sdp_priority_order": list(torch._C._get_sdp_priority_order()),
        }

        def dispatch(q, k, v, *args, **kwargs):
            call = {
                "kind": "self" if int(k.shape[1]) > 1024 else "cross",
                "q_shape": list(q.shape),
                "k_shape": list(k.shape),
                "v_shape": list(v.shape),
                "q_dtype": str(q.dtype),
                "k_dtype": str(k.dtype),
                "v_dtype": str(v.dtype),
                "q_strides": list(q.stride()),
                "k_strides": list(k.stride()),
                "v_strides": list(v.stride()),
                "q_contiguous": bool(q.is_contiguous()),
                "k_contiguous": bool(k.is_contiguous()),
                "v_contiguous": bool(v.is_contiguous()),
                "q_lens_present": kwargs.get("q_lens") is not None,
                "k_lens_present": kwargs.get("k_lens") is not None,
                "causal": bool(kwargs.get("causal", False)),
                "window_size": kwargs.get("window_size", (-1, -1)),
            }
            start = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream())
            previous = self.active
            self.active = call
            out = self.original_dispatch(q, k, v, *args, **kwargs)
            self.active = previous
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream())
            call["dispatch_start"] = start
            call["dispatch_end"] = end
            call["output_shape"] = list(out.shape) if isinstance(out, torch.Tensor) else None
            self.calls.append(call)
            return out

        def sdpa(q, k, v, *args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream())
            out = self.original_sdpa(q, k, v, *args, **kwargs)
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream())
            if self.active is not None:
                self.active["sdpa_start"] = start
                self.active["sdpa_end"] = end
                self.active["sdpa_q_shape"] = list(q.shape)
                self.active["sdpa_k_shape"] = list(k.shape)
                self.active["sdpa_v_shape"] = list(v.shape)
                self.active["sdpa_q_strides"] = list(q.stride())
                self.active["sdpa_k_strides"] = list(k.stride())
                self.active["sdpa_v_strides"] = list(v.stride())
                self.active["sdpa_is_causal"] = bool(kwargs.get("is_causal", False))
                self.active["sdpa_mask_present"] = kwargs.get("attn_mask") is not None
            return out

        model_fast.attention = dispatch
        F.scaled_dot_product_attention = sdpa

    def close(self) -> None:
        self.model_fast.attention = self.original_dispatch
        F.scaled_dot_product_attention = self.original_sdpa

    def report(self) -> dict[str, object]:
        self.close()
        rows = []
        for call in self.calls:
            dispatch_ms = float(call["dispatch_start"].elapsed_time(call["dispatch_end"]))
            sdpa_ms = float(call["sdpa_start"].elapsed_time(call["sdpa_end"]))
            row = {key: value for key, value in call.items() if not isinstance(value, torch.cuda.Event)}
            row["dispatch_ms"] = dispatch_ms
            row["sdpa_ms"] = sdpa_ms
            row["dispatch_overhead_ms"] = max(0.0, dispatch_ms - sdpa_ms)
            rows.append(row)
        grouped: dict[str, dict[str, object]] = {}
        for row in rows:
            group = grouped.setdefault(row["kind"], {
                "kind": row["kind"],
                "calls": 0,
                "dispatch_ms": 0.0,
                "sdpa_ms": 0.0,
                "dispatch_overhead_ms": 0.0,
                "q_shapes": {},
                "k_shapes": {},
                "sdpa_q_shapes": {},
                "sdpa_k_shapes": {},
                "q_strides": {},
                "k_strides": {},
                "sdpa_k_strides": {},
                "dtypes": {},
                "mask_modes": {},
            })
            group["calls"] += 1
            for field in ("dispatch_ms", "sdpa_ms", "dispatch_overhead_ms"):
                group[field] += float(row[field])
            for field, value in (
                ("q_shapes", tuple(row["q_shape"])),
                ("k_shapes", tuple(row["k_shape"])),
                ("sdpa_q_shapes", tuple(row["sdpa_q_shape"])),
                ("sdpa_k_shapes", tuple(row["sdpa_k_shape"])),
                ("q_strides", tuple(row["q_strides"])),
                ("k_strides", tuple(row["k_strides"])),
                ("sdpa_k_strides", tuple(row["sdpa_k_strides"])),
                ("dtypes", (row["q_dtype"], row["k_dtype"], row["v_dtype"])),
                ("mask_modes", (row["sdpa_is_causal"], row["sdpa_mask_present"])),
            ):
                counts = group[field]
                key = str(value)
                counts[key] = counts.get(key, 0) + 1
        return {
            "backend_flags": self.backend_flags,
            "calls_recorded": len(rows),
            "calls": rows,
            "groups": list(grouped.values()),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", choices=sorted(MAX_AREA_CONFIGS), default="480*832")
    parser.add_argument(
        "--max-area-pixels",
        type=int,
        default=None,
        help="optional custom Wan-compatible area; overrides --size for one controlled geometry test",
    )
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument(
        "--denoise-schedule",
        choices=("4", "3-drop-957"),
        default="4",
        help="causal-fast schedule: upstream four points, or one controlled three-point drop of 957",
    )
    parser.add_argument("--local-attn-size", type=int, default=18)
    parser.add_argument("--sink-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--action-path", required=True)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="save the accepted x0 latent stream for decoder-only comparisons",
    )
    parser.add_argument("--max-chunks", type=int, default=0, help="debug limit; zero processes the full session")
    parser.add_argument(
        "--vae-attention-backend",
        choices=("math", "default"),
        default="math",
        help="VAE spatial attention backend; math is the validated streaming-correctness fallback",
    )
    parser.add_argument(
        "--vae-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
        help="persistent VAE decoder compute/weight dtype",
    )
    parser.add_argument(
        "--display-decoder",
        choices=("canonical", "taehv"),
        default="canonical",
        help="presentation decoder; taehv is an opt-in pinned StreamingTAEHV path",
    )
    parser.add_argument(
        "--taehv-dir",
        default=".upstream/taehv",
        help="pinned TAEHV checkout used by --display-decoder taehv",
    )
    parser.add_argument(
        "--taehv-weights",
        default=None,
        help="optional TAEW2.1 weight path; defaults to taehv-dir/taew2_1.pth",
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="experimental queued VAE/DiT stream overlap; serial is the default",
    )
    parser.add_argument(
        "--profile-vae",
        action="store_true",
        help="record CUDA-event inclusive/exclusive timing for decoder modules",
    )
    parser.add_argument(
        "--vae-temporal-split-module",
        default=None,
        help="opt-in exact Conv3d->Conv2d split for one decoder module path",
    )
    parser.add_argument(
        "--profile-dit-from-chunk",
        type=int,
        default=-1,
        help="profile the DiT module tree starting at this chunk (steady-state use: 18)",
    )
    parser.add_argument(
        "--defer-clean-kv",
        action="store_true",
        help="display the decoded frame before the exact clean-latent KV commit (transactional scheduling experiment)",
    )
    parser.add_argument(
        "--profile-attention-from-chunk",
        type=int,
        default=-1,
        help="record DiT attention/SDPA layouts starting at this chunk for three chunks",
    )
    return parser


def prepare_session(pipe: WanI2VCausal, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    """Prepare prompt, image conditioning, camera controls, noise and caches."""
    action_dir = Path(args.action_path)
    c2ws = np.load(action_dir / "poses.npy")
    len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
    frame_num = ((args.frames - 1) // 4) * 4 + 1
    frame_num = min(frame_num, len_c2ws)
    c2ws = c2ws[:frame_num]

    image = Image.open(args.image).convert("RGB")
    img = TF.to_tensor(image).sub_(0.5).div_(0.5).to(device)
    F = frame_num
    h0, w0 = img.shape[1:]
    aspect_ratio = h0 / w0
    max_area = args.max_area_pixels or MAX_AREA_CONFIGS[args.size]
    lat_h = round(np.sqrt(max_area * aspect_ratio) // pipe.vae_stride[1] // pipe.patch_size[1] * pipe.patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // pipe.vae_stride[2] // pipe.patch_size[2] * pipe.patch_size[2])
    h = lat_h * pipe.vae_stride[1]
    w = lat_w * pipe.vae_stride[2]
    lat_f = (F - 1) // pipe.vae_stride[0] + 1
    lat_f = int(lat_f - (lat_f % 1))
    F = (lat_f - 1) * 4 + 1

    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(args.seed)
    noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=device)

    mask = torch.ones(1, F, lat_h, lat_w, device=device)
    mask[:, 1:] = 0
    mask = torch.concat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
    mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]

    pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=5.0)
    full_timestep_indices = [0, 179, 358, 679]
    if args.denoise_schedule == "4":
        timestep_indices = full_timestep_indices
    else:
        # The released causal-fast path hard-codes these four distilled
        # points.  This is one controlled approximation: remove the second
        # point while retaining the trained endpoints and lower-noise stages.
        timestep_indices = [0, 358, 679]
    timesteps = pipe.scheduler.timesteps[timestep_indices]
    prompt_key = hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()
    if prompt_key in pipe._t5_cache:
        context = pipe._t5_cache[prompt_key]
        t5_ms = None
        t5_cache_hit = True
    else:
        t5_t0 = time.perf_counter()
        pipe.text_encoder.model.to(device)
        context = pipe.text_encoder([args.prompt], device)
        sync(device)
        t5_ms = (time.perf_counter() - t5_t0) * 1000.0
        pipe._t5_cache[prompt_key] = context
        t5_cache_hit = False

    Ks = torch.from_numpy(np.load(action_dir / "intrinsics.npy")).float()
    Ks = get_Ks_transformed(Ks, height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)[0]
    len_c2ws_ = int((len(c2ws) - 1) // 4) + 1
    len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % 1))
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len(c2ws) - 1, len(c2ws)),
        src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, len(c2ws) - 1, len_c2ws_),
    )
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
    Ks = Ks.repeat(len(c2ws_infer), 1).to(device)
    c2ws_infer = c2ws_infer.to(device)
    plucker = get_plucker_embeddings(c2ws_infer, Ks, h, w)
    plucker = rearrange(plucker, "f (h c1) (w c2) c -> (f h w) (c c1 c2)", c1=int(h // lat_h), c2=int(w // lat_w))
    plucker = rearrange(plucker[None, ...], "b (f h w) c -> b c f h w", f=lat_f, h=lat_h, w=lat_w).to(pipe.param_dtype)

    vae_encode_t0 = time.perf_counter()
    y = pipe.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
            torch.zeros(3, F - 1, h, w),
        ], dim=1).to(device)
    ])[0]
    sync(device)
    vae_encode_ms = (time.perf_counter() - vae_encode_t0) * 1000.0
    y = torch.concat([mask, y])

    model_args = pipe.model.config
    frame_seqlen = int(lat_h * lat_w // 4)
    kv_size = frame_seqlen * args.local_attn_size
    head_dim = model_args.dim // model_args.num_heads
    self_kv_cache = pipe._initialize_self_kv_cache(
        num_layers=model_args.num_layers,
        shape=[1, kv_size, model_args.num_heads, head_dim],
        dtype=pipe.pipe_dtype,
        device=device,
    )
    cross_kv_cache = pipe._initialize_crossattn_cache(
        num_layers=model_args.num_layers,
        shape=[1, 512, model_args.num_heads, head_dim],
        dtype=pipe.pipe_dtype,
        device=device,
    )
    pipe._cross_attn_initialized = False
    return {
        "image": image,
        "noise_chunks": noise.split(1, dim=1),
        "condition_chunks": y.split(1, dim=1),
        "plucker_chunks": plucker.split(1, dim=2),
        "context": context,
        "timesteps": timesteps,
        "timestep_indices": timestep_indices,
        "timestep_values": [int(timestep) for timestep in timesteps],
        "denoise_schedule": args.denoise_schedule,
        "seed_g": seed_g,
        "self_kv_cache": self_kv_cache,
        "cross_kv_cache": cross_kv_cache,
        "lat_f": lat_f,
        "lat_h": lat_h,
        "lat_w": lat_w,
        "height": h,
        "width": w,
        "max_area_pixels": int(max_area),
        "frame_seqlen": frame_seqlen,
        "kv_size": kv_size,
        "frames": F,
        "t5_encode_ms": t5_ms,
        "t5_cache_hit": t5_cache_hit,
        "vae_encode_ms": vae_encode_ms,
    }


def generate_chunk(
    pipe: WanI2VCausal,
    state: dict[str, object],
    chunk_id: int,
    current_latent: torch.Tensor,
    current_condition: torch.Tensor,
    current_plucker: torch.Tensor,
    device: torch.device,
    sync_fn,
    defer_clean_kv: bool = False,
) -> dict[str, object]:
    """Run the five causal-fast DiT forwards for one latent chunk."""
    chunk_t0 = time.perf_counter()
    prepare_t0 = chunk_t0
    max_seq_len = state["frame_seqlen"]
    kwargs = {
        "context": [state["context"][0]],
        "seq_len": max_seq_len,
        "y": [current_condition],
        "dit_cond_dict": {"c2ws_plucker_emb": current_plucker.chunk(1, dim=0)},
        "kv_cache": state["self_kv_cache"],
        "crossattn_cache": state["cross_kv_cache"],
        "current_start": chunk_id * state["frame_seqlen"],
        "max_attention_size": state["kv_size"],
        "frame_seqlen": state["frame_seqlen"],
    }
    action_prepare_ms = (time.perf_counter() - prepare_t0) * 1000.0
    forward_records: list[dict[str, object]] = []
    latent_postprocess_ms = 0.0
    transformer_t0 = time.perf_counter()
    for timestep_idx, current_timestep in enumerate(state["timesteps"]):
        forward_t0 = time.perf_counter()
        cache_before = cache_positions(state["self_kv_cache"])
        memory_before = cuda_memory(device)
        noise_pred = pipe.model(
            x=[current_latent],
            t=torch.stack([current_timestep]).to(device),
            cross_attn_first_call=not pipe._cross_attn_initialized,
            **kwargs,
        )[0]
        sync_fn()
        memory_after = cuda_memory(device)
        forward_ms = (time.perf_counter() - forward_t0) * 1000.0
        forward_records.append({
            "kind": "denoise",
            "index": timestep_idx,
            "timestep": float(current_timestep),
            "elapsed_ms": forward_ms,
            "purpose": "denoise and clean-latent update",
            "query_tokens": int(state["frame_seqlen"]),
            "kv_length_before": cache_before["local_end_index"],
            "kv_length_after": cache_positions(state["self_kv_cache"])["local_end_index"],
            "kv_capacity_tokens": int(state["kv_size"]),
            "cache_before": cache_before,
            "cache_after": cache_positions(state["self_kv_cache"]),
            "allocated_before_bytes": memory_before.get("allocated_bytes"),
            "allocated_after_bytes": memory_after.get("allocated_bytes"),
            "reserved_before_bytes": memory_before.get("reserved_bytes"),
            "reserved_after_bytes": memory_after.get("reserved_bytes"),
            "finite": bool(torch.isfinite(noise_pred).all()),
        })
        pipe._cross_attn_initialized = True
        post_t0 = time.perf_counter()
        x0 = pipe._convert_flow_pred_to_x0(noise_pred, current_latent, current_timestep, pipe.scheduler)
        if timestep_idx < len(state["timesteps"]) - 1:
            next_timestep = state["timesteps"][timestep_idx + 1]
            current_latent = pipe.scheduler.add_noise(
                x0,
                torch.randn(x0.shape, generator=state["seed_g"], device=device, dtype=x0.dtype),
                next_timestep,
            )
        sync_fn()
        latent_postprocess_ms += (time.perf_counter() - post_t0) * 1000.0

    generated = {
        "chunk_id": chunk_id,
        "chunk_t0": chunk_t0,
        "kwargs": kwargs,
        "x0": x0,
        "cache": cache_positions(state["self_kv_cache"]),
        "cross_attention_initialized": int(state["cross_kv_cache"][0]["is_init"].item()),
        "transformer_ms": (time.perf_counter() - transformer_t0) * 1000.0,
        "action_prepare_ms": action_prepare_ms,
        "latent_postprocess_ms": latent_postprocess_ms,
        "forward_records": forward_records,
        "clean_kv_deferred": defer_clean_kv,
        "clean_kv_ms": None,
        "state_ready_ms": None,
    }
    if not defer_clean_kv:
        commit_clean_kv(pipe, state, generated, device, sync_fn)
    generated["transformer_ms"] = (time.perf_counter() - transformer_t0) * 1000.0
    return generated


def commit_clean_kv(
    pipe: WanI2VCausal,
    state: dict[str, object],
    generated: dict[str, object],
    device: torch.device,
    sync_fn,
) -> dict[str, object]:
    """Commit the exact clean-latent KV state and make it transactionally ready."""
    cache_t0 = time.perf_counter()
    zero_timestep = state["timesteps"][-1] * 0.0
    cache_before = cache_positions(state["self_kv_cache"])
    memory_before = cuda_memory(device)
    pipe.model(
        x=[generated["x0"]],
        t=torch.stack([zero_timestep]).to(device),
        cross_attn_first_call=False,
        **generated["kwargs"],
    )
    sync_fn()
    memory_after = cuda_memory(device)
    cache_after = cache_positions(state["self_kv_cache"])
    elapsed_ms = (time.perf_counter() - cache_t0) * 1000.0
    generated["forward_records"].append({
        "kind": "cache_update",
        "index": len(state["timesteps"]),
        "timestep": 0.0,
        "elapsed_ms": elapsed_ms,
        "purpose": "clean-latent self-attention KV-cache write; output discarded",
        "query_tokens": int(state["frame_seqlen"]),
        "kv_length_before": cache_before["local_end_index"],
        "kv_length_after": cache_after["local_end_index"],
        "kv_capacity_tokens": int(state["kv_size"]),
        "cache_before": cache_before,
        "cache_after": cache_after,
        "allocated_before_bytes": memory_before.get("allocated_bytes"),
        "allocated_after_bytes": memory_after.get("allocated_bytes"),
        "reserved_before_bytes": memory_before.get("reserved_bytes"),
        "reserved_after_bytes": memory_after.get("reserved_bytes"),
    })
    generated["cache"] = cache_after
    generated["cross_attention_initialized"] = int(state["cross_kv_cache"][0]["is_init"].item())
    generated["clean_kv_ms"] = elapsed_ms
    generated["state_ready_ms"] = (time.perf_counter() - generated["chunk_t0"]) * 1000.0
    generated["clean_kv_committed"] = True
    return generated


def summarize_module_profile(profile: dict[str, object] | None) -> dict[str, object] | None:
    """Aggregate a module-tree profile into operator class/shape buckets."""
    if profile is None:
        return None
    groups: dict[tuple[str, str | None, str | None], dict[str, object]] = {}
    for row in profile.get("operators", []):
        key = (str(row["class"]), str(row.get("input_shape")), str(row.get("output_shape")))
        group = groups.setdefault(key, {
            "class": row["class"],
            "input_shape": row.get("input_shape"),
            "output_shape": row.get("output_shape"),
            "calls": 0,
            "exclusive_ms": 0.0,
            "inclusive_ms": 0.0,
            "modules": [],
        })
        group["calls"] += int(row["calls"])
        group["exclusive_ms"] += float(row["exclusive_ms"])
        group["inclusive_ms"] += float(row["inclusive_ms"])
        group["modules"].append(row["module"])
    rows = sorted(groups.values(), key=lambda row: -float(row["exclusive_ms"]))
    total = sum(float(row["exclusive_ms"]) for row in rows)
    cumulative = 0.0
    for row in rows:
        row["share_of_exclusive_ms"] = float(row["exclusive_ms"]) / total if total else 0.0
        cumulative += float(row["exclusive_ms"])
        row["cumulative_exclusive_share"] = cumulative / total if total else 0.0
    return {
        "exclusive_total_ms": total,
        "operators": rows,
    }


def finish_action(
    pipe: WanI2VCausal,
    state: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    decoder: IncrementalCausalDecoder,
    generated: dict[str, object],
    decoded_gpu: torch.Tensor,
    decode_ms: float,
    actions: list[dict[str, object]],
    outputs: list[torch.Tensor],
    first_visible_ms: float | None,
    full_action_ms: float | None,
    overlap: bool = False,
    decoder_stats: dict[str, object] | None = None,
    post_visible_hook=None,
) -> None:
    post_t0 = time.perf_counter()
    decoded_hwc_gpu = decoded_gpu.permute(1, 2, 3, 0).contiguous()
    rgb_postprocess_ms = (time.perf_counter() - post_t0) * 1000.0
    visible_index = 1 if generated["chunk_id"] == 0 and decoded_hwc_gpu.shape[0] > 1 else 0
    # Copy all frames only after the VAE stream has completed.  The visible
    # frame copy is retained to make the first-visible boundary explicit.
    visible_copy_t0 = time.perf_counter()
    _visible_frame = decoded_hwc_gpu[visible_index].float().cpu()
    visible_copy_ms = (time.perf_counter() - visible_copy_t0) * 1000.0
    host_first_visible_ms = (time.perf_counter() - generated["chunk_t0"]) * 1000.0
    if post_visible_hook is not None:
        post_visible_hook()
    all_copy_t0 = time.perf_counter()
    all_frames = decoded_hwc_gpu.float().cpu()
    all_copy_ms = (time.perf_counter() - all_copy_t0) * 1000.0
    host_all_frames_ms = (time.perf_counter() - generated["chunk_t0"]) * 1000.0
    if not overlap:
        first_visible_ms = host_first_visible_ms if generated["chunk_id"] > 0 else None
        full_action_ms = host_all_frames_ms if generated["chunk_id"] > 0 else None
    outputs.append(all_frames)
    x0 = generated["x0"]
    cache = generated.get("cache", cache_positions(state["self_kv_cache"]))
    stats = decoder_stats if decoder_stats is not None else decoder.last_stats.copy()
    waterfall = {
        "action_camera_conditioning_prepare": generated["action_prepare_ms"],
        "clean_kv_on_first_visible_path": (
            generated["clean_kv_ms"]
            if not generated.get("clean_kv_deferred", False) else 0.0
        ),
        "clean_kv_after_first_visible": (
            generated["clean_kv_ms"]
            if generated.get("clean_kv_deferred", False) else 0.0
        ),
        "latent_postprocessing": generated["latent_postprocess_ms"],
        "vae_decode": decode_ms,
        "rgb_postprocessing": rgb_postprocess_ms,
        "device_to_host_first_frame": visible_copy_ms,
        "device_to_host_all_frames": all_copy_ms,
        "presentation": 0.0,
        "serialization": 0.0,
        "sum_to_first_visible": first_visible_ms,
        "sum_to_all_frames_host": host_all_frames_ms,
        "state_ready": generated.get("state_ready_ms"),
        "next_action_ready": host_all_frames_ms,
    }
    for record in generated["forward_records"]:
        if record["kind"] == "denoise":
            waterfall[f"dit_forward_{int(record['index']) + 1}"] = record["elapsed_ms"]
        else:
            waterfall["clean_kv_forward"] = record["elapsed_ms"]
    ordered_waterfall = {"action_camera_conditioning_prepare": waterfall.pop("action_camera_conditioning_prepare")}
    for record in generated["forward_records"]:
        key = (
            f"dit_forward_{int(record['index']) + 1}"
            if record["kind"] == "denoise" else "clean_kv_forward"
        )
        ordered_waterfall[key] = waterfall.pop(key)
    ordered_waterfall.update(waterfall)

    actions.append({
        "chunk_id": generated["chunk_id"],
        "current_start": generated["kwargs"]["current_start"],
        "frames_decoded": int(all_frames.shape[0]),
        "new_visible_frame_index": visible_index,
        "transformer_ms": generated["transformer_ms"],
        "forward_ms": generated["forward_records"],
        "denoise_forward_count": sum(record["kind"] == "denoise" for record in generated["forward_records"]),
        "vae_decode_ms": decode_ms,
        "keypress_to_first_visible_ms": first_visible_ms,
        "full_action_ms": full_action_ms,
        "cache": cache,
        "cross_attention_initialized": generated.get(
            "cross_attention_initialized",
            int(state["cross_kv_cache"][0]["is_init"].item()),
        ),
        "memory_after_action": cuda_memory(device),
        "rss_bytes": current_rss_bytes(),
        "host_memory": host_memory(),
        "output_finite": bool(torch.isfinite(all_frames).all()),
        "latent_finite": bool(torch.isfinite(x0).all()),
        "decoded_finite": bool(torch.isfinite(decoded_gpu).all()),
        "decoded_min": float(decoded_gpu.min()) if torch.isfinite(decoded_gpu).all() else None,
        "decoded_max": float(decoded_gpu.max()) if torch.isfinite(decoded_gpu).all() else None,
        "decoder_stats": stats,
        "overlap": overlap,
        "clean_kv_deferred": bool(generated.get("clean_kv_deferred", False)),
        "clean_kv_ms": generated.get("clean_kv_ms"),
        "state_ready_ms": generated.get("state_ready_ms"),
        "next_action_ready_ms": host_all_frames_ms,
        "waterfall_ms": ordered_waterfall,
    })


def finish_taehv_action(
    pipe: WanI2VCausal,
    state: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    decoder: StreamingTAEHVDecoder,
    generated: dict[str, object],
    first_pending: torch.Tensor,
    tae_first_rgb_ms: float,
    actions: list[dict[str, object]],
    outputs: list[torch.Tensor],
    first_visible_ms: float | None,
    full_action_ms: float | None,
    post_visible_hook=None,
) -> None:
    """Present TAE's first frame before draining the remaining frame group."""
    first_frame_gpu = first_pending[:, 0]
    first_post_t0 = time.perf_counter()
    first_hwc_gpu = first_frame_gpu[0].permute(1, 2, 0).contiguous()
    first_rgb_postprocess_ms = (time.perf_counter() - first_post_t0) * 1000.0
    first_copy_t0 = time.perf_counter()
    first_host = first_hwc_gpu.float().clamp(0, 1).cpu()
    first_copy_ms = (time.perf_counter() - first_copy_t0) * 1000.0
    host_first_visible_ms = (time.perf_counter() - generated["chunk_t0"]) * 1000.0

    # The exact clean-KV transaction is allowed to begin only after the first
    # RGB is host-ready. It remains a prerequisite for the next DiT action.
    if post_visible_hook is not None:
        post_visible_hook()

    decoded_gpu, tae_remaining_rgb_ms = decoder.drain_latent(first_pending, device)
    decoded_hwc_gpu = decoded_gpu.permute(1, 2, 3, 0).contiguous()
    tail_copy_t0 = time.perf_counter()
    if decoded_hwc_gpu.shape[0] > 1:
        tail_host = decoded_hwc_gpu[1:].float().clamp(0, 1).cpu()
        all_frames = torch.cat([first_host.unsqueeze(0), tail_host], dim=0)
    else:
        all_frames = first_host.unsqueeze(0)
    all_copy_ms = (time.perf_counter() - tail_copy_t0) * 1000.0
    host_all_frames_ms = (time.perf_counter() - generated["chunk_t0"]) * 1000.0
    tae_all_rgb_ms = tae_first_rgb_ms + tae_remaining_rgb_ms
    output_model = all_frames.mul(2.0).sub(1.0)
    outputs.append(output_model)

    if generated["chunk_id"] > 0:
        first_visible_ms = host_first_visible_ms
        full_action_ms = host_all_frames_ms
    state_ready_ms = generated.get("state_ready_ms")
    if state_ready_ms is None:
        state_ready_ms = host_all_frames_ms
    generated["state_ready_ms"] = max(float(state_ready_ms), 0.0)

    ordered_waterfall: dict[str, object] = {
        "action_camera_conditioning_prepare": generated["action_prepare_ms"],
    }
    for record in generated["forward_records"]:
        if record["kind"] == "denoise":
            ordered_waterfall[f"dit_forward_{int(record['index']) + 1}"] = record["elapsed_ms"]
    if not generated.get("clean_kv_deferred", False):
        for record in generated["forward_records"]:
            if record["kind"] == "cache_update":
                ordered_waterfall["clean_kv_forward"] = record["elapsed_ms"]
    ordered_waterfall.update({
        "taehv_first_rgb_gpu": tae_first_rgb_ms,
        "taehv_first_rgb_host_copy": first_copy_ms,
    })
    if generated.get("clean_kv_deferred", False):
        for record in generated["forward_records"]:
            if record["kind"] == "cache_update":
                ordered_waterfall["clean_kv_forward"] = record["elapsed_ms"]
    ordered_waterfall.update({
        "taehv_remaining_rgb_gpu": tae_remaining_rgb_ms,
        "taehv_all_rgb_gpu": tae_all_rgb_ms,
        "latent_postprocessing": generated["latent_postprocess_ms"],
        "vae_decode": tae_all_rgb_ms,
        "rgb_postprocessing": first_rgb_postprocess_ms,
        "device_to_host_first_frame": first_copy_ms,
        "device_to_host_remaining_frames": all_copy_ms,
        "device_to_host_all_frames": first_copy_ms + all_copy_ms,
        "presentation": 0.0,
        "serialization": 0.0,
        "sum_to_first_visible": first_visible_ms,
        "sum_to_all_frames_host": host_all_frames_ms,
        "state_ready": generated["state_ready_ms"],
        "next_action_ready": host_all_frames_ms,
    })
    cache = generated.get("cache", cache_positions(state["self_kv_cache"]))
    actions.append({
        "chunk_id": generated["chunk_id"],
        "current_start": generated["kwargs"]["current_start"],
        "frames_decoded": int(all_frames.shape[0]),
        "new_visible_frame_index": 0,
        "transformer_ms": generated["transformer_ms"],
        "forward_ms": generated["forward_records"],
        "denoise_forward_count": sum(record["kind"] == "denoise" for record in generated["forward_records"]),
        "vae_decode_ms": tae_all_rgb_ms,
        "taehv_first_rgb_gpu_ms": tae_first_rgb_ms,
        "taehv_remaining_rgb_gpu_ms": tae_remaining_rgb_ms,
        "taehv_all_rgb_gpu_ms": tae_all_rgb_ms,
        "keypress_to_first_visible_ms": first_visible_ms,
        "full_action_ms": full_action_ms,
        "cache": cache,
        "cross_attention_initialized": generated.get(
            "cross_attention_initialized",
            int(state["cross_kv_cache"][0]["is_init"].item()),
        ),
        "memory_after_action": cuda_memory(device),
        "rss_bytes": current_rss_bytes(),
        "host_memory": host_memory(),
        "output_finite": bool(torch.isfinite(output_model).all()),
        "latent_finite": bool(torch.isfinite(generated["x0"]).all()),
        "decoded_finite": bool(torch.isfinite(decoded_gpu).all()),
        "decoded_min": float(output_model.min()),
        "decoded_max": float(output_model.max()),
        "decoder_stats": decoder.last_stats.copy(),
        "overlap": False,
        "clean_kv_deferred": bool(generated.get("clean_kv_deferred", False)),
        "clean_kv_ms": generated.get("clean_kv_ms"),
        "state_ready_ms": generated.get("state_ready_ms"),
        "next_action_ready_ms": host_all_frames_ms,
        "waterfall_ms": ordered_waterfall,
    })


def run_session(pipe: WanI2VCausal, state: dict[str, object], args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    outputs: list[torch.Tensor] = []
    accepted_latents: list[torch.Tensor] = []
    actions: list[dict[str, object]] = []
    if args.display_decoder == "taehv":
        if args.overlap:
            raise ValueError("TAEHV display mode is serial-only in this experiment")
        decoder = StreamingTAEHVDecoder(
            args.taehv_dir,
            device,
            args.taehv_weights,
        )
    else:
        decoder = IncrementalCausalDecoder(
            pipe.vae,
            args.vae_attention_backend,
            args.vae_dtype,
            profile=args.profile_vae,
            temporal_split_module=args.vae_temporal_split_module,
        )
    dit_profiler = None
    attention_probe = None
    @contextmanager
    def noop_no_sync():
        yield
    no_sync_model = getattr(pipe.model, "no_sync", noop_no_sync)

    session_t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad(), no_sync_model():
        main_stream = torch.cuda.current_stream(device)
        if args.overlap:
            vae_stream = torch.cuda.Stream(device=device)
            pending: list[tuple[dict[str, object], torch.Tensor, torch.cuda.Event, torch.cuda.Event, dict[str, object], torch.cuda.Event]] = []
            for chunk_id, (current_latent, current_condition, current_plucker) in enumerate(zip(
                state["noise_chunks"], state["condition_chunks"], state["plucker_chunks"]
            )):
                if args.max_chunks and chunk_id >= args.max_chunks:
                    break
                if args.profile_dit_from_chunk >= 0 and chunk_id == args.profile_dit_from_chunk:
                    dit_profiler = DecoderProfiler(pipe.model)
                if args.profile_attention_from_chunk >= 0 and chunk_id == args.profile_attention_from_chunk:
                    attention_probe = DitAttentionProbe()
                main_start = torch.cuda.Event(enable_timing=True)
                main_start.record(main_stream)
                generated = generate_chunk(
                    pipe, state, chunk_id, current_latent, current_condition,
                    current_plucker, device, main_stream.synchronize,
                    defer_clean_kv=False,
                )
                if attention_probe is not None and chunk_id >= args.profile_attention_from_chunk + 2:
                    attention_probe.close()
                # x0 is produced on the main stream.  The explicit dependency
                # plus record_stream keeps the allocator from recycling it
                # while the VAE stream consumes it.
                vae_stream.wait_stream(main_stream)
                generated["x0"].record_stream(vae_stream)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(vae_stream):
                    start_event.record(vae_stream)
                    decoded_gpu = decoder.decode_latent(generated["x0"], device, synchronize=False)
                    decoder_stats = decoder.last_stats.copy()
                    end_event.record(vae_stream)
                pending.append((generated, decoded_gpu, start_event, end_event, decoder_stats, main_start))

            vae_stream.synchronize()
            for generated, decoded_gpu, start_event, end_event, decoder_stats, main_start in pending:
                decode_ms = float(start_event.elapsed_time(end_event))
                completion_ms = float(main_start.elapsed_time(end_event))
                finish_action(
                    pipe, state, args, device, decoder, generated, decoded_gpu,
                    decode_ms, actions, outputs,
                    completion_ms if generated["chunk_id"] > 0 else None,
                    completion_ms if generated["chunk_id"] > 0 else None,
                    overlap=True, decoder_stats=decoder_stats,
                )
        else:
            for chunk_id, (current_latent, current_condition, current_plucker) in enumerate(zip(
                state["noise_chunks"], state["condition_chunks"], state["plucker_chunks"]
            )):
                if args.max_chunks and chunk_id >= args.max_chunks:
                    break
                if args.profile_dit_from_chunk >= 0 and chunk_id == args.profile_dit_from_chunk:
                    dit_profiler = DecoderProfiler(pipe.model)
                if args.profile_attention_from_chunk >= 0 and chunk_id == args.profile_attention_from_chunk:
                    attention_probe = DitAttentionProbe()
                generated = generate_chunk(
                    pipe, state, chunk_id, current_latent, current_condition,
                    current_plucker, device, lambda: sync(device),
                    defer_clean_kv=args.defer_clean_kv,
                )
                if args.save_latents:
                    accepted_latents.append(generated["x0"].detach().cpu())
                if attention_probe is not None and chunk_id >= args.profile_attention_from_chunk + 2:
                    attention_probe.close()
                post_visible_hook = (
                    lambda generated=generated: commit_clean_kv(
                        pipe, state, generated, device, lambda: sync(device)
                    )
                ) if args.defer_clean_kv else None
                if args.display_decoder == "taehv":
                    first_pending, first_rgb_ms, _ = decoder.begin_latent(generated["x0"], device)
                    finish_taehv_action(
                        pipe, state, args, device, decoder, generated,
                        first_pending, first_rgb_ms, actions, outputs,
                        None, None, post_visible_hook=post_visible_hook,
                    )
                else:
                    decode_t0 = time.perf_counter()
                    decoded_gpu = decoder.decode_latent(generated["x0"], device)
                    decode_ms = (time.perf_counter() - decode_t0) * 1000.0
                    action_ms = (time.perf_counter() - generated["chunk_t0"]) * 1000.0
                    finish_action(
                        pipe, state, args, device, decoder, generated, decoded_gpu,
                        decode_ms, actions, outputs,
                        action_ms if chunk_id > 0 else None,
                        action_ms if chunk_id > 0 else None,
                        post_visible_hook=post_visible_hook,
                    )

    profile = decoder.profile_report()
    dit_profile = dit_profiler.report() if dit_profiler is not None else None
    attention_profile = attention_probe.report() if attention_probe is not None else None
    output = torch.cat(outputs, dim=0)
    if args.save_latents:
        latent_stream = torch.cat(accepted_latents, dim=1)
        torch.save(
            {
                "latents": latent_stream,
                "dtype": str(latent_stream.dtype),
                "shape": list(latent_stream.shape),
                "geometry": [int(state["height"]), int(state["width"])],
                "latent_shape_cthw": [
                    int(latent_stream.shape[0]),
                    int(latent_stream.shape[1]),
                    int(latent_stream.shape[2]),
                    int(latent_stream.shape[3]),
                ],
                "denoise_schedule": state["denoise_schedule"],
                "timestep_indices": state["timestep_indices"],
                "timestep_values": state["timestep_values"],
                "seed": int(args.seed),
                "local_attn_size": int(args.local_attn_size),
                "sink_size": int(args.sink_size),
            },
            Path(args.output_dir) / "accepted_latents.pt",
        )
    decoder.clear()
    session_ms = (time.perf_counter() - session_t0) * 1000.0
    action_rows = [row for row in actions if row["full_action_ms"] is not None]
    result: dict[str, object] = {
        "status": "success",
        "overlap": bool(args.overlap),
        "display_decoder": args.display_decoder,
        "defer_clean_kv": bool(args.defer_clean_kv),
        "denoise_schedule": state["denoise_schedule"],
        "timestep_indices": state["timestep_indices"],
        "timestep_values": state["timestep_values"],
        "vae_profile": profile,
        "dit_profile": dit_profile,
        "dit_operator_summary": summarize_module_profile(dit_profile),
        "profile_dit_from_chunk": args.profile_dit_from_chunk,
        "attention_profile": attention_profile,
        "profile_attention_from_chunk": args.profile_attention_from_chunk,
        "frames": int(output.shape[0]),
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(torch.isfinite(output).all()),
            "min": float(output.min()),
            "max": float(output.max()),
            "mean": float(output.mean()),
            "std": float(output.std()),
            "mean_adjacent_frame_delta": float((output[1:] - output[:-1]).abs().mean()),
        },
        "session_ms": session_ms,
        "bootstrap": actions[0],
        "actions": actions,
        "action_latency_ms": {
            "first": action_rows[0]["full_action_ms"] if action_rows else None,
            "mean": float(np.mean([row["full_action_ms"] for row in action_rows])) if action_rows else None,
            "median": float(np.median([row["full_action_ms"] for row in action_rows])) if action_rows else None,
            "last_five_mean": float(np.mean([row["full_action_ms"] for row in action_rows[-5:]])) if action_rows else None,
            "first_visible_first_action": action_rows[0]["keypress_to_first_visible_ms"] if action_rows else None,
            "state_ready_first": action_rows[0]["state_ready_ms"] if action_rows else None,
            "state_ready_median": float(np.median([row["state_ready_ms"] for row in action_rows])) if action_rows else None,
            "next_action_ready_median": float(np.median([row["next_action_ready_ms"] for row in action_rows])) if action_rows else None,
        },
        "memory_after_session": cuda_memory(device),
        "max_rss_bytes": rss_bytes(),
        "host_memory_after_session": host_memory(),
        "rocm_smi_after_session": rocm_smi_snapshot(),
    }
    return {"result": result, "output": output}


def main() -> int:
    args = build_parser().parse_args()
    if args.frames < 5:
        raise SystemExit("use at least 5 frames so the causal session has a bootstrap and an action")
    result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    report: dict[str, object] = {
        "status": "running",
        "arguments": vars(args),
        "host": {"environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("HIP", "ROCR", "HSA", "PYTORCH", "TORCH", "CUDA", "ROCM"))}},
        "torch": {
            "version": torch.__version__,
            "hip": torch.version.hip,
            "device_name": torch.cuda.get_device_name(device),
            "properties": str(torch.cuda.get_device_properties(device)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
    }
    try:
        init_t0 = time.perf_counter()
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
            metrics=None,
        )
        sync(device)
        report["model_initialization_ms"] = (time.perf_counter() - init_t0) * 1000.0
        report["parameter_report"] = parameter_report(pipe)
        state_t0 = time.perf_counter()
        state = prepare_session(pipe, args, device)
        report["session_prepare_ms"] = (time.perf_counter() - state_t0) * 1000.0
        report["session_config"] = {
            key: value for key, value in state.items()
            if key in ("lat_f", "lat_h", "lat_w", "height", "width", "max_area_pixels", "frame_seqlen", "kv_size", "frames", "timestep_indices", "timestep_values", "denoise_schedule", "t5_encode_ms", "t5_cache_hit", "vae_encode_ms")
        }
        current = run_session(pipe, state, args, device)
        report.update(current["result"])
        output = current["output"]
        if args.save_video:
            path = result_dir / "interactive.mp4"
            write_video(output, path)
            report["video_path"] = str(path)
        report["status"] = "success"
    except Exception as exc:
        report["status"] = "failure"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        (result_dir / "failure.txt").write_text(report["traceback"])
        (result_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    (result_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "frames": report.get("frames"),
        "first_visible_ms": report.get("action_latency_ms", {}).get("first_visible_first_action"),
        "action_median_ms": report.get("action_latency_ms", {}).get("median"),
        "session_ms": report.get("session_ms"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
