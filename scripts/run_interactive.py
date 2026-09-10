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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", choices=sorted(MAX_AREA_CONFIGS), default="480*832")
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--local-attn-size", type=int, default=18)
    parser.add_argument("--sink-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--action-path", required=True)
    parser.add_argument("--save-video", action="store_true")
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
    max_area = MAX_AREA_CONFIGS[args.size]
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
    timesteps = pipe.scheduler.timesteps[[0, 179, 358, 679]]
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
        "seed_g": seed_g,
        "self_kv_cache": self_kv_cache,
        "cross_kv_cache": cross_kv_cache,
        "lat_f": lat_f,
        "lat_h": lat_h,
        "lat_w": lat_w,
        "height": h,
        "width": w,
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
) -> dict[str, object]:
    """Run the five causal-fast DiT forwards for one latent chunk."""
    chunk_t0 = time.perf_counter()
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
    forward_records: list[dict[str, object]] = []
    transformer_t0 = time.perf_counter()
    for timestep_idx, current_timestep in enumerate(state["timesteps"]):
        forward_t0 = time.perf_counter()
        noise_pred = pipe.model(
            x=[current_latent],
            t=torch.stack([current_timestep]).to(device),
            cross_attn_first_call=not pipe._cross_attn_initialized,
            **kwargs,
        )[0]
        sync_fn()
        forward_records.append({
            "kind": "denoise",
            "index": timestep_idx,
            "timestep": float(current_timestep),
            "elapsed_ms": (time.perf_counter() - forward_t0) * 1000.0,
            "finite": bool(torch.isfinite(noise_pred).all()),
        })
        pipe._cross_attn_initialized = True
        x0 = pipe._convert_flow_pred_to_x0(noise_pred, current_latent, current_timestep, pipe.scheduler)
        if timestep_idx < len(state["timesteps"]) - 1:
            next_timestep = state["timesteps"][timestep_idx + 1]
            current_latent = pipe.scheduler.add_noise(
                x0,
                torch.randn(x0.shape, generator=state["seed_g"], device=device, dtype=x0.dtype),
                next_timestep,
            )

    cache_t0 = time.perf_counter()
    zero_timestep = state["timesteps"][-1] * 0.0
    pipe.model(
        x=[x0],
        t=torch.stack([zero_timestep]).to(device),
        cross_attn_first_call=False,
        **kwargs,
    )
    sync_fn()
    forward_records.append({
        "kind": "cache_update",
        "index": len(state["timesteps"]),
        "timestep": 0.0,
        "elapsed_ms": (time.perf_counter() - cache_t0) * 1000.0,
    })
    return {
        "chunk_id": chunk_id,
        "chunk_t0": chunk_t0,
        "kwargs": kwargs,
        "x0": x0,
        "cache": cache_positions(state["self_kv_cache"]),
        "cross_attention_initialized": int(state["cross_kv_cache"][0]["is_init"].item()),
        "transformer_ms": (time.perf_counter() - transformer_t0) * 1000.0,
        "forward_records": forward_records,
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
) -> None:
    decoded_hwc_gpu = decoded_gpu.permute(1, 2, 3, 0).contiguous()
    visible_index = 1 if generated["chunk_id"] == 0 and decoded_hwc_gpu.shape[0] > 1 else 0
    # Copy all frames only after the VAE stream has completed.  The visible
    # frame copy is retained to make the first-visible boundary explicit.
    _visible_frame = decoded_hwc_gpu[visible_index].float().cpu()
    all_frames = decoded_hwc_gpu.float().cpu()
    outputs.append(all_frames)
    x0 = generated["x0"]
    cache = generated.get("cache", cache_positions(state["self_kv_cache"]))
    stats = decoder_stats if decoder_stats is not None else decoder.last_stats.copy()
    actions.append({
        "chunk_id": generated["chunk_id"],
        "current_start": generated["kwargs"]["current_start"],
        "frames_decoded": int(all_frames.shape[0]),
        "new_visible_frame_index": visible_index,
        "transformer_ms": generated["transformer_ms"],
        "forward_ms": generated["forward_records"],
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
    })


def run_session(pipe: WanI2VCausal, state: dict[str, object], args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    outputs: list[torch.Tensor] = []
    actions: list[dict[str, object]] = []
    decoder = IncrementalCausalDecoder(
        pipe.vae,
        args.vae_attention_backend,
        args.vae_dtype,
        profile=args.profile_vae,
        temporal_split_module=args.vae_temporal_split_module,
    )
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
                main_start = torch.cuda.Event(enable_timing=True)
                main_start.record(main_stream)
                generated = generate_chunk(
                    pipe, state, chunk_id, current_latent, current_condition,
                    current_plucker, device, main_stream.synchronize,
                )
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
                generated = generate_chunk(
                    pipe, state, chunk_id, current_latent, current_condition,
                    current_plucker, device, lambda: sync(device),
                )
                decode_t0 = time.perf_counter()
                decoded_gpu = decoder.decode_latent(generated["x0"], device)
                decode_ms = (time.perf_counter() - decode_t0) * 1000.0
                action_ms = (time.perf_counter() - generated["chunk_t0"]) * 1000.0
                finish_action(
                    pipe, state, args, device, decoder, generated, decoded_gpu,
                    decode_ms, actions, outputs,
                    action_ms if chunk_id > 0 else None,
                    action_ms if chunk_id > 0 else None,
                )

    profile = decoder.profile_report()
    output = torch.cat(outputs, dim=0)
    decoder.clear()
    session_ms = (time.perf_counter() - session_t0) * 1000.0
    action_rows = [row for row in actions if row["full_action_ms"] is not None]
    result: dict[str, object] = {
        "status": "success",
        "overlap": bool(args.overlap),
        "vae_profile": profile,
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
            if key in ("lat_f", "lat_h", "lat_w", "height", "width", "frame_seqlen", "kv_size", "frames", "t5_encode_ms", "t5_cache_hit", "vae_encode_ms")
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
