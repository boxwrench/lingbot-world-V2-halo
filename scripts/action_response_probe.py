#!/usr/bin/env python3
"""Measure when a live action becomes visible and action-dependent.

This is a bounded diagnostic harness.  It generates one common persistent
prefix, clones that complete world state, and runs matched action branches.
The model, sampler, cache semantics, and deferred clean-KV transaction are
the same as the accepted live path; no optimization is implemented here.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from run_interactive import (
    StreamingTAEHVDecoder,
    cache_positions,
    commit_clean_kv,
    generate_chunk,
    prepare_session,
    sync,
)
from run_live import action_from_key, make_plucker
from wan.configs import WAN_CONFIGS
from wan.image2video import WanI2VCausal


PREFIX_ACTIONS = ["w", "w", "j", "w", "l", "l", "s", "s", "j", "w", "d", "d", "w", "l", "a", "a", "w", "j", "s", "l"]
BRANCH_PAIRS = {
    "stay_vs_strong_turn": (" ", "l"),
    "forward_vs_reverse": ("w", "s"),
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream-dir", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--taehv-dir", required=True)
    p.add_argument("--local-attn-size", type=int, default=12)
    p.add_argument("--sink-size", type=int, default=6)
    p.add_argument("--prefix-actions", default=",".join(PREFIX_ACTIONS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default="A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky, while the camera glides smoothly forward.")
    p.add_argument("--image", required=True)
    p.add_argument("--action-path", required=True)
    p.add_argument("--input-delay-ms", type=float, default=50.0)
    p.add_argument("--max-area-pixels", type=int, default=264192)
    p.add_argument("--size", default="480*832")
    return p


def runtime_args(args: argparse.Namespace, frames: int) -> argparse.Namespace:
    return argparse.Namespace(
        upstream_dir=args.upstream_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        size=args.size,
        max_area_pixels=args.max_area_pixels,
        frames=frames,
        denoise_schedule="3-drop-957",
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        seed=args.seed,
        prompt=args.prompt,
        image=args.image,
        action_path=args.action_path,
        save_video=False,
        save_latents=False,
        max_chunks=0,
        vae_attention_backend="math",
        vae_dtype="fp16",
        display_decoder="taehv",
        taehv_dir=args.taehv_dir,
        taehv_weights=None,
        overlap=False,
        profile_vae=False,
        vae_temporal_split_module=None,
        profile_dit_from_chunk=-1,
        defer_clean_kv=True,
        profile_attention_from_chunk=-1,
    )


def clone_cache(cache: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [
        {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in layer.items()}
        for layer in cache
    ]


def clone_state(base: dict[str, object]) -> dict[str, object]:
    state = dict(base)
    state["self_kv_cache"] = clone_cache(base["self_kv_cache"])
    state["cross_kv_cache"] = clone_cache(base["cross_kv_cache"])
    generator = torch.Generator(device="cuda")
    generator.set_state(base["seed_g"].get_state())
    state["seed_g"] = generator
    return state


def frame_host(frame: torch.Tensor) -> torch.Tensor:
    return frame.detach().float().clamp(0.0, 1.0).cpu()


def save_frame(frame: torch.Tensor, path: Path) -> None:
    array = frame_host(frame)
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] == 3:
        array = array.permute(1, 2, 0)
    image = (array.numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def action_pose(key: str) -> tuple[str, torch.Tensor]:
    selected = action_from_key(key)
    if selected is None:
        raise ValueError(f"unsupported branch action {key!r}")
    return selected


def decode_stream(
    decoder: StreamingTAEHVDecoder,
    latent: torch.Tensor,
    device: torch.device,
    action_start: float,
    output_dir: Path,
    branch_name: str,
    input_probe: bool,
    input_delay_ms: float,
    pipe: WanI2VCausal,
    state: dict[str, object],
    generated: dict[str, object],
) -> dict[str, object]:
    """Decode RGB0..RGBn with individual ready/presentation boundaries."""
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    tae_latent = latent.to(device=device, dtype=torch.float16).permute(0, 2, 1, 3, 4).contiguous()
    frames: list[torch.Tensor] = []
    frame_records: list[dict[str, object]] = []

    def decode_one(index: int, input_tensor: torch.Tensor | None) -> bool:
        t0 = time.perf_counter()
        pending = decoder.streaming.decode(input_tensor)
        sync(device)
        ready = time.perf_counter()
        if pending is None:
            return False
        frame = pending[:, 0]
        copy_t0 = time.perf_counter()
        host = frame_host(frame)
        host_ready = time.perf_counter()
        # In this headless probe, host-ready is the presentation proxy.  The
        # real Tk viewer adds its own short compositor/update interval.
        frames.append(host)
        frame_records.append({
            "index": index,
            "gpu_ready_ms": (ready - action_start) * 1000.0,
            "host_ready_ms": (host_ready - action_start) * 1000.0,
            "presented_proxy_ms": (host_ready - action_start) * 1000.0,
            "decode_call_ms": (ready - t0) * 1000.0,
            "host_copy_ms": (host_ready - copy_t0) * 1000.0,
            "finite": bool(torch.isfinite(host).all()),
        })
        return True

    decode_one(0, tae_latent)
    input_probe_result: dict[str, object] | None = None
    arrival: dict[str, float] = {}
    clean_start = time.perf_counter()
    if input_probe:
        def receive_input() -> None:
            time.sleep(max(0.0, input_delay_ms) / 1000.0)
            arrival["arrival"] = time.perf_counter()

        worker = threading.Thread(target=receive_input, name="action-arrival-probe", daemon=True)
        worker.start()
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
        commit_clean_kv(pipe, state, generated, device, lambda: sync(device))
    clean_end = time.perf_counter()
    if input_probe:
        worker.join(timeout=1.0)
        arrived = arrival.get("arrival")
        input_probe_result = {
            "command": "forward",
            "scheduled_after_first_present_ms": input_delay_ms,
            "arrival_ms_from_action_start": ((arrived - action_start) * 1000.0) if arrived else None,
            "clean_start_ms_from_action_start": (clean_start - action_start) * 1000.0,
            "clean_end_ms_from_action_start": (clean_end - action_start) * 1000.0,
            "arrived_during_clean": bool(arrived is not None and clean_start <= arrived <= clean_end),
            "observer_recorded_arrival": bool(arrived is not None),
            "current_live_runner_queue": "none",
            "current_live_runner_selection": "not selected until the next input wait after clean/remaining RGB; no GPU generation may start before clean KV commit",
        }
    while decoder.streaming.decoder_work_queue:
        if not decode_one(len(frames), None):
            break
    next_ready = time.perf_counter()
    for index, frame in enumerate(frames):
        save_frame(frame, output_dir / f"{branch_name}-rgb{index}.png")
    return {
        "branch": branch_name,
        "action_start_ms": 0.0,
        "x0_accepted_ms": (generated["x0_accepted_t"] - action_start) * 1000.0,
        "rgb_frames": frame_records,
        "first_rgb_presented_ms": frame_records[0]["presented_proxy_ms"],
        "all_rgb_complete_ms": (next_ready - action_start) * 1000.0,
        "clean_start_ms": (clean_start - action_start) * 1000.0,
        "clean_commit_ms": generated["clean_kv_ms"],
        "clean_end_ms": (clean_end - action_start) * 1000.0,
        "next_action_permitted_ms": (next_ready - action_start) * 1000.0,
        "input_probe": input_probe_result,
        "x0_stats": {
            "finite": bool(torch.isfinite(generated["x0"]).all()),
            "min": float(generated["x0"].min()),
            "max": float(generated["x0"].max()),
            "mean": float(generated["x0"].mean()),
            "std": float(generated["x0"].std()),
        },
        "cache_after": cache_positions(state["self_kv_cache"]),
        "output_finite": all(bool(record["finite"]) for record in frame_records),
        "frames_cpu": frames,
    }


def measure_branch(
    pipe: WanI2VCausal,
    base_state: dict[str, object],
    prefix_latents: list[torch.Tensor],
    args: argparse.Namespace,
    runtime: argparse.Namespace,
    device: torch.device,
    branch_key: str,
    output_dir: Path,
    input_probe: bool,
) -> dict[str, object]:
    state = clone_state(base_state)
    decoder = StreamingTAEHVDecoder(args.taehv_dir, device)
    try:
        for latent in prefix_latents:
            pending, _, _ = decoder.begin_latent(latent, device)
            decoder.drain_latent(pending, device)
        name, pose = action_pose(branch_key)
        selected_t0 = time.perf_counter()
        plucker = make_plucker(pose, runtime, state, device, pipe.param_dtype)
        generation_start = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
            generated = generate_chunk(
                pipe,
                state,
                len(prefix_latents),
                state["noise_chunks"][len(prefix_latents)],
                state["condition_chunks"][len(prefix_latents)],
                plucker,
                device,
                lambda: sync(device),
                defer_clean_kv=True,
            )
        generated["x0_accepted_t"] = time.perf_counter()
        measured = decode_stream(
            decoder,
            generated["x0"],
            device,
            selected_t0,
            output_dir,
            name,
            input_probe,
            args.input_delay_ms,
            pipe,
            state,
            generated,
        )
        measured["input_received_ms"] = 0.0
        measured["input_selected_ms"] = 0.0
        measured["action_conditioning_prepare_ms"] = (generation_start - selected_t0) * 1000.0
        measured["generation_start_ms"] = (generation_start - selected_t0) * 1000.0
        measured["denoise_ms"] = generated["transformer_ms"]
        measured["clean_forward_records"] = [r for r in generated["forward_records"] if r["kind"] == "cache_update"]
        measured["frames_cpu"] = torch.stack([frame.squeeze(0) for frame in measured.pop("frames_cpu")])
        return measured
    finally:
        decoder.clear()


def compare_pair(a: dict[str, object], b: dict[str, object], name: str) -> dict[str, object]:
    frames_a = a["frames_cpu"]
    frames_b = b["frames_cpu"]
    count = min(frames_a.shape[0], frames_b.shape[0])
    rows = []
    for index in range(count):
        diff = (frames_a[index] - frames_b[index]).abs()
        rows.append({
            "index": index,
            "mean_abs_diff": float(diff.mean()),
            "p95_abs_diff": float(torch.quantile(diff.flatten(), 0.95)),
            "fraction_over_0.05": float((diff > 0.05).float().mean()),
            "a_presented_ms": a["rgb_frames"][index]["presented_proxy_ms"],
            "b_presented_ms": b["rgb_frames"][index]["presented_proxy_ms"],
        })
    # This is a diagnostic screening threshold, not a definition of quality.
    first_material = next((row for row in rows if row["mean_abs_diff"] >= 0.02 and row["fraction_over_0.05"] >= 0.10), None)
    return {
        "pair": name,
        "branches": [a["branch"], b["branch"]],
        "frame_differences": rows,
        "first_numeric_material_difference": first_material,
    }


def main() -> int:
    args = parser().parse_args()
    prefix_actions = [key.strip().lower() for key in args.prefix_actions.split(",") if key.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    runtime = runtime_args(args, frames=4 * (len(prefix_actions) + 3) + 1)
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
    state = prepare_session(pipe, runtime, device)
    prefix_decoder = StreamingTAEHVDecoder(args.taehv_dir, device)
    prefix_latents: list[torch.Tensor] = []
    try:
        with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
            bootstrap = generate_chunk(
                pipe,
                state,
                0,
                state["noise_chunks"][0],
                state["condition_chunks"][0],
                state["plucker_chunks"][0],
                device,
                lambda: sync(device),
                defer_clean_kv=True,
            )
        prefix_latents.append(bootstrap["x0"].detach().clone())
        pending, _, _ = prefix_decoder.begin_latent(bootstrap["x0"], device)
        prefix_decoder.drain_latent(pending, device)
        with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
            commit_clean_kv(pipe, state, bootstrap, device, lambda: sync(device))
        for index, key in enumerate(prefix_actions, start=1):
            _, pose = action_pose(key)
            plucker = make_plucker(pose, runtime, state, device, pipe.param_dtype)
            with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
                generated = generate_chunk(
                    pipe,
                    state,
                    index,
                    state["noise_chunks"][index],
                    state["condition_chunks"][index],
                    plucker,
                    device,
                    lambda: sync(device),
                    defer_clean_kv=True,
                )
            prefix_latents.append(generated["x0"].detach().clone())
            pending, _, _ = prefix_decoder.begin_latent(generated["x0"], device)
            prefix_decoder.drain_latent(pending, device)
            with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
                commit_clean_kv(pipe, state, generated, device, lambda: sync(device))
    finally:
        prefix_decoder.clear()

    common_state = clone_state(state)
    common_pipe_flag = pipe._cross_attn_initialized
    results: dict[str, object] = {
        "status": "success",
        "configuration": {
            "geometry": [int(state["height"]), int(state["width"])],
            "latent_geometry": [int(state["lat_h"]), int(state["lat_w"])],
            "tokens_per_frame": int(state["frame_seqlen"]),
            "local_attn_size": args.local_attn_size,
            "sink_size": args.sink_size,
            "timestep_values": state["timestep_values"],
            "prefix_actions": prefix_actions,
            "prefix_chunks": len(prefix_latents),
            "branch_chunk_id": len(prefix_latents),
        },
        "common_state_after_prefix": cache_positions(state["self_kv_cache"]),
        "pairs": {},
    }
    for pair_name, (first_key, second_key) in BRANCH_PAIRS.items():
        first = measure_branch(pipe, common_state, prefix_latents, args, runtime, device, first_key, output_dir, input_probe=pair_name == "stay_vs_strong_turn")
        pipe._cross_attn_initialized = common_pipe_flag
        second = measure_branch(pipe, common_state, prefix_latents, args, runtime, device, second_key, output_dir, input_probe=False)
        pipe._cross_attn_initialized = common_pipe_flag
        results["pairs"][pair_name] = {
            "branches": [first, second],
            "comparison": compare_pair(first, second, pair_name),
        }
    for pair in results["pairs"].values():
        for branch in pair["branches"]:
            branch.pop("frames_cpu", None)
    results["common_state_after_prefix"] = cache_positions(common_state["self_kv_cache"])
    (output_dir / "action_response_metrics.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps({
        "status": results["status"],
        "metrics": str(output_dir / "action_response_metrics.json"),
        "pairs": list(results["pairs"]),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
