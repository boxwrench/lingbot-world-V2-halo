#!/usr/bin/env python3
"""C2 bounded chunk/context screen: one matched-state quality trajectory.

Runs a fixed scripted action sequence on the live chunk-size-1 path and
captures per-chunk x0 latents plus TAEHV RGB. Fresh process per
configuration; same seed/script/prompt/image across configs makes chunk
indices comparable. The first WARMUP_GENERATIONS chunks are excluded from
scoring (first-encode guard) but their hashes are recorded.

No optimization is implemented here. Run via /tmp/run-c2-screen.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from action_response_probe import decode_stream
from run_interactive import (
    StreamingTAEHVDecoder,
    cache_positions,
    generate_chunk,
    prepare_session,
    sync,
)
from run_live import action_from_key, make_plucker
from wan.configs import WAN_CONFIGS
from wan.image2video import WanI2VCausal

SCREEN_ACTIONS = ["w", "w", "j", "w", "l", "l", "s", "s", "j", "w", "d", "d", "w", "l", "a", "a"]
WARMUP_GENERATIONS = 2  # chunk 0 (bootstrap) + chunk 1; score from chunk 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream-dir", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--taehv-dir", required=True)
    p.add_argument("--local-attn-size", type=int, default=12)
    p.add_argument("--sink-size", type=int, default=6)
    p.add_argument("--denoise-schedule", default="3-drop-957")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prompt",
        default="A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky, while the camera glides smoothly forward.",
    )
    p.add_argument("--image", required=True)
    p.add_argument("--action-path", required=True)
    p.add_argument("--screen-actions", default=",".join(SCREEN_ACTIONS))
    p.add_argument("--size", default="480*832")
    p.add_argument("--max-area-pixels", type=int, default=264192)
    return p


def runtime_args(args: argparse.Namespace, frames: int) -> argparse.Namespace:
    return argparse.Namespace(
        upstream_dir=args.upstream_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        size=args.size,
        max_area_pixels=args.max_area_pixels,
        frames=frames,
        denoise_schedule=args.denoise_schedule,
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


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().numpy().tobytes()).hexdigest()


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    args = parser().parse_args()
    keys = [key.strip().lower() for key in args.screen_actions.split(",") if key.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    total_generations = WARMUP_GENERATIONS + len(keys)
    runtime = runtime_args(args, frames=4 * (total_generations + 2) + 1)
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
    decoder = StreamingTAEHVDecoder(args.taehv_dir, device)
    scored_x0: list[torch.Tensor] = []
    chunks: list[dict[str, object]] = []
    warmup: list[dict[str, object]] = []
    try:
        with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
            bootstrap = generate_chunk(
                pipe, state, 0,
                state["noise_chunks"][0], state["condition_chunks"][0],
                state["plucker_chunks"][0], device, lambda: sync(device),
                defer_clean_kv=True,
            )
        warmup.append({
            "chunk_id": 0,
            "condition_sha256": tensor_sha(state["condition_chunks"][0]),
            "x0_sha256": tensor_sha(bootstrap["x0"]),
        })
        pending, _, _ = decoder.begin_latent(bootstrap["x0"], device)
        decoder.drain_latent(pending, device)
        with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
            from run_interactive import commit_clean_kv
            commit_clean_kv(pipe, state, bootstrap, device, lambda: sync(device))
        for index in range(1, total_generations):
            if index < WARMUP_GENERATIONS:
                # Warmup continuation: reuse the script prefix head so noise
                # indexing stays aligned with the scored trajectory.
                use_key = keys[index - 1]
            else:
                use_key = keys[index - WARMUP_GENERATIONS]
            selected = action_from_key(use_key)
            if selected is None:
                raise ValueError(f"unsupported action key {use_key!r}")
            _, pose = selected
            plucker = make_plucker(pose, runtime, state, device, pipe.param_dtype)
            action_start = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
                generated = generate_chunk(
                    pipe, state, index,
                    state["noise_chunks"][index], state["condition_chunks"][index],
                    plucker, device, lambda: sync(device),
                    defer_clean_kv=True,
                )
            generated["x0_accepted_t"] = time.perf_counter()
            if index < WARMUP_GENERATIONS:
                warmup.append({
                    "chunk_id": index,
                    "action_key": use_key,
                    "condition_sha256": tensor_sha(state["condition_chunks"][index]),
                    "x0_sha256": tensor_sha(generated["x0"]),
                })
                pending, _, _ = decoder.begin_latent(generated["x0"], device)
                decoder.drain_latent(pending, device)
                with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
                    commit_clean_kv(pipe, state, generated, device, lambda: sync(device))
                continue
            measured = decode_stream(
                decoder, generated["x0"], device, action_start,
                output_dir, f"chunk{index:02d}", False, 0.0,
                pipe, state, generated,
            )
            scored_x0.append(generated["x0"].detach().float().cpu().clone())
            chunks.append({
                "chunk_id": index,
                "action_key": use_key,
                "condition_sha256": tensor_sha(state["condition_chunks"][index]),
                "x0_sha256": tensor_sha(generated["x0"]),
                "x0_stats": measured["x0_stats"],
                "rgb_frames": measured["rgb_frames"],
                "denoise_ms": measured["denoise_ms"],
                "clean_commit_ms": measured["clean_commit_ms"],
                "output_finite": measured["output_finite"],
                "cache_after": {k: (int(v) if not hasattr(v, "item") else int(v.item())) for k, v in measured["cache_after"].items()},
            })
    finally:
        decoder.clear()
    latent_stream = torch.cat(scored_x0, dim=1)
    torch.save(
        {
            "latents": latent_stream,
            "dtype": str(latent_stream.dtype),
            "shape": list(latent_stream.shape),
            "geometry": [int(state["height"]), int(state["width"])],
            "denoise_schedule": state["denoise_schedule"],
            "timestep_values": state["timestep_values"],
            "seed": int(args.seed),
            "local_attn_size": int(args.local_attn_size),
            "sink_size": int(args.sink_size),
            "screen_actions": keys,
            "warmup_generations": WARMUP_GENERATIONS,
        },
        output_dir / "accepted_latents.pt",
    )
    payload = {
        "kind": "c2_context_screen_trajectory",
        "configuration": {
            "local_attn_size": int(args.local_attn_size),
            "sink_size": int(args.sink_size),
            "denoise_schedule": args.denoise_schedule,
            "timestep_values": state["timestep_values"],
            "geometry": [int(state["height"]), int(state["width"])],
            "seed": int(args.seed),
            "screen_actions": keys,
            "warmup_generations": WARMUP_GENERATIONS,
            "scored_chunks": [c["chunk_id"] for c in chunks],
        },
        "provenance": {
            "base_commit": command_output(["git", "rev-parse", "HEAD"]),
            "upstream_commit": command_output(["git", "-C", ".upstream/lingbot-world-v2", "rev-parse", "HEAD"]),
            "model_dir": str(Path(args.model_dir).resolve()),
        },
        "warmup_hashes": warmup,
        "chunks": chunks,
    }
    (output_dir / "screen_trajectory.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({
        "status": "success",
        "chunks_scored": len(chunks),
        "all_finite": all(c["output_finite"] for c in chunks),
        "metrics": str(output_dir / "screen_trajectory.json"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
