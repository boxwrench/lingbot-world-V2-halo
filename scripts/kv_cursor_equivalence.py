#!/usr/bin/env python3
"""Compare tensor-cursor and host-cursor causal-fast state transitions.

This is a correctness harness for the opt-in host-KV-cursor experiment.  It
uses identical prepared latent/conditioning chunks and independent cloned
caches, then compares x0, K/V contents, cursor mirrors, and finite status
through cache fill, rollover, and the exact clean-latent commit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_interactive import (
    commit_clean_kv,
    generate_chunk,
    prepare_session,
    sync,
)
from wan.configs import WAN_CONFIGS
from wan.image2video import WanI2VCausal


PROMPT = (
    "A sweeping cinematic journey along the Great Wall of China, winding "
    "through golden autumn hills under a brilliant blue sky, while the "
    "camera glides smoothly forward."
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream-dir", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--action-path", required=True)
    p.add_argument("--size", default="480*832")
    p.add_argument("--max-area-pixels", type=int, default=264192)
    p.add_argument("--frames", type=int, default=69)
    p.add_argument("--local-attn-size", type=int, default=12)
    p.add_argument("--sink-size", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunks", type=int, default=16)
    return p


def runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        upstream_dir=args.upstream_dir,
        model_dir=args.model_dir,
        output_dir=str(Path(args.output).parent),
        size=args.size,
        max_area_pixels=args.max_area_pixels,
        frames=args.frames,
        denoise_schedule="3-drop-957",
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        host_kv_cursor=False,
        seed=args.seed,
        prompt=PROMPT,
        image=args.image,
        action_path=args.action_path,
        save_video=False,
        save_latents=False,
        max_chunks=0,
        vae_attention_backend="math",
        vae_dtype="fp16",
        display_decoder="taehv",
        taehv_dir="",
        taehv_weights=None,
        overlap=False,
        profile_vae=False,
        vae_temporal_split_module=None,
        profile_dit_from_chunk=-1,
        defer_clean_kv=True,
        profile_attention_from_chunk=-1,
    )


def clone_cache(cache: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in layer.items()
        }
        for layer in cache
    ]


def clone_state(base: dict[str, object], host_cursor: bool) -> dict[str, object]:
    state = dict(base)
    state["self_kv_cache"] = clone_cache(base["self_kv_cache"])
    state["cross_kv_cache"] = clone_cache(base["cross_kv_cache"])
    generator = torch.Generator(device="cuda")
    generator.set_state(base["seed_g"].get_state())
    state["seed_g"] = generator
    if host_cursor:
        for layer in state["self_kv_cache"]:
            layer["global_end_index_py"] = 0
            layer["local_end_index_py"] = 0
    return state


def cache_summary(cache: list[dict[str, object]]) -> dict[str, object]:
    tensor_pairs = [
        (int(layer["global_end_index"].item()), int(layer["local_end_index"].item()))
        for layer in cache
    ]
    host_pairs = [
        (layer.get("global_end_index_py"), layer.get("local_end_index_py"))
        for layer in cache
    ]
    return {
        "tensor_unique": sorted(set(tensor_pairs)),
        "host_unique": sorted(set(host_pairs)),
        "layer_count": len(cache),
        "all_tensor_layers_equal": len(set(tensor_pairs)) == 1,
        "all_host_layers_equal": len(set(host_pairs)) == 1 if any(host_pairs[0]) else None,
    }


def compare_caches(
    control: list[dict[str, object]],
    candidate: list[dict[str, object]],
) -> dict[str, object]:
    max_k = 0.0
    max_v = 0.0
    exact_k = True
    exact_v = True
    tensor_cursor_equal = True
    host_mirrors_equal = True
    for left, right in zip(control, candidate):
        tensor_cursor_equal &= (
            torch.equal(left["global_end_index"], right["global_end_index"])
            and torch.equal(left["local_end_index"], right["local_end_index"])
        )
        host_mirrors_equal &= (
            int(right["global_end_index_py"]) == int(left["global_end_index"].item())
            and int(right["local_end_index_py"]) == int(left["local_end_index"].item())
        )
        k_delta = (left["k"] - right["k"]).abs()
        v_delta = (left["v"] - right["v"]).abs()
        max_k = max(max_k, float(k_delta.max().item()))
        max_v = max(max_v, float(v_delta.max().item()))
        exact_k &= bool(torch.equal(left["k"], right["k"]))
        exact_v &= bool(torch.equal(left["v"], right["v"]))
    return {
        "tensor_cursor_equal": tensor_cursor_equal,
        "host_mirrors_equal": host_mirrors_equal,
        "k_exact": exact_k,
        "v_exact": exact_v,
        "max_abs_k": max_k,
        "max_abs_v": max_v,
    }


def main() -> int:
    args = parser().parse_args()
    device = torch.device("cuda:0")
    runtime = runtime_args(args)
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
    base = prepare_session(pipe, runtime, device)
    control = clone_state(base, host_cursor=False)
    candidate = clone_state(base, host_cursor=True)
    control_cross_init = False
    candidate_cross_init = False
    rows: list[dict[str, object]] = []
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
        for chunk_id in range(min(args.chunks, len(base["noise_chunks"]))):
            common = {
                "chunk_id": chunk_id,
                "control_before": cache_summary(control["self_kv_cache"]),
                "candidate_before": cache_summary(candidate["self_kv_cache"]),
            }
            pipe._cross_attn_initialized = control_cross_init
            generated_control = generate_chunk(
                pipe, control, chunk_id, control["noise_chunks"][chunk_id],
                control["condition_chunks"][chunk_id], control["plucker_chunks"][chunk_id],
                device, lambda: sync(device), defer_clean_kv=True,
            )
            control_cross_init = True
            pipe._cross_attn_initialized = candidate_cross_init
            generated_candidate = generate_chunk(
                pipe, candidate, chunk_id, candidate["noise_chunks"][chunk_id],
                candidate["condition_chunks"][chunk_id], candidate["plucker_chunks"][chunk_id],
                device, lambda: sync(device), defer_clean_kv=True,
            )
            candidate_cross_init = True
            x0_delta = (generated_control["x0"] - generated_candidate["x0"]).abs()
            common.update({
                "x0_finite_control": bool(torch.isfinite(generated_control["x0"]).all()),
                "x0_finite_candidate": bool(torch.isfinite(generated_candidate["x0"]).all()),
                "x0_exact": bool(torch.equal(generated_control["x0"], generated_candidate["x0"])),
                "x0_max_abs": float(x0_delta.max().item()),
                "after_denoise": compare_caches(control["self_kv_cache"], candidate["self_kv_cache"]),
            })
            pipe._cross_attn_initialized = control_cross_init
            commit_clean_kv(pipe, control, generated_control, device, lambda: sync(device))
            control_cross_init = True
            pipe._cross_attn_initialized = candidate_cross_init
            commit_clean_kv(pipe, candidate, generated_candidate, device, lambda: sync(device))
            candidate_cross_init = True
            common["after_clean"] = compare_caches(control["self_kv_cache"], candidate["self_kv_cache"])
            common["control_after"] = cache_summary(control["self_kv_cache"])
            common["candidate_after"] = cache_summary(candidate["self_kv_cache"])
            rows.append(common)

    final = rows[-1]
    payload = {
        "status": "success",
        "configuration": {
            "chunks": len(rows),
            "local_attn_size": args.local_attn_size,
            "sink_size": args.sink_size,
            "tokens_per_frame": int(base["frame_seqlen"]),
            "capacity_tokens": int(base["kv_size"]),
        },
        "all_x0_exact": all(row["x0_exact"] for row in rows),
        "all_x0_finite": all(row["x0_finite_control"] and row["x0_finite_candidate"] for row in rows),
        "all_denoise_caches_equal": all(row["after_denoise"]["k_exact"] and row["after_denoise"]["v_exact"] for row in rows),
        "all_clean_caches_equal": all(row["after_clean"]["k_exact"] and row["after_clean"]["v_exact"] for row in rows),
        "all_host_mirrors_equal": all(row["after_clean"]["host_mirrors_equal"] for row in rows),
        "rows": rows,
        "final": final,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "chunks": len(rows),
        "all_x0_exact": payload["all_x0_exact"],
        "all_denoise_caches_equal": payload["all_denoise_caches_equal"],
        "all_clean_caches_equal": payload["all_clean_caches_equal"],
        "all_host_mirrors_equal": payload["all_host_mirrors_equal"],
        "final_cache": final["candidate_after"],
        "output": str(path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
