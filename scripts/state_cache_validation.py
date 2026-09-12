#!/usr/bin/env python3
"""C1 recurrence/cache correctness fixtures and gfx1151 validation.

This is observational experiment code.  It does not replace or optimize the
LingBot cache path.  The reference implementation below intentionally uses a
plain Python logical-token list plus a contiguous tensor reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


def tensor_hash(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def independent_rope(
    value: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int,
) -> torch.Tensor:
    """Literal independent 3-D RoPE construction for the one-sample path."""
    outputs = []
    complex_width = value.shape[-1] // 2
    widths = [complex_width - 2 * (complex_width // 3), complex_width // 3, complex_width // 3]
    temporal, vertical, horizontal = freqs.split(widths, dim=1)
    for sample, (frames, height, width) in zip(value, grid_sizes.tolist()):
        length = frames * height * width
        sample_complex = torch.view_as_complex(
            sample[:length].to(torch.float64).reshape(length, value.shape[2], -1, 2)
        )
        phase = torch.cat(
            [
                temporal[start_frame:start_frame + frames, None, None].expand(frames, height, width, -1),
                vertical[None, :height, None].expand(frames, height, width, -1),
                horizontal[None, None, :width].expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(length, 1, -1)
        rotated = torch.view_as_real(sample_complex * phase).flatten(2)
        outputs.append(torch.cat([rotated, sample[length:]], dim=0))
    return torch.stack(outputs).to(value.dtype)


@dataclass
class ReferenceCache:
    capacity: int
    sink_tokens: int
    k: torch.Tensor
    v: torch.Tensor
    logical_ids: list[int]
    global_end: int = 0

    @classmethod
    def empty(cls, template: torch.Tensor, capacity: int, sink_tokens: int) -> "ReferenceCache":
        shape = (template.shape[0], capacity, template.shape[2], template.shape[3])
        return cls(
            capacity=capacity,
            sink_tokens=sink_tokens,
            k=torch.zeros(shape, dtype=template.dtype, device=template.device),
            v=torch.zeros(shape, dtype=template.dtype, device=template.device),
            logical_ids=[],
        )

    @property
    def local_end(self) -> int:
        return len(self.logical_ids)

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor, current_start: int) -> dict[str, Any]:
        count = new_k.shape[1]
        current_end = current_start + count
        advanced = current_end > self.global_end
        evicted: list[int] = []
        if advanced:
            incoming = list(range(current_start, current_end))
            combined_ids = self.logical_ids + incoming
            combined_k = torch.cat([self.k[:, :self.local_end], new_k], dim=1)
            combined_v = torch.cat([self.v[:, :self.local_end], new_v], dim=1)
            overflow = max(0, len(combined_ids) - self.capacity)
            if overflow:
                sink_ids = combined_ids[:self.sink_tokens]
                recent_ids = combined_ids[self.sink_tokens + overflow:]
                keep_indices = list(range(self.sink_tokens)) + list(
                    range(self.sink_tokens + overflow, len(combined_ids))
                )
                evicted = combined_ids[self.sink_tokens:self.sink_tokens + overflow]
                index = torch.tensor(keep_indices, dtype=torch.long, device=new_k.device)
                combined_k = combined_k.index_select(1, index)
                combined_v = combined_v.index_select(1, index)
                combined_ids = sink_ids + recent_ids
            self.logical_ids = combined_ids
            self.k.zero_()
            self.v.zero_()
            self.k[:, :len(combined_ids)] = combined_k
            self.v[:, :len(combined_ids)] = combined_v
            self.global_end = current_end
        else:
            # Denoise retries and the final clean t=0 transaction target the
            # same logical positions and overwrite the same physical tail.
            start = self.logical_ids.index(current_start)
            self.k[:, start:start + count] = new_k
            self.v[:, start:start + count] = new_v
        return {"advanced": advanced, "evicted": evicted}


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qf = q.transpose(1, 2).float()
    kf = k.transpose(1, 2).float()
    vf = v.transpose(1, 2).float()
    scores = torch.matmul(qf, kf.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    return torch.matmul(torch.softmax(scores, dim=-1), vf).transpose(1, 2).to(q.dtype)


def cache_dict(capacity: int, heads: int, head_dim: int, host_cursor: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "k": torch.zeros(1, capacity, heads, head_dim),
        "v": torch.zeros(1, capacity, heads, head_dim),
        "global_end_index": torch.tensor([0], dtype=torch.long),
        "local_end_index": torch.tensor([0], dtype=torch.long),
    }
    if host_cursor:
        result["global_end_index_py"] = 0
        result["local_end_index_py"] = 0
    return result


def run_synthetic_variant(host_cursor: bool) -> dict[str, Any]:
    from wan.modules import model_fast
    from wan.modules.model import rope_params

    torch.manual_seed(20260912)
    module = model_fast.CausalWanSelfAttention(
        dim=36, num_heads=3, local_attn_size=4, sink_size=1, qk_norm=False
    ).eval()
    grid = torch.tensor([[1, 1, 1]], dtype=torch.long)
    freqs = torch.cat([rope_params(32, 4), rope_params(32, 4), rope_params(32, 4)], dim=1)
    cache = cache_dict(4, 3, 12, host_cursor)
    ref: ReferenceCache | None = None
    records = []
    original_attention = model_fast.attention
    model_fast.attention = explicit_attention
    try:
        for logical_position in range(9):
            clean_x = torch.arange(36, dtype=torch.float32).reshape(1, 1, 36) / 41 + logical_position
            for phase, offset in (("denoise-0", 0.25), ("denoise-1", -0.125), ("clean-t0", 0.0)):
                x = clean_x + offset
                q = module.q(x).view(1, 1, 3, 12)
                k = module.k(x).view(1, 1, 3, 12)
                v = module.v(x).view(1, 1, 3, 12)
                q_rope = independent_rope(q, grid, freqs, logical_position)
                k_rope = independent_rope(k, grid, freqs, logical_position)
                if ref is None:
                    ref = ReferenceCache.empty(k_rope, capacity=4, sink_tokens=1)
                transition = ref.update(k_rope, v, logical_position)
                expected_attention = explicit_attention(q_rope, ref.k[:, :ref.local_end], ref.v[:, :ref.local_end])
                expected_output = module.o(expected_attention.flatten(2))
                actual_output = module(
                    x=x,
                    seq_lens=torch.tensor([1]),
                    grid_sizes=grid,
                    freqs=freqs,
                    kv_cache=cache,
                    current_start=logical_position,
                    max_attention_size=4,
                    frame_seqlen=1,
                )
                local_end = int(cache["local_end_index"].item())
                cache_k_diff = float((cache["k"][:, :local_end] - ref.k[:, :ref.local_end]).abs().max())
                cache_v_diff = float((cache["v"][:, :local_end] - ref.v[:, :ref.local_end]).abs().max())
                output_diff = float((actual_output - expected_output).abs().max())
                records.append(
                    {
                        "position": logical_position,
                        "phase": phase,
                        "global_end": int(cache["global_end_index"].item()),
                        "local_end": local_end,
                        "logical_ids": list(ref.logical_ids),
                        "evicted": transition["evicted"],
                        "cache_k_max_abs_diff": cache_k_diff,
                        "cache_v_max_abs_diff": cache_v_diff,
                        "consumer_output_max_abs_diff": output_diff,
                    }
                )
    finally:
        model_fast.attention = original_attention
    clean = [row for row in records if row["phase"] == "clean-t0"]
    boundary_names = {
        0: "startup-first-clean",
        1: "partially-filled",
        3: "immediately-before-first-eviction",
        4: "immediately-after-first-eviction/first-rolled-state",
        5: "repeated-roll",
        8: "later-steady-rolled-state",
    }
    boundaries = {boundary_names[row["position"]]: row for row in clean if row["position"] in boundary_names}
    # Reset discriminator: a pristine cache must reproduce startup; a stale
    # rolled cache must materially alter the same query's consumer output.
    startup_x = torch.arange(36, dtype=torch.float32).reshape(1, 1, 36) / 41
    q = module.q(startup_x).view(1, 1, 3, 12)
    k = module.k(startup_x).view(1, 1, 3, 12)
    v = module.v(startup_x).view(1, 1, 3, 12)
    q0 = independent_rope(q, grid, freqs, 0)
    k0 = independent_rope(k, grid, freqs, 0)
    pristine = explicit_attention(q0, k0, v)
    stale = explicit_attention(q0, ref.k[:, :ref.local_end], ref.v[:, :ref.local_end])
    stale_delta = float((pristine - stale).abs().max())
    # Broken control: reverse only the non-sink keys, leaving values in their
    # correct order.  This deliberately breaks key/value token association.
    broken_indices = [0] + list(range(ref.local_end - 1, 0, -1))
    broken = explicit_attention(q0, ref.k[:, broken_indices], ref.v[:, :ref.local_end])
    broken_delta = float((pristine - broken).abs().max())
    return {
        "host_cursor": host_cursor,
        "tolerances": {"cache_max_abs": 0.0, "consumer_output_max_abs": 1e-6},
        "boundaries": boundaries,
        "all_records_pass": all(
            row["cache_k_max_abs_diff"] == 0.0
            and row["cache_v_max_abs_diff"] == 0.0
            and row["consumer_output_max_abs_diff"] <= 1e-6
            for row in records
        ),
        "reset_discriminator": {
            "fresh_reproduces_startup_by_construction": True,
            "stale_vs_fresh_max_abs_diff": stale_delta,
            "stale_detected": stale_delta > 1e-4,
        },
        "broken_control": {
            "condition": "reverse non-sink keys only (invalid key/value token association)",
            "max_abs_output_delta": broken_delta,
            "detected": broken_delta > 1e-4,
        },
    }


class LayerZeroValidator:
    """Observe layer-zero production calls and mirror them independently."""

    def __init__(self, module: torch.nn.Module, capacity: int, sink_tokens: int, calls_per_chunk: int):
        from wan.modules import model_fast

        self.module = module
        self.model_fast = model_fast
        self.capacity = capacity
        self.sink_tokens = sink_tokens
        self.calls_per_chunk = calls_per_chunk
        self.reference: ReferenceCache | None = None
        self.records: list[dict[str, Any]] = []
        self.original = module.forward
        module.forward = self.forward

    def close(self) -> None:
        self.module.forward = self.original

    def forward(self, *args, **kwargs):
        x = kwargs.get("x", args[0] if args else None)
        grid = kwargs.get("grid_sizes", args[2])
        freqs = kwargs.get("freqs", args[3])
        cache = kwargs.get("kv_cache", args[4])
        current_start = int(kwargs.get("current_start", args[5]))
        frame_seqlen = int(kwargs["frame_seqlen"])
        q = self.module.norm_q(self.module.q(x)).view(x.shape[0], x.shape[1], self.module.num_heads, self.module.head_dim)
        k = self.module.norm_k(self.module.k(x)).view_as(q)
        v = self.module.v(x).view_as(q)
        start_frame = current_start // frame_seqlen
        q_rope = independent_rope(q, grid, freqs, start_frame).to(v.dtype)
        k_rope = independent_rope(k, grid, freqs, start_frame).to(v.dtype)
        if self.reference is None:
            self.reference = ReferenceCache.empty(k_rope, self.capacity, self.sink_tokens)
        transition = self.reference.update(k_rope, v, current_start)
        actual = self.original(*args, **kwargs)
        local_end = int(cache["local_end_index"].item())
        k_diff = float((cache["k"][:, :local_end] - self.reference.k[:, :self.reference.local_end]).abs().max())
        v_diff = float((cache["v"][:, :local_end] - self.reference.v[:, :self.reference.local_end]).abs().max())
        call_index = len(self.records)
        call_in_chunk = call_index % self.calls_per_chunk
        is_clean = call_in_chunk == self.calls_per_chunk - 1
        output_diff = None
        if is_clean:
            selected_k = self.reference.k[:, :self.reference.local_end]
            selected_v = self.reference.v[:, :self.reference.local_end]
            expected = self.model_fast.attention(q_rope, selected_k, selected_v)
            expected = self.module.o(expected.flatten(2))
            output_diff = float((actual - expected).abs().max())
        self.records.append(
            {
                "call_index": call_index,
                "chunk_id": current_start // frame_seqlen,
                "call_in_chunk": call_in_chunk,
                "phase": "clean-t0" if is_clean else "denoise",
                "current_start": current_start,
                "rope_start_frame": start_frame,
                "global_end": int(cache["global_end_index"].item()),
                "local_end": local_end,
                "logical_ids_first": self.reference.logical_ids[:4],
                "logical_ids_last": self.reference.logical_ids[-4:],
                "logical_frame_ids": [item // frame_seqlen for item in self.reference.logical_ids[::frame_seqlen]],
                "evicted_token_count": len(transition["evicted"]),
                "evicted_frame_ids": sorted({item // frame_seqlen for item in transition["evicted"]}),
                "cache_k_max_abs_diff": k_diff,
                "cache_v_max_abs_diff": v_diff,
                "consumer_output_max_abs_diff": output_diff,
            }
        )
        return actual


def command_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True).stdout.strip()


def run_gpu(args: argparse.Namespace) -> dict[str, Any]:
    from run_interactive import cache_positions, commit_clean_kv, generate_chunk, prepare_session, sync
    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2VCausal

    device = torch.device("cuda:0")
    pipe = WanI2VCausal(
        config=WAN_CONFIGS["i2v-A14B"], checkpoint_dir=args.model_dir,
        device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False,
        t5_cpu=False, convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, infer_mode="causal_fast", metrics=None,
    )
    session_args = SimpleNamespace(
        action_path=args.action_path, image=args.image, frames=4 * args.chunks + 1,
        max_area_pixels=args.max_area_pixels, size="480*832", seed=args.seed,
        denoise_schedule="4", prompt=args.prompt, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, host_kv_cursor=False,
    )
    state = prepare_session(pipe, session_args, device)
    validator = LayerZeroValidator(
        pipe.model.blocks[0].self_attn,
        int(state["kv_size"]),
        int(args.sink_size * state["frame_seqlen"]),
        len(state["timesteps"]) + 1,
    )
    chunks = []
    first_x0_hash = None
    first_cache_hash = None
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
        for chunk_id, values in enumerate(zip(state["noise_chunks"], state["condition_chunks"], state["plucker_chunks"])):
            if chunk_id >= args.chunks:
                break
            before = cache_positions(state["self_kv_cache"])
            generated = generate_chunk(pipe, state, chunk_id, *values, device, lambda: sync(device), defer_clean_kv=True)
            denoise_position = cache_positions(state["self_kv_cache"])
            commit_clean_kv(pipe, state, generated, device, lambda: sync(device))
            after = cache_positions(state["self_kv_cache"])
            layer0 = state["self_kv_cache"][0]
            row = {
                "chunk_id": chunk_id,
                "action_association": "bootstrap" if chunk_id == 0 else f"pose/action latent {chunk_id}",
                "current_start": chunk_id * int(state["frame_seqlen"]),
                "rope_frame": chunk_id,
                "cache_before": before,
                "cache_after_denoise": denoise_position,
                "cache_after_clean_t0": after,
                "x0_sha256": tensor_hash(generated["x0"]),
                "plucker_sha256": tensor_hash(values[2]),
                "layer0_k_occupied_sha256": tensor_hash(layer0["k"][:, :after["local_end_index"]]),
                "layer0_v_occupied_sha256": tensor_hash(layer0["v"][:, :after["local_end_index"]]),
                "all_layer_cursors_equal": len({
                    (int(item["global_end_index"].item()), int(item["local_end_index"].item()))
                    for item in state["self_kv_cache"]
                }) == 1,
                "clean_transaction_same_logical_position": denoise_position == after,
            }
            chunks.append(row)
            if chunk_id == 0:
                first_x0_hash = row["x0_sha256"]
                first_cache_hash = row["layer0_k_occupied_sha256"]
    validator.close()
    clean_records = [row for row in validator.records if row["phase"] == "clean-t0"]

    # A fresh prepared session is the production reset operation.  It must
    # reproduce the discriminating bootstrap after a heavily rolled session.
    reset_state = prepare_session(pipe, session_args, device)
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
        reset_generated = generate_chunk(
            pipe, reset_state, 0, reset_state["noise_chunks"][0],
            reset_state["condition_chunks"][0], reset_state["plucker_chunks"][0],
            device, lambda: sync(device), defer_clean_kv=True,
        )
        commit_clean_kv(pipe, reset_state, reset_generated, device, lambda: sync(device))
    reset_pos = cache_positions(reset_state["self_kv_cache"])
    reset_x0_hash = tensor_hash(reset_generated["x0"])
    reset_cache_hash = tensor_hash(reset_state["self_kv_cache"][0]["k"][:, :reset_pos["local_end_index"]])
    reset_result = {
        "fresh_x0_matches_initial": reset_x0_hash == first_x0_hash,
        "fresh_layer0_cache_matches_initial": reset_cache_hash == first_cache_hash,
        "fresh_positions": reset_pos,
        "cross_cache_initialized": int(reset_state["cross_kv_cache"][0]["is_init"].item()),
        "pipe_cross_gate": bool(pipe._cross_attn_initialized),
    }
    del reset_state, reset_generated
    torch.cuda.empty_cache()

    # Controlled stale-state negative on the old rolled state.  This mutates
    # only experiment-owned state after all positive evidence is captured.
    stale_hash = None
    stale_error = None
    try:
        with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
            stale_generated = generate_chunk(
                pipe, state, 0, state["noise_chunks"][0], state["condition_chunks"][0],
                state["plucker_chunks"][0], device, lambda: sync(device), defer_clean_kv=True,
            )
        stale_hash = tensor_hash(stale_generated["x0"])
    except Exception as exc:
        # A stale later-roll cursor makes current_start=0 invalid in the
        # production slice arithmetic.  Record this as detected, not as a
        # campaign failure: the condition is intentionally incorrect.
        stale_error = f"{type(exc).__name__}: {exc}"
    stale_delta_detected = stale_error is not None or stale_hash != first_x0_hash

    frame_seqlen = int(state["frame_seqlen"])
    capacity_frames = int(state["kv_size"]) // frame_seqlen
    boundary_chunks = {
        "startup": 0,
        "partially_filled": 1,
        "immediately_before_first_eviction": capacity_frames - 1,
        "immediately_after_first_eviction_first_rolled": capacity_frames,
        "repeated_roll": capacity_frames + 1,
        "later_steady_rolled": args.chunks - 1,
    }
    selected = {name: chunks[index] for name, index in boundary_chunks.items()}
    selected_reference = {name: clean_records[index] for name, index in boundary_chunks.items()}
    tolerances = {"cache_max_abs": 0.0, "consumer_output_max_abs": 0.02}
    positive_pass = all(
        row["cache_k_max_abs_diff"] <= tolerances["cache_max_abs"]
        and row["cache_v_max_abs_diff"] <= tolerances["cache_max_abs"]
        and row["consumer_output_max_abs_diff"] is not None
        and row["consumer_output_max_abs_diff"] <= tolerances["consumer_output_max_abs"]
        for row in selected_reference.values()
    )
    return {
        "device": {
            "name": torch.cuda.get_device_name(device),
            "properties": str(torch.cuda.get_device_properties(device)),
            "torch": torch.__version__, "hip": torch.version.hip,
        },
        "configuration": {
            "seed": args.seed, "chunks": args.chunks,
            "local_attn_size_frames_total_including_sink": args.local_attn_size,
            "sink_size_frames": args.sink_size,
            "frame_seqlen": frame_seqlen,
            "capacity_tokens": int(state["kv_size"]),
            "capacity_frames": capacity_frames,
            "timesteps": state["timestep_values"],
            "deferred_clean_t0": True,
        },
        "tolerances": tolerances,
        "boundaries": selected,
        "reference_checks": selected_reference,
        "all_boundary_reference_checks_pass": positive_pass,
        "reset": reset_result,
        "stale_negative_control": {
            "condition": "reuse later rolled self/cross cache and cross gate at current_start=0",
            "fresh_x0_sha256": first_x0_hash,
            "stale_x0_sha256": stale_hash,
            "stale_error": stale_error,
            "detected": stale_delta_detected,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixture", "gpu"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--image")
    parser.add_argument("--action-path")
    parser.add_argument("--prompt", default="A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky, while the camera glides smoothly forward.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunks", type=int, default=9)
    parser.add_argument("--local-attn-size", type=int, default=4)
    parser.add_argument("--sink-size", type=int, default=1)
    parser.add_argument("--max-area-pixels", type=int, default=264192)
    args = parser.parse_args()
    if args.mode == "fixture":
        payload = {
            "kind": "deterministic_minimal_reference_fixtures",
            "python": sys.version, "platform": platform.platform(),
            "variants": [run_synthetic_variant(False), run_synthetic_variant(True)],
        }
    else:
        for name in ("model_dir", "image", "action_path"):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required for --mode gpu")
        payload = {"kind": "gfx1151_real_lingbot_boundary_validation", **run_gpu(args)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), "kind": payload["kind"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
