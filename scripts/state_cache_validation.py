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


C1R_PHASE0_ACTIONS = ("w", "w", "j", "w", "l", "l", "s", "s", "j", "w", "d", "d", "w", "l", "a")


def parse_action_keys(spec: str | None) -> list[str] | None:
    if spec is None:
        return None
    keys = [item.strip().lower() for item in spec.split(",") if item.strip()]
    valid = set("wsadjlik") | {"space"}
    invalid = [key for key in keys if key not in valid]
    if invalid:
        raise ValueError(f"unsupported action key(s): {', '.join(invalid)}")
    return [" " if key == "space" else key for key in keys]


def scenario_labels(keys: list[str]) -> list[list[str]]:
    """Label protocol coverage without claiming accumulated-pose closure."""
    seen: set[str] = set()
    labels = []
    for key in keys:
        current = []
        if key == "w":
            current.append("forward_motion")
        if key == "s":
            current.append("reversal")
        if key in ("j", "l"):
            current.append("turn")
        if key in seen:
            current.append("exact_conditioning_revisit")
        seen.add(key)
        labels.append(current)
    return labels


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
        self.broken_control: dict[str, Any] | None = None
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
            if self.broken_control is None and self.reference.global_end > self.capacity:
                # Reverse only non-sink keys while keeping values fixed.  One
                # rolled clean call is enough to prove this validator detects
                # a broken key/value association.
                indices = list(range(self.sink_tokens)) + list(
                    range(self.reference.local_end - 1, self.sink_tokens - 1, -1)
                )
                broken_k = selected_k[:, indices]
                broken = self.model_fast.attention(q_rope, broken_k, selected_v)
                broken = self.module.o(broken.flatten(2))
                delta = float((actual - broken).abs().max())
                self.broken_control = {
                    "condition": "reverse non-sink keys only while retaining value order",
                    "chunk_id": current_start // frame_seqlen,
                    "consumer_output_max_abs_delta": delta,
                    "detected": delta > 1e-4,
                }
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


def tail_hashes(cache: dict[str, Any], frame_seqlen: int) -> dict[str, str]:
    local_end = int(cache["local_end_index"].item())
    start = max(0, local_end - frame_seqlen)
    return {
        "k_sha256": tensor_hash(cache["k"][:, start:local_end]),
        "v_sha256": tensor_hash(cache["v"][:, start:local_end]),
    }


def assertion(assertion_id: str, expected: Any, observed: Any, passed: bool, kind: str) -> dict[str, Any]:
    return {
        "id": assertion_id,
        "kind": kind,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def compute_verdict(assertions: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [item for item in assertions if not item["passed"]]
    if not failed:
        classification = "PASS"
    elif any(item["kind"] == "correctness" for item in failed):
        classification = "FAIL"
    else:
        classification = "INCONCLUSIVE"
    return {
        "classification": classification,
        "all_assertions_pass": not failed,
        "failed_assertion_ids": [item["id"] for item in failed],
        "assertions": assertions,
    }


def build_pipe(args: argparse.Namespace, device: torch.device):
    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2VCausal
    return WanI2VCausal(
        config=WAN_CONFIGS["i2v-A14B"], checkpoint_dir=args.model_dir,
        device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False,
        t5_cpu=False, convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, infer_mode="causal_fast", metrics=None,
    )


def make_session_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        action_path=args.action_path, image=args.image, frames=4 * args.chunks + 1,
        max_area_pixels=args.max_area_pixels, size="480*832", seed=args.seed,
        denoise_schedule=args.denoise_schedule, prompt=args.prompt, local_attn_size=args.local_attn_size,
        sink_size=args.sink_size, host_kv_cursor=False,
    )


def freeze_value(value: Any) -> dict[str, Any]:
    if torch.is_tensor(value):
        return {
            "kind": "tensor",
            "sha256": tensor_hash(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "items": [freeze_value(item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {key: freeze_value(value[key]) for key in sorted(value)}}
    return {"kind": type(value).__name__, "sha256": None}


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.shape != right.shape:
        return None
    return float((left.detach().float().cpu() - right.detach().float().cpu()).abs().max())


def run_bootstrap_capture(pipe: Any, session_args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """One fresh prepare_session plus the exact chunk-0 path the reset check uses."""
    from run_interactive import cache_positions, commit_clean_kv, generate_chunk, prepare_session, sync
    state = prepare_session(pipe, session_args, device)
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
        generated = generate_chunk(
            pipe, state, 0, state["noise_chunks"][0],
            state["condition_chunks"][0], state["plucker_chunks"][0],
            device, lambda: sync(device), defer_clean_kv=True,
        )
        commit_clean_kv(pipe, state, generated, device, lambda: sync(device))
    position = cache_positions(state["self_kv_cache"])
    local_end = position["local_end_index"]
    layer0 = state["self_kv_cache"][0]
    tensors = {
        "noise_chunk_0": state["noise_chunks"][0],
        "condition_chunk_0": state["condition_chunks"][0],
        "plucker_chunk_0": state["plucker_chunks"][0],
        "text_context": state["context"],
        "bootstrap_x0": generated["x0"],
        "layer0_clean_k": layer0["k"][:, :local_end],
        "layer0_clean_v": layer0["v"][:, :local_end],
    }
    frozen = {name: freeze_value(value) for name, value in tensors.items()}
    kept = {
        name: value.detach().float().cpu().clone()
        for name, value in tensors.items() if torch.is_tensor(value)
    }
    timestep_values = [int(timestep) for timestep in state["timesteps"]]
    del state, generated, tensors, layer0
    torch.cuda.empty_cache()
    return {
        "position": position,
        "prompt_sha256": hashlib.sha256(session_args.prompt.encode("utf-8")).hexdigest(),
        "timestep_values": timestep_values,
        "frozen": frozen,
        "tensors": kept,
    }


def compare_captures(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    order = (
        "noise_chunk_0", "condition_chunk_0", "plucker_chunk_0", "text_context",
        "bootstrap_x0", "layer0_clean_k", "layer0_clean_v",
    )
    entries = {}
    for name in order:
        frozen_left = left["frozen"][name]
        frozen_right = right["frozen"][name]
        sha_equal = (
            frozen_left.get("sha256") is not None
            and frozen_left.get("sha256") == frozen_right.get("sha256")
        )
        diff = None
        if name in left["tensors"] and name in right["tensors"]:
            diff = max_abs_diff(left["tensors"][name], right["tensors"][name])
        entries[name] = {"sha_equal": bool(sha_equal), "max_abs_diff": diff}
    return entries


def strip_tensors(capture: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in capture.items() if key != "tensors"}


INPUT_ORDER = ("noise_chunk_0", "condition_chunk_0", "plucker_chunk_0", "text_context")


def classify_reset(ab: dict[str, Any], ac: dict[str, Any] | None) -> dict[str, Any]:
    if not all(entry["sha_equal"] for entry in ab.values()):
        return {
            "classification": "BASELINE_NONDETERMINISM_OR_STRICT_HASH",
            "detail": "Fresh bootstraps A and B differ without any rollout between them; "
                      "do not claim persistent-state leakage. Use max_abs_diff to judge tolerance.",
        }
    if ac is None:
        return {"classification": "NOT_RUN", "detail": "A==B gate failed; rollout and C were skipped."}
    if all(entry["sha_equal"] for entry in ac.values()):
        return {
            "classification": "RESET_REPRODUCES",
            "detail": "C matches A bitwise after the rollout; the earlier FAIL needs re-examination, not a leak claim.",
        }
    for name in INPUT_ORDER:
        if not ac[name]["sha_equal"]:
            return {
                "classification": "PREPARATION_SIDE_STATE",
                "detail": f"First differing input after rollout: {name}.",
                "first_differing_input": name,
            }
    if not ac["bootstrap_x0"]["sha_equal"]:
        return {
            "classification": "MODEL_RUNTIME_STATE",
            "detail": "C inputs identical to A, but bootstrap x0 differs: model/runtime persistent state.",
        }
    return {
        "classification": "CACHE_VALIDATOR_PATH",
        "detail": "x0 identical but clean K/V differ: cache or validator path issue.",
    }


def run_reset_discriminator(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda:0")
    pipe = build_pipe(args, device)
    session_args = make_session_args(args)
    capture_a = run_bootstrap_capture(pipe, session_args, device)
    capture_b = run_bootstrap_capture(pipe, session_args, device)
    comparison_ab = compare_captures(capture_a, capture_b)
    gate_pass = all(entry["sha_equal"] for entry in comparison_ab.values())
    rollout = None
    capture_c = None
    comparison_ac = None
    if gate_pass:
        rollout = run_gpu(args, pipe=pipe)
        capture_c = run_bootstrap_capture(pipe, session_args, device)
        comparison_ac = compare_captures(capture_a, capture_c)
    state_probe = prepare_session_probe(pipe, session_args, device)
    return {
        "configuration": {
            "seed": args.seed,
            "chunks": args.chunks,
            "local_attn_size_frames_total_including_sink": args.local_attn_size,
            "sink_size_frames": args.sink_size,
            "denoise_schedule": args.denoise_schedule,
            "scripted_actions": args.scripted_actions,
        },
        "provenance": {
            "base_commit": command_output(["git", "rev-parse", "HEAD"]),
            "upstream_commit": command_output(["git", "-C", ".upstream/lingbot-world-v2", "rev-parse", "HEAD"]),
            "model_dir": str(Path(args.model_dir).resolve()),
            "image": str(Path(args.image).resolve()),
            "action_path": str(Path(args.action_path).resolve()),
        },
        "capture_a": strip_tensors(capture_a),
        "capture_b": strip_tensors(capture_b),
        "comparison_ab": comparison_ab,
        "gate_pass_ab_bitwise": gate_pass,
        "rollout": rollout,
        "capture_c": strip_tensors(capture_c) if capture_c is not None else None,
        "comparison_ac": comparison_ac,
        "classification": classify_reset(comparison_ab, comparison_ac),
        "state_probe": state_probe,
    }


def prepare_session_probe(pipe: Any, session_args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from run_interactive import prepare_session
    state = prepare_session(pipe, session_args, device)
    return {
        "timestep_values": [int(timestep) for timestep in state["timesteps"]],
        "context": freeze_value(state["context"]),
        "noise_0": freeze_value(state["noise_chunks"][0]),
        "condition_0": freeze_value(state["condition_chunks"][0]),
        "plucker_0": freeze_value(state["plucker_chunks"][0]),
    }


def run_gpu(args: argparse.Namespace, pipe: Any = None) -> dict[str, Any]:
    from run_interactive import cache_positions, commit_clean_kv, generate_chunk, prepare_session, sync
    from run_live import action_from_key, make_plucker

    device = torch.device("cuda:0")
    if pipe is None:
        pipe = build_pipe(args, device)
    session_args = make_session_args(args)
    state = prepare_session(pipe, session_args, device)
    validator = LayerZeroValidator(
        pipe.model.blocks[0].self_attn,
        int(state["kv_size"]),
        int(args.sink_size * state["frame_seqlen"]),
        len(state["timesteps"]) + 1,
    )
    chunks = []
    action_keys = parse_action_keys(args.scripted_actions)
    action_labels = scenario_labels(action_keys or [])
    first_x0_hash = None
    first_cache_hash = None
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
        for chunk_id, values in enumerate(zip(state["noise_chunks"], state["condition_chunks"], state["plucker_chunks"])):
            if chunk_id >= args.chunks:
                break
            action_key = None
            action_name = "bootstrap"
            labels = ["bootstrap"]
            plucker = values[2]
            if chunk_id > 0 and action_keys is not None:
                action_key = action_keys[chunk_id - 1]
                selected = action_from_key(action_key)
                if selected is None or selected[0] == "ignored":
                    raise RuntimeError(f"invalid scripted action at chunk {chunk_id}: {action_key!r}")
                action_name, pose = selected
                labels = action_labels[chunk_id - 1]
                plucker = make_plucker(pose, args, state, device, pipe.param_dtype)
            values = (values[0], values[1], plucker)
            before = cache_positions(state["self_kv_cache"])
            generated = generate_chunk(pipe, state, chunk_id, *values, device, lambda: sync(device), defer_clean_kv=True)
            denoise_position = cache_positions(state["self_kv_cache"])
            denoise_tail = tail_hashes(state["self_kv_cache"][0], int(state["frame_seqlen"]))
            commit_clean_kv(pipe, state, generated, device, lambda: sync(device))
            after = cache_positions(state["self_kv_cache"])
            clean_tail = tail_hashes(state["self_kv_cache"][0], int(state["frame_seqlen"]))
            layer0 = state["self_kv_cache"][0]
            row = {
                "chunk_id": chunk_id,
                "action_key": action_key,
                "action_name": action_name,
                "scenario_labels": labels,
                "action_association": "bootstrap" if chunk_id == 0 else f"scripted action {chunk_id - 1}",
                "current_start": chunk_id * int(state["frame_seqlen"]),
                "rope_frame": chunk_id,
                "cache_before": before,
                "cache_after_denoise": denoise_position,
                "cache_after_clean_t0": after,
                "x0_sha256": tensor_hash(generated["x0"]),
                "plucker_sha256": tensor_hash(values[2]),
                "tail_after_denoise": denoise_tail,
                "tail_after_clean_t0": clean_tail,
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
    reset_cache_v_hash = tensor_hash(reset_state["self_kv_cache"][0]["v"][:, :reset_pos["local_end_index"]])
    first_cache_v_hash = chunks[0]["layer0_v_occupied_sha256"]
    reset_result = {
        "fresh_x0_matches_initial": reset_x0_hash == first_x0_hash,
        "fresh_layer0_cache_matches_initial": reset_cache_hash == first_cache_hash,
        "fresh_layer0_v_matches_initial": reset_cache_v_hash == first_cache_v_hash,
        "fresh_positions": reset_pos,
        "all_layer_cursors_equal": len({
            (int(item["global_end_index"].item()), int(item["local_end_index"].item()))
            for item in reset_state["self_kv_cache"]
        }) == 1,
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
        "capacity_filled": capacity_frames - 1,
        "first_eviction_first_rolled": capacity_frames,
        "repeated_roll_1": capacity_frames + 1,
        "repeated_roll_2": capacity_frames + 2,
        "repeated_roll_3": args.chunks - 1,
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
    repeated_w_hashes = {
        row["plucker_sha256"] for row in chunks if row["action_key"] == "w"
    }
    expected_timesteps = [999, 899, 702] if args.denoise_schedule == "3-drop-957" else [999, 957, 899, 702]
    calls_expected = args.chunks * (len(expected_timesteps) + 1)
    all_clean_overwrite = all(row["clean_transaction_same_logical_position"] for row in selected.values())
    reset_pass = all((
        reset_result["fresh_x0_matches_initial"],
        reset_result["fresh_layer0_cache_matches_initial"],
        reset_result["fresh_layer0_v_matches_initial"],
        reset_result["all_layer_cursors_equal"],
        reset_pos == {"global_end_index": frame_seqlen, "local_end_index": frame_seqlen},
    ))
    coverage = {label for row in chunks for label in row["scenario_labels"]}
    assertions = [
        assertion("exact_local_capacity", 12, args.local_attn_size, args.local_attn_size == 12, "configuration"),
        assertion("exact_sink_size", 6, args.sink_size, args.sink_size == 6, "configuration"),
        assertion("exact_timesteps", [999, 899, 702], state["timestep_values"], state["timestep_values"] == [999, 899, 702], "configuration"),
        assertion("exact_chunk_count", 16, len(chunks), len(chunks) == 16, "coverage"),
        assertion("exact_geometry", [384, 672], [int(state["height"]), int(state["width"])], [int(state["height"]), int(state["width"])] == [384, 672], "configuration"),
        assertion("bf16_dit", "torch.bfloat16", str(pipe.param_dtype), str(pipe.param_dtype) == "torch.bfloat16", "configuration"),
        assertion("deferred_clean_t0", True, True, True, "configuration"),
        assertion("all_64_layer0_calls_recorded", 64, len(validator.records), len(validator.records) == 64 and len(validator.records) == calls_expected, "coverage"),
        assertion("selected_boundaries_present", sorted(boundary_chunks), sorted(selected), len(selected) == len(boundary_chunks), "coverage"),
        assertion("selected_reference_checks", True, positive_pass, positive_pass, "correctness"),
        assertion("selected_clean_t0_overwrites", True, all_clean_overwrite, all_clean_overwrite, "correctness"),
        assertion("scenario_coverage", ["exact_conditioning_revisit", "forward_motion", "reversal", "turn"], sorted(coverage - {"bootstrap"}), {"exact_conditioning_revisit", "forward_motion", "reversal", "turn"}.issubset(coverage), "coverage"),
        assertion("repeated_w_exact_conditioning", 1, len(repeated_w_hashes), len(repeated_w_hashes) == 1, "coverage"),
        assertion("fresh_reset", True, reset_pass, reset_pass, "correctness"),
        assertion("stale_state_control", True, stale_delta_detected, stale_delta_detected, "discriminator"),
        assertion("broken_kv_control", True, bool(validator.broken_control and validator.broken_control["detected"]), bool(validator.broken_control and validator.broken_control["detected"]), "discriminator"),
    ]
    result = {
        "device": {
            "name": torch.cuda.get_device_name(device),
            "properties": str(torch.cuda.get_device_properties(device)),
            "torch": torch.__version__, "hip": torch.version.hip,
        },
        "configuration": {
            "seed": args.seed, "chunks": args.chunks,
            "chunk_size": 1,
            "scripted_action_keys": action_keys,
            "local_attn_size_frames_total_including_sink": args.local_attn_size,
            "sink_size_frames": args.sink_size,
            "frame_seqlen": frame_seqlen,
            "capacity_tokens": int(state["kv_size"]),
            "capacity_frames": capacity_frames,
            "timesteps": state["timestep_values"],
            "denoise_schedule": args.denoise_schedule,
            "height": int(state["height"]),
            "width": int(state["width"]),
            "dit_dtype": str(pipe.param_dtype),
            "taehv_dtype": "torch.float16 (RC1 presentation contract; decoder not invoked by this cache validator)",
            "deferred_clean_t0": True,
        },
        "provenance": {
            "base_commit": command_output(["git", "rev-parse", "HEAD"]),
            "upstream_commit": command_output(["git", "-C", ".upstream/lingbot-world-v2", "rev-parse", "HEAD"]),
            "model_dir": str(Path(args.model_dir).resolve()),
            "image": str(Path(args.image).resolve()),
            "action_path": str(Path(args.action_path).resolve()),
        },
        "tolerances": tolerances,
        "chunks": chunks,
        "validator_records": validator.records,
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
        "broken_kv_negative_control": validator.broken_control,
    }
    result["verdict"] = compute_verdict(assertions)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixture", "gpu", "reset-discriminator"), required=True)
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
    parser.add_argument("--denoise-schedule", choices=("4", "3-drop-957"), default="4")
    parser.add_argument(
        "--scripted-actions",
        help="comma-separated production run_live action keys; use 'space' for no-op",
    )
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
                parser.error(f"--{name.replace('_', '-')} is required for --mode {args.mode}")
        try:
            keys = parse_action_keys(args.scripted_actions)
        except ValueError as exc:
            parser.error(str(exc))
        if keys is not None and len(keys) != args.chunks - 1:
            parser.error("--scripted-actions must contain exactly --chunks minus one keys (bootstrap is implicit)")
        if args.mode == "gpu":
            payload = {"kind": "gfx1151_real_lingbot_boundary_validation", **run_gpu(args)}
        else:
            payload = {"kind": "gfx1151_reset_discriminator", **run_reset_discriminator(args)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), "kind": payload["kind"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
