#!/usr/bin/env python3
"""Small keyboard-driven viewer for the accepted persistent TAEHV path.

This deliberately reuses the experiment harness' generation and cache code.
The only new behavior is choosing one framewise camera delta per keypress and
showing the first TAEHV RGB frame in a Tk/Pillow window.  It is a viewer, not a
new sampler or an alternative world-state implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageTk
import tkinter as tk

from run_experiment import (
    cuda_memory,
    current_rss_bytes,
    host_memory,
    normalize_video,
    write_video,
)
from run_interactive import (
    DecoderProfiler,
    DitAttentionProbe,
    StreamingTAEHVDecoder,
    cache_positions,
    commit_clean_kv,
    generate_chunk,
    prepare_session,
    sync,
)
from wan.configs import WAN_CONFIGS
from wan.image2video import WanI2VCausal
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings


# The pure-helper startup wrapper injects this shared dictionary before
# calling main(). Values are milliseconds from the wrapper process start.
# Ordinary runs leave it unset and retain the existing report shape.
STARTUP_TIMESTAMPS: dict[str, float] | None = None
STARTUP_PROCESS_START: float | None = None


def startup_mark(name: str) -> None:
    if STARTUP_TIMESTAMPS is None or STARTUP_PROCESS_START is None:
        return
    STARTUP_TIMESTAMPS[name] = (time.perf_counter() - STARTUP_PROCESS_START) * 1000.0


KEY_ACTIONS = {
    ord("w"): ("forward", 0.0, 0.0, 0.10),
    ord("s"): ("backward", 0.0, 0.0, -0.10),
    ord("a"): ("strafe_left", -0.10, 0.0, 0.0),
    ord("d"): ("strafe_right", 0.10, 0.0, 0.0),
    ord("j"): ("turn_left", 0.0, 0.06, 0.0),
    ord("l"): ("turn_right", 0.0, -0.06, 0.0),
    ord("i"): ("look_up", 0.0, 0.0, 0.0),
    ord("k"): ("look_down", 0.0, 0.0, 0.0),
    ord(" "): ("no_op", 0.0, 0.0, 0.0),
}


def normalize_input_key(key: str) -> str:
    """Map Tk keysyms to the canonical one-character action keys."""
    aliases = {
        "up": "i",
        "down": "k",
        "left": "j",
        "right": "l",
        "space": " ",
        "esc": "escape",
    }
    return aliases.get(str(key).lower(), str(key).lower())


class PendingInputState:
    """Bounded input policy independent of Tk and model execution.

    Tk may deliver several buffered key events at the next ``root.update``.
    While an action is busy, retain only the newest valid action key. Quit is
    sticky and always wins over ordinary movement. This class deliberately
    contains no model or threading behavior so its policy can be tested alone.
    """

    def __init__(self, time_origin: float) -> None:
        self.time_origin = time_origin
        self.busy = False
        self.current_action: str | None = None
        self.pending_key: str | None = None
        self.pending_record: dict[str, object] | None = None
        self.waiting_key: str | None = None
        self.waiting_record: dict[str, object] | None = None
        self.quit_requested = False
        self._active_events: list[dict[str, object]] = []
        self._selection: dict[str, object] | None = None
        self._generation_start_ms: float | None = None

    def now_ms(self) -> float:
        return (time.perf_counter() - self.time_origin) * 1000.0

    def request_quit(self) -> None:
        self.quit_requested = True

    def begin_action(self, action: str) -> None:
        self.busy = True
        self.current_action = action
        self._active_events = []
        self._generation_start_ms = self.now_ms()
        # A key delivered in the small interval between the previous action
        # and this call belongs to the action after the one now in flight.
        if self.waiting_key is not None:
            key = self.waiting_key
            record = dict(self.waiting_record or {"key": key})
            record["stored_ms"] = self.now_ms()
            record["replaced_pending"] = self.pending_key is not None
            self.pending_key = key
            self.pending_record = record
            self._active_events.append(dict(record))
            self.waiting_key = None
            self.waiting_record = None

    def finish_action(self) -> dict[str, object]:
        report = self.action_report()
        self.busy = False
        self.current_action = None
        self._active_events = []
        self._generation_start_ms = None
        return report

    def observe(self, raw_key: str) -> str | None:
        key = normalize_input_key(raw_key)
        if key in ("q", "escape"):
            self.request_quit()
            return "escape"
        if self.quit_requested:
            return None
        if len(key) != 1 or ord(key) not in KEY_ACTIONS:
            # Invalid input must not erase either waiting or pending input.
            return None

        observed_ms = self.now_ms()
        if self.busy:
            replaced = self.pending_key is not None
            record: dict[str, object] = {
                "key": key,
                "observed_ms": observed_ms,
                "stored_ms": observed_ms,
                "replaced_pending": replaced,
            }
            self.pending_key = key
            self.pending_record = record
            self._active_events.append(dict(record))
        else:
            self.waiting_key = key
            self.waiting_record = {
                "key": key,
                "observed_ms": observed_ms,
            }
        return key

    def take_waiting(self) -> str | None:
        key = self.waiting_key
        if key is None:
            return None
        record = dict(self.waiting_record or {"key": key})
        record["selected_ms"] = self.now_ms()
        self._selection = record
        self.waiting_key = None
        self.waiting_record = None
        return key

    def take_pending(self) -> str | None:
        key = self.pending_key
        if key is None:
            return None
        record = dict(self.pending_record or {"key": key})
        record["selected_ms"] = self.now_ms()
        self._selection = record
        self.pending_key = None
        self.pending_record = None
        return key

    def action_report(self) -> dict[str, object]:
        pending = dict(self.pending_record) if self.pending_record is not None else None
        selected = dict(self._selection) if self._selection is not None else None
        return {
            "current_action": self.current_action,
            "generation_start_ms": self._generation_start_ms,
            "selected_input": selected,
            "busy_input_events": [dict(item) for item in self._active_events],
            "pending_key_after_action": self.pending_key,
            "pending_record_after_action": pending,
            "quit_requested": self.quit_requested,
        }


def live_parser() -> argparse.ArgumentParser:
    from run_interactive import build_parser

    parser = build_parser()
    parser.set_defaults(
        display_decoder="taehv",
        denoise_schedule="3-drop-957",
        defer_clean_kv=True,
        size="480*832",
        max_area_pixels=264192,
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=20,
        help="maximum keypress actions after the automatic bootstrap",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=900.0,
        help="in-process safety limit; use the shell wrapper for hard timeout",
    )
    parser.add_argument(
        "--window-title",
        default="LingBot World live (Q/ESC quits)",
    )
    parser.add_argument(
        "--scripted-actions",
        default=None,
        help="comma-separated deterministic keys (for example w,w,j,l); bypasses manual key waiting",
    )
    parser.add_argument(
        "--profile-contexts",
        default="",
        help="comma-separated chunk IDs for detailed DiT/SDPA and clean-KV profiling",
    )
    return parser


def relative_pose(name: str, x: float, yaw: float, z: float) -> torch.Tensor:
    """Return a modest OpenCV-camera-frame incremental pose.

    The released path normalizes framewise camera translations to roughly one
    standard deviation.  These values are intentionally conservative controls
    for a first live session, not a claim about a canonical game mapping.
    """
    pitch = 0.0
    if name == "look_up":
        pitch = 0.05
    elif name == "look_down":
        pitch = -0.05
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = ry @ rx
    pose[:3, 3] = (x, 0.0, z)
    return torch.from_numpy(pose)


def make_plucker(
    pose: torch.Tensor,
    args: argparse.Namespace,
    state: dict[str, object],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    action_dir = Path(args.action_path)
    raw_k = torch.from_numpy(np.load(action_dir / "intrinsics.npy")).float()[:1]
    k = get_Ks_transformed(
        raw_k,
        height_org=480,
        width_org=832,
        height_resize=int(state["height"]),
        width_resize=int(state["width"]),
        height_final=int(state["height"]),
        width_final=int(state["width"]),
    )
    pose = pose.to(device=device, dtype=torch.float32).unsqueeze(0)
    embeddings = get_plucker_embeddings(
        pose,
        k.to(device=device),
        int(state["height"]),
        int(state["width"]),
    )
    lat_h, lat_w = int(state["lat_h"]), int(state["lat_w"])
    embeddings = rearrange(
        embeddings,
        "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
        c1=int(state["height"]) // lat_h,
        c2=int(state["width"]) // lat_w,
    )
    embeddings = rearrange(
        embeddings[None],
        "b (f h w) c -> b c f h w",
        f=1,
        h=lat_h,
        w=lat_w,
    )
    return embeddings.to(device=device, dtype=dtype)


class LiveViewer:
    def __init__(self, title: str, time_origin: float) -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.configure(background="black")
        self.image_label = tk.Label(self.root, background="black")
        self.image_label.pack(fill="both", expand=True)
        self.status_label = tk.Label(
            self.root,
            text="Loading...",
            anchor="w",
            justify="left",
            background="black",
            foreground="white",
            font=("TkFixedFont", 11),
        )
        self.status_label.pack(fill="x")
        self.input_state = PendingInputState(time_origin)
        self.closed = False
        self.photo: ImageTk.PhotoImage | None = None
        self.root.bind("<KeyPress>", self.on_key)
        self.root.focus_force()
        self.update()

    def on_key(self, event) -> None:
        self.input_state.observe(str(event.keysym))

    def close(self) -> None:
        self.closed = True
        self.input_state.request_quit()

    def update(self) -> None:
        if self.closed:
            return
        self.root.update_idletasks()
        self.root.update()

    def show(self, frame: torch.Tensor, status: str) -> str | None:
        array = frame_rgb(frame)
        image = Image.fromarray(array, mode="RGB")
        image.thumbnail((1100, 700), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)
        self.status_label.configure(text=status)
        self.update()
        if self.input_state.quit_requested:
            return "escape"
        return None

    def wait_for_key(self) -> str | None:
        self.input_state.waiting_key = None
        self.input_state.waiting_record = None
        while not self.closed and not self.input_state.quit_requested and self.input_state.waiting_key is None:
            self.update()
            time.sleep(0.01)
        if self.closed or self.input_state.quit_requested:
            return "escape"
        return self.input_state.take_waiting()

    def begin_action(self, action: str) -> None:
        self.input_state.begin_action(action)

    def finish_action(self) -> dict[str, object]:
        return self.input_state.finish_action()

    def take_pending_action(self) -> str | None:
        return self.input_state.take_pending()


def frame_rgb(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().float().clamp(0.0, 1.0).cpu()
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] == 3:
        array = array.permute(1, 2, 0)
    array = (array.numpy() * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(array)


def show_frame(viewer: LiveViewer, frame: torch.Tensor, status: str) -> str | None:
    return viewer.show(
        frame,
        f"{status}\nWASD move | J/L turn | I/K look | SPACE no-op | Q/ESC quit | Ctrl-C emergency",
    )


def action_from_key(key: str) -> tuple[str, torch.Tensor] | None:
    key = normalize_input_key(key)
    if len(key) != 1 or ord(key) not in KEY_ACTIONS:
        return "ignored", torch.eye(4)
    name, x, yaw, z = KEY_ACTIONS[ord(key)]
    return name, relative_pose(name, x, yaw, z)


def choose_action(viewer: LiveViewer) -> tuple[str, torch.Tensor] | None:
    key = viewer.take_pending_action()
    if key is None:
        key = viewer.wait_for_key()
    if key in (None, "q", "escape"):
        return None
    return action_from_key(key)


def parse_scripted_actions(spec: str | None) -> list[str] | None:
    if spec is None:
        return None
    actions = [item.strip().lower() for item in spec.split(",") if item.strip()]
    if not actions:
        raise SystemExit("--scripted-actions must contain at least one key")
    valid = set("wsadjlik ") | {"q", "escape", "esc"}
    for key in actions:
        if key not in valid:
            raise SystemExit(f"unsupported scripted action {key!r}; use movement/look keys or q")
    return actions


def parse_profile_contexts(spec: str) -> set[int]:
    if not spec.strip():
        return set()
    try:
        contexts = {int(item.strip()) for item in spec.split(",") if item.strip()}
    except ValueError as exc:
        raise SystemExit("--profile-contexts must be comma-separated non-negative chunk IDs") from exc
    if any(item < 0 for item in contexts):
        raise SystemExit("--profile-contexts must contain non-negative chunk IDs")
    return contexts


def occupancy_record(
    chunk_id: int,
    cache_before: dict[str, int],
    cache_after: dict[str, int],
    frame_seqlen: int,
    sink_size: int,
) -> dict[str, object]:
    capacity = int(cache_after["cache_capacity_tokens"])
    before = int(cache_before["local_end_index"])
    after = int(cache_after["local_end_index"])
    if before <= frame_seqlen:
        regime = "early"
    elif before < capacity and after >= capacity:
        regime = "full"
    elif before >= capacity:
        regime = "rolled"
    else:
        regime = "mid"
    current = int(frame_seqlen)
    sink_capacity = int(sink_size) * current
    # The upstream cache is one contiguous tensor. Before the first eviction,
    # sink frames are only a logical retention policy; after rollover the
    # first sink_capacity tokens are physically retained at the front.
    rolled = int(cache_after["global_end_index"]) > capacity
    retained_sink = min(sink_capacity, max(0, after - current)) if rolled else 0
    recent = max(0, after - current - retained_sink)
    return {
        "regime": regime,
        "chunk_id": int(chunk_id),
        "frame_seqlen_tokens": current,
        "cache_capacity_tokens": capacity,
        "global_k_tokens_before": int(cache_before["global_end_index"]),
        "global_k_tokens_after": int(cache_after["global_end_index"]),
        "local_k_tokens_before": before,
        "local_k_tokens_after": after,
        "attention_k_tokens_after": after,
        "current_frame_tokens": current,
        "rolled": rolled,
        "sink_tokens_configured": sink_capacity,
        "sink_tokens_retained": retained_sink,
        "recent_history_tokens_accounting": recent,
        "sink_frames_configured": int(sink_size),
        "local_frames_after": after // current if current else None,
    }


def main() -> int:
    startup_mark("live_main_start")
    args = live_parser().parse_args()
    if args.display_decoder != "taehv":
        raise SystemExit("run_live.py is intentionally limited to the TAEHV presentation path")
    if args.overlap:
        raise SystemExit("TAEHV live viewer is serial-only")
    if args.max_actions < 0:
        raise SystemExit("--max-actions must be non-negative")
    scripted_actions = parse_scripted_actions(args.scripted_actions)
    profile_contexts = parse_profile_contexts(args.profile_contexts)

    # The session preparation uses the existing action path for the initial
    # image-conditioning layout and allocates enough deterministic noise/state
    # for the bounded viewer session.  Later camera embeddings come from keys.
    # The live action count, rather than the batch runner's historical 81-frame
    # default, determines the requested session length.  prepare_session still
    # caps it at the available pose-path length (67 user actions here).
    args.frames = 4 * (int(args.max_actions) + 1) + 1
    if args.frames < 5:
        raise SystemExit("use at least one action; --frames must allow a bootstrap and action")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    report: dict[str, object] = {
        "status": "running",
        "mode": "keyboard_live_taehv",
        "arguments": vars(args),
        "input_policy": {
            "max_pending_actions": 1,
            "replacement": "latest_valid_command",
            "quit_priority": True,
            "generation_barrier": "exact_clean_kv_commit",
        },
        "actions": [],
    }
    if STARTUP_TIMESTAMPS is not None:
        report["startup"] = STARTUP_TIMESTAMPS
    viewer: LiveViewer | None = None
    start = time.perf_counter()
    pipe: WanI2VCausal | None = None
    decoder: StreamingTAEHVDecoder | None = None
    recorded_video_chunks: list[torch.Tensor] = []
    stopped = False
    try:
        viewer = LiveViewer(args.window_title, start)
        print("Loading LingBot and preparing the persistent session; Ctrl-C is the emergency stop.", flush=True)
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
        startup_mark("session_preparation_start")
        state = prepare_session(pipe, args, device)
        decoder = StreamingTAEHVDecoder(args.taehv_dir, device, args.taehv_weights)
        startup_mark("session_preparation_end")
        startup_mark("runtime_ready")
        print("Runtime ready; automatic bootstrap is starting.", flush=True)
        report["session_config"] = {
            key: value for key, value in state.items()
            if key in ("lat_f", "lat_h", "lat_w", "height", "width", "frame_seqlen", "kv_size", "frames", "timestep_values", "denoise_schedule")
        }
        print("Bootstrap is running. The first generated frame will appear in the window.", flush=True)

        def run_one(chunk_id: int, plucker: torch.Tensor, label: str) -> tuple[torch.Tensor, dict[str, object]]:
            # Match the accepted runner: the DiT has BF16 parameters and must
            # receive the same CUDA/HIP autocast context as run_session().
            viewer.begin_action(label)
            if chunk_id == 0:
                startup_mark("bootstrap_generation_start")
            elif chunk_id == 1:
                startup_mark("first_action_generation_start")
            cache_before_action = cache_positions(state["self_kv_cache"])
            profile_payload: dict[str, object] = {}
            dit_profiler = None
            attention_probe = None
            with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
                if chunk_id in profile_contexts:
                    # Attach only for this action. The detailed hooks/events
                    # intentionally do not define product latency numbers.
                    dit_profiler = DecoderProfiler(pipe.model)
                    attention_probe = DitAttentionProbe(len(pipe.model.blocks))
                generated = generate_chunk(
                    pipe,
                    state,
                    chunk_id,
                    state["noise_chunks"][chunk_id],
                    state["condition_chunks"][chunk_id],
                    plucker,
                    device,
                    lambda: sync(device),
                    defer_clean_kv=True,
                )
                if chunk_id == 0:
                    startup_mark("bootstrap_transformer_end")
                elif chunk_id == 1:
                    startup_mark("first_action_transformer_end")
                if dit_profiler is not None:
                    profile_payload["denoise_module_profile"] = dit_profiler.report()
                if attention_probe is not None:
                    profile_payload["denoise_attention_profile"] = attention_probe.report()
                first_pending, first_gpu_ms, _ = decoder.begin_latent(generated["x0"], device)
                first_frame = first_pending[0, 0]
                first_host_t0 = time.perf_counter()
                first_frame_host = first_frame.float().clamp(0, 1).cpu()
                first_host_ms = (time.perf_counter() - first_host_t0) * 1000.0
                first_visible_ms = (time.perf_counter() - generated["chunk_t0"]) * 1000.0
                key = show_frame(
                    viewer,
                    first_frame_host,
                    f"{label} | first RGB {first_visible_ms:.0f} ms | Ctrl-C/Q/ESC emergency exit",
                )
                if chunk_id == 0:
                    startup_mark("bootstrap_first_rgb_presented")
                elif chunk_id == 1:
                    startup_mark("first_action_first_rgb_presented")
                if key in ("q", "escape"):
                    raise KeyboardInterrupt
                first_rgb_presented_ms = viewer.input_state.now_ms()
                clean_profiler = DecoderProfiler(pipe.model) if chunk_id in profile_contexts else None
                clean_attention_probe = (
                    DitAttentionProbe(len(pipe.model.blocks))
                    if chunk_id in profile_contexts else None
                )
                clean_kv_start_ms = viewer.input_state.now_ms()
                commit_clean_kv(pipe, state, generated, device, lambda: sync(device))
                clean_kv_end_ms = viewer.input_state.now_ms()
                if clean_profiler is not None:
                    profile_payload["clean_module_profile"] = clean_profiler.report()
                if clean_attention_probe is not None:
                    profile_payload["clean_attention_profile"] = clean_attention_probe.report()
                decoded, remaining_gpu_ms = decoder.drain_latent(first_pending, device)
                # Keep the newest decoded frame in the viewer while waiting for
                # the next key.  The first frame was already displayed at the
                # latency boundary; no video is serialized by this viewer.
                key = show_frame(
                    viewer,
                    decoded[:, -1],
                    f"{label} | next action ready {((time.perf_counter() - generated['chunk_t0']) * 1000.0):.0f} ms | Q/ESC quit",
                )
                if key in ("q", "escape"):
                    raise KeyboardInterrupt
                next_action_permitted_ms = viewer.input_state.now_ms()
                input_control = viewer.finish_action()
                input_control.update({
                    "first_rgb_presented_ms": first_rgb_presented_ms,
                    "clean_kv_start_ms": clean_kv_start_ms,
                    "clean_kv_end_ms": clean_kv_end_ms,
                    "next_action_permitted_ms": next_action_permitted_ms,
                })
                row = {
                    "chunk_id": chunk_id,
                    "action": label,
                    "transformer_ms": generated["transformer_ms"],
                    "action_prepare_ms": generated["action_prepare_ms"],
                    "latent_postprocess_ms": generated["latent_postprocess_ms"],
                    "forward_records": generated["forward_records"],
                    "denoise_forward_count": len(state["timesteps"]),
                    "taehv_first_rgb_gpu_ms": first_gpu_ms,
                    "taehv_remaining_rgb_gpu_ms": remaining_gpu_ms,
                    "taehv_all_rgb_gpu_ms": first_gpu_ms + remaining_gpu_ms,
                    "first_visible_ms": first_visible_ms,
                    "first_host_copy_ms": first_host_ms,
                    "clean_kv_ms": generated["clean_kv_ms"],
                    "next_action_ready_ms": (time.perf_counter() - generated["chunk_t0"]) * 1000.0,
                    "cache": cache_positions(state["self_kv_cache"]),
                    "output_finite": bool(torch.isfinite(decoded).all()),
                    "latent_finite": bool(torch.isfinite(generated["x0"]).all()),
                    "memory": cuda_memory(device),
                    "rss_bytes": current_rss_bytes(),
                    "host_memory": host_memory(),
                    "input_control": input_control,
                }
                cache_after_action = cache_positions(state["self_kv_cache"])
                row["occupancy"] = occupancy_record(
                    chunk_id,
                    cache_before_action,
                    cache_after_action,
                    int(state["frame_seqlen"]),
                    int(args.sink_size),
                )
                if profile_payload:
                    row["profile"] = profile_payload
                if args.save_video:
                    # TAEHV emits RGB in [0, 1], while the shared MP4 helper
                    # consumes the repository's canonical [-1, 1] convention.
                    # Copy only after first presentation and after recording
                    # the latency row, so capture cannot delay first RGB.
                    recorded_video_chunks.append(
                        decoded.detach().float().clamp(0.0, 1.0).mul(2.0).sub(1.0).cpu()
                    )
            report["actions"].append(row)
            return decoded[:, -1], row

        # Bootstrap uses the existing identity camera conditioning.  It is not
        # counted as a user action.
        run_one(0, state["plucker_chunks"][0], "bootstrap")
        startup_mark("bootstrap_complete")
        print("Ready. Press W/A/S/D, J/L, I/K, SPACE; Q or ESC quits. Ctrl-C is the hard escape.", flush=True)

        available = min(args.max_actions, len(state["noise_chunks"]) - 1)
        for action_index in range(available):
            if time.perf_counter() - start >= args.max_seconds:
                print("Reached --max-seconds safety limit; stopping cleanly.", flush=True)
                stopped = True
                break
            if scripted_actions is None:
                selected = choose_action(viewer)
            else:
                if action_index >= len(scripted_actions):
                    stopped = True
                    break
                scripted_key = scripted_actions[action_index]
                if scripted_key in ("q", "esc", "escape"):
                    stopped = True
                    break
                selected = action_from_key(scripted_key)
            if selected is None:
                stopped = True
                break
            label, pose = selected
            if label == "ignored":
                continue
            chunk_id = action_index + 1
            plucker = make_plucker(pose, args, state, device, pipe.param_dtype)
            run_one(chunk_id, plucker, label)
            print(json.dumps(report["actions"][-1], default=str), flush=True)
        report["status"] = "stopped" if stopped else "success"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        print("\nLive session interrupted; cleaning up without starting another action.", flush=True)
    except Exception as exc:
        report["status"] = "failure"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if viewer is not None and viewer.input_state.busy:
            viewer.finish_action()
        if decoder is not None:
            decoder.clear()
        if viewer is not None:
            try:
                viewer.root.destroy()
            except tk.TclError:
                pass
        if args.save_video and recorded_video_chunks:
            try:
                captured = normalize_video(torch.cat(recorded_video_chunks, dim=1))
                video_path = root / "live.mp4"
                write_video(captured, video_path)
                report["video_path"] = str(video_path)
                report["video_frames"] = int(captured.shape[0])
            except Exception as exc:
                report["video_error"] = f"{type(exc).__name__}: {exc}"
        report["elapsed_seconds"] = time.perf_counter() - start
        (root / "live_metrics.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({
        "status": report["status"],
        "actions": len(report["actions"]),
        "metrics": str(root / "live_metrics.json"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
