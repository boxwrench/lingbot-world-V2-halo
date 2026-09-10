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
    def __init__(self, title: str) -> None:
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
        self.last_key: str | None = None
        self.closed = False
        self.photo: ImageTk.PhotoImage | None = None
        self.root.bind("<KeyPress>", self.on_key)
        self.root.focus_force()
        self.update()

    def on_key(self, event) -> None:
        self.last_key = str(event.keysym).lower()

    def close(self) -> None:
        self.closed = True
        self.last_key = "escape"

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
        key, self.last_key = self.last_key, None
        return key

    def wait_for_key(self) -> str | None:
        self.last_key = None
        while not self.closed and self.last_key is None:
            self.update()
            time.sleep(0.01)
        key, self.last_key = self.last_key, None
        return key


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


def choose_action(viewer: LiveViewer) -> tuple[str, torch.Tensor] | None:
    key = viewer.wait_for_key()
    if key in (None, "q", "escape"):
        return None
    aliases = {
        "up": "i",
        "down": "k",
        "left": "j",
        "right": "l",
    }
    key = aliases.get(key, key)
    if len(key) != 1 or ord(key) not in KEY_ACTIONS:
        return "ignored", torch.eye(4)
    name, x, yaw, z = KEY_ACTIONS[ord(key)]
    return name, relative_pose(name, x, yaw, z)


def main() -> int:
    args = live_parser().parse_args()
    if args.display_decoder != "taehv":
        raise SystemExit("run_live.py is intentionally limited to the TAEHV presentation path")
    if args.overlap:
        raise SystemExit("TAEHV live viewer is serial-only")
    if args.max_actions < 0:
        raise SystemExit("--max-actions must be non-negative")

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
        "actions": [],
    }
    viewer: LiveViewer | None = None
    start = time.perf_counter()
    pipe: WanI2VCausal | None = None
    decoder: StreamingTAEHVDecoder | None = None
    recorded_video_chunks: list[torch.Tensor] = []
    stopped = False
    try:
        viewer = LiveViewer(args.window_title)
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
        state = prepare_session(pipe, args, device)
        decoder = StreamingTAEHVDecoder(args.taehv_dir, device, args.taehv_weights)
        report["session_config"] = {
            key: value for key, value in state.items()
            if key in ("lat_f", "lat_h", "lat_w", "height", "width", "frame_seqlen", "kv_size", "frames", "timestep_values", "denoise_schedule")
        }
        print("Bootstrap is running. The first generated frame will appear in the window.", flush=True)

        def run_one(chunk_id: int, plucker: torch.Tensor, label: str) -> tuple[torch.Tensor, dict[str, object]]:
            # Match the accepted runner: the DiT has BF16 parameters and must
            # receive the same CUDA/HIP autocast context as run_session().
            with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
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
                if key in ("q", "escape"):
                    raise KeyboardInterrupt
                commit_clean_kv(pipe, state, generated, device, lambda: sync(device))
                decoded, remaining_gpu_ms = decoder.drain_latent(first_pending, device)
                # Keep the newest decoded frame in the viewer while waiting for
                # the next key.  The first frame was already displayed at the
                # latency boundary; no video is serialized by this viewer.
                show_frame(
                    viewer,
                    decoded[:, -1],
                    f"{label} | next action ready {((time.perf_counter() - generated['chunk_t0']) * 1000.0):.0f} ms | Q/ESC quit",
                )
                row = {
                    "chunk_id": chunk_id,
                    "action": label,
                    "transformer_ms": generated["transformer_ms"],
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
                }
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
        print("Ready. Press W/A/S/D, J/L, I/K, SPACE; Q or ESC quits. Ctrl-C is the hard escape.", flush=True)

        available = min(args.max_actions, len(state["noise_chunks"]) - 1)
        for action_index in range(available):
            if time.perf_counter() - start >= args.max_seconds:
                print("Reached --max-seconds safety limit; stopping cleanly.", flush=True)
                stopped = True
                break
            selected = choose_action(viewer)
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
