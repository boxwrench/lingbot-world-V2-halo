#!/usr/bin/env python3
"""Build the blinded A5 clip package from four screen runs.

Inputs (run dirs from c2_context_screen.py):
  S1-3step S1-4step S2-3step S2-4step
For each scene: verify matched conditions (identical image/seed/actions/
geometry, differing ONLY in denoise schedule), extract mature chunks
10..21 (not startup), build a side-by-side blinded clip (ClipA/ClipB,
randomized per scene, seed recorded only in KEY.json), and copy
representative stills for close inspection.

Usage:
  build_a5_clips.py --runs /tmp/a5-clips-20260912 --out <package-dir> --seed 20260912
Requires ffmpeg on PATH. No GPU needed.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

CLIP_CHUNKS = list(range(10, 22))  # mature rollout, not startup
FPS = 8


def load_traj(run_dir: Path) -> dict:
    return json.loads((run_dir / "screen_trajectory.json").read_text())


def check_matched(ref: dict, other: dict, scene: str) -> None:
    rc, oc = ref["configuration"], other["configuration"]
    for key in ("local_attn_size", "sink_size", "geometry", "seed", "screen_actions"):
        if rc[key] != oc[key]:
            raise SystemExit(f"mismatch in {scene}: {key}: {rc[key]!r} vs {oc[key]!r}")
    if rc["denoise_schedule"] == oc["denoise_schedule"]:
        raise SystemExit(f"{scene}: schedules do not differ")
    if {c["chunk_id"] for c in ref["chunks"]} != {c["chunk_id"] for c in other["chunks"]}:
        raise SystemExit(f"{scene}: chunk sets differ")


def chunk_frames(run_dir: Path, chunk_id: int) -> list[Path]:
    frames = sorted(run_dir.glob(f"chunk{chunk_id:02d}-rgb*.png"))
    if not frames:
        raise SystemExit(f"missing frames for chunk {chunk_id} in {run_dir}")
    return frames


def build_clip(frames3: list[Path], frames4: list[Path], out_mp4: Path,
               assign: dict[str, str]) -> None:
    if len(frames3) != len(frames4):
        raise SystemExit(f"frame count mismatch: {len(frames3)} vs {len(frames4)}")
    seq = {"3-step": frames3, "4-step": frames4}
    left, right = seq[assign["left"]], seq[assign["right"]]
    work = out_mp4.parent / "work"
    ld, rd = work / "left", work / "right"
    shutil.rmtree(work, ignore_errors=True)
    ld.mkdir(parents=True)
    rd.mkdir(parents=True)
    for i, (l, r) in enumerate(zip(left, right)):
        shutil.copy(l, ld / f"f{i:03d}.png")
        shutil.copy(r, rd / f"f{i:03d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
         "-i", str(ld / "f%03d.png"), "-framerate", str(FPS),
         "-i", str(rd / "f%03d.png"),
         "-filter_complex", "[0:v][1:v]hstack=inputs=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         str(out_mp4)], check=True)
    shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()
    runs = Path(args.runs)
    out = Path(args.out)
    rng = random.Random(args.seed)
    key: dict[str, object] = {
        "key": "DO NOT OPEN until scene judgments are recorded",
        "seed": args.seed,
        "chunks": CLIP_CHUNKS,
        "fps": FPS,
    }
    for scene, img in (("scene1", "S1"), ("scene2", "S2")):
        d3 = runs / f"{img}-3step"
        d4 = runs / f"{img}-4step"
        t3, t4 = load_traj(d3), load_traj(d4)
        check_matched(t3, t4, scene)
        frames3 = [f for c in CLIP_CHUNKS for f in chunk_frames(d3, c)]
        frames4 = [f for c in CLIP_CHUNKS for f in chunk_frames(d4, c)]
        sides = ["3-step", "4-step"]
        rng.shuffle(sides)
        assign = {"left": sides[0], "right": sides[1]}
        sdir = out / scene
        sdir.mkdir(parents=True, exist_ok=True)
        build_clip(frames3, frames4, sdir / "clip.mp4", assign)
        # Representative stills for close inspection (mid + late chunk).
        for chunk_id in (CLIP_CHUNKS[len(CLIP_CHUNKS) // 2], CLIP_CHUNKS[-1]):
            for tag, run_dir in (("clip-left", d3 if assign["left"] == "3-step" else d4),
                                 ("clip-right", d3 if assign["right"] == "3-step" else d4)):
                for f in chunk_frames(run_dir, chunk_id)[:1]:
                    shutil.copy(f, sdir / f"still-chunk{chunk_id:02d}-{tag}.png")
        key[scene] = {"clip-left": assign["left"], "clip-right": assign["right"],
                      "frames": len(frames3)}
        print(f"{scene}: left={assign['left']} right={assign['right']} frames={len(frames3)}",
              flush=True)
    (out / "KEY.json").write_text(json.dumps(key, indent=2) + "\n")
    print("KEY SEALED:", out / "KEY.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
