#!/usr/bin/env python3
"""C2 screen analysis (CPU-only): compare matched-state trajectories.

Inputs: output dirs of screen runs (each with accepted_latents.pt,
screen_trajectory.json, chunkNN-rgb*.png). Compares x0 latents and RGB
frames chunk-aligned, using the A/A repeat pair as the noise-floor ruler.

Usage:
  python3 scripts/c2_screen_compare.py --ref DIR_A1 --aa DIR_A2 \
    --window DIR_B --denoise DIR_C --output DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_latents(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    latents = payload["latents"] if isinstance(payload, dict) else payload
    return latents.float()


def split_chunks(stream: torch.Tensor, count: int) -> list[torch.Tensor]:
    if stream.shape[1] == count:
        return [stream[:, i:i + 1].clone() for i in range(count)]
    raise ValueError(f"cannot split {tuple(stream.shape)} into {count} chunks")


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def load_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def rgb_pair_stats(png_a: Path, png_b: Path) -> dict[str, float]:
    a, b = load_png(png_a), load_png(png_b)
    diff = np.abs(a - b)
    return {
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff": float(np.percentile(diff, 95)),
        "fraction_over_0.05": float((diff > 0.05).mean()),
    }


def temporal_stats(pngs: list[Path]) -> dict[str, float | None]:
    if len(pngs) < 2:
        return {"mean_adjacent_abs": None, "p95_adjacent_abs": None}
    frames = [load_png(p) for p in pngs]
    deltas = [float(np.abs(frames[i] - frames[i - 1]).mean()) for i in range(1, len(frames))]
    return {"mean_adjacent_abs": float(np.mean(deltas)), "p95_adjacent_abs": float(np.percentile(deltas, 95))}


def first_pngs(run_dir: Path, chunk_id: int) -> list[Path]:
    return sorted(run_dir.glob(f"chunk{chunk_id:02d}-rgb*.png"))


def analyze_run(run_dir: Path) -> dict:
    traj = json.loads((run_dir / "screen_trajectory.json").read_text())
    latents = load_latents(run_dir / "accepted_latents.pt")
    chunks = traj["chunks"]
    first_frames = [first_pngs(run_dir, c["chunk_id"])[0] for c in chunks]
    return {
        "dir": str(run_dir),
        "configuration": traj["configuration"],
        "provenance": traj["provenance"],
        "warmup_hashes": traj["warmup_hashes"],
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "all_finite": all(c["output_finite"] for c in chunks),
        "latents": latents,
        "first_frames": first_frames,
        "temporal": temporal_stats(first_frames),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--aa", required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--denoise", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ref = analyze_run(Path(args.ref))
    aa = analyze_run(Path(args.aa))
    win = analyze_run(Path(args.window))
    den = analyze_run(Path(args.denoise))
    assert ref["chunk_ids"] == aa["chunk_ids"] == win["chunk_ids"] == den["chunk_ids"]
    n = len(ref["chunk_ids"])
    ref_c = split_chunks(ref["latents"], n)
    aa_c = split_chunks(aa["latents"], n)
    win_c = split_chunks(win["latents"], n)
    den_c = split_chunks(den["latents"], n)
    floor = [max_abs(a, b) for a, b in zip(ref_c, aa_c)]
    floor_max = max(floor)
    axes = {}
    for name, run_c, run in (("window_18_vs_12", win_c, win), ("denoise_4_vs_3", den_c, den)):
        per_chunk = [max_abs(a, b) for a, b in zip(ref_c, run_c)]
        rgb_rows = [rgb_pair_stats(a, b) for a, b in zip(ref["first_frames"], run["first_frames"])]
        axes[name] = {
            "x0_max_abs_per_chunk": per_chunk,
            "x0_max_abs_max": max(per_chunk),
            "aa_floor_max": floor_max,
            "ratio_to_floor": (max(per_chunk) / floor_max) if floor_max > 0 else None,
            "rgb_mean_abs_max": max(r["mean_abs_diff"] for r in rgb_rows),
            "rgb_rows": rgb_rows,
        }
    aa_rgb = [rgb_pair_stats(a, b) for a, b in zip(ref["first_frames"], aa["first_frames"])]
    verdict = {
        "kind": "c2_context_screen_analysis",
        "runs": {k: {"dir": v["dir"], "configuration": v["configuration"]} for k, v in
                 (("ref_A1", ref), ("repeat_A2", aa), ("window_B", win), ("denoise_C", den))},
        "gates": {
            "all_finite": all(v["all_finite"] for v in (ref, aa, win, den)),
            "chunk_alignment": True,
        },
        "aa_noise_floor": {
            "x0_max_abs_per_chunk": floor,
            "x0_max_abs_max": floor_max,
            "rgb_mean_abs_max": max(r["mean_abs_diff"] for r in aa_rgb),
        },
        "axes": axes,
        "temporal_by_run": {k: v["temporal"] for k, v in
                            (("ref_A1", ref), ("repeat_A2", aa), ("window_B", win), ("denoise_C", den))},
        "warmup_hashes": {k: v["warmup_hashes"] for k, v in
                          (("ref_A1", ref), ("repeat_A2", aa), ("window_B", win), ("denoise_C", den))},
    }
    (out / "c2_screen_analysis.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({
        "floor_x0_max": floor_max,
        "window_ratio": axes["window_18_vs_12"]["ratio_to_floor"],
        "denoise_ratio": axes["denoise_4_vs_3"]["ratio_to_floor"],
        "output": str(out / "c2_screen_analysis.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
