#!/usr/bin/env python3
"""Build the compact C4 evidence package from the frozen-point captures.

Inputs: <raw>/waterfall-run/benchmark.json (browser waterfall run) and
<raw>/detailed-profile/live_metrics.json (intrusive action-13 profile).
Outputs: waterfall.json, profile-summary.json, ranking.json, README.md
under the branch docs/artifacts/c2.../c4-profile-20260912/ directory.
Mirrors scripts/build_phase0_evidence.py field derivations.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def rolled(record: dict) -> bool:
    cache = record["cache"]
    return int(cache["global_end_index"]) > int(cache["capacity_tokens"])


def summary(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bench_files = sorted((raw / "waterfall-run").glob("benchmark*.json"))
    if not bench_files:
        raise SystemExit("no benchmark files in waterfall-run")
    rows = [r for f in bench_files for r in load(f)["records"] if rolled(r)]
    run_label = f"{len(bench_files)} run(s): " + ", ".join(f.name for f in bench_files)
    if not rows:
        raise SystemExit("no rolled actions in waterfall run")
    med = {
        "action_to_base_rgb_ms": summary([float(r["derived"]["action_to_base_rgb_ms"]) for r in rows]),
        "action_to_next_ready_ms": summary([float(r["derived"]["action_to_next_ready_ms"]) for r in rows]),
        "denoise_ms": summary([float(r["denoise_ms"]) for r in rows]),
        "transformer_ms": summary([float(r["transformer_ms"]) for r in rows]),
        "clean_kv_ms": summary([float(r["clean_kv_ms"]) for r in rows]),
    }
    rep = min(
        rows,
        key=lambda r: abs(r["derived"]["action_to_base_rgb_ms"] - med["action_to_base_rgb_ms"]["median"])
        + abs(r["derived"]["action_to_next_ready_ms"] - med["action_to_next_ready_ms"]["median"]),
    )
    forwards = rep["forward_records"]
    denoise = [dict(r) for r in forwards if r["kind"] == "denoise"]
    clean = next(dict(r) for r in forwards if r["kind"] == "cache_update")
    waterfall = {
        "selection": {
            "action_id": rep["action_id"],
            "rule": "rolled action nearest base-RGB and next-ready medians",
            "source": run_label,
        },
        "medians_ms": med,
        "rolled_actions": len(rows),
        "gpu_synchronized_forwards_ms": {
            f"denoise_{r['index'] + 1}_t{int(r['timestep'])}": r["elapsed_ms"] for r in denoise
        }
        | {"clean_kv_t0": clean["elapsed_ms"]},
        "finite": rep["finite"],
    }

    detailed = load(raw / "detailed-profile/live_metrics.json")
    profiled = next(row for row in detailed["actions"] if row["chunk_id"] == 13)
    profile = profiled["profile"]
    phases = {}
    for phase in ("denoise", "clean"):
        module = profile[f"{phase}_module_profile"]
        attention = profile[f"{phase}_attention_profile"]
        classes: dict[str, float] = defaultdict(float)
        for row in module["operators"]:
            classes[row["class"]] += float(row["exclusive_ms"])
        groups = {row["kind"]: row for row in attention["groups"]}
        phases[phase] = {
            "intrusive_wall_ms": profiled["transformer_ms"] if phase == "denoise" else profiled["clean_kv_ms"],
            "module_exclusive_total_ms": module["exclusive_total_ms"],
            "module_class_exclusive_ms": dict(sorted(classes.items(), key=lambda item: -item[1])),
            "self_sdpa_ms": groups["self"]["sdpa_ms"],
            "cross_sdpa_ms": groups["cross"]["sdpa_ms"],
        }
    denoise_wall = phases["denoise"]["intrusive_wall_ms"]
    clean_wall = phases["clean"]["intrusive_wall_ms"]
    denoise_self = phases["denoise"]["self_sdpa_ms"]
    clean_self = phases["clean"]["self_sdpa_ms"]
    denoise_non_sdpa = denoise_wall - denoise_self
    clean_non_sdpa = clean_wall - clean_self
    areas = [
        ("denoise self-SDPA (rolled rectangular)", denoise_self),
        ("denoise non-SDPA (Linears, norms, blocks)", denoise_non_sdpa),
        ("clean-KV non-SDPA recomputation", clean_non_sdpa),
        ("clean-KV self-SDPA", clean_self),
    ]
    areas.sort(key=lambda item: -item[1])
    ranking = {
        "rule": "intrusive module-exclusive attribution on rolled chunk 13; raw-ms order only, not optimizability",
        "hotspots": [{"rank": i + 1, "area": name, "ms": ms} for i, (name, ms) in enumerate(areas)],
        "note": "Raw ms does not equal opportunity (denoise Linears are GEMM/TunableOp-covered). C5 research decides "
                "what is actionable; PF1/PF2 are now active inputs since C3/C4 confirm the hotspots they describe.",
    }
    (out / "waterfall.json").write_text(json.dumps(waterfall, indent=2) + "\n")
    (out / "profile-summary.json").write_text(json.dumps({
        "warning": "Intrusive CUDA-event/module-hook run; use only for attribution, not acceptance latency.",
        "profiled_chunk": 13,
        "rolled": profiled["occupancy"]["rolled"],
        "phases": phases,
    }, indent=2) + "\n")
    (out / "ranking.json").write_text(json.dumps(ranking, indent=2) + "\n")
    print(json.dumps({
        "rolled_actions": len(rows),
        "base_rgb_median": med["action_to_base_rgb_ms"]["median"],
        "next_ready_median": med["action_to_next_ready_ms"]["median"],
        "denoise_self_sdpa_ms": denoise_self,
        "clean_non_sdpa_ms": clean_non_sdpa,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
