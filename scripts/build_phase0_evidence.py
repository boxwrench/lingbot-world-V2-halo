#!/usr/bin/env python3
"""Build the compact Phase 0 evidence package from ignored raw RC1 runs."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results/raw/phase0-20260912"
OUT = ROOT / "docs/artifacts/phase0-20260912"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rolled(record: dict) -> bool:
    cache = record["cache"]
    return int(cache["global_end_index"]) > int(cache["capacity_tokens"])


def compact_record(run: str, record: dict, include_forwards: bool = False) -> dict:
    value = {
        "run": run,
        "action_id": record["action_id"],
        "action": record["action"],
        "action_to_base_rgb_ms": record["derived"]["action_to_base_rgb_ms"],
        "action_to_next_ready_ms": record["derived"]["action_to_next_ready_ms"],
        "denoise_ms": record["denoise_ms"],
        "transformer_ms": record["transformer_ms"],
        "clean_kv_ms": record["clean_kv_ms"],
        "taehv_first_rgb_gpu_ms": record["tae_first_rgb_gpu_ms"],
        "taehv_remaining_rgb_gpu_ms": record["tae_remaining_rgb_gpu_ms"],
        "cache": record["cache"],
        "finite": record["finite"],
    }
    if include_forwards:
        value["server"] = record["server"]
        value["forward_records"] = record["forward_records"]
        value["action_prepare_ms"] = record["action_prepare_ms"]
        value["latent_postprocess_ms"] = record["latent_postprocess_ms"]
    return value


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    aa_runs = [load(RAW / f"aa-run-{index}/benchmark.json") for index in range(1, 4)]
    aa_rows = [
        compact_record(f"aa-run-{index}", record)
        for index, run in enumerate(aa_runs, start=1)
        for record in run["records"]
        if rolled(record)
    ]
    waterfall_run = load(RAW / "waterfall-run/benchmark.json")
    waterfall_rows = [record for record in waterfall_run["records"] if rolled(record)]
    pooled_base = statistics.median([row["action_to_base_rgb_ms"] for row in aa_rows])
    pooled_ready = statistics.median([row["action_to_next_ready_ms"] for row in aa_rows])
    representative = min(
        waterfall_rows,
        key=lambda row: abs(row["derived"]["action_to_base_rgb_ms"] - pooled_base)
        + abs(row["derived"]["action_to_next_ready_ms"] - pooled_ready),
    )

    with (OUT / "timings.jsonl").open("w") as stream:
        for row in aa_rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    metrics = {
        "protocol": {
            "independent_rollouts": 3,
            "actions_per_rollout": 15,
            "rolled_actions_per_rollout": 4,
            "pooled_rolled_actions": len(aa_rows),
            "profiling_enabled": False,
        },
        "pooled_rolled": {
            key: summary([float(row[field]) for row in aa_rows])
            for key, field in (
                ("action_to_base_rgb_ms", "action_to_base_rgb_ms"),
                ("action_to_next_ready_ms", "action_to_next_ready_ms"),
                ("denoise_ms", "denoise_ms"),
                ("transformer_ms", "transformer_ms"),
                ("clean_kv_ms", "clean_kv_ms"),
                ("taehv_first_rgb_gpu_ms", "taehv_first_rgb_gpu_ms"),
                ("taehv_remaining_rgb_gpu_ms", "taehv_remaining_rgb_gpu_ms"),
            )
        },
        "per_rollout_rolled_medians": [run["rolled_server_metrics"] for run in aa_runs],
    }
    decoder_total = [
        row["taehv_first_rgb_gpu_ms"] + row["taehv_remaining_rgb_gpu_ms"]
        for row in aa_rows
    ]
    metrics["pooled_rolled"]["taehv_total_gpu_ms"] = summary(decoder_total)
    write_json("metrics-summary.json", metrics)

    server = representative["server"]
    forwards = representative["forward_records"]
    denoise = [row for row in forwards if row["kind"] == "denoise"]
    clean = next(row for row in forwards if row["kind"] == "cache_update")
    waterfall = {
        "selection": {
            "source_run": "waterfall-run",
            "action_id": representative["action_id"],
            "rule": "rolled action nearest pooled A/A base-RGB and next-ready medians",
            "profiling_enabled": False,
        },
        "wall_clock_ms": {
            "input_to_selection": server["input_selected_ms"] - server["input_received_ms"],
            "selection_to_generation_start": server["generation_start_ms"] - server["input_selected_ms"],
            "generation_to_accepted_x0": server["accepted_x0_ms"] - server["generation_start_ms"],
            "accepted_x0_to_base_rgb": server["base_rgb_ready_ms"] - server["accepted_x0_ms"],
            "base_rgb_to_clean_kv_start": server["clean_kv_start_ms"] - server["base_rgb_ready_ms"],
            "clean_kv": server["clean_kv_complete_ms"] - server["clean_kv_start_ms"],
            "clean_kv_complete_to_next_ready": server["next_ready_ms"] - server["clean_kv_complete_ms"],
            "input_to_base_rgb": representative["derived"]["action_to_base_rgb_ms"],
            "input_to_next_ready": representative["derived"]["action_to_next_ready_ms"],
        },
        "gpu_synchronized_forwards_ms": {
            f"denoise_{row['index'] + 1}_t{int(row['timestep'])}": row["elapsed_ms"]
            for row in denoise
        }
        | {"clean_kv_t0": clean["elapsed_ms"]},
        "other_measured_ms": {
            "transformer_total": representative["transformer_ms"],
            "latent_postprocess_total": representative["latent_postprocess_ms"],
            "taehv_first_rgb_gpu": representative["tae_first_rgb_gpu_ms"],
            "taehv_remaining_rgb_gpu": representative["tae_remaining_rgb_gpu_ms"],
            "action_prepare_internal": representative["action_prepare_ms"],
        },
        "cache": representative["cache"],
        "finite": representative["finite"],
        "critical_path": {
            "before_base_rgb": [
                "input selection",
                "camera/action preparation",
                "three serialized DiT denoise forwards",
                "latent postprocess between forwards",
                "TAEHV first-RGB decode",
                "first-frame device-to-host copy",
            ],
            "after_base_rgb_before_next_ready": [
                "exact t=0 clean-KV DiT forward",
                "TAEHV remaining-frame decode",
                "remaining frame copies",
            ],
            "overlap": ["CPU JPEG encoding/presentation worker after base-frame submission"],
            "not_generation_state_critical": ["JPEG encoding", "WebSocket transport", "browser decode/presentation"],
        },
    }
    write_json("waterfall.json", waterfall)

    detailed = load(RAW / "detailed-profile/live_metrics.json")
    profiled = next(row for row in detailed["actions"] if row["chunk_id"] == 13)
    profile = profiled["profile"]
    phase_summary = {}
    for phase in ("denoise", "clean"):
        module = profile[f"{phase}_module_profile"]
        attention = profile[f"{phase}_attention_profile"]
        classes: dict[str, float] = defaultdict(float)
        for row in module["operators"]:
            classes[row["class"]] += float(row["exclusive_ms"])
        groups = {row["kind"]: row for row in attention["groups"]}
        phase_summary[phase] = {
            "intrusive_wall_ms": profiled["transformer_ms"] if phase == "denoise" else profiled["clean_kv_ms"],
            "module_exclusive_total_ms": module["exclusive_total_ms"],
            "module_class_exclusive_ms": dict(sorted(classes.items(), key=lambda item: -item[1])),
            "self_sdpa_ms": groups["self"]["sdpa_ms"],
            "cross_sdpa_ms": groups["cross"]["sdpa_ms"],
            "self_sdpa_contract": {
                "calls": groups["self"]["calls"],
                "q_shapes": groups["self"]["sdpa_q_shapes"],
                "k_shapes": groups["self"]["sdpa_k_shapes"],
                "k_strides": groups["self"]["sdpa_k_strides"],
                "dtypes": groups["self"]["dtypes"],
                "mask_modes": groups["self"]["mask_modes"],
            },
        }
    write_json("profile-summary.json", {
        "warning": "Intrusive CUDA-event/module-hook run; use only for attribution, not acceptance latency.",
        "profiled_chunk": 13,
        "rolled": profiled["occupancy"]["rolled"],
        "phases": phase_summary,
        "current_exact_shape_dispatch_probe": {
            "operator": "aten::_scaled_dot_product_flash_attention",
            "kernel": "attn_fwd.kd",
            "device": "AMD Radeon 8060S Graphics",
            "gfx_target": "gfx1151",
            "finite": True,
        },
    })

    session = detailed["session_config"]
    write_json("runtime-config.json", {
        "source": "loaded runtime, unprofiled action ledger, and rolled action-13 probe",
        "width": session["width"],
        "height": session["height"],
        "session_latent_dimensions": [session["lat_f"], session["lat_h"], session["lat_w"]],
        "accepted_chunk_latent_shape": [1, 16, 1, session["lat_h"], session["lat_w"]],
        "chunk_size": 1,
        "tokens_per_frame": session["frame_seqlen"],
        "local_attention_frames": detailed["arguments"]["local_attn_size"],
        "sink_frames": detailed["arguments"]["sink_size"],
        "kv_capacity_tokens": session["kv_size"],
        "local_capacity_includes_sink_tokens": True,
        "sink_tokens": detailed["arguments"]["sink_size"] * session["frame_seqlen"],
        "rolled_recent_history_tokens_excluding_current": 5040,
        "denoise_timesteps": session["timestep_values"],
        "dit_blocks": detailed["pure_compiler"]["total_model_blocks"],
        "attention_heads": 12,
        "head_dimension": 128,
        "dit_precision": "torch.bfloat16 (observed Q/K/V and loaded runtime autocast)",
        "decoder": "TAEHV taew2_1",
        "decoder_precision": "torch.float16",
        "decoder_weight_sha256": "d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e",
        "attention_backend": "PyTorch flash SDPA / aten::_scaled_dot_product_flash_attention / attn_fwd.kd",
        "serial_execution": True,
        "clean_kv": "deferred exact t=0 full DiT forward before next-ready",
        "tunableop": detailed["pure_compiler"]["tunableop"],
        "pure_helper_compilation": {
            "compiled_blocks": detailed["pure_compiler"]["requested_block_count"],
            "compiled_helpers": detailed["pure_compiler"]["compiled_helpers"],
            "prewarmed": detailed["pure_compiler"]["prewarm_requested"],
        },
    })

    write_json("environment.json", {
        "gate": "PASS",
        "repo": {
            "path": str(ROOT),
            "git_metadata": ".git-experiment/.git",
            "branch": "main",
            "head": "5f858c53d88f0462de7ff44aae8d3ea70e0b9be8",
            "initial_dirty_paths": ["scripts/run_window_sweep.sh"],
            "initial_dirty_diff_sha256": "2f8f41a04b97eaf5db3050af6cd2fe50f8bb10866dd87baddc0bbc4fed1af521",
        },
        "upstream": {
            "revision": "45fa40673607c9acba6cf96a1f9396c95bcef25f",
            "patched_paths": ["wan/image2video.py", "wan/modules/model_fast.py"],
            "working_diff_sha256": "14222b0ddb9d13bd86e8e1a9d5b7743a7c4f418b66092ef247c27353e6ce668e",
        },
        "model": {
            "repository": "robbyant/lingbot-world-v2-1.3b-causal-fast",
            "revision": "7e36a5f919f86cb4255cc9bfc30adb44963fbde1",
            "index_sha256": "b0409d663b57810af19443c9e8d8dde43320193c8dbb979d86f6618102922e42",
        },
        "hardware": {
            "cpu_apu": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
            "gpu_count": 1,
            "gpu": "AMD Radeon 8060S Graphics",
            "gfx_target": "gfx1151",
            "device_memory_bytes": 120259084288,
            "system_ram_bytes": 130459455488,
            "bf16_supported": True,
            "bf16_matmul_finite": True,
            "performance_level": "auto",
        },
        "software": {
            "kernel": "6.17.0-35-generic",
            "torch": "2.13.0+rocm7.15.0a20260728",
            "torch_hip": "7.15.0",
            "system_hipconfig": "7.2.53211-671d39a71e",
            "system_rocm_libraries": "7.2.2.70202-86~24.04",
            "triton": "3.8.0+git4cff872c.rocm7.15.0a20260728",
        },
        "visibility": {
            "PYTORCH_ROCM_ARCH": "gfx1151",
            "HIP_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        "memory": {
            "uma_gtt_total_bytes": 120259084288,
            "rocm_smi_vram_aperture_note": "APU reports unified GTT as the usable device pool",
            "kernel_cmdline": "amd_iommu=on iommu.passthrough=0",
        },
        "raw_gate_path": "results/raw/phase0-20260912/environment-gate",
    })

    raw_files = [
        RAW / f"aa-run-{index}/benchmark.json" for index in range(1, 4)
    ] + [
        RAW / "waterfall-run/benchmark.json",
        RAW / "waterfall-run/browser_metrics.json",
        RAW / "detailed-profile/live_metrics.json",
        RAW / "detailed-profile/pure_compile_metrics.json",
    ]
    write_json("raw-artifacts.json", {
        "policy": "Large/generated raw outputs remain gitignored; paths and SHA-256 digests are committed.",
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in raw_files
        ],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
