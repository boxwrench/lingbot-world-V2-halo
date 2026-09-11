#!/usr/bin/env python3
"""Opt-in live probe compiling only state-free non-GEMM tensor islands.

The normal runner and accepted defaults remain unchanged.  This launcher keeps
attention, all Linear modules, cache management, and rollover logic eager,
then delegates generation to the existing live path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _option_value(args: list[str], option: str) -> str | None:
    try:
        return args[args.index(option) + 1]
    except (ValueError, IndexError):
        return None


def main() -> int:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--pure-compile-block-count", type=int, required=True)
    wrapper.add_argument("--tunableop-results", required=True)
    wrapper.add_argument(
        "--prewarm-pure-compile",
        action="store_true",
        help="prewarm the installed live helper instances before runtime READY",
    )
    parsed, live_args = wrapper.parse_known_args()
    if parsed.pure_compile_block_count < 1:
        raise SystemExit("--pure-compile-block-count must be positive")

    process_start = time.perf_counter()
    startup_timestamps: dict[str, float] = {"process_start_ms": 0.0}

    def mark(name: str) -> None:
        startup_timestamps[name] = (time.perf_counter() - process_start) * 1000.0

    import torch
    import torch._dynamo
    import torch._dynamo.utils
    import torch.cuda.tunable as tunable
    from wan.image2video import WanI2VCausal
    from pure_compile_helpers import (
        install_on_blocks,
        prewarm_pure_tensor_islands,
        restore_blocks,
    )

    result_path = Path(parsed.tunableop_results).resolve()
    if not result_path.is_file():
        raise SystemExit(f"missing TunableOp result file: {result_path}")
    tunable.set_filename(str(result_path))
    tunable.enable(True)
    tunable.tuning_enable(False)
    tunable.record_untuned_enable(False)
    if not tunable.read_file(str(result_path)):
        raise SystemExit(f"TunableOp rejected result file: {result_path}")
    mark("tunableop_loaded")
    tunable_info = {
        "enabled": tunable.is_enabled(),
        "tuning_enabled": tunable.tuning_is_enabled(),
        "record_untuned": tunable.record_untuned_is_enabled(),
        "filename": tunable.get_filename(),
        "validators": _jsonable(tunable.get_validators()),
        "loaded_results": len(tunable.get_results()),
    }
    print(json.dumps({"tunableop": tunable_info}, default=str), flush=True)

    compile_info: dict[str, object] = {
        "requested_block_count": int(parsed.pure_compile_block_count),
        "mode": "default",
        "backend": "inductor",
        "fullgraph": True,
        "target": "state-free tensor islands only",
        "compiled_block_indices": [],
        "compiled_helpers": [],
        "tunableop": tunable_info,
        "compile_setup_seconds": None,
        "prewarm_requested": bool(parsed.prewarm_pure_compile),
        "prewarm": None,
        "startup_timestamps": startup_timestamps,
    }
    installation: dict[str, object] | None = None
    original_init = WanI2VCausal.__init__

    def patched_init(self, *args, **kwargs):
        mark("model_initialization_start")
        original_init(self, *args, **kwargs)
        mark("model_initialization_end")

        setup_t0 = time.perf_counter()
        nonlocal installation
        mark("pure_helper_install_start")
        installation = install_on_blocks(self.model, parsed.pure_compile_block_count)
        mark("pure_helper_install_end")
        compile_info["compile_setup_seconds"] = time.perf_counter() - setup_t0
        compile_info["compiled_block_indices"] = installation["compiled_block_indices"]
        compile_info["total_model_blocks"] = installation["total_model_blocks"]
        compile_info["compiled_helpers"] = list(installation["islands"].helper_names)
        compile_info["compiled_region_contract"] = {
            "eager": [
                "self-attention",
                "cross-attention",
                "all Linear/GEMM modules",
                "KV reads/writes",
                "rollover/index state",
                "cache dictionaries",
            ],
            "compiled": list(installation["islands"].helper_names),
        }
        if parsed.prewarm_pure_compile:
            mark("explicit_prewarm_start")
            prewarm = prewarm_pure_tensor_islands(installation["islands"], self.device)
            mark("explicit_prewarm_end")
            compile_info["prewarm"] = prewarm

    WanI2VCausal.__init__ = patched_init
    sys.argv = [sys.argv[0], *live_args]
    import run_live as live_module

    live_module.STARTUP_TIMESTAMPS = startup_timestamps
    live_module.STARTUP_PROCESS_START = process_start

    run_status = 0
    try:
        run_status = int(live_module.main())
    finally:
        if installation is not None:
            restore_blocks(installation)
        WanI2VCausal.__init__ = original_init
        compile_info["dynamo_counters"] = _jsonable(dict(torch._dynamo.utils.counters))
        compile_info["dynamo_compile_times"] = _jsonable(torch._dynamo.utils.compile_times())
        output_dir = _option_value(live_args, "--output-dir")
        if output_dir:
            path = Path(output_dir).resolve() / "pure_compile_metrics.json"
            path.write_text(json.dumps(compile_info, indent=2) + "\n")
            live_metrics = path.parent / "live_metrics.json"
            if live_metrics.is_file():
                report = json.loads(live_metrics.read_text())
                report["pure_compiler"] = compile_info
                live_metrics.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return run_status


if __name__ == "__main__":
    raise SystemExit(main())
