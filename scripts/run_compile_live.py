#!/usr/bin/env python3
"""Opt-in regional Inductor probe for the existing LingBot live runner.

The normal runner remains eager.  This launcher replaces only the requested
prefix of the real repeated ``CausalWanAttentionBlock`` list with
``torch.compile`` wrappers, then delegates all generation and state handling to
``run_live.main``.  It is deliberately an experiment launcher, not a new
inference implementation.
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
    wrapper.add_argument(
        "--compile-block-count",
        type=int,
        required=True,
        help="number of leading real transformer blocks to compile",
    )
    wrapper.add_argument(
        "--tunableop-results",
        required=True,
        help="validator-matched persisted TunableOp result CSV",
    )
    parsed, live_args = wrapper.parse_known_args()
    if parsed.compile_block_count < 1:
        raise SystemExit("--compile-block-count must be positive")

    import torch
    import torch.cuda.tunable as tunable
    import torch._dynamo
    import torch._dynamo.utils
    from wan.image2video import WanI2VCausal

    result_path = Path(parsed.tunableop_results).resolve()
    if not result_path.is_file():
        raise SystemExit(f"missing TunableOp result file: {result_path}")
    tunable.set_filename(str(result_path))
    tunable.enable(True)
    tunable.tuning_enable(False)
    tunable.record_untuned_enable(False)
    if not tunable.read_file(str(result_path)):
        raise SystemExit(f"TunableOp rejected result file: {result_path}")
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
        "requested_block_count": int(parsed.compile_block_count),
        "mode": "default",
        "backend": "inductor",
        "fullgraph": False,
        "compile_constructed": False,
        "compiled_block_indices": [],
        "compile_wrapper_setup_seconds": None,
        "tunableop": tunable_info,
    }
    original_init = WanI2VCausal.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        blocks = list(self.model.blocks)
        count = min(parsed.compile_block_count, len(blocks))
        setup_t0 = time.perf_counter()
        compiled = []
        for index, block in enumerate(blocks):
            if index < count:
                block = torch.compile(
                    block,
                    backend="inductor",
                    mode="default",
                    fullgraph=False,
                    name=f"lingbot_causal_block_{index}",
                )
                compile_info["compiled_block_indices"].append(index)
            compiled.append(block)
        self.model.blocks = torch.nn.ModuleList(compiled)
        compile_info["compile_wrapper_setup_seconds"] = time.perf_counter() - setup_t0
        compile_info["compile_constructed"] = True
        compile_info["total_model_blocks"] = len(blocks)

    WanI2VCausal.__init__ = patched_init
    sys.argv = [sys.argv[0], *live_args]
    from run_live import main as live_main

    run_status = 0
    try:
        run_status = int(live_main())
    finally:
        # Restore the class for callers embedding this launcher in a process.
        WanI2VCausal.__init__ = original_init
        compile_info["dynamo_counters"] = _jsonable(dict(torch._dynamo.utils.counters))
        compile_info["dynamo_compile_times"] = _jsonable(torch._dynamo.utils.compile_times())
        output_dir = _option_value(live_args, "--output-dir")
        if output_dir:
            path = Path(output_dir).resolve() / "compile_metrics.json"
            path.write_text(json.dumps(compile_info, indent=2) + "\n")
            live_metrics = path.parent / "live_metrics.json"
            if live_metrics.is_file():
                report = json.loads(live_metrics.read_text())
                report["compiler"] = compile_info
                live_metrics.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return run_status


if __name__ == "__main__":
    raise SystemExit(main())
