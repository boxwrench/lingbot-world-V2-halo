#!/usr/bin/env python3
"""Run the existing live viewer with a persisted TunableOp result file.

This wrapper intentionally does not alter the LingBot runner.  It enables the
PyTorch TunableOp dispatcher, loads a validator-checked result file before the
first model operation, disables online tuning, and then delegates all live
arguments to ``run_live.main``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--tunableop-results", required=True)
    parsed, live_args = wrapper.parse_known_args()

    import torch.cuda.tunable as tunable

    result_path = Path(parsed.tunableop_results).resolve()
    if not result_path.is_file():
        raise SystemExit(f"missing TunableOp result file: {result_path}")
    tunable.set_filename(str(result_path))
    tunable.enable(True)
    tunable.tuning_enable(False)
    tunable.record_untuned_enable(False)
    if not tunable.read_file(str(result_path)):
        raise SystemExit(f"TunableOp rejected result file: {result_path}")
    print(json.dumps({
        "tunableop_enabled": tunable.is_enabled(),
        "tuning_enabled": tunable.tuning_is_enabled(),
        "record_untuned": tunable.record_untuned_is_enabled(),
        "filename": tunable.get_filename(),
        "validators": tunable.get_validators(),
        "loaded_results": len(tunable.get_results()),
    }, default=str), flush=True)

    sys.argv = [sys.argv[0], *live_args]
    from run_live import main as live_main

    return int(live_main())


if __name__ == "__main__":
    raise SystemExit(main())
