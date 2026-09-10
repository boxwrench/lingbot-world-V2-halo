#!/usr/bin/env python3
"""Tune only GEMM signatures collected by an actual LingBot workload."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.cuda.tunable as tunable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--untuned", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    untuned = args.untuned.resolve()
    results = args.results.resolve()
    if not untuned.is_file():
        raise SystemExit(f"missing untuned file: {untuned}")
    results.parent.mkdir(parents=True, exist_ok=True)

    tunable.set_filename(str(results))
    tunable.enable(True)
    tunable.tuning_enable(True)
    tunable.record_untuned_enable(False)
    torch.cuda.init()
    start = time.perf_counter()
    tunable.tune_gemm_in_file(str(untuned))
    torch.cuda.synchronize()
    print(json.dumps({
        "untuned": str(untuned),
        "results": str(results),
        "elapsed_seconds": time.perf_counter() - start,
        "validators": tunable.get_validators(),
        "result_count": len(tunable.get_results()),
        "results_in_process": tunable.get_results(),
    }, default=str, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
