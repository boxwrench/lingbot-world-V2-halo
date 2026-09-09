#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM_DIR=${LINGBOT_UPSTREAM_DIR:-"$ROOT_DIR/.upstream/lingbot-world-v2"}
MODEL_DIR=${LINGBOT_MODEL_DIR:-"$ROOT_DIR/models/lingbot-world-v2-1.3b-causal-fast"}
PYTHON=${LINGBOT_PYTHON:-"$ROOT_DIR/.venv/bin/python"}

export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx1151}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

die() { echo "error: $*" >&2; exit 1; }

require_file() { test -f "$1" || die "missing file: $1"; }

require_runtime() {
    require_file "$PYTHON"
    test -d "$UPSTREAM_DIR" || die "upstream checkout missing; run scripts/prepare_upstream.sh"
    test -d "$MODEL_DIR" || die "model missing; run scripts/prepare_model.sh"
}

run_python() {
    require_runtime
    PYTHONPATH="$UPSTREAM_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$@"
}

run_experiment() {
    require_runtime
    local result_dir=$1
    shift
    mkdir -p "$ROOT_DIR/results/raw/$result_dir"
    run_python "$ROOT_DIR/scripts/run_experiment.py" \
        --upstream-dir "$UPSTREAM_DIR" \
        --model-dir "$MODEL_DIR" \
        --output-dir "$ROOT_DIR/results/raw/$result_dir" \
        "$@"
}

