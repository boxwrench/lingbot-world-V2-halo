#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
RUNTIME_DIR=${RUNTIME_DIR:-"$ROOT_DIR/model-cache/lingbot-world-v2-14b-boundary-runtime"}
MODE=${1:-baseline}
shift || true

case "$MODE" in
  baseline)
    OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/results/raw/14b/q4-480x832-18plus6-phase-baseline"}
    EXTRA_ARGS=(--phase-memory)
    ;;
  cleanup)
    OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/results/raw/14b/q4-480x832-18plus6-phase-cleanup"}
    EXTRA_ARGS=(--phase-memory --cleanup-before-vae)
    ;;
  *)
    echo "usage: $0 [baseline|cleanup] [additional runner args...]" >&2
    exit 2
    ;;
esac

RUNTIME_DIR="$RUNTIME_DIR" bash \
  "$ROOT_DIR/experiments/lingbot-world-v2-14b/scripts/setup_boundary_runtime.sh"

RUNTIME_DIR="$RUNTIME_DIR" OUTPUT_DIR="$OUTPUT_DIR" bash \
  "$ROOT_DIR/experiments/lingbot-world-v2-14b/scripts/run_q4_baseline.sh" \
  --local-attn-size 18 --sink-size 6 --repeat 1 "${EXTRA_ARGS[@]}" "$@"
