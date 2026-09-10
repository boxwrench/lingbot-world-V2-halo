#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/common.sh"

bash "$ROOT_DIR/scripts/prepare_upstream.sh"
if [[ ! -d "$MODEL_DIR" ]]; then bash "$ROOT_DIR/scripts/prepare_model.sh"; fi

OUTPUT_DIR=${LINGBOT_RESPONSE_OUTPUT_DIR:-"$ROOT_DIR/results/raw/action-response-12"}
mkdir -p "$OUTPUT_DIR"
LIVE_PYTHONPATH="$UPSTREAM_DIR:$ROOT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec env PYTHONUNBUFFERED=1 PYTHONPATH="$LIVE_PYTHONPATH" \
  "$PYTHON" "$ROOT_DIR/scripts/action_response_probe.py" \
  --upstream-dir "$UPSTREAM_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --taehv-dir "$ROOT_DIR/.upstream/taehv" \
  --image "$UPSTREAM_DIR/examples/03/image.jpg" \
  --action-path "$UPSTREAM_DIR/examples/03" \
  --local-attn-size 12 \
  --sink-size 6 \
  "$@"
