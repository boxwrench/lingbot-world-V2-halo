#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/common.sh"

TIMEOUT_SECONDS=${LINGBOT_LIVE_TIMEOUT_SECONDS:-900}
MAX_ACTIONS=${LINGBOT_LIVE_MAX_ACTIONS:-20}
OUTPUT_DIR=${LINGBOT_LIVE_OUTPUT_DIR:-"$ROOT_DIR/results/raw/live-taehv-384x672"}

bash "$ROOT_DIR/scripts/prepare_upstream.sh"
if [[ ! -d "$MODEL_DIR" ]]; then bash "$ROOT_DIR/scripts/prepare_model.sh"; fi

mkdir -p "$OUTPUT_DIR"
echo "Live viewer safety: press Q/ESC in the viewer, Ctrl-C in this terminal, or wait for the ${TIMEOUT_SECONDS}s hard timeout."

exec timeout --foreground --signal=INT --kill-after=20s "${TIMEOUT_SECONDS}s" \
  env PYTHONUNBUFFERED=1 "$PYTHON" "$ROOT_DIR/scripts/run_live.py" \
    --upstream-dir "$UPSTREAM_DIR" \
    --model-dir "$MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --size '480*832' \
    --max-area-pixels 264192 \
    --frames 81 \
    --max-actions "$MAX_ACTIONS" \
    --denoise-schedule 3-drop-957 \
    --local-attn-size 18 \
    --sink-size 6 \
    --seed 42 \
    --prompt "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky, while the camera glides smoothly forward." \
    --image "$UPSTREAM_DIR/examples/03/image.jpg" \
    --action-path "$UPSTREAM_DIR/examples/03" \
    --vae-attention-backend math \
    --vae-dtype fp16 \
    --display-decoder taehv \
    --taehv-dir "$ROOT_DIR/.upstream/taehv" \
    --defer-clean-kv \
    --max-seconds "$TIMEOUT_SECONDS" \
    "$@"
