#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/common.sh"

HOST=${LINGBOT_BROWSER_HOST:-127.0.0.1}
PORT=${LINGBOT_BROWSER_PORT:-8765}
TIMEOUT_SECONDS=${LINGBOT_BROWSER_TIMEOUT_SECONDS:-3600}
MAX_ACTIONS=${LINGBOT_BROWSER_MAX_ACTIONS:-40}
OUTPUT_DIR=${LINGBOT_BROWSER_OUTPUT_DIR:-"$ROOT_DIR/results/raw/browser-serving-20260910"}

if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" ]]; then
    die "RC1 browser server is localhost-only; set LINGBOT_BROWSER_HOST to 127.0.0.1 or localhost"
fi

bash "$ROOT_DIR/scripts/prepare_upstream.sh"
if [[ ! -d "$MODEL_DIR" ]]; then bash "$ROOT_DIR/scripts/prepare_model.sh"; fi
mkdir -p "$OUTPUT_DIR"

echo "LingBot browser safety: Ctrl-C or Q/ESC stops the local model owner."
echo "Opening http://$HOST:$PORT/ after the server starts; hard timeout ${TIMEOUT_SECONDS}s."

LIVE_PYTHONPATH="$UPSTREAM_DIR:$ROOT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
OPEN_ARGS=()
if [[ "${LINGBOT_BROWSER_OPEN_BROWSER:-1}" == "0" ]]; then
    OPEN_ARGS+=(--no-open-browser)
fi

exec timeout --foreground --signal=INT --kill-after=20s "${TIMEOUT_SECONDS}s" \
  env PYTHONUNBUFFERED=1 PYTHONPATH="$LIVE_PYTHONPATH" \
  "$PYTHON" "$ROOT_DIR/scripts/run_browser.py" \
    --upstream-dir "$UPSTREAM_DIR" \
    --model-dir "$MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --size '480*832' \
    --max-area-pixels 264192 \
    --frames "$((4 * (MAX_ACTIONS + 1) + 1))" \
    --max-actions "$MAX_ACTIONS" \
    --max-seconds "$TIMEOUT_SECONDS" \
    --denoise-schedule 3-drop-957 \
    --local-attn-size 12 \
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
    --tunableop-results "$ROOT_DIR/docs/artifacts/tunable-op-20260910/tunableop_results.csv" \
    --pure-compile-block-count 30 \
    --prewarm-pure-compile \
    --host "$HOST" \
    --port "$PORT" \
    "${OPEN_ARGS[@]}" \
    "$@"
