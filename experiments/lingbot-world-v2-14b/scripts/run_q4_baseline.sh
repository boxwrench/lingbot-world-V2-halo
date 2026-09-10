#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODEL_DIR=${MODEL_DIR:-"$ROOT_DIR/models/lingbot-world-v2-14b-community"}
RUNTIME_DIR=${RUNTIME_DIR:-"$ROOT_DIR/model-cache/lingbot-world-v2-14b-runtime"}
COMFY_DIR=${COMFY_DIR:-"$RUNTIME_DIR/ComfyUI"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/results/raw/14b/q4-480x832-6plus2"}

bash "$(dirname "${BASH_SOURCE[0]}")/setup_runtime.sh"

export PYTHONPATH="$RUNTIME_DIR/python-extra:$COMFY_DIR:$RUNTIME_DIR/community:$RUNTIME_DIR/ComfyUI-GGUF:$ROOT_DIR/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export COMFY_DIR MODEL_DIR RUNTIME_DIR
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

exec "$ROOT_DIR/.venv/bin/python" \
  "$ROOT_DIR/experiments/lingbot-world-v2-14b/scripts/run_q4_baseline.py" \
  --comfy-dir "$COMFY_DIR" \
  --community-dir "$RUNTIME_DIR/community" \
  --gguf-dir "$RUNTIME_DIR/ComfyUI-GGUF" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
