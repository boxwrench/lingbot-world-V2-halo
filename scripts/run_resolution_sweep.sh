#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/common.sh"

bash "$ROOT_DIR/scripts/prepare_upstream.sh"
if [[ ! -d "$MODEL_DIR" ]]; then bash "$ROOT_DIR/scripts/prepare_model.sh"; fi

# The upstream CLI accepts a finite SIZE_CONFIGS set. Use only entries known
# to be supported by that source, and keep prompt/seed/frame/window constant.
for size in 480\*832 720\*1280; do
    run_experiment "resolution-${size//\*/x}" \
        --size "$size" \
        --frames 21 \
        --chunk-size 4 \
        --local-attn-size 18 \
        --sink-size 6 \
        --seed 42 \
        --prompt "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky, while the camera glides smoothly forward." \
        --image "$UPSTREAM_DIR/examples/03/image.jpg" \
        --action-path "$UPSTREAM_DIR/examples/03" \
        --offload-model false
done
