#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/common.sh"

bash "$ROOT_DIR/scripts/prepare_upstream.sh"
if [[ ! -d "$MODEL_DIR" ]]; then bash "$ROOT_DIR/scripts/prepare_model.sh"; fi

for window in 6 12 18 24 36; do
    run_experiment "window-${window}" \
        --size 480*832 \
        --frames 81 \
        --chunk-size 4 \
        --local-attn-size "$window" \
        --sink-size 6 \
        --seed 42 \
        --prompt "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under an brilliant blue sky, while the camera glides smoothly forward." \
        --image "$UPSTREAM_DIR/examples/03/image.jpg" \
        --action-path "$UPSTREAM_DIR/examples/03" \
        --offload-model false
done

