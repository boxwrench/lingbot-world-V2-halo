#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/common.sh"

bash "$ROOT_DIR/scripts/prepare_upstream.sh"
if [[ ! -d "$MODEL_DIR" ]]; then bash "$ROOT_DIR/scripts/prepare_model.sh"; fi

# 81 frames gives multiple latent chunks while remaining a practical first
# baseline. The exact 480x832, 4n+1, BF16 configuration is recorded in JSON.
run_experiment baseline-480x832 \
    --size 480*832 \
    --frames 81 \
    --chunk-size 4 \
    --local-attn-size -1 \
    --sink-size 0 \
    --seed 42 \
    --prompt "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky, while the camera glides smoothly forward." \
    --image "$UPSTREAM_DIR/examples/03/image.jpg" \
    --action-path "$UPSTREAM_DIR/examples/03" \
    --repeat 2 \
    --save-video \
    --offload-model false
