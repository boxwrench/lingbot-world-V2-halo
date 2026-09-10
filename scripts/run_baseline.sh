#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/common.sh"

bash "$ROOT_DIR/scripts/prepare_upstream.sh"
if [[ ! -d "$MODEL_DIR" ]]; then bash "$ROOT_DIR/scripts/prepare_model.sh"; fi

# Keep this first baseline short enough for a rapid cold/warm cycle while
# preserving 21 valid output frames: the upstream pipeline's chunk size of 3
# makes 21 (6 latent frames) an exact causal request.
run_experiment baseline-480x832-21f \
    --size 480*832 \
    --frames 21 \
    --chunk-size 3 \
    --local-attn-size 18 \
    --sink-size 6 \
    --seed 42 \
    --prompt "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky, while the camera glides smoothly forward." \
    --image "$UPSTREAM_DIR/examples/03/image.jpg" \
    --action-path "$UPSTREAM_DIR/examples/03" \
    --repeat 2 \
    --save-video \
    --offload-model false
