#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/common.sh"

bash "$ROOT_DIR/scripts/prepare_upstream.sh"
if [[ ! -d "$MODEL_DIR" ]]; then bash "$ROOT_DIR/scripts/prepare_model.sh"; fi

# One latent frame per causal action.  81 requested frames produce 21 latent
# chunks, enough to fill and roll the 18-frame attention window while keeping
# the prompt/image/actions identical to the accepted baseline.
run_experiment interactive-chunk1-480x832-81f \
    --size 480*832 \
    --frames 81 \
    --chunk-size 1 \
    --local-attn-size 18 \
    --sink-size 6 \
    --seed 42 \
    --prompt "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky, while the camera glides smoothly forward." \
    --image "$UPSTREAM_DIR/examples/03/image.jpg" \
    --action-path "$UPSTREAM_DIR/examples/03" \
    --repeat 2 \
    --save-video \
    --offload-model false
