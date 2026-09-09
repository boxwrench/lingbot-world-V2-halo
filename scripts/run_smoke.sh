#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/scripts/common.sh"

bash "$ROOT_DIR/scripts/prepare_upstream.sh"
if [[ ! -d "$MODEL_DIR" ]]; then bash "$ROOT_DIR/scripts/prepare_model.sh"; fi

run_experiment smoke \
    --size 480*832 \
    --frames 21 \
    --chunk-size 4 \
    --local-attn-size 18 \
    --sink-size 6 \
    --seed 42 \
    --prompt "A small red rover moves slowly through a sunlit grassy clearing, with trees and a blue sky in the background." \
    --image "$UPSTREAM_DIR/examples/03/image.jpg" \
    --action-path "$UPSTREAM_DIR/examples/03" \
    --save-video \
    --offload-model false

