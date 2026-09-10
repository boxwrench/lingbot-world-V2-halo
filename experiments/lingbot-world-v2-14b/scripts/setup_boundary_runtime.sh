#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
RUNTIME_DIR=${RUNTIME_DIR:-"$ROOT_DIR/model-cache/lingbot-world-v2-14b-boundary-runtime"}

RUNTIME_DIR="$RUNTIME_DIR" bash "$ROOT_DIR/experiments/lingbot-world-v2-14b/scripts/setup_runtime.sh"

COMMUNITY_DIR="$RUNTIME_DIR/community"
PHASE_PATCH="$ROOT_DIR/experiments/lingbot-world-v2-14b/patches/14b-phase-memory-hooks.patch"
SAMPLER_PATCH="$ROOT_DIR/experiments/lingbot-world-v2-14b/patches/14b-phase-memory-sampler.patch"
IMAGE2VIDEO="$COMMUNITY_DIR/ComfyUI_Rebels_LingBotWorld/wan/image2video.py"
SAMPLER="$COMMUNITY_DIR/ComfyUI_Rebels_LingBotWorld/lbworld_nodes.py"

if ! rg -q '_experiment_phase_hook' "$IMAGE2VIDEO"; then
  patch --directory "$COMMUNITY_DIR" --strip=1 --forward < "$PHASE_PATCH" >/dev/null
fi
if ! rg -q '_experiment_before_vae_cleanup' "$SAMPLER"; then
  patch --directory "$COMMUNITY_DIR" --strip=1 --forward < "$SAMPLER_PATCH" >/dev/null
fi

echo "boundary runtime ready: $(git -C "$COMMUNITY_DIR" rev-parse HEAD)" >&2
