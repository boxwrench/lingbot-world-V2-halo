#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODEL_DIR=${MODEL_DIR:-"$ROOT_DIR/models/lingbot-world-v2-14b-community"}
RUNTIME_DIR=${RUNTIME_DIR:-"$ROOT_DIR/model-cache/lingbot-world-v2-14b-runtime"}
COMFY_DIR="$RUNTIME_DIR/ComfyUI"
COMMUNITY_DIR="$RUNTIME_DIR/community"
GGUF_DIR="$RUNTIME_DIR/ComfyUI-GGUF"
EXTRA_DIR="$RUNTIME_DIR/python-extra"
PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}

for required in \
  "$MODEL_DIR/LingBot-World-14B-causal-fast_merged-Q4_K_M.gguf" \
  "$MODEL_DIR/umt5-xxl-encoder-Q4_K_S.gguf" \
  "$MODEL_DIR/Wan2.1_VAE.pth"; do
  if [[ ! -f "$required" ]]; then
    echo "missing asset: $required" >&2
    echo "run download_q4_assets.sh first" >&2
    exit 2
  fi
done

clone_at() {
  local url=$1 dest=$2 rev=$3
  if [[ ! -d "$dest/.git" ]]; then
    mkdir -p "$(dirname "$dest")"
    git clone "$url" "$dest"
  fi
  git -C "$dest" fetch --quiet --depth 1 origin "$rev" || true
  git -C "$dest" checkout --quiet --detach "$rev"
}

clone_at https://github.com/comfyanonymous/ComfyUI.git \
  "$COMFY_DIR" be92396834f9b6e3e361cfe30e7eed693137a448
clone_at https://github.com/RealRebelAI/Rebels_LingBot-World-V2_GGUF_ComfyUI.git \
  "$COMMUNITY_DIR" f310acc1109e7af447d5c6efb94f29fe746408a1
clone_at https://github.com/city96/ComfyUI-GGUF.git \
  "$GGUF_DIR" 6ea2651e7df66d7585f6ffee804b20e92fb38b8a

mkdir -p "$COMFY_DIR/models/diffusion_models" "$COMFY_DIR/models/text_encoders" \
  "$COMFY_DIR/models/vae" "$COMFY_DIR/input/lingbot_actions"

link_file() {
  local source=$1 dest=$2
  if [[ ! -e "$dest" && ! -L "$dest" ]]; then
    ln -s "$source" "$dest"
  fi
}
link_file "$MODEL_DIR/LingBot-World-14B-causal-fast_merged-Q4_K_M.gguf" \
  "$COMFY_DIR/models/diffusion_models/LingBot-World-14B-causal-fast_merged-Q4_K_M.gguf"
link_file "$MODEL_DIR/umt5-xxl-encoder-Q4_K_S.gguf" \
  "$COMFY_DIR/models/text_encoders/umt5-xxl-encoder-Q4_K_S.gguf"
link_file "$MODEL_DIR/Wan2.1_VAE.pth" "$COMFY_DIR/models/vae/Wan2.1_VAE.pth"
link_file "$ROOT_DIR/.upstream/lingbot-world-v2/examples/03" \
  "$COMFY_DIR/input/lingbot_actions/strix-example"
# The pinned community node discovers ComfyUI-GGUF relative to its custom-node
# directory. Keep the actual clone isolated above, but expose that expected
# sibling path without copying or modifying either upstream checkout.
link_file "$GGUF_DIR" "$COMMUNITY_DIR/ComfyUI-GGUF"

mkdir -p "$EXTRA_DIR"
"$PYTHON_BIN" -m pip install --disable-pip-version-check --no-deps --upgrade --target "$EXTRA_DIR" \
  gguf torchsde trampoline pyyaml av simpleeval \
  comfy-kitchen==0.2.33 comfy-aimdo==0.5.3 >/dev/null

test -f "$COMFY_DIR/models/diffusion_models/LingBot-World-14B-causal-fast_merged-Q4_K_M.gguf"
test -f "$COMFY_DIR/models/text_encoders/umt5-xxl-encoder-Q4_K_S.gguf"
test -f "$COMFY_DIR/models/vae/Wan2.1_VAE.pth"
echo "runtime ready: comfy=$(git -C "$COMFY_DIR" rev-parse HEAD) community=$(git -C "$COMMUNITY_DIR" rev-parse HEAD) gguf=$(git -C "$GGUF_DIR" rev-parse HEAD)" >&2
