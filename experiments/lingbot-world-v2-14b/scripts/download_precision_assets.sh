#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODEL_DIR=${MODEL_DIR:-"$ROOT_DIR/models/lingbot-world-v2-14b-community"}
HF_REPO=${HF_REPO:-realrebelai/LingBot_World_V2_ComfyUI}
HF_REVISION=${HF_REVISION:-79be5d5f30d63327ac6020ced7f653cdbc301f49}

command -v hf >/dev/null || {
  echo "hf CLI is required; see download_q4_assets.sh" >&2
  exit 2
}

mkdir -p "$MODEL_DIR"
echo "Downloading Q6_K and Q8_0 DiT assets into $MODEL_DIR"
hf download "$HF_REPO" \
  --revision "$HF_REVISION" \
  --local-dir "$MODEL_DIR" \
  --include 'LingBot-World-14B-Q6_K.gguf' \
  --include 'LingBot-World-14B-Q8_0.gguf'

for name in LingBot-World-14B-Q6_K.gguf LingBot-World-14B-Q8_0.gguf; do
  test -f "$MODEL_DIR/$name" || { echo "missing $name" >&2; exit 1; }
  stat -c '%s %n' "$MODEL_DIR/$name"
done
