#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODEL_DIR=${MODEL_DIR:-"$ROOT_DIR/models/lingbot-world-v2-14b-community"}
HF_REPO=${HF_REPO:-realrebelai/LingBot_World_V2_ComfyUI}
HF_REVISION=${HF_REVISION:-79be5d5f30d63327ac6020ced7f653cdbc301f49}

command -v hf >/dev/null || {
  echo "hf CLI is required; see the repository's Hugging Face setup notes." >&2
  exit 2
}

mkdir -p "$MODEL_DIR"
echo "Downloading the Q4 baseline assets into $MODEL_DIR"
hf download "$HF_REPO" \
  --revision "$HF_REVISION" \
  --local-dir "$MODEL_DIR" \
  --include 'LingBot-World-14B-causal-fast_merged-Q4_K_M.gguf' \
  --include 'umt5-xxl-encoder-Q4_K_S.gguf' \
  --include 'Wan2.1_VAE.pth'

echo
echo "Expected files:"
for name in \
  LingBot-World-14B-causal-fast_merged-Q4_K_M.gguf \
  umt5-xxl-encoder-Q4_K_S.gguf \
  Wan2.1_VAE.pth; do
  test -f "$MODEL_DIR/$name" || { echo "missing $name" >&2; exit 1; }
  stat -c '%s %n' "$MODEL_DIR/$name"
done
