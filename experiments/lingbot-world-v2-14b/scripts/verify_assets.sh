#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODEL_DIR=${MODEL_DIR:-"$ROOT_DIR/models/lingbot-world-v2-14b-community"}

declare -A EXPECTED=(
  [LingBot-World-14B-causal-fast_merged-Q4_K_M.gguf]=71e0ae7d4d245288713c1aaf8132018d977605831f00c4ab2fc56939aac394d1
  [umt5-xxl-encoder-Q4_K_S.gguf]=4a3176f32fd70c0a335b4419fcbf8c86cc875e23498c0fc06f5b4aa0930889e0
  [Wan2.1_VAE.pth]=38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981
)

for name in "${!EXPECTED[@]}"; do
  path="$MODEL_DIR/$name"
  test -f "$path" || { echo "missing: $path" >&2; exit 1; }
  actual=$(sha256sum "$path" | awk '{print $1}')
  printf '%s  %s\n' "$actual" "$name"
  test "$actual" = "${EXPECTED[$name]}" || {
    echo "hash mismatch: $name" >&2
    exit 1
  }
done
