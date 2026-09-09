#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT_DIR=${1:-"$ROOT_DIR/results/raw/14b/provenance"}
mkdir -p "$OUT_DIR"

{
  echo "# captured $(date --iso-8601=seconds)"
  echo
  echo '## environment'
  env | sort | rg '^(CUDA|HIP|HSA|ROCR|ROCM|PYTORCH|TORCH|HF_|MIOPEN|GGML|COMFY)' || true
  echo
  echo '## host'
  uname -a
  cat /etc/os-release || true
  lscpu || true
  free -h
  grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo || true
  echo
  echo '## commands'
  command -v rocminfo && rocminfo || true
  command -v rocm-smi && rocm-smi || true
  echo
  echo '## Python'
  "$ROOT_DIR/.venv/bin/python" - <<'PY' || true
import platform
import sys
import torch
print(sys.version)
print('platform', platform.platform())
print('torch', torch.__version__)
print('hip', torch.version.hip)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    d = torch.device('cuda:0')
    print('device', torch.cuda.get_device_name(d))
    print('properties', torch.cuda.get_device_properties(d))
    print('bf16', torch.cuda.is_bf16_supported())
PY
} | tee "$OUT_DIR/report.txt"

git -C "$ROOT_DIR/.upstream/lingbot-world-v2" rev-parse HEAD >"$OUT_DIR/upstream-sha.txt" 2>/dev/null || true
cp "$ROOT_DIR/experiments/lingbot-world-v2-14b/asset-manifest.json" "$OUT_DIR/asset-manifest.json"
