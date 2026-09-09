#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-"$ROOT_DIR/results/raw/provenance-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$OUT"

uname -a > "$OUT/uname.txt"
cat /etc/os-release > "$OUT/os-release.txt"
lscpu > "$OUT/lscpu.txt"
free -h > "$OUT/free.txt"
lspci -nnk > "$OUT/lspci.txt" 2>&1 || true
lsmod > "$OUT/lsmod.txt" 2>&1 || true
env | sort > "$OUT/environment.txt"
env | sort | rg -i '^(HIP|ROCR|HSA|PYTORCH|TORCH|CUDA|ROCM|HF_|HUGGING|PATH)=' > "$OUT/relevant-environment.txt" || true

rocminfo > "$OUT/rocminfo.txt" 2>&1 || true
rocm-smi > "$OUT/rocm-smi.txt" 2>&1 || true
amd-smi list > "$OUT/amd-smi-list.txt" 2>&1 || true

{
    echo "--- DRM/KFD sysfs ---"
    for x in /sys/class/drm/card*/device/{uevent,mem_info_vram_total,mem_info_vram_used,mem_info_gtt_total,mem_info_gtt_used}; do
        if [[ -r "$x" ]]; then echo "[$x]"; cat "$x"; fi
    done
    echo "--- devices ---"
    ls -l /dev/kfd /dev/dri/renderD* 2>&1 || true
} > "$OUT/uma-sysfs.txt"

dmidecode -t system -t bios > "$OUT/dmidecode.txt" 2>&1 || true

PYTHON=${LINGBOT_PYTHON:-"$ROOT_DIR/.venv/bin/python"}
if [[ ! -x "$PYTHON" ]]; then PYTHON=/home/keith/ciru-ling-runtime/.venv/bin/python; fi
"$PYTHON" - <<'PY' > "$OUT/pytorch.txt" 2>&1 || true
import importlib.metadata as md
import os
import platform
import sys
import torch

print("python", sys.version)
print("executable", sys.executable)
print("platform", platform.platform())
for name in ("torch", "torchvision", "transformers", "tokenizers", "accelerate", "numpy", "scipy", "pillow", "diffusers"):
    try: print(name, md.version(name))
    except Exception: print(name, "MISSING")
print("torch_version", torch.__version__)
print("hip_version", torch.version.hip)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("device", i, torch.cuda.get_device_name(i))
    print("properties", torch.cuda.get_device_properties(i))
    print("bf16_supported", torch.cuda.is_bf16_supported())
    try: print("mem_get_info", torch.cuda.mem_get_info(i))
    except Exception as e: print("mem_get_info_error", type(e).__name__, e)
try:
    x = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    print("bf16_matmul", y.dtype, y.shape, bool(torch.isfinite(y).all()))
except Exception as e:
    print("bf16_matmul_error", type(e).__name__, e)
PY

if [[ -d "$ROOT_DIR/.upstream/lingbot-world-v2/.git" ]]; then
    git -C "$ROOT_DIR/.upstream/lingbot-world-v2" rev-parse HEAD > "$OUT/upstream-sha.txt"
fi

echo "provenance written to $OUT"

